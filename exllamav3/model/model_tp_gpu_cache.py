import os, json, pickle, mmap, time, torch, hashlib, struct
from enum import Enum
from pathlib import Path
from .model_tp_cuda import cuda_load_file_direct

TP_CACHE_ENV_VAR                   = "EXL3_TP_CACHE_ENABLE"
TP_CACHE_BASE_PATH                 = ".tensorsplit"
TP_CACHE_MAGIC                     = "TPC "
TP_CACHE_FOOTER_SIZE               = 4 + 16 # magic + 16-byte md5 hash

CUDA_BLOCK_SIZE                    = 32
CACHE_LOAD_CHUNK_SIZE              = 64 * 1024 * 1024

class TPCacheManagerMode(Enum):
    DISABLED = 0
    CAPTURING = 1
    SERVING = 2    

class TPCacheManager:
    _instance = None

    @staticmethod
    def enabled() -> bool:
        return os.environ.get(TP_CACHE_ENV_VAR) == "1"    

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
    
            cls._instance._initialized = False
            cls._instance._tensors = {}            
            cls._instance._mode = TPCacheManagerMode.DISABLED                            
        return cls._instance

    def __init__(self, directory = None):
        if self._initialized or directory is None:
            return

        self.directory = directory
        self.base_path = os.path.join(self.directory, TP_CACHE_BASE_PATH)
        self.manifest_path = os.path.join(self.base_path, "manifest.json")
        self.modules_path = os.path.join(self.base_path, "modules.pkl")     

        self._tensors = None
        self._modules = None

    def _get_tensor_dump_path(self, device: int, temp: bool = False):
        postfix = ".tmp" if temp == True else ""
        return os.path.join(self.base_path, f"cuda{device}.bin{postfix}")

    def _get_tensor_list_path(self, device: int):            
        return os.path.join(self.base_path, f"cuda{device}.manifest.json")

    def _write_aligned(self, file, data, length: int) -> int:
        offset = file.tell()
        pad = (CUDA_BLOCK_SIZE - 
                (length % CUDA_BLOCK_SIZE)) % CUDA_BLOCK_SIZE
            
        file.write(data)
        file.write(b'\x00' * pad)
        return offset
    
    def set_mode(self, mode: TPCacheManagerMode):                
        self._mode = mode

        if mode == TPCacheManagerMode.CAPTURING:
            if os.path.exists(self.base_path):
                folder = Path(self.base_path)                        
                for file in folder.iterdir():
                    os.remove(file)
            else:                
                os.mkdir(self.base_path)                    
                
    def valid(self, plan: list) -> bool:
        # 1. Check if cache folder exists at all
        if not os.path.exists(self.base_path):
            return False        

        # 2. Check if manifest and modules files exist:
        if not os.path.exists(self.manifest_path) or not os.path.exists(self.modules_path):
            return False

        # 3. Validate plan hash
        plan_hash = hashlib.md5(json.dumps(plan).encode())
        with open(self.manifest_path) as f:
            manifest = json.load(f)
            if manifest["plan"] != plan_hash.hexdigest():
                print(f"Cache does not match current split plan, will invalidate.")
                return False            

        # 4. Check if device dumps exist and have proper footer:
        for device, subplan in enumerate(plan):
            path = self._get_tensor_dump_path(device)
            if not os.path.exists(path):
                return False        

            with open(path, "rb") as f:
                f.seek(os.path.getsize(path) - TP_CACHE_FOOTER_SIZE)
                footer = f.read(TP_CACHE_FOOTER_SIZE)
                if footer[:len(TP_CACHE_MAGIC)] != TP_CACHE_MAGIC.encode('ascii') or \
                   footer[len(TP_CACHE_MAGIC):TP_CACHE_FOOTER_SIZE] != plan_hash.digest():
                    print(f"Cache dump file is corrupt on mismatches manifest, will invaliidate.")
                    return False

        return True

    def capture(self, device: int, key: str, tensor: torch.Tensor):
        if self._mode != TPCacheManagerMode.CAPTURING:
            return        

        if key is None:
            raise KeyError("Key cannot be none when capturing gpu cache")
        
        t = tensor.detach().cpu().contiguous()       
        flat = t.view(-1).view(torch.uint8)         
        data = flat.numpy().data          

        dump_path = self._get_tensor_dump_path(device, temp=True)
        with open(dump_path, "ab") as f_dump:
            offset = self._write_aligned(f_dump, data, tensor.nbytes)

        tensor = {
            "key": key,
            "offset": str(offset),
            "nbytes": str(tensor.nbytes),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape)            
        }        

        list_path = self._get_tensor_list_path(device)
        with open(list_path, "a") as f_list:
            f_list.write(json.dumps(tensor))
            f_list.write("\n")   

    def finalize(self, active_devices: list, exported_modules: dict, plan: list):
        if self._mode != TPCacheManagerMode.CAPTURING:
            return

        tensors = {}    

        # 1. Combine tensor lists into a manifest and move unique tensors        
        plan_hash = hashlib.md5(json.dumps(plan).encode())

        for device in active_devices:
            tensors[device] = {}

            list_path = self._get_tensor_list_path(device)
            with open(list_path, "r") as f_list:
                tensors_list = f_list.readlines()

            source_path = self._get_tensor_dump_path(device, temp=True)
            dest_path = self._get_tensor_dump_path(device, temp=False)

            def actual_size(nbytes: int):
                return ((nbytes + CUDA_BLOCK_SIZE - 1)
                    // CUDA_BLOCK_SIZE) * CUDA_BLOCK_SIZE
            
            for tensor in tensors_list:
                tensor = json.loads(tensor)
                key = tensor.pop("key")

                if key not in tensors[device]:                                                        
                    tensors[device][key] = tensor                                

            with open(source_path, "a+b") as f:
                total_size = 0
                with mmap.mmap(f.fileno(), length=os.path.getsize(source_path), access=mmap.ACCESS_WRITE) as dst:
                    with memoryview(dst) as src:
                        dst_offset = 0                    
                        for tensor in sorted(tensors[device].values(), key=lambda x: int(x["offset"])):                        
                            src_offset = int(tensor["offset"])                 
                            
                            size = actual_size(int(tensor["nbytes"]))
                            total_size += size
                            
                            if src_offset != dst_offset:
                                tensor["offset"] = str(dst_offset)                        
                                data = src[src_offset:src_offset + size].tobytes()
                                
                                dst.seek(dst_offset)
                                dst.write(data)

                            dst_offset += size  

                
                f.write(TP_CACHE_MAGIC.encode('ascii'))                
                f.write(plan_hash.digest())

                f.truncate(total_size + TP_CACHE_FOOTER_SIZE)
                        
            os.remove(list_path)         
            os.rename(source_path, dest_path)

        manifest = {
            "plan": plan_hash.hexdigest(),
            "tensors": tensors
        }        
        with open(self.manifest_path, "w") as f_manifest:
            json.dump(manifest, f_manifest)

        # 2. Export modules
        with open(self.modules_path, "wb") as f_modules:
            pickle.dump(exported_modules, f_modules)        

    def init(self, device: int):
        if self._mode != TPCacheManagerMode.SERVING:
            return

        # 1. Load tensor manifest
        with open(self.manifest_path, "r") as f_manifest:
            self._manifest = json.load(f_manifest)

        # 2. Prepare dump descriptor
        path = self._get_tensor_dump_path(device)
            
        dump_size = os.path.getsize(path)                                            
        
        gpu_buf = torch.empty(dump_size, dtype=torch.uint8, device=device)              
        read_bytes = cuda_load_file_direct(gpu_buf.data_ptr(), path, dump_size)

        if read_bytes <= 0: # Fallback
            print("Warning: cuda_load_file_direct failed, using fallback path (slower)")
            dump = open(path, "rb")        
            dump_mm = mmap.mmap(dump.fileno(), length=0, access=mmap.ACCESS_READ)
            dump_mm_mv = memoryview(dump_mm)            
            cpu_buffer = torch.empty(min(CACHE_LOAD_CHUNK_SIZE, dump_size), dtype=torch.uint8, pin_memory=True)
            cpu_buffer_mv = memoryview(cpu_buffer.numpy())
        
            offset = 0        
            while offset < dump_size:
                chunk_size = min(CACHE_LOAD_CHUNK_SIZE, dump_size - offset)
                start_time = time.perf_counter()
                cpu_buffer_mv[:chunk_size] = dump_mm_mv[offset:(offset + chunk_size)]
                gpu_buf[offset:(offset + chunk_size)].copy_(cpu_buffer[:chunk_size], non_blocking=True)
                offset += chunk_size                    
            
            cpu_buffer_mv.release()
            dump_mm_mv.release()
            dump_mm.close()
            dump.close()

        self._tensors = {}
        for key, tensor in self._manifest["tensors"][str(device)].items():
            dtype  = getattr(torch, tensor["dtype"].replace("torch.", ""))
            shape  = tuple(tensor["shape"])
            nbytes = int(tensor["nbytes"])
            start  = int(tensor["offset"])
            end    = start + nbytes

            if start < 0 or end > dump_size:
                raise ValueError(
                    f"Corrupt device dump: tensor {key!r} offset [{start}:{end}] is out of bounds; "
                    f"delete the .tensorsplit cache and reload"
                )
                        
            self._tensors[key] = gpu_buf[start:end].view(dtype).view(shape)

    def get_module(self, key: str) -> object:
        if self._mode != TPCacheManagerMode.SERVING:
            return None

        if self._modules == None:
            with open(self.modules_path, "rb") as f_modules:
                self._modules = pickle.load(f_modules)

        return self._modules.get(key)

    def get_tensor(self, device: int, key: str, peek: bool = False) -> torch.Tensor:
        if self._mode != TPCacheManagerMode.SERVING:
            return None

        if key is None or not key in self._tensors:
            if peek: return None
            raise KeyError(f"Key {key} not found in gpu cache for current device.")        
        
        return self._tensors[key]

    