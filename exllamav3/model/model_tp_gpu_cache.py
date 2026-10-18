import os
from enum import Enum
import torch
from pathlib import Path
import json, pickle
import mmap

GPU_CACHE_ENV_VAR                   = "EXL3_TP_CACHE_ENABLE"
GPU_CACHE_BASE_PATH                 = ".tensorsplit"
GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE    = 32
CPU_CACHE_LOAD_CHUNK_SIZE_MB        = 16

class GpuCacheManagerMode(Enum):
    DISABLED = 0
    CAPTURING = 1
    SERVING = 2    

class GpuCacheManager:
    _instance = None

    @staticmethod
    def enabled() -> bool:
        return os.environ.get(GPU_CACHE_ENV_VAR) == "1"    

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
    
            cls._instance._initialized = False
            cls._instance._tensors = {}            
            cls._instance._mode = GpuCacheManagerMode.DISABLED                        
    
        return cls._instance

    def __init__(self, directory = None):
        if self._initialized or directory is None:
            return

        self.directory = directory
        self.base_path = os.path.join(self.directory, GPU_CACHE_BASE_PATH)
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
        pad = (GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE - 
                (length % GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE)) % GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE
            
        file.write(data)
        file.write(b'\x00' * pad)
        return offset
    
    def set_mode(self, mode: GpuCacheManagerMode):                
        if mode == GpuCacheManagerMode.CAPTURING:
            if os.path.exists(self.base_path):
                folder = Path(self.base_path)                        
                for file in folder.iterdir():
                    os.remove(file)
            else:                
                os.mkdir(self.base_path)

        if self._mode == GpuCacheManagerMode.SERVING and mode == GpuCacheManagerMode.DISABLED:
            self._cpu_buffer_mv.release()
            self._dump_mm_mv.release()
            self._dump_mm.close()
            self._dump.close()
            
                
    def valid(self, plan: list) -> bool:
        # 1. Check if cache folder exists at all
        if not os.path.exists(self.base_path):
            return False        

        # 2. Check if manifest and modules files exist:
        if not os.path.exists(self.manifest_path) or not os.path.exists(self.modules_path):
            return False

        # 3. Check if device dumps exist:
        for device, subplan in enumerate(plan):
            if not os.path.exists(self._get_tensor_dump_path(device)):
                return False

        # 4. TODO: Validate actual plan

        return True

    def capture(self, device: int, key: str, tensor: torch.Tensor):
        if self._mode != GpuCacheManagerMode.CAPTURING:
            return        

        if key is None:
            raise KeyError("Key cannot be none when capturing gpu cache")
        
        t = tensor.detach().cpu().contiguous()       
        flat = t.view(-1).view(torch.uint8)         
        data = flat.numpy().data          

        dump_path = self._get_tensor_dump_path(device, temp=True)
        with open(dump_path, "ab") as f_dump:
            offset = self._write_aligned(f_dump, data, tensor.nbytes)

        tensor_manifest = {
            "key": key,
            "offset": str(offset),
            "nbytes": str(tensor.nbytes),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape)            
        }        

        list_path = self._get_tensor_list_path(device)
        with open(list_path, "a") as f_list:
            f_list.write(json.dumps(tensor_manifest))
            f_list.write("\n")   

    def finalize(self, active_devices: list, exported_modules: dict):
        if self._mode != GpuCacheManagerMode.CAPTURING:
            return

        tensor_manifest = {}

        # 1. Combine tensor lists into a manifest and move unique tensors
        for device in active_devices:
            tensor_manifest[device] = {}

            list_path = self._get_tensor_list_path(device)
            with open(list_path, "r") as f_list:
                tensor_list = f_list.readlines()

            source_path = self._get_tensor_dump_path(device, temp=True)
            dest_path = self._get_tensor_dump_path(device, temp=False)
            
            with open(source_path, "rb") as f_src:
                with open(dest_path, "wb") as f_dst:
                    for tensor in tensor_list:
                        tensor = json.loads(tensor)
                        key = tensor.pop("key")

                        if key not in tensor_manifest[device]:                                                        
                            offset = int(tensor["offset"])
                            nbytes = int(tensor["nbytes"])                                 

                            f_src.seek(offset, 0)
                            data = f_src.read(nbytes)

                            offset = self._write_aligned(f_dst, data, nbytes)                            
                            tensor["offset"] = str(offset)

                            tensor_manifest[device][key] = tensor 
            
            os.remove(list_path)         
            os.remove(source_path)
        
        with open(self.manifest_path, "w") as f_manifest:
            json.dump(tensor_manifest, f_manifest)

        # 2. Export modules
        with open(self.modules_path, "wb") as f_modules:
            pickle.dump(exported_modules, f_modules)        

    def init(self, device: int):
        if self._mode != GpuCacheManagerMode.SERVING:
            return

        # 1. Load tensor manifest
        with open(self.manifest_path, "r") as f_manifest:
            self._manifest = json.load(f_manifest)[str(device)]

        # 2. Prepare dump descriptor
        path = self._get_tensor_dump_path(device)
            
        self._dump_size = os.path.getsize(path)                                            
        self._dump = open(path, "rb", buffering=4*1024*1024)        
        self._dump_mm = mmap.mmap(self._dump.fileno(), length=0, access=mmap.ACCESS_READ)
        self._dump_mm_mv = memoryview(self._dump_mm)

        max_nbytes = max((int(t["nbytes"]) for t in self._manifest.values()), default = 0)
        self._cpu_buffer = torch.empty(max_nbytes, dtype=torch.uint8, pin_memory=True)
        self._cpu_buffer_mv = memoryview(self._cpu_buffer.numpy())

    def get_module(self, key: str) -> object:
        if self._mode != GpuCacheManagerMode.SERVING:
            return None

        if self._modules == None:
            with open(self.modules_path, "rb") as f_modules:
                self._modules = pickle.load(f_modules)

        return self._modules.get(key)

    def get_tensor(self, device: int, key: str) -> torch.Tensor:
        if self._mode != GpuCacheManagerMode.SERVING:
            return None

        if key is None or not key in self._manifest:
            raise KeyError(f"Key {key} not found in gpu cache for current device.")
        
        tensor = self._manifest[key]
        dtype  = getattr(torch, tensor["dtype"].replace("torch.", ""))
        shape  = tuple(tensor["shape"])
        nbytes = int(tensor["nbytes"])
        start  = int(tensor["offset"])
        end    = start + nbytes

        if start < 0 or end > self._dump_size:
            raise ValueError(
                f"Corrupt device dump: tensor {key!r} offset [{start}:{end}] is out of bounds; "
                f"delete the .tensorsplit cache and reload"
            )

        torch.cuda.set_device(device)  
        
        gpu_buf = torch.empty(nbytes, dtype=torch.uint8, device=device)                
        self._cpu_buffer_mv[:nbytes] = self._dump_mm_mv[start:end]
        gpu_buf.copy_(self._cpu_buffer[:nbytes], non_blocking=True)
        tensor = gpu_buf.view(dtype).view(shape)        
        
        torch.cuda.current_stream().synchronize() 
        
        return tensor

    