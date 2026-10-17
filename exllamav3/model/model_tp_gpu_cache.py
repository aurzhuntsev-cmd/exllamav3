import os
from ..model.config import Config
from enum import Enum
import torch
import shutil
import json, pickle

GPU_CACHE_ENV_VAR                   = "EXL3_TP_CACHE_ENABLE"
GPU_CACHE_BASE_PATH                 = ".tensorsplit/"
GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE    = 256
CPU_CACHE_LOAD_CHUNK_SIZE_MB        = 64

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

    def __init__(self, config: Config = None):
        if self._initialized or config is None:
            return

        self.config = config
        self.base_path = os.path.join(self.config.directory, GPU_CACHE_BASE_PATH)
        self.manifest_path = os.path.join(self.base_path, "manifest.json")
        self.modules_path = os.path.join(self.base_path, "modules.pkl")     

        self._tensors = None
        self._modules = None

    def _get_tensor_dump_path(self, device: int, temp: bool = False):
        postfix = ".tmp" if temp == True else ""
        return os.path.join(self.base_path, f"cuda{device}.bin{postfix}")

    def _get_tensor_list_path(self, device: int):            
        return os.path.join(self.base_path, f"cuda{device}.manifest.json")
    
    def set_mode(self, mode: GpuCacheManagerMode):
        self._mode = mode
        
        if mode == GpuCacheManagerMode.CAPTURING:
            if os.path.exists(self.base_path):
                shutil.rmtree(self.base_path)

            os.mkdir(self.base_path)
                
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
            offset = f_dump.tell()
            pad = (GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE - 
                                (tensor.nbytes % GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE)) % GPU_CACHE_CUDA_ALIGN_SECTOR_SIZE
            
            f_dump.write(data)
            f_dump.write(b'\x00' * pad)

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

        # 1. Combine tensor lists into a manifest
        for device in active_devices:
            tensor_manifest[device] = {}

            list_path = self._get_tensor_list_path(device)
            with open(list_path, "r") as f_list:
                tensor_list = f_list.readlines()

            for tensor in tensor_list:
                tensor = json.loads(tensor)
                key = tensor.pop("key")
                tensor_manifest[device][key] = tensor

            os.remove(list_path)                
            
        with open(self.manifest_path, "w") as f_manifest:
            json.dump(tensor_manifest, f_manifest)

        # 2. Export modules
        with open(self.modules_path, "wb") as f_modules:
            pickle.dump(exported_modules, f_modules)

        # 3. Rename dumps (non-atomic, huh)
        for device in active_devices:
            os.rename(
                self._get_tensor_dump_path(device, temp=True),
                self._get_tensor_dump_path(device, temp=False)
            )            

    def load(self, device: int):
        if self._mode != GpuCacheManagerMode.SERVING:
            return

        # 1. Load tensor manifest
        with open(self.manifest_path, "r") as f_manifest:
            tensor_manifest = json.load(f_manifest)

        # 2. Load dump into gpu memory
        path = self._get_tensor_dump_path(device)
        size = os.path.getsize(path)                            

        torch.cuda.set_device(device)  
        """
        gpu_buf = torch.empty(size, dtype=torch.uint8, device=device)
        cpu_buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)            
        
        with open(path, "rb") as f_bin:
            f_bin.readinto(cpu_buf.numpy())            
                
        gpu_buf.copy_(cpu_buf, non_blocking=True)
        """

        chunk = CPU_CACHE_LOAD_CHUNK_SIZE_MB * 1024 * 1024
        gpu_buf = torch.empty(size, dtype=torch.uint8, device=device)
        cpu_chunk = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)            
        mv = memoryview(cpu_chunk.numpy())        
        
        with open(path, "rb") as f:
            offset = 0
            while offset < size:
                n = min(chunk, size - offset)
                got = f.readinto(mv[:n])
                if not got:
                    break
                gpu_buf[offset:offset + got].copy_(cpu_chunk[:got], non_blocking=True)                
                torch.cuda.current_stream().synchronize()
                offset += got


        #torch.cuda.current_stream().synchronize() 

        # 3. Reconstruct tensors
        self._tensors = {}
        for key, tensor in tensor_manifest[str(device)].items():            
            dtype  = getattr(torch, tensor["dtype"].replace("torch.", ""))
            shape  = tuple(tensor["shape"])    
            nbytes = int(tensor["nbytes"])
            start  = int(tensor["offset"])                                
            end    = start + nbytes            
            
            if start < 0 or end > gpu_buf.numel():
                raise ValueError(
                    f"Corrupt device dump: tensor {key!r} offset [{start}:{end}] is out of bounds "
                    f"of device{device}.bin ({gpu_buf.numel()} bytes); delete the .tensorsplit cache and reload"
                )                    

            tensor = gpu_buf[start:end].view(dtype).view(shape).contiguous()
            self._tensors[key] = tensor

    def get_module(self, key: str) -> object:
        if self._mode != GpuCacheManagerMode.SERVING:
            return None

        if self._modules == None:
            with open(self.modules_path, "rb") as f_modules:
                self._modules = pickle.load(f_modules)

        return self._modules.get(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        if self._mode != GpuCacheManagerMode.SERVING:
            return None

        if key is None or not key in self._tensors:
            raise KeyError(f"Key {key} not found in gpu cache for current device.")

        return self._tensors[key]

    