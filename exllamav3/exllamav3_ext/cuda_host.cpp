#include "cuda_host.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"

#include <fcntl.h>
#include <unistd.h>

// The runtime records a failed call's error as the last error as well. Errors these functions report (or deliberately
// ignore) must be consumed, or the next cudaGetLastError() elsewhere (torch's launch checks) attributes them to
// something unrelated
static void consume_error(cudaError_t cr)
{
    if (cr != cudaSuccess) cudaGetLastError();
}

void cuda_host_register(uintptr_t ptr, size_t nbytes, unsigned int flags)
{
    cudaError_t cr = cudaHostRegister(reinterpret_cast<void*>(ptr), nbytes, flags);

    // A region can only be pinned once per process, but ranks sharing an arena may each try to
    // register it
    consume_error(cr);
    if (cr == cudaErrorHostMemoryAlreadyRegistered) return;

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostRegister(", reinterpret_cast<void*>(ptr), ", ", nbytes, ") failed: ",
        cudaGetErrorString(cr)
    );
}

void cuda_host_unregister(uintptr_t ptr)
{
    cudaError_t cr = cudaHostUnregister(reinterpret_cast<void*>(ptr));

    // Teardown is racy by nature: the region may already have been released, or the runtime may
    // be unloading while shared segments are still being torn down. Both are benign here
    consume_error(cr);
    if (cr == cudaErrorHostMemoryNotRegistered || cr == cudaErrorCudartUnloading) return;

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostUnregister(", reinterpret_cast<void*>(ptr), ") failed: ", cudaGetErrorString(cr)
    );
}

void cuda_memcpy_async(uintptr_t dst, uintptr_t src, size_t nbytes, int kind)
{
    TORCH_CHECK(kind == 1 || kind == 2, "cuda_memcpy_async: kind must be 1 (host to device) or 2 (device to host)");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t cr = cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), nbytes,
                                     kind == 1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream);
    consume_error(cr);
    TORCH_CHECK(cr == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(cr));
}

uintptr_t cuda_host_get_device_pointer(uintptr_t ptr)
{
    // Device-side alias of memory registered with cudaHostRegisterMapped. Equal to the host
    // pointer under UVA (Linux desktop), but distinct under WDDM (native Windows, and potentially
    // WSL2), where the host pointer is not usable in kernels and this alias must be passed instead
    void* dev_ptr = nullptr;
    cudaError_t cr = cudaHostGetDevicePointer(&dev_ptr, reinterpret_cast<void*>(ptr), 0);
    consume_error(cr);

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostGetDevicePointer(", reinterpret_cast<void*>(ptr), ") failed: ",
        cudaGetErrorString(cr)
    );

    return reinterpret_cast<uintptr_t>(dev_ptr);
}

int cuda_device_get_attribute(int attr, int device)
{
    int value = 0;
#if defined(USE_ROCM)
    // The caller passes CUDA's cudaDeviceAttr numbering (model_tp_cuda.py), which HIP's attribute enum does
    // not share: translate the values in use rather than query whatever attribute HIP has at that number
    TORCH_CHECK(attr == 91, "cuda_device_get_attribute: CUDA attribute ", attr, " has no HIP mapping here");
    cudaError_t cr = cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
#else
    cudaError_t cr = cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(attr), device);
#endif
    consume_error(cr);

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaDeviceGetAttribute(", attr, ", ", device, ") failed: ", cudaGetErrorString(cr)
    );

    return value;
}

at::Tensor pinned_cuda_view(const at::Tensor& t, int64_t device)
{
    // CUDA-device alias of a pinned host tensor: same storage, but with a device dtype/layout
    // so torch ops (matmul/cuBLAS) and extension kernels accept it directly, reading over PCIe
    // (zero-copy). The view does NOT own the memory; the caller must keep the pinned source
    // tensor alive for the alias's lifetime. Uses cudaHostGetDevicePointer rather than assuming
    // UVA pointer equality so the alias also holds under WDDM
    TORCH_CHECK(t.device().is_cpu(), "pinned_cuda_view: tensor must be a CPU tensor");
    TORCH_CHECK(t.is_pinned(), "pinned_cuda_view: tensor must be pinned");
    void* dev_ptr = nullptr;
    cudaError_t cr = cudaHostGetDevicePointer(&dev_ptr, t.data_ptr(), 0);
    consume_error(cr);
    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostGetDevicePointer(", t.data_ptr(), ") failed: ", cudaGetErrorString(cr)
    );
    auto options = t.options().device(at::kCUDA, static_cast<c10::DeviceIndex>(device));
    return at::from_blob(dev_ptr, t.sizes(), t.strides(), [](void*) {}, options);
}

ssize_t cuda_load_file_direct(uintptr_t dst_gpu_addr,
                              const char* path,
                              size_t      file_size)
{
    constexpr size_t CHUNK = 64ull * 1024 * 1024;  
    constexpr int    NBUFS = 4;                    
    constexpr size_t BLKSZ = 4096;                 

    if (file_size == 0) return 0;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return -1;

    void* host[NBUFS]   = {};
    cudaStream_t stream[NBUFS] = {};

    for (int i = 0; i < NBUFS; ++i) {
        if (cudaHostAlloc(&host[i], CHUNK, cudaHostAllocDefault) != cudaSuccess) {
            for (int j = 0; j < i; ++j)
                cudaFreeHost(host[j]);
    
                close(fd);
            return -1;
        }
    
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }

    char* dst = reinterpret_cast<char*>(dst_gpu_addr);
    size_t offset = 0;
    int i = 0;
    int rc = 0;

    while (offset < file_size) {
        int slot = i % NBUFS;
        size_t remaining = file_size - offset;
        size_t want = (remaining >= CHUNK)
                      ? CHUNK
                      : ((remaining + BLKSZ - 1) & ~(BLKSZ - 1));

        cudaStreamSynchronize(stream[slot]);

        ssize_t n = pread(fd, host[slot], want, static_cast<off_t>(offset));        
        if (n < 0) {
             rc = -1;
              break;
        }

        if (n == 0) {
            rc = -1;
             break;
        }                       

        if ((size_t)n < want && offset + (size_t)n < file_size) {
            rc = -1;
            break;                                    
        }

        size_t push = (static_cast<size_t>(n) < remaining)
                    ? static_cast<size_t>(n)
                    : remaining;

        if (cudaSuccess != cudaMemcpyAsync(dst + offset, host[slot], push,
                        cudaMemcpyHostToDevice, stream[slot])) {
            rc = -1;
            break;
        }                             

        offset += push;   
        ++i;
    }

    for (int s = 0; s < NBUFS; ++s)
        cudaStreamSynchronize(stream[s]);

    for (int s = 0; s < NBUFS; ++s) {
        cudaStreamDestroy(stream[s]);
        cudaFreeHost(host[s]);
    }

    close(fd);

    return (rc == 0) ? static_cast<ssize_t>(offset) : -1;
}