#include "cuda_api.h"
#include <spdlog/spdlog.h>
#include <dlfcn.h>

namespace cuda_api {

static void *g_handle = nullptr;
static bool g_attempted = false;
static bool g_ok = false;

#define SANDBOX_CUDA_FN(name) name##_fn name = nullptr;
SANDBOX_CUDA_FN(cuInit)
SANDBOX_CUDA_FN(cuDeviceGet)
SANDBOX_CUDA_FN(cuDevicePrimaryCtxRetain)
SANDBOX_CUDA_FN(cuDevicePrimaryCtxRelease)
SANDBOX_CUDA_FN(cuCtxSetCurrent)
SANDBOX_CUDA_FN(cuCtxGetCurrent)
SANDBOX_CUDA_FN(cuGraphicsGLRegisterBuffer)
SANDBOX_CUDA_FN(cuGraphicsUnregisterResource)
SANDBOX_CUDA_FN(cuGraphicsMapResources)
SANDBOX_CUDA_FN(cuGraphicsUnmapResources)
SANDBOX_CUDA_FN(cuGraphicsResourceGetMappedPointer)
SANDBOX_CUDA_FN(cuMemcpyDtoD)
SANDBOX_CUDA_FN(cuStreamSynchronize)
#undef SANDBOX_CUDA_FN
cuGetErrorName_fn cuGetErrorName = nullptr;

bool load() {
    if (g_attempted) return g_ok;
    g_attempted = true;

    // The driver library carries both the core and GL-interop entry points.
    g_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!g_handle) g_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_handle) {
        spdlog::info("[cuda_api] libcuda not found — CUDA display interop "
                     "unavailable");
        return false;
    }

    // Several entry points have _v2 revisions with the modern signature;
    // prefer those and fall back to the unsuffixed symbol.
    auto resolve = [](const char *name, const char *v2) -> void * {
        void *p = v2 ? dlsym(g_handle, v2) : nullptr;
        if (!p) p = dlsym(g_handle, name);
        return p;
    };

#define RESOLVE(name, v2)                                                  \
    name = reinterpret_cast<name##_fn>(resolve(#name, v2));                \
    if (!name) {                                                           \
        spdlog::warn("[cuda_api] missing symbol " #name);                  \
        dlclose(g_handle);                                                 \
        g_handle = nullptr;                                                \
        return false;                                                      \
    }

    RESOLVE(cuInit, nullptr)
    RESOLVE(cuDeviceGet, nullptr)
    RESOLVE(cuDevicePrimaryCtxRetain, nullptr)
    RESOLVE(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2")
    RESOLVE(cuCtxSetCurrent, nullptr)
    RESOLVE(cuCtxGetCurrent, nullptr)
    RESOLVE(cuGraphicsGLRegisterBuffer, nullptr)
    RESOLVE(cuGraphicsUnregisterResource, nullptr)
    RESOLVE(cuGraphicsMapResources, nullptr)
    RESOLVE(cuGraphicsUnmapResources, nullptr)
    RESOLVE(cuGraphicsResourceGetMappedPointer,
            "cuGraphicsResourceGetMappedPointer_v2")
    RESOLVE(cuMemcpyDtoD, "cuMemcpyDtoD_v2")
    RESOLVE(cuStreamSynchronize, nullptr)
#undef RESOLVE

    cuGetErrorName =
        reinterpret_cast<cuGetErrorName_fn>(dlsym(g_handle, "cuGetErrorName"));

    g_ok = true;
    spdlog::info("[cuda_api] libcuda loaded (driver API ready)");
    return true;
}

bool is_available() { return g_ok; }

const char *error_name(CUresult r) {
    const char *name = nullptr;
    if (cuGetErrorName && cuGetErrorName(r, &name) == CUDA_SUCCESS && name)
        return name;
    return "?";
}

} // namespace cuda_api
