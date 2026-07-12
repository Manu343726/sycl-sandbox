#pragma once

/// @file
/// CUDA *driver* API loader via dlopen — no compile-time CUDA dependency.
///
/// Loads libcuda.so.1 (the driver, which also hosts the GL interop entry
/// points) and resolves the handful of functions the zero-copy display
/// target needs.  Safe to call on non-NVIDIA systems: dlopen simply fails
/// and load() returns false.
///
/// This is the ONLY layer in the codebase that speaks CUDA.  Everything
/// above it (render loop, tone-map, UI) sees portable SYCL plus the
/// DisplayTarget interface.

#include <cstddef>
#include <cstdint>

namespace cuda_api {

// ── Driver API types (mirrors cuda.h; kept minimal on purpose) ────────

using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUstream = void *;
using CUgraphicsResource = void *;
using CUdeviceptr = unsigned long long;

inline constexpr CUresult CUDA_SUCCESS = 0;

// ── Function pointers ────────────────────────────────────────────────

#define SANDBOX_CUDA_FN(name, args) \
    using name##_fn = CUresult (*) args; \
    extern name##_fn name;

SANDBOX_CUDA_FN(cuInit, (unsigned int flags))
SANDBOX_CUDA_FN(cuDeviceGet, (CUdevice *device, int ordinal))
SANDBOX_CUDA_FN(cuDevicePrimaryCtxRetain, (CUcontext *ctx, CUdevice dev))
SANDBOX_CUDA_FN(cuDevicePrimaryCtxRelease, (CUdevice dev))
SANDBOX_CUDA_FN(cuCtxSetCurrent, (CUcontext ctx))
SANDBOX_CUDA_FN(cuCtxGetCurrent, (CUcontext *ctx))
SANDBOX_CUDA_FN(cuGraphicsGLRegisterBuffer,
                (CUgraphicsResource *resource, unsigned int gl_buffer,
                 unsigned int flags))
SANDBOX_CUDA_FN(cuGraphicsUnregisterResource, (CUgraphicsResource resource))
SANDBOX_CUDA_FN(cuGraphicsMapResources,
                (unsigned int count, CUgraphicsResource *resources,
                 CUstream stream))
SANDBOX_CUDA_FN(cuGraphicsUnmapResources,
                (unsigned int count, CUgraphicsResource *resources,
                 CUstream stream))
SANDBOX_CUDA_FN(cuGraphicsResourceGetMappedPointer,
                (CUdeviceptr *dev_ptr, size_t *size,
                 CUgraphicsResource resource))
SANDBOX_CUDA_FN(cuMemcpyDtoD, (CUdeviceptr dst, CUdeviceptr src, size_t bytes))
SANDBOX_CUDA_FN(cuStreamSynchronize, (CUstream stream))

#undef SANDBOX_CUDA_FN

// cuGetErrorName returns the name through an out-parameter.
using cuGetErrorName_fn = CUresult (*)(CUresult, const char **);
extern cuGetErrorName_fn cuGetErrorName;

// ── Loader ────────────────────────────────────────────────────────────

/// Load libcuda.so.1 and resolve all functions.  Idempotent; returns true
/// on success.  Does NOT call cuInit — the display target does that.
bool load();

/// True after a successful load().
bool is_available();

/// Human-readable name for a CUresult ("?" if unresolvable).
const char *error_name(CUresult r);

} // namespace cuda_api
