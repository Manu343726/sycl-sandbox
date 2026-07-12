#pragma once
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/scene/data.h>

#include <dlfcn.h>
#include <string>
#include <cstdio>

// ── SceneDebugInfo forward declaration ───────────────────────────────
struct SceneDebugInfo;

// ── Kernel dispatch helpers ───────────────────────────────────────────

/// Call init_kernel() on a loaded kernel .so.
/// `q` is null on the software (native) backend — the kernel .so was built
/// with KERNEL_NATIVE and receives it as a plain void*.
inline void call_init_kernel(void *handle, sycl::queue *q, int w, int h,
                              const void *params, size_t sz) {
    PROFILER_FUNCTION();
    using fn_t = void (*)(sycl::queue *, int, int, const void *, size_t);
    auto *fn = reinterpret_cast<fn_t>(dlsym(handle, "init_kernel"));
    if ( fn ) {
        fn(q, w, h, params, sz);
    } else {
        // Not an error — some kernels don't export init_kernel
    }
}

/// Call render_kernel() on a loaded kernel .so (ABI v2).
/// ENQUEUE-ONLY: the kernel submits work to the queue and returns without
/// waiting — the caller chains the tone-map + display copy after it.
/// `q` is null on the software backend (synchronous execution).
inline void call_render_kernel(void *handle, sycl::queue *q,
                               const RenderContext &ctx) {
    PROFILER_FUNCTION();
    using fn_t = void (*)(sycl::queue *, const RenderContext *);
    auto *fn = reinterpret_cast<fn_t>(dlsym(handle, "render_kernel"));
    if ( fn ) {
        fn(q, &ctx);
    } else {
        // Not an error — some kernels don't export render_kernel
    }
}

// ── Optional kernel API resolvers ─────────────────────────────────────

/// Type of the get_scene_debug_info() function from kernel.
using get_scene_debug_info_fn = const SceneDebugInfo *(*)();

/// Resolve get_scene_debug_info() from a loaded kernel.
inline get_scene_debug_info_fn resolve_scene_debug(void *handle) {
    return reinterpret_cast<get_scene_debug_info_fn>(dlsym(handle, "get_scene_debug_info"));
}

/// Type of set_param_lookup() function from kernel.
using set_param_lookup_fn = void (*)(const rt::ParamLookup *);

/// Resolve set_param_lookup() from a loaded kernel.
inline set_param_lookup_fn resolve_set_param_lookup(void *handle) {
    return reinterpret_cast<set_param_lookup_fn>(dlsym(handle, "set_param_lookup"));
}

/// Type of set_stat_lookup() function from kernel.
using set_stat_lookup_fn = void (*)(const rt::StatWriter *);

/// Resolve set_stat_lookup() from a loaded kernel.
inline set_stat_lookup_fn resolve_set_stat_lookup(void *handle) {
    return reinterpret_cast<set_stat_lookup_fn>(dlsym(handle, "set_stat_lookup"));
}

/// Type of set_scene_view() function from kernel.
using set_scene_view_fn = void (*)(const rt::SceneView *);

/// Resolve set_scene_view() from a loaded kernel.
inline set_scene_view_fn resolve_set_scene_view(void *handle) {
    return reinterpret_cast<set_scene_view_fn>(dlsym(handle, "set_scene_view"));
}

/// Type of set_scene_debug_info() function from kernel.
using set_scene_debug_info_fn = void (*)(const SceneDebugInfo *);

/// Resolve set_scene_debug_info() from a loaded kernel.
inline set_scene_debug_info_fn resolve_set_scene_debug_info(void *handle) {
    return reinterpret_cast<set_scene_debug_info_fn>(dlsym(handle, "set_scene_debug_info"));
}

/// Type of set_profiler_buffer() function from kernel.
using set_profiler_buffer_fn = void (*)(void *);

/// Resolve set_profiler_buffer() from a loaded kernel.
inline set_profiler_buffer_fn resolve_set_profiler_buffer(void *handle) {
    return reinterpret_cast<set_profiler_buffer_fn>(dlsym(handle, "set_profiler_buffer"));
}

/// Type of set_runtime() function from kernel.
using set_runtime_fn = void (*)(void *);

/// Resolve set_runtime() from a loaded kernel.
inline set_runtime_fn resolve_set_runtime(void *handle) {
    return reinterpret_cast<set_runtime_fn>(dlsym(handle, "set_runtime"));
}
