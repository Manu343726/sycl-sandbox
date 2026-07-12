#pragma once
#include <sycl-sandbox/param_types.h>
#include <sycl-sandbox/profiler_device.h>
#include <cstddef>
#include <cstdint>

// ── Kernel API ABI abstraction ───────────────────────────────────────
// Kernels compiled in KERNEL_NATIVE mode (CPU Software backend) are pure
// C++ shared libraries without any SYCL dependency.  The host dispatches
// to the correct kernel variant at runtime.
//
// KERNEL_QUEUE_PARAM resolves to the appropriate queue pointer type for
// the kernel's compilation mode:
//   SYCL mode:    sycl::queue *   (real SYCL queue)
//   Native mode:  void *          (always nullptr in practice)

#ifdef KERNEL_NATIVE
#define KERNEL_QUEUE_PARAM void *
#else
#include <sycl/sycl.hpp>
#define KERNEL_QUEUE_PARAM sycl::queue *
#endif

namespace rt {
struct StatWriter;
}

struct KernelDesc {
    const char *name;
    const char *description;
    int32_t max_spp; // max samples/pixel this kernel benefits from (1 = single-frame)
    int32_t source_count;
    const char **sources;
};

// ── RenderContext (ABI v2) ────────────────────────────────────────────
/// Everything a render call needs, bundled in one struct.  Valid only for
/// the duration of the render_kernel() call — kernels must not stash the
/// pointer.  POD and SYCL-free under KERNEL_NATIVE.
struct RenderContext {
    int32_t width;
    int32_t height;

    /// Host-visible parameter snapshot for this frame.  Read it HOST-side
    /// (before enqueuing the parallel_for) and capture scalars by value
    /// into the kernel lambda — device code must never dereference it.
    const void *params;

    /// Persistent accumulation buffer: device float4[width*height].
    /// Kernels ADD samples (rgb += sample; w += 1 per sample) and must
    /// NEVER clear it — the host clears it when accumulation restarts.
    float *accum;

    /// Number of samples to add in this call (host's read of "spp_frame").
    uint32_t spp_frame;

    /// Samples already accumulated before this call.  Use for RNG stream
    /// decorrelation (replaces the old sample_index argument).
    uint32_t spp_total;

    /// Monotone frame counter (animation tick).
    uint64_t frame_index;

    /// Device profiler ring.  Capture BY VALUE into kernel lambdas and
    /// use PROFILER_DEVICE_ZONE; inactive ring makes it a no-op.
    profiler::DeviceRing prof;

    /// Per-frame statistic writer (host-side writes only, race-free
    /// publication handled by the host).  May be null.
    const rt::StatWriter *stats;
};

// ── Kernel entry points ───────────────────────────────────────────────

extern "C" KernelDesc *get_kernel_desc();
extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int w, int h, const void *params, size_t params_size);

/// ENQUEUE-ONLY CONTRACT: render_kernel must submit its device work to the
/// queue and return WITHOUT waiting for completion.  The host chains the
/// tone-map and display publication after it on the same in-order queue.
/// (Native mode executes synchronously — the contract is trivially met.)
extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *ctx);

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM);

// ── Debug geometry types ──────────────────────────────────────────────

/// Lightweight host-side description of a sphere for debug rendering.
struct DebugSphere {
    float center[3];
    float radius;
    float color[3];
};

/// Lightweight host-side description of a quad for debug rendering.
struct DebugQuad {
    float base[3];
    float edge_u[3];
    float edge_v[3];
    float color[3];
};

/// Lightweight host-side description of a box for debug rendering.
struct DebugBox {
    float box_min[3];
    float box_max[3];
    float color[3];
};

/// Optional debug data returned by kernels that support scene visualization.
struct SceneDebugInfo {
    // AABB data (flat array: [min_x,min_y,min_z,max_x,max_y,max_z] repeated)
    const float *aabb_data;
    int num_aabbs;

    // Real geometry data for proper 3D rendering
    const DebugSphere *spheres;
    int num_spheres;
    const DebugQuad *quads;
    int num_quads;
    const DebugBox *boxes;
    int num_boxes;
};

/// Optional: kernels that support scene debug visualization implement this.
/// Return nullptr if debug info is unavailable.  Host calls once after
/// init_kernel and caches the result.
extern "C" const SceneDebugInfo *get_scene_debug_info();
