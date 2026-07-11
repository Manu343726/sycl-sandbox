#pragma once
#include <sycl-sandbox/param_types.h>
#include <sycl/sycl.hpp>
#include <cstdint>

struct KernelDesc {
    const char *name;
    const char *description;
    int32_t param_count;
    const ParamMeta *params;
    size_t params_buffer_size;
    int32_t max_spp; // max samples/pixel this kernel benefits from (1 = single-frame)
    int32_t source_count;
    const char **sources;
};

extern "C" KernelDesc *get_kernel_desc();
extern "C" void init_kernel(sycl::queue *, int w, int h, const void *params, size_t params_size);
extern "C" void render_kernel(sycl::queue *,
                              int w,
                              int h,
                              const void *params,
                              void *accum_buffer,
                              int sample_index);
extern "C" void shutdown_kernel(sycl::queue *);

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
