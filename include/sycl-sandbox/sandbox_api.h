#pragma once
#include <sycl-sandbox/context.h>
#include <sycl-sandbox/param_types.h>

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
