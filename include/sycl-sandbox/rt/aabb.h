#pragma once

/// @file
/// Axis-aligned bounding boxes and the flat BVH node.
///
/// The scene BVH and the per-mesh BVHs are flat arrays of BvhNode, each
/// carrying the Aabb of its subtree.  The hot traversal paths call
/// aabb_hit() once per visited node — it accepts an optional rt::Context
/// so the device profiler can record the slab tests (PROFILER_ZONE); the
/// defaulted Context{} keeps callers that have none (host diagnostics,
/// scene building) working unchanged.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/math.h>

namespace rt {

/// Axis-aligned bounding box: two extreme corners enclosing a volume.
struct Aabb {
    float3 min; ///< Minimum corner (most negative components).
    float3 max; ///< Maximum corner (most positive components).
};

/// Build an Aabb from two arbitrary points (order-independent).
inline Aabb aabb_from_points(float3 a, float3 b) {
    return {{math::fmin(a.x, b.x), math::fmin(a.y, b.y), math::fmin(a.z, b.z)},
            {math::fmax(a.x, b.x), math::fmax(a.y, b.y), math::fmax(a.z, b.z)}};
}

/// Merge two Aabbs into the smallest box containing both.
inline Aabb aabb_merge(Aabb a, Aabb b) {
    return {
        {math::fmin(a.min.x, b.min.x), math::fmin(a.min.y, b.min.y), math::fmin(a.min.z, b.min.z)},
        {math::fmax(a.max.x, b.max.x), math::fmax(a.max.y, b.max.y), math::fmax(a.max.z, b.max.z)}};
}

/// Sentinel value indicating a BVH leaf node (left child = invalid).
static constexpr uint32_t BVH_LEAF = 0xFFFFFFFF;

/// Flat BVH node stored in a contiguous array.
/// Leaf nodes: left == BVH_LEAF, right == object index (scene handle
/// index for the scene BVH, triangle index for a per-mesh BVH).
/// Internal nodes: left and right are indices into the bvh_nodes array.
struct BvhNode {
    Aabb bounds;    ///< Bounding box enclosing both children.
    uint32_t left;  ///< Left child index (BVH_LEAF for leaf).
    uint32_t right; ///< Right child index or object index (leaf).
};

/// Ray-AABB intersection test using the slab method.
/// Returns true if the ray intersects the box within [t_min, t_max].
inline bool aabb_hit(Aabb box, const Ray &ray, float t_min, float t_max,
                     const Context &ctx = Context{}) {
    PROFILER_FUNCTION();

    for ( int axis = 0; axis < 3; axis++ ) {
        PROFILER_ZONE("aabb_slab");

        float orig = (axis == 0) ? ray.orig.x : (axis == 1) ? ray.orig.y : ray.orig.z;
        float dir = (axis == 0) ? ray.dir.x : (axis == 1) ? ray.dir.y : ray.dir.z;
        float bmin = (axis == 0) ? box.min.x : (axis == 1) ? box.min.y : box.min.z;
        float bmax = (axis == 0) ? box.max.x : (axis == 1) ? box.max.y : box.max.z;

        if ( dir == 0.f ) {
            if ( orig < bmin || orig > bmax ) {
                return false;
            }
            continue;
        }

        float inv_dir = 1.f / dir;
        float t0 = (bmin - orig) * inv_dir;
        float t1 = (bmax - orig) * inv_dir;
        if ( t0 > t1 ) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = (t0 > t_min) ? t0 : t_min;
        t_max = (t1 < t_max) ? t1 : t_max;
        if ( t_max < t_min ) {
            return false;
        }
    }
    return true;
}

} // namespace rt
