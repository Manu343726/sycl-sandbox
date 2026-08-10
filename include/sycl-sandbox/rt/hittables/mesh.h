#pragma once

/// @file
/// Triangle mesh hittable — a window into the scene's global triangle
/// array.
///
/// A mesh is ONE handle pointing at a contiguous range of the scene's
/// per-type triangle array (`first_triangle` + `num_triangles` index
/// into `SceneView::triangles`).  No per-mesh device allocations are
/// needed: the mesh's triangles are pushed into the same staging array
/// as standalone `Triangle` hittables and uploaded as a single buffer.
/// The whole mesh is culled by the BVH through its one per-handle AABB,
/// and `hit()` scans its triangles linearly — the same "compound
/// primitive" composition used by `Box` (six `Quad` faces), extended to
/// an arbitrary triangle count.
///
/// Textures sample the winning triangle's barycentric coordinates
/// (u weights vertex b, v weights vertex c), exactly like a standalone
/// `Triangle` hittable.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/aabb.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/hittables/triangle.h>
#include <sycl-sandbox/optional.h>
#include <sycl-sandbox/math.h>

namespace rt::hittables {

class Mesh {
public:
    /// Index of the first triangle in the scene's triangle array.
    uint32_t first_triangle = 0;
    /// Number of consecutive triangles belonging to this mesh.
    uint32_t num_triangles = 0;
    /// Root of this mesh's per-mesh BVH: index into the flat
    /// `SceneView::mesh_bvh_nodes` array (leaves reference ABSOLUTE
    /// triangle indices into the scene's triangle array).  -1 when no
    /// per-mesh BVH was built (see SceneBuilder::build_mesh_bvhs()) —
    /// hit() then falls back to a linear scan of the triangle window.
    int bvh_root = -1;

    Mesh() = default;
    Mesh(uint32_t first_triangle_, uint32_t num_triangles_)
        : first_triangle(first_triangle_), num_triangles(num_triangles_) {
    }

    /// Return the axis-aligned bounding box enclosing every triangle
    /// of the mesh (union of the per-triangle AABBs).
    Aabb aabb(const Triangle *triangles) const {
        if ( num_triangles == 0 ) {
            return {{0, 0, 0}, {0, 0, 0}};
        }
        Aabb box = triangles[first_triangle].aabb();
        for ( uint32_t i = 1; i < num_triangles; i++ ) {
            box = aabb_merge(box, triangles[first_triangle + i].aabb());
        }
        return box;
    }

    /// Iterative stack-based traversal of the mesh's flat BVH.  Keeps the
    /// closest triangle hit in [t_min, t_max]; leaves reference absolute
    /// triangle indices into the scene triangle array.  Every node whose
    /// AABB the ray enters is reported through ctx.collector
    /// on_hittable_bvh_node(), so the scene-debug view can highlight the
    /// mesh's interior traversal distinctly from the scene BVH.
    optional<HitRecord> bvh_hit(const Ray &ray, float t_min, float t_max,
                                const BvhNode *nodes, uint32_t root,
                                const Triangle *triangles,
                                const Context &ctx) const {
        optional<HitRecord> closest;
        uint32_t stack[64];
        int stack_size = 0;
        stack[stack_size++] = root;
        while ( stack_size > 0 ) {
            uint32_t node_index = stack[--stack_size];
            const BvhNode &node = nodes[node_index];
            if ( !aabb_hit(node.bounds, ray, t_min, t_max, ctx) ) {
                continue;
            }
            ctx.collector.on_hittable_bvh_node(node_index);
            if ( node.left == BVH_LEAF ) {
                auto hit = triangles[node.right].hit(ray, t_min, t_max, ctx);
                if ( hit && ( !closest || hit->t < closest->t ) ) {
                    closest = hit;
                    t_max = hit->t;
                }
                continue;
            }
            stack[stack_size++] = node.right;
            stack[stack_size++] = node.left;
        }
        return closest;
    }

    /// Ray-mesh intersection: the per-mesh BVH (when built) culls the
    /// interior of the mesh's triangle window; without one the window is
    /// scanned linearly.  The scene BVH culls the mesh as a whole by its
    /// per-handle AABB before this is reached.
    ///
    /// Profiling: the whole intersection is one "hit_mesh" zone — the per-
    /// triangle inner loop does NOT record profiler zones (a large mesh
    /// would flood the ring in a single hit test).  The debug collector
    /// still sees every per-triangle hit test and every per-mesh BVH node
    /// entered, so debuggers can record the scan in full.
    optional<HitRecord> hit(const Ray &ray, float t_min, float t_max,
                            const Triangle *triangles,
                            const BvhNode *mesh_bvh_nodes,
                            const Context &ctx = Context{}) const {
        PROFILER_FUNCTION();
        ctx.collector.on_hit_test(HittableType::Mesh, ray, t_min, t_max);
        if ( mesh_bvh_nodes && bvh_root >= 0 ) {
            return bvh_hit(ray, t_min, t_max, mesh_bvh_nodes,
                           (uint32_t)bvh_root, triangles, ctx);
        }
        optional<HitRecord> closest;
        for ( uint32_t i = 0; i < num_triangles; i++ ) {
            auto hit = triangles[first_triangle + i].hit(ray, t_min, t_max, ctx);
            if ( hit && ( !closest || hit->t < closest->t ) ) {
                closest = hit;
            }
        }
        return closest;
    }
};

inline Mesh mesh(uint32_t first_triangle, uint32_t num_triangles) {
    return Mesh(first_triangle, num_triangles);
}

} // namespace rt::hittables
