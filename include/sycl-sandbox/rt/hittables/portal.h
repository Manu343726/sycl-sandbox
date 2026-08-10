#pragma once

/// @file
/// Portals — a pair of hittables (entry <-> exit) that teleport rays.
///
/// Portals are BIDIRECTIONAL: when a ray hits EITHER shape, the
/// parametric coordinates (u, v) of the hit are mapped onto the OTHER
/// shape's surface (via point_at_uv(), the inverse of each primitive's
/// hit() UV convention).  The resulting HitRecord carries the other
/// surface's point/normal, keeps the hit shape's distance for closest-
/// hit ordering, and marks the record with `is_portal` + the
/// continuation ray (`portal_origin`/`portal_dir`).  The trace loop
/// stays generic: materials scatter portal records as a pure teleport
/// (see rt::portal_scatter()), so the portal is instanced as an object
/// with a material — by default a dummy white Lambertian.  Real
/// materials are allowed too, e.g. DiffuseLight makes an emissive
/// portal (glowing surface; emission is handled before scatter).
///
/// Direction mapping is position-only by default: the ray keeps its
/// direction (classic "window into another place" behaviour).  The
/// teleport origin is nudged along the other-surface normal in the
/// direction of travel so the continuation ray starts just outside the
/// surface.
///
/// Single form:
///   - `Portal`               — concrete type for the data-oriented
///     scene layout: entry/exit are a `PortalShape` variant, so one
///     per-type array holds mixed pairs (quad -> quad, quad -> sphere,
///     ...).  Used by SceneBuilder/SceneView and by kernels.
///
/// Supported shapes: Sphere, Triangle, Quad (they provide point_at_uv()).
/// Box is NOT supported as a portal shape — its hit() does not record
/// which face was hit, so UVs would be ambiguous.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/aabb.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/hittables/sphere.h>
#include <sycl-sandbox/rt/hittables/triangle.h>
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/optional.h>
#include <sycl-sandbox/variant.h>
#include <variant>
#include <utility>
#include <sycl-sandbox/profiler.h>

namespace rt::hittables {

// ── Portal ────────────────────────────────────────────────────────────

/// The portal-capable primitive shapes (all provide point_at_uv()).
using PortalShape = std::variant<Sphere, Triangle, Quad>;

/// Concrete portal for the data-oriented scene layout: entry and exit are
/// stored as PortalShape variants, so a single per-type array can hold
/// mixed pairs (e.g. quad -> sphere).  Trivially copyable (all shapes are).
class Portal {
public:
    PortalShape entry;
    PortalShape exit;

    Portal() = default;
    Portal(PortalShape e, PortalShape x) : entry(std::move(e)), exit(std::move(x)) {
    }

    Aabb aabb() const {
        Aabb box = {};
        visit(entry, [&](const auto &shape) { box = shape.aabb(); });
        Aabb exit_box = {};
        visit(exit, [&](const auto &shape) { exit_box = shape.aabb(); });
        return aabb_merge(box, exit_box);
    }

    /// Hit either surface: a ray hitting the ENTRY shape teleports to the
    /// EXIT shape, and a ray hitting the EXIT shape teleports back to the
    /// ENTRY (bidirectional).  Both shapes are tested with the same
    /// t-range and the CLOSEST hit wins (the pair may overlap).  The
    /// record carries the hit shape's distance (closest-hit ordering)
    /// and the OTHER shape's surface point, UV-mapped via point_at_uv().
    ///
    /// \param ctx per-call kernel context: records a "hit_portal"
    ///        profiler zone and reports the hit test to the trace
    ///        collector; forwarded to the entry/exit shape hit tests.
    optional<HitRecord> hit(const Ray &ray, float t_min, float t_max,
                            const Context &ctx = Context{}) const {
        PROFILER_FUNCTION();
        ctx.collector.on_hit_test(HittableType::Portal, ray, t_min, t_max);
        optional<HitRecord> entry_hit;
        visit(entry, [&](const auto &shape) {
            entry_hit = shape.hit(ray, t_min, t_max, ctx);
        });
        optional<HitRecord> exit_hit;
        visit(exit, [&](const auto &shape) {
            exit_hit = shape.hit(ray, t_min, t_max, ctx);
        });
        if ( !entry_hit && !exit_hit ) {
            return nullopt;
        }

        // Whichever shape was hit, the ray teleports to the other one.
        bool from_entry = true;
        if ( entry_hit && exit_hit ) {
            from_entry = entry_hit->t <= exit_hit->t;   // closest wins
        } else if ( exit_hit ) {
            from_entry = false;
        }

        HitRecord rec;
        if ( from_entry ) {
            visit(exit, [&](const auto &shape) {
                rec = shape.point_at_uv(entry_hit->u, entry_hit->v);
            });
            rec.t = entry_hit->t;   // keep the HIT distance for ordering
            rec.u = entry_hit->u;
            rec.v = entry_hit->v;
        } else {
            visit(entry, [&](const auto &shape) {
                rec = shape.point_at_uv(exit_hit->u, exit_hit->v);
            });
            rec.t = exit_hit->t;
            rec.u = exit_hit->u;
            rec.v = exit_hit->v;
        }
        rec.is_portal = true;

        float3 eps = scale(rec.normal, 0.001f);
        if ( dot(ray.dir, rec.normal) < 0.f ) {
            eps = scale(eps, -1.f);
        }
        rec.portal_origin = add(rec.p, eps);
        rec.portal_dir = ray.dir;
        return rec;
    }
};

inline Portal portal(PortalShape e, PortalShape x) {
    return Portal(std::move(e), std::move(x));
}

} // namespace rt::hittables
