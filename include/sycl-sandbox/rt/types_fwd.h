#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/math.h>
#include <sycl-sandbox/optional.h>

/// Forward declarations used by individual hittable/material headers.
namespace rt {

/// A ray in 3D space: origin point + direction vector.
struct Ray {
    float3 orig, dir;
    /// Animation time of the ray (from the scene's `time` param).  Used
    /// by time-varying materials/textures (e.g. animated procedural
    /// textures).  Defaults to 0 for rays built without a time.
    float time = 0.f;
};

/// Record filled by Hittable::hit() when a ray-geometry intersection is found.
struct HitRecord {
    float3 p;        ///< World-space intersection point.
    float3 normal;   ///< Surface normal at intersection (pointing outward).
    float t;         ///< Distance along the ray where the hit occurred.
    bool front_face; ///< True if the ray hit from outside the surface.
    /// Parametric surface coordinates of the hit point, in [0,1] within
    /// the primitive's own UV space (quad/triangle: affine/barycentric
    /// coordinates, sphere: spherical mapping, box: the hit face's quad
    /// coordinates).  Input to texture samplers; textures decide their
    /// own behavior for out-of-range values.
    float u = 0.f;
    float v = 0.f;
    /// Portal teleport marker.  When true, p/normal are the OTHER surface
    /// point and normal: whichever portal shape the ray hit (entry or
    /// exit — portals are bidirectional), its parametric coordinates were
    /// mapped onto the counterpart hittable.  Materials scatter portal
    /// records as a pure teleport: the ray continues from `portal_origin`
    /// along `portal_dir` with unit attenuation (see portal_scatter()).
    /// `t` is the HIT shape's distance (entry or exit), so closest-hit
    /// ordering sees the portal at the surface the ray actually crossed.
    bool is_portal = false;
    float3 portal_origin = {0, 0, 0}; ///< where the continuation ray starts
    float3 portal_dir = {0, 0, 0};    ///< direction of the continuation ray
};

/// Record returned by Material::scatter() when scattering occurs.
struct ScatterRecord {
    float3 attenuation; ///< Colour attenuation (albedo × path throughput).
    Ray scattered;      ///< New ray direction after scattering.
};

/// Build the ScatterRecord for a portal hit: the ray continues from the
/// exit surface (`portal_origin`/`portal_dir`) with unit attenuation —
/// the path passes through the portal unchanged.  Called by every
/// material's scatter() on is_portal records, so the trace loop stays
/// generic.  Emission is handled before scatter, so a portal instanced
/// with an emissive material (e.g. DiffuseLight) glows instead of
/// teleporting.
inline optional<ScatterRecord> portal_scatter(const Ray &incoming_ray,
                                              const HitRecord &rec) {
    return ScatterRecord {{1, 1, 1},
                          Ray {rec.portal_origin, rec.portal_dir, incoming_ray.time}};
}

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

/// Ray-AABB intersection test using the slab method.
/// Returns true if the ray intersects the box within [t_min, t_max].
inline bool aabb_hit(Aabb box, const Ray &ray, float t_min, float t_max) {
    for ( int axis = 0; axis < 3; axis++ ) {
        float orig = (axis == 0) ? ray.orig.x : (axis == 1) ? ray.orig.y : ray.orig.z;
        float dir = (axis == 0) ? ray.dir.x : (axis == 1) ? ray.dir.y : ray.dir.z;
        float bmin = (axis == 0) ? box.min.x : (axis == 1) ? box.min.y : box.min.z;
        float bmax = (axis == 0) ? box.max.x : (axis == 1) ? box.max.y : box.max.z;

        float inv_dir = 1.f / dir;
        float t0 = (bmin - orig) * inv_dir;
        float t1 = (bmax - orig) * inv_dir;
        if ( inv_dir < 0.f ) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = (t0 > t_min) ? t0 : t_min;
        t_max = (t1 < t_max) ? t1 : t_max;
        if ( t_max <= t_min ) {
            return false;
        }
    }
    return true;
}

} // namespace rt
