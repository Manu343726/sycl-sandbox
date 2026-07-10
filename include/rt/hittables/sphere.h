#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

/// Spherical geometry primitive.
///
/// Factory:  rt::hittables::sphere(center, radius) -> Sphere
namespace rt::hittables {

class Sphere {
public:
    float3 center; ///< Sphere centre in world space.
    float  radius; ///< Sphere radius.

    Sphere() = default;
    Sphere(float3 center_, float radius_) : center(center_), radius(radius_) {}

    /// Ray-sphere intersection using the quadratic formula.
    ///
    /// Solves ‖ray.origin + t·ray.direction − sphere.center‖² = sphere.radius²
    /// for t.  This expands to a·t² + b·t + c = 0 where:
    ///   a = ‖dir‖²,  b = 2·dot(oc, dir),  c = ‖oc‖² − r²
    /// The conventional factor of 2 in b is omitted and compensated by
    /// dividing by a instead of 2a in the quadratic formula.
    ///
    /// @returns HitRecord if the ray hits the sphere within [t_min, t_max].
    std::optional<HitRecord> hit(const Ray& ray, float t_min, float t_max) const {
        float3 oc = sub(ray.orig, center);
        float a = dot(ray.dir, ray.dir);
        float half_b = dot(oc, ray.dir);
        float c_ = dot(oc, oc) - radius * radius;
        float discriminant = half_b*half_b - a*c_;
        if (discriminant <= 0) return std::nullopt;
        float sqrt_d = sycl::sqrt(discriminant);
        float t = (-half_b - sqrt_d) / a;
        if (t < t_min || t > t_max) t = (-half_b + sqrt_d) / a;
        if (t < t_min || t > t_max) return std::nullopt;
        HitRecord rec;
        rec.t = t;
        rec.p = add(ray.orig, scale(ray.dir, t));
        rec.normal = scale(sub(rec.p, center), 1.f / radius);
        rec.front_face = dot(ray.dir, rec.normal) < 0;
        if (!rec.front_face) rec.normal = scale(rec.normal, -1);
        return rec;
    }
};

/// Creates a Sphere from centre and radius.
inline Sphere sphere(float3 center, float radius) {
    return Sphere(center, radius);
}

} // namespace rt::hittables
