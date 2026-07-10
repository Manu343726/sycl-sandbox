#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <optional>

namespace rt::hittables {

class Sphere {
public:
    float3 center;
    float radius;

    Sphere() = default;
    Sphere(float3 center_, float radius_) : center(center_), radius(radius_) {
    }

    /// Ray-sphere intersection using the reduced quadratic formula.
    std::optional<HitRecord> hit(const Ray &ray, float t_min, float t_max) const {
        // Compute the vector from the sphere centre to the ray origin
        float3 oc = sub(ray.orig, center);

        // Compute quadratic coefficients: a·t² + 2·half_b·t + c = 0
        // a = ‖dir‖²,  half_b = oc·dir,  c = ‖oc‖² − r²
        float a = dot(ray.dir, ray.dir);
        float half_b = dot(oc, ray.dir);
        float c_ = dot(oc, oc) - radius * radius;

        // Solve the reduced quadratic: t = (-half_b ± √(half_b² − a·c)) / a
        float discriminant = half_b * half_b - a * c_;
        if ( discriminant <= 0 ) {
            return std::nullopt;
        }

        float sqrt_d = sycl::sqrt(discriminant);
        float t = (-half_b - sqrt_d) / a;
        if ( t < t_min || t > t_max ) {
            t = (-half_b + sqrt_d) / a;
        }
        if ( t < t_min || t > t_max ) {
            return std::nullopt;
        }

        // Fill the HitRecord with the intersection point and the outward-facing normal
        HitRecord rec;
        rec.t = t;
        rec.p = add(ray.orig, scale(ray.dir, t));
        rec.normal = scale(sub(rec.p, center), 1.f / radius);
        rec.front_face = dot(ray.dir, rec.normal) < 0;
        if ( !rec.front_face ) {
            rec.normal = scale(rec.normal, -1);
        }
        return rec;
    }
};

inline Sphere sphere(float3 center, float radius) {
    return Sphere(center, radius);
}

} // namespace rt::hittables
