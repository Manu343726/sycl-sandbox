#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"
#include <optional>

/// Dielectric (transparent) material: refracts and reflects light based on
/// Snell's law with the Schlick Fresnel approximation.
///
/// Examples: glass (ir=1.5), diamond (ir=2.4), water (ir=1.33).
///
/// Factory:  rt::materials::dielectric(refractive_index) -> Dielectric
namespace rt::materials {

class Dielectric {
public:
    float refractive_index; ///< Index of refraction (>1 for denser media).

    Dielectric() = default;
    explicit Dielectric(float index) : refractive_index(index) {}

    /// Refracts or reflects the incoming ray.
    ///
    /// The behaviour depends on whether the ray is entering the dielectric
    /// (dot(ray.dir, normal) > 0) or exiting it:
    ///
    /// - **Entering:** the incident medium is air (η≈1), the interior medium
    ///   has index `refractive_index`.  The surface normal points outward,
    ///   so we flip it and use η = refractive_index.
    ///
    /// - **Exiting:** the incident medium is the dielectric, the exterior is
    ///   air.  The normal already points outward, η = 1/refractive_index.
    ///
    /// The cosine of the transmitted angle is computed from Snell's law.
    /// If the discriminant is negative, total internal reflection occurs.
    /// Otherwise the Schlick approximation determines the reflection
    /// probability; a random sample decides between refraction and reflection.
    std::optional<ScatterRecord> scatter(const Ray& incoming_ray, const HitRecord& hit,
                                          RNG& rng) const {
        float3 outward_normal;
        float eta_ratio;
        float cos_theta_i;

        if (dot(incoming_ray.dir, hit.normal) > 0) {
            // Ray is exiting the dielectric
            outward_normal = scale(hit.normal, -1);
            eta_ratio = refractive_index;
            cos_theta_i = refractive_index * dot(incoming_ray.dir, hit.normal)
                          / len(incoming_ray.dir);
        } else {
            // Ray is entering the dielectric
            outward_normal = hit.normal;
            eta_ratio = 1.f / refractive_index;
            cos_theta_i = -dot(incoming_ray.dir, hit.normal) / len(incoming_ray.dir);
        }

        float3 refracted_direction;
        if (refract(incoming_ray.dir, outward_normal, eta_ratio, refracted_direction)
            && rng.next() >= schlick(cos_theta_i, refractive_index)) {
            return ScatterRecord{{1,1,1}, Ray{hit.p, refracted_direction}};
        } else {
            return ScatterRecord{{1,1,1}, Ray{hit.p, reflect(norm(incoming_ray.dir), hit.normal)}};
        }
    }

    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

inline Dielectric dielectric(float refractive_index) {
    return Dielectric(refractive_index);
}

} // namespace rt::materials
