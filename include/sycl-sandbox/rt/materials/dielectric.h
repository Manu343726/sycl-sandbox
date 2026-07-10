#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/helpers.h>
#include <optional>

namespace rt::materials {

class Dielectric {
public:
    float refractive_index;

    Dielectric() = default;
    explicit Dielectric(float index) : refractive_index(index) {
    }

    /// Refract or reflect the incoming ray using Snell's law + Schlick Fresnel.
    std::optional<ScatterRecord>
    scatter(const Ray &incoming_ray, const HitRecord &hit, RNG &rng) const {
        // Determine whether the ray is entering the dielectric or exiting it
        float3 outward_normal;
        float eta_ratio;
        float cos_theta_i;

        if ( dot(incoming_ray.dir, hit.normal) > 0 ) {
            // Ray is exiting the dielectric (going from dense to air)
            outward_normal = scale(hit.normal, -1);
            eta_ratio = refractive_index;
            cos_theta_i =
                refractive_index * dot(incoming_ray.dir, hit.normal) / len(incoming_ray.dir);
        } else {
            // Ray is entering the dielectric (going from air to dense)
            outward_normal = hit.normal;
            eta_ratio = 1.f / refractive_index;
            cos_theta_i = -dot(incoming_ray.dir, hit.normal) / len(incoming_ray.dir);
        }

        // Use Schlick's approximation to decide between refraction and reflection
        float3 refracted_direction;
        if ( refract(incoming_ray.dir, outward_normal, eta_ratio, refracted_direction) &&
             rng.next() >= schlick(cos_theta_i, refractive_index) ) {
            return ScatterRecord {{1, 1, 1}, Ray {hit.p, refracted_direction}};
        } else {
            return ScatterRecord {{1, 1, 1},
                                  Ray {hit.p, reflect(norm(incoming_ray.dir), hit.normal)}};
        }
    }

    float3 emit(const HitRecord &) const {
        return {0, 0, 0};
    }
};

inline Dielectric dielectric(float refractive_index) {
    return Dielectric(refractive_index);
}

} // namespace rt::materials
