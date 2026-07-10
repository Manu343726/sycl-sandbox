#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/helpers.h>
#include <optional>

/// Metallic (reflective) material: reflects incoming rays with a fuzzy
/// perturbation controlled by `fuzz`.  fuzz=0 gives a perfect mirror.
///
/// Factory:  rt::materials::metal(albedo, fuzz) -> Metal
namespace rt::materials {

class Metal {
public:
    float3 albedo; ///< Surface colour.
    float fuzz;    ///< Roughness: 0 = perfect mirror, 1 = very rough.

    Metal() = default;
    Metal(float3 albedo_, float fuzz_amount) : albedo(albedo_), fuzz(fuzz_amount) {
    }

    /// Reflects the incoming ray about the surface normal, adding a random
    /// perturbation scaled by `fuzz`.  Returns std::nullopt if the scattered
    /// ray points into the surface (absorption).
    std::optional<ScatterRecord>
    scatter(const Ray &incoming_ray, const HitRecord &hit, RNG &rng) const {
        float3 reflected = reflect(norm(incoming_ray.dir), hit.normal);
        Ray scattered {hit.p, add(reflected, scale(random_in_unit_sphere(rng), fuzz))};
        if ( dot(scattered.dir, hit.normal) <= 0 ) {
            return std::nullopt;
        }
        return ScatterRecord {albedo, scattered};
    }

    float3 emit(const HitRecord &) const {
        return {0, 0, 0};
    }
};

inline Metal metal(float3 albedo, float fuzz) {
    return Metal(albedo, fuzz);
}

} // namespace rt::materials
