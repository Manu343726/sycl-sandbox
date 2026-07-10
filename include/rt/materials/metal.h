#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"

/// Metallic (reflective) material: reflects incoming rays with a fuzzy
/// perturbation controlled by `fuzz`.  fuzz=0 gives perfect mirror reflection.
///
/// Factory:  rt::materials::metal(albedo, fuzz) -> Metal
///
/// Example:
///   Object sphere = {sphere({0,0,0}, 1), metal({0.9f,0.9f,0.9f}, 0.1f)};
namespace rt::materials {

class Metal {
public:
    float3 albedo; ///< Surface colour.
    float  fuzz;   ///< Roughness: 0 = perfect mirror, 1 = very rough.

    Metal() = default;
    Metal(float3 a, float f) : albedo(a), fuzz(f) {}

    /// Reflects the incoming ray about the surface normal, adding a random
    /// perturbation scaled by `fuzz`.  Returns false if the scattered ray
    /// points into the surface (total internal reflection / absorption).
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const {
        float3 reflected = reflect(norm(in.dir), rec.normal);
        scattered = {rec.p, add(reflected, scale(random_in_unit_sphere(rng), fuzz))};
        attenuation = albedo;
        return dot(scattered.dir, rec.normal) > 0;
    }

    /// Metal materials do not emit light.
    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

/// Creates a Metal material with the given albedo and fuzz factor.
inline Metal metal(float3 albedo, float fuzz) {
    return Metal(albedo, fuzz);
}

} // namespace rt::materials
