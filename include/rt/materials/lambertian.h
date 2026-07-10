#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"

/// Diffuse (Lambertian) material: scatters incoming rays in random directions
/// on the hemisphere oriented by the surface normal.  Produces a matte finish.
///
/// Factory:  rt::materials::lambertian(albedo) -> Lambertian
///
/// Example:
///   Object wall = {quad(…), lambertian({0.8f, 0.2f, 0.2f})};
namespace rt::materials {

class Lambertian {
public:
    float3 albedo; ///< Surface colour (reflectance).

    Lambertian() = default;
    explicit Lambertian(float3 a) : albedo(a) {}

    /// Scatters the ray in a cosine-weighted random hemisphere direction.
    /// Always returns true (always scatters).
    bool scatter(const Ray&, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const {
        float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
        scattered = {rec.p, sub(target, rec.p)};
        attenuation = albedo;
        return true;
    }

    /// Lambertian materials do not emit light.
    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

/// Creates a Lambertian material with the given albedo.
inline Lambertian lambertian(float3 albedo) {
    return Lambertian(albedo);
}

} // namespace rt::materials
