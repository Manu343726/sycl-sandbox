#pragma once

/// @file
/// Textured lambertian (diffuse) material.
///
/// Same diffuse scattering as Lambertian, but the albedo is sampled
/// from a texture at the hit point's parametric coordinates (u, v) and
/// the ray's animation time:
/// @code
///   float3 albedo = textures::sample(texture, rec.u, rec.v, incoming_ray.time);
/// @endcode
/// The texture is a polymorphic `textures::Texture` variant (see
/// rt/textures/texture.h), so the material stays a plain POD that can
/// be uploaded into device material arrays.

#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/helpers.h>
#include <sycl-sandbox/rt/textures/texture.h>
#include <sycl-sandbox/optional.h>

namespace rt::materials {

class TexturedLambertian {
public:
    textures::Texture texture; ///< Polymorphic procedural texture.

    TexturedLambertian() = default;

    explicit TexturedLambertian(textures::Texture t) : texture(t) {
    }

    optional<ScatterRecord>
    scatter(const Ray &incoming_ray, const HitRecord &rec, RNG &rng) const {
        if ( rec.is_portal ) {
            return portal_scatter(incoming_ray, rec);
        }
        // Sample the texture at the hit's parametric coordinates and the
        // ray's time, passing the path RNG through (stochastic textures
        // may use it); UV out-of-range behavior is defined by the texture.
        float3 albedo = textures::sample(texture, rec.u, rec.v, incoming_ray.time, rng);
        float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
        return ScatterRecord {albedo, Ray {rec.p, sub(target, rec.p)}};
    }

    float3 emit(const HitRecord &) const {
        return {0, 0, 0};
    }
};

inline TexturedLambertian textured_lambertian(textures::Texture t) {
    return TexturedLambertian(t);
}

} // namespace rt::materials
