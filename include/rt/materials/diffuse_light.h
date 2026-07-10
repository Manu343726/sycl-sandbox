#pragma once
#include "../math.h"
#include "../types_fwd.h"

/// Diffuse light (emissive) material: does not scatter rays — instead it
/// emits light with a given colour/intensity.  Rays that hit a DiffuseLight
/// surface terminate and return the emitted colour.
///
/// Factory:  rt::materials::diffuse_light(emit) -> DiffuseLight
///
/// Example:
///   Object light = {quad(…), diffuse_light({15,15,15})};
namespace rt::materials {

class DiffuseLight {
public:
    float3 emit_color; ///< Emitted radiance (colour × intensity).

    DiffuseLight() = default;
    explicit DiffuseLight(float3 e) : emit_color(e) {}

    /// DiffuseLight does not scatter — ray terminates at the hit point.
    bool scatter(const Ray&, const HitRecord&, float3&, Ray&, RNG&) const { return false; }

    /// Returns the emitted colour.
    float3 emit(const HitRecord&) const { return emit_color; }
};

/// Creates a DiffuseLight material with the given emission colour.
inline DiffuseLight diffuse_light(float3 emit) {
    return DiffuseLight(emit);
}

} // namespace rt::materials
