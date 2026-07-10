#pragma once
#include "../math.h"
#include "../types_fwd.h"

namespace rt::materials {

class DiffuseLight {
public:
    float3 emit_color;
    DiffuseLight() = default;
    explicit DiffuseLight(float3 e) : emit_color(e) {}

    bool scatter(const Ray&, const HitRecord&, float3&, Ray&, RNG&) const { return false; }
    float3 emit(const HitRecord&) const { return emit_color; }
};

inline DiffuseLight make_diffuse_light(float3 emit) {
    return DiffuseLight(emit);
}

} // namespace rt::materials
