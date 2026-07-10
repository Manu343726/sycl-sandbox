#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

namespace rt::materials {

class DiffuseLight {
public:
    float3 emit_color;
    DiffuseLight() = default;
    explicit DiffuseLight(float3 e) : emit_color(e) {}

    std::optional<ScatterRecord> scatter(const Ray&, const HitRecord&, RNG&) const { return std::nullopt; }
    float3 emit(const HitRecord&) const { return emit_color; }
};

inline DiffuseLight diffuse_light(float3 emit) { return DiffuseLight(emit); }

} // namespace rt::materials
