#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/optional.h>

namespace rt::materials {

class DiffuseLight {
public:
    float3 emit_color;
    DiffuseLight() = default;
    explicit DiffuseLight(float3 e) : emit_color(e) {
    }

    Optional<ScatterRecord> scatter(const Ray &, const HitRecord &, RNG &) const {
        return Optional<ScatterRecord>{};
    }
    float3 emit(const HitRecord &) const {
        return emit_color;
    }
};

inline DiffuseLight diffuse_light(float3 emit) {
    return DiffuseLight(emit);
}

} // namespace rt::materials
