#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/optional.h>
#include <sycl-sandbox/profiler.h>

namespace rt::materials {

class DiffuseLight {
public:
    float3 emit_color;
    DiffuseLight() = default;
    explicit DiffuseLight(float3 e) : emit_color(e) {
    }

    optional<ScatterRecord> scatter(const Ray &incoming_ray, const HitRecord &rec, RNG &) const {
        // Portal records teleport (emission is checked before scatter, so
        // an emissive portal terminates the path instead).
        if ( rec.is_portal ) {
            return portal_scatter(incoming_ray, rec);
        }
        return nullopt;
    }
    float3 emit(const HitRecord &) const {
        return emit_color;
    }
};

inline DiffuseLight diffuse_light(float3 emit) {
    return DiffuseLight(emit);
}

} // namespace rt::materials
