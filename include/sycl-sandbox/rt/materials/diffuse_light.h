#pragma once
#include <sycl-sandbox/context.h>
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

    optional<ScatterRecord> scatter(const Ray &incoming_ray, const HitRecord &rec, RNG &,
                                    const Context &ctx = Context{}) const {
        PROFILER_FUNCTION();
        ctx.collector.on_scatter(MaterialType::DiffuseLight, incoming_ray, rec);
        // Portal records teleport (emission is checked before scatter, so
        // an emissive portal terminates the path instead).
        if ( rec.is_portal ) {
            return portal_scatter(incoming_ray, rec);
        }
        return nullopt;
    }
    float3 emit(const HitRecord &hit, const Context &ctx = Context{}) const {
        PROFILER_FUNCTION();
        ctx.collector.on_emit(MaterialType::DiffuseLight, hit);
        return emit_color;
    }
};

inline DiffuseLight diffuse_light(float3 emit) {
    return DiffuseLight(emit);
}

} // namespace rt::materials
