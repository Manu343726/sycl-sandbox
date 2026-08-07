#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/helpers.h>
#include <sycl-sandbox/optional.h>
#include <sycl-sandbox/profiler.h>

namespace rt::materials {

class Lambertian {
public:
    float3 albedo;
    Lambertian() = default;
    explicit Lambertian(float3 a) : albedo(a) {
    }

    optional<ScatterRecord> scatter(const Ray &incoming_ray, const HitRecord &rec, RNG &rng) const {

        if ( rec.is_portal ) {
            return portal_scatter(incoming_ray, rec);
        }
        PROFILER_ZONE("Lambertian_scatter");
        float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
        return ScatterRecord {albedo, Ray {rec.p, sub(target, rec.p)}};
    }

    float3 emit(const HitRecord &) const {

        PROFILER_ZONE("Lambertian_emit");
        return {0, 0, 0};
    }
};

inline Lambertian lambertian(float3 albedo) {
    return Lambertian(albedo);
}

} // namespace rt::materials
