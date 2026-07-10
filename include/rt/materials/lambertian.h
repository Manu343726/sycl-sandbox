#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"
#include <optional>

namespace rt::materials {

class Lambertian {
public:
    float3 albedo;
    Lambertian() = default;
    explicit Lambertian(float3 a) : albedo(a) {
    }

    std::optional<ScatterRecord> scatter(const Ray &, const HitRecord &rec, RNG &rng) const {
        float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
        return ScatterRecord {albedo, Ray {rec.p, sub(target, rec.p)}};
    }

    float3 emit(const HitRecord &) const {
        return {0, 0, 0};
    }
};

inline Lambertian lambertian(float3 albedo) {
    return Lambertian(albedo);
}

} // namespace rt::materials
