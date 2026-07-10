#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"

namespace rt::materials {

class Lambertian {
public:
    float3 albedo;
    Lambertian() = default;
    explicit Lambertian(float3 a) : albedo(a) {}

    bool scatter(const Ray&, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const {
        float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
        scattered = {rec.p, sub(target, rec.p)};
        attenuation = albedo;
        return true;
    }

    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

inline Lambertian make_lambertian(float3 albedo) {
    return Lambertian(albedo);
}

} // namespace rt::materials
