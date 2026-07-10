#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"

namespace rt::materials {

class Metal {
public:
    float3 albedo;
    float  fuzz;
    Metal() = default;
    Metal(float3 a, float f) : albedo(a), fuzz(f) {}

    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const {
        float3 reflected = reflect(norm(in.dir), rec.normal);
        scattered = {rec.p, add(reflected, scale(random_in_unit_sphere(rng), fuzz))};
        attenuation = albedo;
        return dot(scattered.dir, rec.normal) > 0;
    }

    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

inline Metal make_metal(float3 albedo, float fuzz) {
    return Metal(albedo, fuzz);
}

} // namespace rt::materials
