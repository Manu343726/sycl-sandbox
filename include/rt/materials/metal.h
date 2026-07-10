#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"
#include <optional>

namespace rt::materials {

class Metal {
public:
    float3 albedo;
    float  fuzz;
    Metal() = default;
    Metal(float3 a, float f) : albedo(a), fuzz(f) {}

    std::optional<ScatterRecord> scatter(const Ray& in, const HitRecord& rec, RNG& rng) const {
        float3 reflected = reflect(norm(in.dir), rec.normal);
        Ray scattered{rec.p, add(reflected, scale(random_in_unit_sphere(rng), fuzz))};
        if (dot(scattered.dir, rec.normal) <= 0) return std::nullopt;
        return ScatterRecord{albedo, scattered};
    }

    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

inline Metal metal(float3 albedo, float fuzz) { return Metal(albedo, fuzz); }

} // namespace rt::materials
