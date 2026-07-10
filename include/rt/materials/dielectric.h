#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"

namespace rt::materials {

class Dielectric {
public:
    float ir;
    Dielectric() = default;
    explicit Dielectric(float i) : ir(i) {}

    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const {
        attenuation = {1,1,1};
        float3 outward_n;
        float eta_ratio, cos;
        if (dot(in.dir, rec.normal) > 0) {
            outward_n = scale(rec.normal, -1);
            eta_ratio = ir;
            cos = ir * dot(in.dir, rec.normal) / len(in.dir);
        } else {
            outward_n = rec.normal;
            eta_ratio = 1.f / ir;
            cos = -dot(in.dir, rec.normal) / len(in.dir);
        }
        float3 refracted;
        if (refract(in.dir, outward_n, eta_ratio, refracted) && rng.next() >= schlick(cos, ir))
            scattered = {rec.p, refracted};
        else
            scattered = {rec.p, reflect(norm(in.dir), rec.normal)};
        return true;
    }

    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

inline Dielectric dielectric(float ir) {
    return Dielectric(ir);
}

} // namespace rt::materials
