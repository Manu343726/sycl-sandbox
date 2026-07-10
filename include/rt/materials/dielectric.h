#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "../helpers.h"

/// Dielectric (transparent) material: refracts and reflects light based on
/// the relative index of refraction and the incident angle (Snell's law +
/// Schlick Fresnel approximation).  Examples: glass, water, diamond.
///
/// Factory:  rt::materials::dielectric(ir) -> Dielectric
///   ir = index of refraction (e.g. 1.5 for glass, 2.4 for diamond).
///
/// Example:
///   Object sphere = {sphere({0,0,0}, 1), dielectric(1.5f)};
namespace rt::materials {

class Dielectric {
public:
    float ir; ///< Index of refraction (>1 for denser media).

    Dielectric() = default;
    explicit Dielectric(float i) : ir(i) {}

    /// Refracts or reflects the incoming ray.  The probability of reflection
    /// increases at grazing angles (Schlick's approximation).  Always returns
    /// true (some ray is always produced).
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

    /// Dielectric materials do not emit light.
    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

/// Creates a Dielectric material with the given index of refraction.
inline Dielectric dielectric(float ir) {
    return Dielectric(ir);
}

} // namespace rt::materials
