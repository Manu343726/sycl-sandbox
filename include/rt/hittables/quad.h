#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

/// Parallelogram-quad geometry primitive (two triangles form a quad).
///
/// Factory: rt::hittables::quad(a, b, c) -> Quad
namespace rt::hittables {

class Quad {
public:
    float3 a, b, c, normal;

    Quad() = default;
    Quad(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }

    /// Ray-quad intersection test (barycentric).
    std::optional<HitRecord> hit(const Ray& r, float t_min, float t_max) const {
        float denom = dot(normal, r.dir);
        if (sycl::fabs(denom) < 1e-8f) return std::nullopt;
        float t = dot(sub(a, r.orig), normal) / denom;
        if (t < t_min || t > t_max) return std::nullopt;
        float3 p = add(r.orig, scale(r.dir, t));
        float3 ba = sub(b, a), ca = sub(c, a), pa = sub(p, a);
        float d00 = dot(ba,ba), d01 = dot(ba,ca), d11 = dot(ca,ca);
        float d20 = dot(pa,ba), d21 = dot(pa,ca);
        float den = d00*d11 - d01*d01;
        if (sycl::fabs(den) < 1e-12f) return std::nullopt;
        float u = (d11*d20 - d01*d21)/den, v = (d00*d21 - d01*d20)/den;
        if (u<0||u>1||v<0||v>1) return std::nullopt;
        HitRecord rec;
        rec.t = t; rec.p = p;
        rec.normal = denom<0 ? normal : scale(normal,-1.f);
        rec.front_face = denom < 0;
        return rec;
    }
};

inline Quad quad(float3 a, float3 b, float3 c) { return Quad(a, b, c); }

} // namespace rt::hittables
