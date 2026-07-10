#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

/// Triangle geometry primitive (half of a quad / axis-aligned rectangle).
///
/// Two triangles form a quad: given four corners p0-p1-p2-p3 of a rectangle,
/// use triangle(p0, p1, p2) and triangle(p0, p2, p3).
///
/// Factory: rt::hittables::triangle(a, b, c) -> Triangle
///
/// Example:
///   Object t1 = {triangle(p0, p1, p2), material};
///   Object t2 = {triangle(p0, p2, p3), material};
namespace rt::hittables {

class Triangle {
public:
    float3 a, b, c, normal;

    Triangle() = default;
    Triangle(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }

    /// Ray-triangle intersection (Möller–Trumbore style barycentric).
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
        float u = (d11*d20 - d01*d21)/den;
        float v = (d00*d21 - d01*d20)/den;
        if (u < 0 || v < 0 || u + v > 1) return std::nullopt;
        HitRecord rec;
        rec.t = t; rec.p = p;
        rec.normal = denom<0 ? normal : scale(normal,-1.f);
        rec.front_face = denom < 0;
        return rec;
    }
};

inline Triangle triangle(float3 a, float3 b, float3 c) { return Triangle(a, b, c); }

} // namespace rt::hittables
