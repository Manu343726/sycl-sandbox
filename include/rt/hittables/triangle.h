#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

/// Triangle geometry primitive (half of a quad / arbitrary mesh face).
///
/// Uses barycentric ray–triangle intersection (Möller–Trumbore style).
/// Two triangles form a quad:  triangle(p0,p1,p2) + triangle(p0,p2,p3).
///
/// Factory:  rt::hittables::triangle(a, b, c) -> Triangle
namespace rt::hittables {

class Triangle {
public:
    float3 a, b, c;   ///< Three vertices in counter-clockwise order.
    float3 normal;     ///< Unit surface normal (pre-computed from cross(b-a, c-a)).

    Triangle() = default;
    Triangle(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }

    /// Ray-triangle intersection.
    ///
    /// First computes the ray–plane intersection, then tests whether the
    /// hit point lies inside the triangle using barycentric coordinates.
    ///
    /// The barycentric coordinates (u,v) satisfy:
    ///   hit_point = a + u·(b-a) + v·(c-a)
    /// where u ≥ 0, v ≥ 0, and u+v ≤ 1 for points inside the triangle.
    /// The 2x2 linear system is solved via Cramer's rule.
    std::optional<HitRecord> hit(const Ray& ray, float t_min, float t_max) const {
        float denom = dot(normal, ray.dir);
        if (sycl::fabs(denom) < 1e-8f) return std::nullopt;
        float t = dot(sub(a, ray.orig), normal) / denom;
        if (t < t_min || t > t_max) return std::nullopt;
        float3 hit_point = add(ray.orig, scale(ray.dir, t));
        float3 ba = sub(b, a), ca = sub(c, a), pa = sub(hit_point, a);

        // Dot products for the 2x2 linear system:  [ba·ba  ba·ca] [u] = [pa·ba]
        //                                          [ca·ba  ca·ca] [v]   [pa·ca]
        float d00 = dot(ba, ba), d01 = dot(ba, ca), d11 = dot(ca, ca);
        float d20 = dot(pa, ba), d21 = dot(pa, ca);
        float denominator = d00*d11 - d01*d01;  // |ba × ca|²  (area scaling)
        if (sycl::fabs(denominator) < 1e-12f) return std::nullopt;
        float u = (d11*d20 - d01*d21) / denominator;
        float v = (d00*d21 - d01*d20) / denominator;
        if (u < 0 || v < 0 || u + v > 1) return std::nullopt;  // outside triangle
        HitRecord rec;
        rec.t = t; rec.p = hit_point;
        rec.normal = denom<0 ? normal : scale(normal,-1.f);
        rec.front_face = denom < 0;
        return rec;
    }
};

inline Triangle triangle(float3 a, float3 b, float3 c) { return Triangle(a, b, c); }

} // namespace rt::hittables
