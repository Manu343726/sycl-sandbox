#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <optional>

namespace rt::hittables {

class Triangle {
public:
    float3 a, b, c;
    float3 normal;

    Triangle() = default;
    Triangle(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }

    /// Ray-triangle intersection using barycentric coordinates (Cramer's rule).
    std::optional<HitRecord> hit(const Ray &ray, float t_min, float t_max) const {
        // Compute the ray-plane intersection; reject rays parallel to the plane
        float denom = dot(normal, ray.dir);
        if ( sycl::fabs(denom) < 1e-8f ) {
            return std::nullopt;
        }

        // Compute the distance t along the ray; reject if outside the allowed range
        float t = dot(sub(a, ray.orig), normal) / denom;
        if ( t < t_min || t > t_max ) {
            return std::nullopt;
        }

        // Compute the hit point and the vector from vertex a to it
        float3 hit_point = add(ray.orig, scale(ray.dir, t));

        // Compute edge vectors (b−a, c−a) for barycentric coordinates
        float3 ba = sub(b, a);
        float3 ca = sub(c, a);
        float3 pa = sub(hit_point, a);

        // Solve the 2x2 linear system  [ba·ba  ba·ca] [u] = [pa·ba]
        //                              [ca·ba  ca·ca] [v]   [pa·ca]
        float d00 = dot(ba, ba), d01 = dot(ba, ca), d11 = dot(ca, ca);
        float d20 = dot(pa, ba), d21 = dot(pa, ca);
        float denominator = d00 * d11 - d01 * d01; // |ba × ca|²  (twice the triangle area)
        if ( sycl::fabs(denominator) < 1e-12f ) {
            return std::nullopt;
        }

        float u = (d11 * d20 - d01 * d21) / denominator;
        float v = (d00 * d21 - d01 * d20) / denominator;

        // The hit point is inside the triangle only if u ≥ 0, v ≥ 0, and u+v ≤ 1
        if ( u < 0 || v < 0 || u + v > 1 ) {
            return std::nullopt;
        }

        // Fill the HitRecord; flip the normal if the ray hit from inside
        HitRecord rec;
        rec.t = t;
        rec.p = hit_point;
        rec.normal = (denom < 0) ? normal : scale(normal, -1.f);
        rec.front_face = denom < 0;
        return rec;
    }
};

inline Triangle triangle(float3 a, float3 b, float3 c) {
    return Triangle(a, b, c);
}

} // namespace rt::hittables
