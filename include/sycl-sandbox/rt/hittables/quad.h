#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/optional.h>

namespace rt::hittables {

class Quad {
public:
    float3 base;
    float3 edge_u;
    float3 edge_v;
    float3 normal;

    Quad() = default;
    Quad(float3 base_, float3 edge_u_, float3 edge_v_)
        : base(base_), edge_u(edge_u_), edge_v(edge_v_) {
        normal = norm(cross(edge_u, edge_v));
    }

    static Quad from_corners(float3 a, float3 b, float3 c) {
        return Quad(a, sub(b, a), sub(c, a));
    }

    /// Ray-quad intersection using barycentric (α, β) coordinates.
    optional<HitRecord> hit(const Ray &ray, float t_min, float t_max) const {
        // Compute the ray-plane intersection; reject rays parallel to the plane
        float denom = dot(normal, ray.dir);
        if ( sycl::fabs(denom) < 1e-8f ) {
            return nullopt;
        }

        // Compute the distance t along the ray; reject if outside the allowed range
        float t = dot(sub(base, ray.orig), normal) / denom;
        if ( t < t_min || t > t_max ) {
            return nullopt;
        }

        // Compute the hit point and the vector from the base corner to it
        float3 hit_point = add(ray.orig, scale(ray.dir, t));
        float3 pa = sub(hit_point, base);

        // Solve pa = α·edge_u + β·edge_v using Cramer's rule:
        //   [edge_u·edge_u  edge_u·edge_v] [α] = [pa·edge_u]
        //   [edge_v·edge_u  edge_v·edge_v] [β]   [pa·edge_v]
        float d00 = dot(edge_u, edge_u);
        float d01 = dot(edge_u, edge_v);
        float d11 = dot(edge_v, edge_v);
        float d20 = dot(pa, edge_u);
        float d21 = dot(pa, edge_v);
        float denominator = d00 * d11 - d01 * d01;
        if ( sycl::fabs(denominator) < 1e-12f ) {
            return nullopt;
        }

        float alpha = (d11 * d20 - d01 * d21) / denominator;
        float beta = (d00 * d21 - d01 * d20) / denominator;

        // The hit point is inside the quad only if α, β ∈ [0, 1]
        if ( alpha < 0 || alpha > 1 || beta < 0 || beta > 1 ) {
            return nullopt;
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

inline Quad quad(float3 base, float3 edge_u, float3 edge_v) {
    return Quad(base, edge_u, edge_v);
}

inline Quad quad_from_corners(float3 a, float3 b, float3 c) {
    return Quad::from_corners(a, b, c);
}

/// Creates an axis-aligned Quad from bounds.
/// @param primary  The fixed axis (0=X, 1=Y, 2=Z).
inline Quad
quad(int primary, float axis_value, float min_s, float max_s, float min_t, float max_t) {
    int second = (primary + 1) % 3, third = (primary + 2) % 3;
    auto corner = [&](int index) -> float3 {
        float components[3] = {0, 0, 0};
        components[primary] = axis_value;
        components[second] = (index & 1) ? max_s : min_s;
        components[third] = (index >> 1) ? max_t : min_t;
        return {components[0], components[1], components[2]};
    };
    float3 a = corner(0), b = corner(1), c = corner(2);
    return Quad(a, sub(c, a), sub(b, a));
}

} // namespace rt::hittables
