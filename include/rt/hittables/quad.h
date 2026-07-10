#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

/// Quadrilateral / parallelogram geometry primitive.
///
/// Defined by a base corner Q and two edge vectors U, V.
/// The four vertices are Q, Q+U, Q+V, and Q+U+V.
///
/// Factories:
///   rt::hittables::quad(q, u, v)              — from base + edge vectors
///   rt::hittables::quad_from_corners(a, b, c)  — from three corner points
///
/// Example:
///   Object face = {quad({0,0,0}, {2,0,0}, {0,3,0}), material};
namespace rt::hittables {

class Quad {
public:
    float3 base;   ///< Base corner Q.
    float3 edge_u; ///< First edge vector U (from Q to the next vertex).
    float3 edge_v; ///< Second edge vector V (from Q to the third vertex).
    float3 normal; ///< Unit surface normal (pre-computed from cross(U, V)).

    Quad() = default;

    /// Construct from base corner Q and two edge vectors U, V.
    Quad(float3 base_, float3 edge_u_, float3 edge_v_)
        : base(base_), edge_u(edge_u_), edge_v(edge_v_) {
        normal = norm(cross(edge_u, edge_v));
    }

    /// Construct from three corner points: a, b, c where b = a+U and c = a+V.
    static Quad from_corners(float3 a, float3 b, float3 c) {
        return Quad(a, sub(b, a), sub(c, a));
    }

    /// Ray-quad intersection.
    ///
    /// Solves for the hit point on the plane, then checks if it lies within
    /// the parallelogram:  hit_point = Q + α·U + β·V  with α,β ∈ [0,1].
    /// The 2x2 linear system is solved via Cramer's rule.
    std::optional<HitRecord> hit(const Ray &ray, float t_min, float t_max) const {
        float denom = dot(normal, ray.dir);
        if ( sycl::fabs(denom) < 1e-8f )
            return std::nullopt;
        float t = dot(sub(base, ray.orig), normal) / denom;
        if ( t < t_min || t > t_max )
            return std::nullopt;
        float3 hit_point = add(ray.orig, scale(ray.dir, t));
        float3 pa = sub(hit_point, base);

        // Solve pa = α·edge_u + β·edge_v  using Cramer's rule:
        //   [edge_u·edge_u  edge_u·edge_v] [α] = [pa·edge_u]
        //   [edge_v·edge_u  edge_v·edge_v] [β]   [pa·edge_v]
        float d00 = dot(edge_u, edge_u), d01 = dot(edge_u, edge_v);
        float d11 = dot(edge_v, edge_v), d20 = dot(pa, edge_u), d21 = dot(pa, edge_v);
        float denominator = d00 * d11 - d01 * d01;
        if ( sycl::fabs(denominator) < 1e-12f )
            return std::nullopt;
        float alpha = (d11 * d20 - d01 * d21) / denominator;
        float beta = (d00 * d21 - d01 * d20) / denominator;
        if ( alpha < 0 || alpha > 1 || beta < 0 || beta > 1 )
            return std::nullopt;
        HitRecord rec;
        rec.t = t;
        rec.p = hit_point;
        rec.normal = denom < 0 ? normal : scale(normal, -1.f);
        rec.front_face = denom < 0;
        return rec;
    }
};

/// Creates a Quad from base corner Q and two edge vectors U, V.
inline Quad quad(float3 base, float3 edge_u, float3 edge_v) {
    return Quad(base, edge_u, edge_v);
}

/// Creates a Quad from three corner points (a, b, c).
/// Equivalent to quad(a, b-a, c-a).
inline Quad quad_from_corners(float3 a, float3 b, float3 c) {
    return Quad::from_corners(a, b, c);
}

} // namespace rt::hittables
