#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include <optional>

/// Quadrilateral / parallelogram geometry primitive.
///
/// Defined by three points Q, U, V where the four corners are
/// Q, Q+U, Q+V, Q+U+V  (a parallelogram).
///
/// Factory:  rt::hittables::quad(q, u, v) -> Quad
///
/// You can also use the alternative form with three corner points:
///   quad(a, b, c) where b = a+U, c = a+V.
///
/// Example:
///   Object face = {quad({0,0,0}, {2,0,0}, {0,3,0}), material};
namespace rt::hittables {

class Quad {
public:
    float3 q;       ///< Base corner.
    float3 u, v;    ///< Edge vectors from Q.
    float3 normal;  ///< Unit surface normal (pre-computed).

    Quad() = default;

    /// Construct from base corner Q and two edge vectors U, V.
    Quad(float3 q_, float3 u_, float3 v_)
        : q(q_), u(u_), v(v_) {
        normal = norm(cross(u, v));
    }

    /// Construct from three corner points: a, b, c.
    /// b = a + U,  c = a + V  ⇒  U = b - a,  V = c - a.
    static Quad from_corners(float3 a, float3 b, float3 c) {
        return Quad(a, sub(b, a), sub(c, a));
    }

    /// Ray-quad intersection.  Returns HitRecord on hit.
    std::optional<HitRecord> hit(const Ray& r, float t_min, float t_max) const {
        float denom = dot(normal, r.dir);
        if (sycl::fabs(denom) < 1e-8f) return std::nullopt;
        float t = dot(sub(q, r.orig), normal) / denom;
        if (t < t_min || t > t_max) return std::nullopt;
        float3 p = add(r.orig, scale(r.dir, t));
        float3 pa = sub(p, q);
        // Solve pa = α*u + β*v
        float d00 = dot(u, u), d01 = dot(u, v), d11 = dot(v, v);
        float d20 = dot(pa, u), d21 = dot(pa, v);
        float den = d00*d11 - d01*d01;
        if (sycl::fabs(den) < 1e-12f) return std::nullopt;
        float alpha = (d11*d20 - d01*d21) / den;
        float beta  = (d00*d21 - d01*d20) / den;
        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
            return std::nullopt;
        HitRecord rec;
        rec.t = t; rec.p = p;
        rec.normal = denom<0 ? normal : scale(normal, -1.f);
        rec.front_face = denom < 0;
        return rec;
    }
};

/// Creates a Quad from base corner Q and two edge vectors U, V.
inline Quad quad(float3 q, float3 u, float3 v) {
    return Quad(q, u, v);
}

/// Creates a Quad from three corner points (a, b, c).
/// Equivalent to quad(a, b-a, c-a).
inline Quad quad_from_corners(float3 a, float3 b, float3 c) {
    return Quad::from_corners(a, b, c);
}

} // namespace rt::hittables
