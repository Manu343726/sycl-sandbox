#pragma once
#include "../math.h"
#include "../types_fwd.h"

/// Parallelogram-quad geometry primitive (two triangles form a quad).
///
/// The quad is defined by three corner points a, b, c.  The fourth corner
/// is implicitly b + c - a.  The surface normal is pre-computed in the
/// constructor from (b - a) × (c - a).
///
/// Factory:  rt::hittables::quad(a, b, c) -> Quad
///
/// Example:
///   Object wall = {quad({-2,0,-2}, {2,0,-2}, {-2,3,-2}), lambertian(white)};
namespace rt::hittables {

class Quad {
public:
    float3 a, b, c;   ///< Three corners of the parallelogram.
    float3 normal;    ///< Unit surface normal (pre-computed).

    Quad() = default;

    /// Constructs a quad from three corners.  Normal is computed automatically.
    Quad(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }

    /// Ray-quad intersection test (barycentric).
    /// Returns true and fills `rec` if the ray hits the quad within [t_min, t_max].
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float denom = dot(normal, r.dir);
        if (sycl::fabs(denom) < 1e-8f) return false;
        float t = dot(sub(a, r.orig), normal) / denom;
        if (t < t_min || t > t_max) return false;
        float3 p = add(r.orig, scale(r.dir, t));
        float3 ba = sub(b, a), ca = sub(c, a), pa = sub(p, a);
        float d00 = dot(ba,ba), d01 = dot(ba,ca), d11 = dot(ca,ca);
        float d20 = dot(pa,ba), d21 = dot(pa,ca);
        float den = d00*d11 - d01*d01;
        if (sycl::fabs(den) < 1e-12f) return false;
        float u = (d11*d20 - d01*d21)/den, v = (d00*d21 - d01*d20)/den;
        if (u<0||u>1||v<0||v>1) return false;
        rec.t = t; rec.p = p;
        rec.normal = denom<0 ? normal : scale(normal,-1.f);
        rec.front_face = denom < 0;
        return true;
    }
};

/// Creates a Quad from three corner points.
inline Quad quad(float3 a, float3 b, float3 c) {
    return Quad(a, b, c);
}

} // namespace rt::hittables
