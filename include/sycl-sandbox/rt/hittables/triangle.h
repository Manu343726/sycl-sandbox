#pragma once
#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/aabb.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/optional.h>
#include <sycl-sandbox/math.h>

namespace rt::hittables {

class Triangle {
public:
    float3 a, b, c;
    float3 normal;
    float3 n0{}, n1{}, n2{};  ///< per-vertex normals (smooth shading)
    bool smooth = false;

    Triangle() = default;
    Triangle(float3 a_, float3 b_, float3 c_, bool smooth_ = false)
        : a(a_), b(b_), c(c_), smooth(smooth_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        float3 n = cross(ab, ac);
        normal = len2(n) < 1e-12f ? float3{0, 0, 0} : norm(n);
        if (smooth) { n0 = normal; n1 = normal; n2 = normal; }
    }
    Triangle(float3 a_, float3 b_, float3 c_,
             float3 n0_, float3 n1_, float3 n2_)
        : a(a_), b(b_), c(c_), n0(n0_), n1(n1_), n2(n2_), smooth(true) {
        float3 ab = sub(b, a), ac = sub(c, a);
        float3 n = cross(ab, ac);
        normal = len2(n) < 1e-12f ? float3{0, 0, 0} : norm(n);
    }

    /// Return the axis-aligned bounding box enclosing this triangle.
    Aabb aabb() const {
        Aabb box = aabb_from_points(a, b);
        return aabb_merge(box, aabb_from_points(c, c));
    }

    /// Ray-triangle intersection using barycentric coordinates (Cramer's rule).
    ///
    /// \param ctx per-call kernel context: records a "hit_triangle"
    ///        profiler zone (decimated by work-item id) and reports the
    ///        hit test to the trace collector.
    optional<HitRecord> hit(const Ray &ray, float t_min, float t_max,
                            const Context &ctx = Context{}) const {
        PROFILER_FUNCTION();
        ctx.collector.on_hit_test(HittableType::Triangle, ray, t_min, t_max);
        // Compute the ray-plane intersection; reject rays parallel to the plane
        float denom = dot(normal, ray.dir);
        if ( math::fabs(denom) < 1e-8f ) {
            return nullopt;
        }

        // Compute the distance t along the ray; reject if outside the allowed range
        float t = dot(sub(a, ray.orig), normal) / denom;
        if ( t < t_min || t > t_max ) {
            return nullopt;
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
        if ( math::fabs(denominator) < 1e-12f ) {
            return nullopt;
        }

        float u = (d11 * d20 - d01 * d21) / denominator;
        float v = (d00 * d21 - d01 * d20) / denominator;

        // The hit point is inside the triangle only if u ≥ 0, v ≥ 0, and u+v ≤ 1
        if ( u < 0 || v < 0 || u + v > 1 ) {
            return nullopt;
        }

        // Fill the HitRecord; flip the normal if the ray hit from inside
        HitRecord rec;
        rec.t = t;
        rec.p = hit_point;
        if (smooth) {
            float w = 1.0f - u - v;  // barycentric weight of vertex a
            float3 sn = norm(add(add(scale(n0, w), scale(n1, u)), scale(n2, v)));
            rec.normal = (denom < 0) ? sn : scale(sn, -1.f);
        } else {
            rec.normal = (denom < 0) ? normal : scale(normal, -1.f);
        }
        rec.front_face = denom < 0;
        // Barycentric coordinates double as texture (u, v): u weights
        // vertex b, v weights vertex c (both in [0,1] inside the triangle).
        rec.u = u;
        rec.v = v;
        return rec;
    }

    /// Map parametric coordinates back onto the triangle surface (inverse
    /// of hit()'s barycentric mapping): p = a + u·(b−a) + v·(c−a).  Used by
    /// portals to place the exit hit at the entry hit's UVs.
    HitRecord point_at_uv(float u, float v) const {
        HitRecord rec;
        rec.p = add(add(a, scale(sub(b, a), u)), scale(sub(c, a), v));
        rec.normal = normal;
        rec.t = 0.f;
        rec.front_face = true;
        rec.u = u;
        rec.v = v;
        return rec;
    }
};

inline Triangle triangle(float3 a, float3 b, float3 c) {
    return Triangle(a, b, c);
}

} // namespace rt::hittables
