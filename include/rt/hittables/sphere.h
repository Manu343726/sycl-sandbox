#pragma once
#include "../math.h"
#include "../types_fwd.h"

namespace rt::hittables {

class Sphere {
public:
    float3 center;
    float  radius;

    Sphere() = default;
    Sphere(float3 c, float r) : center(c), radius(r) {}

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float3 oc = sub(r.orig, center);
        float a = dot(r.dir, r.dir);
        float b = dot(oc, r.dir);
        float c_ = dot(oc, oc) - radius * radius;
        float d = b*b - a*c_;
        if (d <= 0) return false;
        float t = (-b - sycl::sqrt(d)) / a;
        if (t < t_min || t > t_max) t = (-b + sycl::sqrt(d)) / a;
        if (t < t_min || t > t_max) return false;
        rec.t = t; rec.p = add(r.orig, scale(r.dir, t));
        rec.normal = scale(sub(rec.p, center), 1.f / radius);
        rec.front_face = dot(r.dir, rec.normal) < 0;
        if (!rec.front_face) rec.normal = scale(rec.normal, -1);
        return true;
    }
};

inline Sphere make_sphere(float3 center, float radius) {
    return Sphere(center, radius);
}

} // namespace rt::hittables
