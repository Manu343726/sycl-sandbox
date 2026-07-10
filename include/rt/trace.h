#pragma once
#include "types.h"
#include "camera.h"

namespace rt {

// ── helpers ────────────────────────────────────────────────────────────
inline float3 random_hemi(RNG& r, float3 n) {
    float u1 = r.next(), u2 = r.next();
    float rr = sycl::sqrt(u1);
    float theta = 2.f * 3.14159265f * u2;
    float x = rr * sycl::cos(theta);
    float y = rr * sycl::sin(theta);
    float z = sycl::sqrt(1.f - u1);
    float3 up = {0,1,0};
    if (sycl::fabs(n.y) > 0.9f) up = (float3){1,0,0};
    float3 u = norm(cross(up, n));
    float3 v = cross(n, u);
    return add(add(scale(u, x), scale(v, y)), scale(n, z));
}

inline float3 random_in_unit_sphere(RNG& r) {
    for (int i = 0; i < 100; i++) {
        float3 p = {2*r.next()-1, 2*r.next()-1, 2*r.next()-1};
        if (len2(p) < 1) return p;
    }
    return {0,0,1};
}

inline float3 reflect(float3 v, float3 n) {
    return sub(v, scale(n, 2 * dot(v, n)));
}

inline bool refract(float3 v, float3 n, float eta, float3& out) {
    float3 uv = norm(v);
    float dt = dot(uv, n);
    float d = 1 - eta*eta*(1 - dt*dt);
    if (d <= 0) return false;
    out = sub(scale(sub(uv, scale(n, dt)), eta), scale(n, sycl::sqrt(d)));
    return true;
}

inline float schlick(float c, float ir) {
    float r0 = (1-ir)/(1+ir);
    r0 *= r0;
    return r0 + (1-r0)*sycl::pow(1-c, 5);
}

// ── hit functions ──────────────────────────────────────────────────────
inline bool hit_sphere(const Sphere& s, const Ray& r, float t_min, float t_max, HitRecord& rec) {
    float3 oc = sub(r.orig, s.center);
    float a = dot(r.dir, r.dir);
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float d = b*b - a*c;
    if (d <= 0) return false;
    float t = (-b - sycl::sqrt(d)) / a;
    if (t < t_min || t > t_max) t = (-b + sycl::sqrt(d)) / a;
    if (t < t_min || t > t_max) return false;
    rec.t = t;
    rec.p = add(r.orig, scale(r.dir, t));
    rec.normal = scale(sub(rec.p, s.center), 1.f / s.radius);
    rec.front_face = dot(r.dir, rec.normal) < 0;
    if (!rec.front_face) rec.normal = scale(rec.normal, -1);
    rec.mat_type = s.mat_type; rec.albedo = s.albedo;
    rec.fuzz = s.fuzz; rec.ir = s.ir; rec.emit = s.emit;
    return true;
}

inline bool hit_quad(const Quad& q, const Ray& r, float t_min, float t_max, HitRecord& rec) {
    float denom = dot(q.normal, r.dir);
    if (sycl::fabs(denom) < 1e-8f) return false;
    float t = dot(sub(q.a, r.orig), q.normal) / denom;
    if (t < t_min || t > t_max) return false;
    float3 p = add(r.orig, scale(r.dir, t));
    float3 ba = sub(q.b, q.a), ca = sub(q.c, q.a), pa = sub(p, q.a);
    float d00 = dot(ba, ba), d01 = dot(ba, ca), d11 = dot(ca, ca);
    float d20 = dot(pa, ba), d21 = dot(pa, ca);
    float den = d00 * d11 - d01 * d01;
    if (sycl::fabs(den) < 1e-12f) return false;
    float u = (d11 * d20 - d01 * d21) / den;
    float v = (d00 * d21 - d01 * d20) / den;
    if (u < 0.f || u > 1.f || v < 0.f || v > 1.f) return false;
    rec.t = t; rec.p = p;
    rec.normal = denom < 0.f ? q.normal : scale(q.normal, -1.f);
    rec.front_face = denom < 0.f;
    rec.mat_type = q.mat_type; rec.albedo = q.albedo; rec.emit = q.emit;
    return true;
}

// ── generic tracer ─────────────────────────────────────────────────────
template <typename PrimT, typename HitFn>
float3 trace(const Ray& ray, const PrimT* prims, int count,
             HitFn&& hit_fn, int bounces, RNG& rng) {
    float3 att = {1,1,1};
    Ray r = ray;
    for (int b = 0; b < bounces; b++) {
        HitRecord rec;
        float closest = 1e30f;
        bool hit_any = false;
        for (int i = 0; i < count; i++) {
            if (hit_fn(prims[i], r, 0.001f, closest, rec)) {
                closest = rec.t;
                hit_any = true;
            }
        }
        if (!hit_any) return {0,0,0};

        switch (rec.mat_type) {
        case MatType::LAMBERTIAN: {
            float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
            r = {rec.p, sub(target, rec.p)};
            att = mul(att, rec.albedo);
            break;
        }
        case MatType::METAL: {
            float3 rfl = reflect(norm(r.dir), rec.normal);
            r = {rec.p, add(rfl, scale(random_in_unit_sphere(rng), rec.fuzz))};
            if (dot(r.dir, rec.normal) <= 0) return {0,0,0};
            att = mul(att, rec.albedo);
            break;
        }
        case MatType::DIELECTRIC: {
            float3 outward_n;
            float eta, cos;
            if (dot(r.dir, rec.normal) > 0) {
                outward_n = scale(rec.normal, -1);
                eta = rec.ir;
                cos = rec.ir * dot(r.dir, rec.normal) / len(r.dir);
            } else {
                outward_n = rec.normal;
                eta = 1.f / rec.ir;
                cos = -dot(r.dir, rec.normal) / len(r.dir);
            }
            float3 refracted;
            float reflect_prob;
            if (refract(r.dir, outward_n, eta, refracted))
                reflect_prob = schlick(cos, rec.ir);
            else
                reflect_prob = 1.f;
            if (rng.next() < reflect_prob)
                r = {rec.p, reflect(norm(r.dir), rec.normal)};
            else
                r = {rec.p, refracted};
            att = mul(att, rec.albedo);
            break;
        }
        case MatType::DIFFUSE_LIGHT:
            return mul(att, rec.emit);
        }
    }
    return {0,0,0};
}

// ── shared render loop ────────────────────────────────────────────────
template <typename PrimT, typename HitFn>
void render(sycl::queue* q, int w, int h, const Camera& cam,
            const PrimT* d_prims, int count, int spp, int bounces,
            float* accum, int sample_index, HitFn&& hit_fn) {

    q->parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                    [=](sycl::item<2> it) {
        int x = it[1], y = it[0], idx = y * w + x;

        for (int s = 0; s < spp; s++) {
            RNG rng{ (uint32_t)(idx * 6364136223846793005ull
                                + (uint64_t)(sample_index * 2654435761u) + s) };
            float u = (x + rng.next()) / (float)w;
            float v = (y + rng.next()) / (float)h;

            Ray ray;
            ray.orig = cam.origin;
            ray.dir  = norm(sub(add(add(cam.lower_left,
                        scale(cam.horizontal, u)), scale(cam.vertical, v)), cam.origin));

            float3 col = trace(ray, d_prims, count, hit_fn, bounces, rng);

            int base = idx * 4;
            accum[base + 0] += col.x;
            accum[base + 1] += col.y;
            accum[base + 2] += col.z;
            accum[base + 3] += 1;
        }
    }).wait();
}

} // namespace rt
