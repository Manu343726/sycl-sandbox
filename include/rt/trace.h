#pragma once
#include "types.h"
#include "camera.h"
#include "params.h"

namespace rt {

// ── helpers ────────────────────────────────────────────────────────────
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

// ── Material implementations ───────────────────────────────────────────
inline bool Lambertian::scatter(const Ray&, const HitRecord& rec,
                                 float3& attenuation, Ray& scattered, RNG& rng) const {
    float3 target = add(rec.p, add(rec.normal, random_in_unit_sphere(rng)));
    scattered = {rec.p, sub(target, rec.p)};
    attenuation = albedo;
    return true;
}

inline bool Metal::scatter(const Ray& in, const HitRecord& rec,
                            float3& attenuation, Ray& scattered, RNG& rng) const {
    float3 reflected = reflect(norm(in.dir), rec.normal);
    scattered = {rec.p, add(reflected, scale(random_in_unit_sphere(rng), fuzz))};
    attenuation = albedo;
    return dot(scattered.dir, rec.normal) > 0;
}

inline bool Dielectric::scatter(const Ray& in, const HitRecord& rec,
                                 float3& attenuation, Ray& scattered, RNG& rng) const {
    attenuation = {1,1,1};
    float3 outward_n;
    float eta_ratio, cos;
    if (dot(in.dir, rec.normal) > 0) {
        outward_n = scale(rec.normal, -1);
        eta_ratio = ir;
        cos = ir * dot(in.dir, rec.normal) / len(in.dir);
    } else {
        outward_n = rec.normal;
        eta_ratio = 1.f / ir;
        cos = -dot(in.dir, rec.normal) / len(in.dir);
    }
    float3 refracted;
    if (refract(in.dir, outward_n, eta_ratio, refracted) && rng.next() >= schlick(cos, ir))
        scattered = {rec.p, refracted};
    else
        scattered = {rec.p, reflect(norm(in.dir), rec.normal)};
    return true;
}

// ── Hittable implementations ───────────────────────────────────────────
inline bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    float3 oc = sub(r.orig, center);
    float a = dot(r.dir, r.dir);
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float d = b*b - a*c;
    if (d <= 0) return false;
    float t = (-b - sycl::sqrt(d)) / a;
    if (t < t_min || t > t_max) t = (-b + sycl::sqrt(d)) / a;
    if (t < t_min || t > t_max) return false;
    rec.t = t;
    rec.p = add(r.orig, scale(r.dir, t));
    rec.normal = scale(sub(rec.p, center), 1.f / radius);
    rec.front_face = dot(r.dir, rec.normal) < 0;
    if (!rec.front_face) rec.normal = scale(rec.normal, -1);
    return true;
}

inline bool Quad::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    float denom = dot(normal, r.dir);
    if (sycl::fabs(denom) < 1e-8f) return false;
    float t = dot(sub(a, r.orig), normal) / denom;
    if (t < t_min || t > t_max) return false;
    float3 p = add(r.orig, scale(r.dir, t));
    float3 ba = sub(b, a), ca = sub(c, a), pa = sub(p, a);
    float d00 = dot(ba, ba), d01 = dot(ba, ca), d11 = dot(ca, ca);
    float d20 = dot(pa, ba), d21 = dot(pa, ca);
    float den = d00 * d11 - d01 * d01;
    if (sycl::fabs(den) < 1e-12f) return false;
    float u = (d11 * d20 - d01 * d21) / den;
    float v = (d00 * d21 - d01 * d20) / den;
    if (u < 0.f || u > 1.f || v < 0.f || v > 1.f) return false;
    rec.t = t; rec.p = p;
    rec.normal = denom < 0.f ? normal : scale(normal, -1.f);
    rec.front_face = denom < 0.f;
    return true;
}

// ── generic tracer ─────────────────────────────────────────────────────
inline float3 trace(const Ray& ray, const Object* objects, int count, int bounces, RNG& rng) {
    float3 att = {1,1,1};
    Ray r = ray;
    for (int b = 0; b < bounces; b++) {
        HitRecord rec;
        float closest = 1e30f;
        bool hit_any = false;
        int hit_idx = -1;
        for (int i = 0; i < count; i++) {
            if (objects[i].geometry->hit(r, 0.001f, closest, rec)) {
                closest = rec.t;
                hit_any = true;
                hit_idx = i;
            }
        }
        if (!hit_any) return {0,0,0};
        rec.material = objects[hit_idx].material;

        // Emissive surfaces terminate the path and return emitted light
        float3 emitted = rec.material->emit(rec);
        if (emitted.x != 0.f || emitted.y != 0.f || emitted.z != 0.f)
            return mul(att, emitted);

        // Scatter (reflection / refraction)
        float3 albedo;
        Ray scattered;
        if (rec.material->scatter(r, rec, albedo, scattered, rng)) {
            att = mul(att, albedo);
            r = scattered;
        } else {
            return {0,0,0};
        }
    }
    return {0,0,0};
}

// ── generic render_kernel body ─────────────────────────────────────────
template <typename BgFn>
void render_main(sycl::queue* q, int w, int h,
                 const float* p, float* accum, int si,
                 const Object* d_objects, int count,
                 BgFn&& bg_fn) {
    int spp     = (int)p[RT_SPP_FRAME];
    int bounces = (int)p[RT_MAX_BOUNCES];
    float3 ce; memcpy(&ce, p + RT_CAM_EYE, 12);
    float3 ca; memcpy(&ca, p + RT_CAM_AT, 12);
    float3 cu; memcpy(&cu, p + RT_CAM_UP, 12);
    float fov = p[RT_CAM_FOV];
    float aperture = p[RT_CAM_APERTURE];
    float aspect = (float)w / (float)h;

    Camera cam = lookat(ce, ca, cu, fov, aspect);

    q->parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                    [=](sycl::item<2> it) {
        int x = it[1], y = it[0], idx = y * w + x;

        for (int s = 0; s < spp; s++) {
            RNG rng{ (uint32_t)(idx * 6364136223846793005ull
                                + (uint64_t)(si * 2654435761u) + s) };

            float u = (x + rng.next()) / (float)w;
            float v = (y + rng.next()) / (float)h;

            float3 ro = cam.origin;
            if (aperture > 0.f) {
                float3 jit = scale(random_in_unit_sphere(rng), aperture * 0.5f);
                ro.x += jit.x; ro.y += jit.y;
            }

            Ray ray;
            ray.orig = ro;
            ray.dir  = norm(sub(add(add(cam.lower_left,
                        scale(cam.horizontal, u)), scale(cam.vertical, v)), ro));

            float3 col = trace(ray, d_objects, count, bounces, rng);

            if (col.x == 0.f && col.y == 0.f && col.z == 0.f)
                col = bg_fn(ray);

            int base = idx * 4;
            accum[base + 0] += col.x;
            accum[base + 1] += col.y;
            accum[base + 2] += col.z;
            accum[base + 3] += 1;
        }
    }).wait();
}

} // namespace rt
