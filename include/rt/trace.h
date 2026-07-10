#pragma once
#include "types.h"
#include "camera.h"
#include "params.h"
#include "variant.h"

namespace rt {

// ── Object dispatch via visit_rt ───────────────────────────────────────
inline bool Object::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    bool found = false;
    visit_rt(hittable, [&](const auto& h) { found = h.hit(r, t_min, t_max, rec); });
    return found;
}

inline bool Object::scatter(const Ray& in, const HitRecord& rec,
                             float3& att, Ray& scattered, RNG& rng) const {
    bool did_scatter = false;
    visit_rt(material, [&](const auto& m) {
        did_scatter = m.scatter(in, rec, att, scattered, rng);
    });
    return did_scatter;
}

inline float3 Object::emit(const HitRecord& rec) const {
    float3 emitted = {0,0,0};
    visit_rt(material, [&](const auto& m) { emitted = m.emit(rec); });
    return emitted;
}

// ── Trace ──────────────────────────────────────────────────────────────
inline float3 trace(const Ray& ray, const Object* objs, int count, int bounces, RNG& rng) {
    float3 att = {1,1,1};
    Ray r = ray;
    for (int b = 0; b < bounces; b++) {
        HitRecord rec;
        float closest = 1e30f;
        int hit_idx = -1;
        for (int i = 0; i < count; i++) {
            if (objs[i].hit(r, 0.001f, closest, rec)) {
                closest = rec.t;
                hit_idx = i;
            }
        }
        if (hit_idx < 0) return {0,0,0};

        float3 emitted = objs[hit_idx].emit(rec);
        if (emitted.x != 0 || emitted.y != 0 || emitted.z != 0)
            return mul(att, emitted);

        float3 albedo;
        Ray scattered;
        if (objs[hit_idx].scatter(r, rec, albedo, scattered, rng)) {
            att = mul(att, albedo);
            r = scattered;
        } else {
            return {0,0,0};
        }
    }
    return {0,0,0};
}

// ── Generic render body ────────────────────────────────────────────────
template <typename BgFn>
void render_main(sycl::queue* q, int w, int h,
                 const float* p, float* accum, int si,
                 const Object* d_objs, int count, BgFn&& bg_fn) {
    int spp = (int)p[RT_SPP_FRAME];
    int bounces = (int)p[RT_MAX_BOUNCES];
    float3 ce, ca, cu;
    memcpy(&ce, p + RT_CAM_EYE, 12);
    memcpy(&ca, p + RT_CAM_AT, 12);
    memcpy(&cu, p + RT_CAM_UP, 12);
    float fov = p[RT_CAM_FOV], aperture = p[RT_CAM_APERTURE];
    float aspect = (float)w / (float)h;
    Camera cam = lookat(ce, ca, cu, fov, aspect);

    q->parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                    [=](sycl::item<2> it) {
        int x = it[1], y = it[0], idx = y * w + x;
        for (int s = 0; s < spp; s++) {
            RNG rng{(uint32_t)(idx * 6364136223846793005ull
                               + (uint64_t)(si * 2654435761u) + s)};
            float u = (x + rng.next()) / (float)w;
            float v_ = (y + rng.next()) / (float)h;
            float3 ro = cam.origin;
            if (aperture > 0.f) {
                float3 jit = scale(random_in_unit_sphere(rng), aperture * 0.5f);
                ro.x += jit.x; ro.y += jit.y;
            }
            Ray ray;
            ray.orig = ro;
            ray.dir  = norm(sub(add(add(cam.lower_left, scale(cam.horizontal, u)),
                                     scale(cam.vertical, v_)), ro));
            float3 col = trace(ray, d_objs, count, bounces, rng);
            if (col.x == 0.f && col.y == 0.f && col.z == 0.f)
                col = bg_fn(ray);
            int base = idx * 4;
            accum[base+0] += col.x; accum[base+1] += col.y;
            accum[base+2] += col.z; accum[base+3] += 1;
        }
    }).wait();
}

} // namespace rt
