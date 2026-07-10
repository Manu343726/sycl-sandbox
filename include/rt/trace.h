#pragma once
#include "types.h"
#include "camera.h"
#include "params.h"
#include "variant.h"
#include <optional>

namespace rt {

// ── Object dispatch via visit_rt ───────────────────────────────────────

inline std::optional<HitRecord> Object::hit(const Ray &r, float t_min, float t_max) const {
    std::optional<HitRecord> result;
    visit_rt(hittable, [&](const auto &h) {
        result = h.hit(r, t_min, t_max);
    });
    return result;
}

inline std::optional<ScatterRecord>
Object::scatter(const Ray &in, const HitRecord &rec, RNG &rng) const {
    std::optional<ScatterRecord> result;
    visit_rt(material, [&](const auto &m) {
        result = m.scatter(in, rec, rng);
    });
    return result;
}

inline float3 Object::emit(const HitRecord &rec) const {
    float3 emitted = {0, 0, 0};
    visit_rt(material, [&](const auto &m) {
        emitted = m.emit(rec);
    });
    return emitted;
}

// ── Path tracing ───────────────────────────────────────────────────────

/// Traces a single ray path through the scene, accumulating colour through bounces.
inline float3 trace(const Ray &ray, const Object *objects, int count, int bounces, RNG &rng) {
    float3 attenuation = {1, 1, 1};
    Ray r = ray;
    for ( int b = 0; b < bounces; b++ ) {
        auto hit = objects[0].hit(r, 0.001f, 1e30f);
        int hit_index = 0;
        for ( int i = 1; i < count; i++ ) {
            auto candidate = objects[i].hit(r, 0.001f, hit ? hit->t : 1e30f);
            if ( candidate ) {
                hit = candidate;
                hit_index = i;
            }
        }
        if ( !hit )
            return {0, 0, 0};

        float3 emitted = objects[hit_index].emit(*hit);
        if ( emitted.x != 0 || emitted.y != 0 || emitted.z != 0 )
            return mul(attenuation, emitted);

        auto scattered = objects[hit_index].scatter(r, *hit, rng);
        if ( !scattered )
            return {0, 0, 0};

        attenuation = mul(attenuation, scattered->attenuation);
        r = scattered->scattered;
    }
    return {0, 0, 0};
}

// ── Render entry point ─────────────────────────────────────────────────

template <typename BgFn>
void render_main(sycl::queue *q,
                 int w,
                 int h,
                 const float *p,
                 float *accum,
                 int si,
                 const Object *d_objs,
                 int count,
                 BgFn &&bg_fn) {
    int samples_per_frame = (int)p[RT_SPP_FRAME];
    int max_bounces = (int)p[RT_MAX_BOUNCES];
    float3 camera_eye, camera_at, camera_up;
    memcpy(&camera_eye, p + RT_CAM_EYE, 12);
    memcpy(&camera_at, p + RT_CAM_AT, 12);
    memcpy(&camera_up, p + RT_CAM_UP, 12);
    float field_of_view = p[RT_CAM_FOV];
    float aperture_size = p[RT_CAM_APERTURE];
    float aspect_ratio = (float)w / (float)h;

    Camera camera = lookat(camera_eye, camera_at, camera_up, field_of_view, aspect_ratio);

    q->parallel_for(sycl::range<2> {(size_t)h, (size_t)w}, [=](sycl::item<2> pixel) {
         int x = pixel[1], y = pixel[0], flat_index = y * w + x;

         for ( int s = 0; s < samples_per_frame; s++ ) {
             RNG rng {(uint32_t)(flat_index * 6364136223846793005ull +
                                 (uint64_t)(si * 2654435761u) + s)};
             float u = (x + rng.next()) / (float)w;
             float v = (y + rng.next()) / (float)h;

             // Depth-of-field: jitter the ray origin within the aperture disk
             float3 ray_origin = camera.origin;
             if ( aperture_size > 0.f ) {
                 float3 jitter = scale(random_in_unit_sphere(rng), aperture_size * 0.5f);
                 ray_origin.x += jitter.x;
                 ray_origin.y += jitter.y;
             }

             Ray ray;
             ray.orig = ray_origin;
             ray.dir = norm(sub(add(add(camera.lower_left, scale(camera.horizontal, u)),
                                    scale(camera.vertical, v)),
                                ray_origin));

             float3 colour = trace(ray, d_objs, count, max_bounces, rng);

             if ( colour.x == 0.f && colour.y == 0.f && colour.z == 0.f )
                 colour = bg_fn(ray);

             int base = flat_index * 4;
             accum[base + 0] += colour.x;
             accum[base + 1] += colour.y;
             accum[base + 2] += colour.z;
             accum[base + 3] += 1.0f;
         }
     }).wait();
}

} // namespace rt
