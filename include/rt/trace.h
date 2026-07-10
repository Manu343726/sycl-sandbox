#pragma once
#include "types.h"
#include "camera.h"
#include "params.h"
#include "variant.h"

/// Top-level raytracing pipeline: Object dispatch, path tracing, and the
/// generic render_main() entry point called by every raytracer kernel.
///
/// Typical usage in a kernel's render_kernel():
///
///   render_main(queue, width, height, params, accum, sample_index,
///               g_scene_objects, g_num_objects,
///               [](const Ray& ray) -> float3 {
///                   return sky_gradient(ray);  // background on miss
///               });
///
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

// ── Path tracing ───────────────────────────────────────────────────────

/// Traces a single ray path through the scene, accumulating colour through
/// bounces.  Returns the final radiance for this ray.
///
/// @param ray      Initial ray (from camera through a pixel).
/// @param objects  Flat array of scene objects.
/// @param count    Number of objects.
/// @param bounces  Maximum number of ray bounces (path depth).
/// @param rng      Per-pixel RNG state.
inline float3 trace(const Ray& ray, const Object* objects, int count,
                    int bounces, RNG& rng) {
    float3 attenuation = {1,1,1};
    Ray r = ray;
    for (int b = 0; b < bounces; b++) {
        HitRecord rec;
        float closest = 1e30f;
        int hit_index = -1;

        for (int i = 0; i < count; i++) {
            if (objects[i].hit(r, 0.001f, closest, rec)) {
                closest = rec.t;
                hit_index = i;
            }
        }
        if (hit_index < 0) return {0,0,0};

        // Emissive surface → terminate and return emitted light
        float3 emitted = objects[hit_index].emit(rec);
        if (emitted.x != 0 || emitted.y != 0 || emitted.z != 0)
            return mul(attenuation, emitted);

        // Scatter (reflection / refraction)
        float3 albedo;
        Ray scattered;
        if (objects[hit_index].scatter(r, rec, albedo, scattered, rng)) {
            attenuation = mul(attenuation, albedo);
            r = scattered;
        } else {
            return {0,0,0};
        }
    }
    return {0,0,0};
}

// ── Render entry point ─────────────────────────────────────────────────

/// Generic render_kernel body used by all raytracer kernels.
///
/// Reads the standard camera/render parameters from the params buffer,
/// sets up the camera frustum, and launches the SYCL parallel_for that
/// traces rays for every pixel.
///
/// @tparam BgFn  Callable `float3(const Ray&)` returning the background
///               colour when a ray misses all objects.
///
/// @param q       SYCL queue.
/// @param w,h     Image dimensions.
/// @param p       Flat float array of kernel parameters (layout per rt_std_param).
/// @param accum   Accumulation buffer (float4 per pixel).
/// @param si      Current sample index (used for RNG seeding).
/// @param d_objs  Device-side array of scene Objects.
/// @param count   Number of objects in the scene.
/// @param bg_fn   Background function for miss rays.
template <typename BgFn>
void render_main(sycl::queue* q, int w, int h,
                 const float* p, float* accum, int si,
                 const Object* d_objs, int count, BgFn&& bg_fn) {
    int samples_per_frame = (int)p[RT_SPP_FRAME];
    int max_bounces       = (int)p[RT_MAX_BOUNCES];
    float3 camera_eye, camera_at, camera_up;
    memcpy(&camera_eye, p + RT_CAM_EYE, 12);
    memcpy(&camera_at,  p + RT_CAM_AT,  12);
    memcpy(&camera_up,  p + RT_CAM_UP,  12);
    float field_of_view      = p[RT_CAM_FOV];
    float aperture_size      = p[RT_CAM_APERTURE];
    float aspect_ratio       = (float)w / (float)h;

    Camera camera = lookat(camera_eye, camera_at, camera_up,
                            field_of_view, aspect_ratio);

    q->parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                    [=](sycl::item<2> pixel) {
        int x = pixel[1], y = pixel[0], flat_index = y * w + x;

        for (int s = 0; s < samples_per_frame; s++) {
            RNG rng{(uint32_t)(flat_index * 6364136223846793005ull
                               + (uint64_t)(si * 2654435761u) + s)};
            float u = (x + rng.next()) / (float)w;
            float v = (y + rng.next()) / (float)h;

            // Depth-of-field: jitter the ray origin within the aperture disk
            float3 ray_origin = camera.origin;
            if (aperture_size > 0.f) {
                float3 jitter = scale(random_in_unit_sphere(rng), aperture_size * 0.5f);
                ray_origin.x += jitter.x;
                ray_origin.y += jitter.y;
            }

            Ray ray;
            ray.orig = ray_origin;
            ray.dir  = norm(sub(add(add(camera.lower_left,
                        scale(camera.horizontal, u)), scale(camera.vertical, v)),
                                ray_origin));

            float3 colour = trace(ray, d_objs, count, max_bounces, rng);

            // Background on complete miss
            if (colour.x == 0.f && colour.y == 0.f && colour.z == 0.f)
                colour = bg_fn(ray);

            int base = flat_index * 4;
            accum[base+0] += colour.x;
            accum[base+1] += colour.y;
            accum[base+2] += colour.z;
            accum[base+3] += 1.0f;
        }
    }).wait();
}

} // namespace rt
