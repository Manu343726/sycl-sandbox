#pragma once
#include <rt/types.h>
#include <rt/camera.h>
#include <rt/params.h>
#include <rt/variant.h>
#include <optional>

namespace rt {

// ── Object dispatch via visit ──────────────────────────────────────────

inline std::optional<HitRecord> Object::hit(const Ray &ray, float t_min, float t_max) const {
    std::optional<HitRecord> result;
    visit(hittable, [&](const auto &h) {
        result = h.hit(ray, t_min, t_max);
    });
    return result;
}

inline std::optional<ScatterRecord>
Object::scatter(const Ray &incoming_ray, const HitRecord &hit, RNG &rng) const {
    std::optional<ScatterRecord> result;
    visit(material, [&](const auto &m) {
        result = m.scatter(incoming_ray, hit, rng);
    });
    return result;
}

inline float3 Object::emit(const HitRecord &hit) const {
    float3 emitted = {0, 0, 0};
    visit(material, [&](const auto &m) {
        emitted = m.emit(hit);
    });
    return emitted;
}

// ── Path tracing ───────────────────────────────────────────────────────

inline float3
trace(const Ray &ray, const Object *objects, int num_objects, int max_bounces, RNG &rng) {
    // Initialise the path throughput (attenuation) and the working ray
    float3 attenuation = {1, 1, 1};
    Ray ray_in_out = ray;

    // Trace the ray through successive bounces
    for ( int bounce = 0; bounce < max_bounces; bounce++ ) {
        // Find the closest object hit by the ray within [0.001, ∞)
        std::optional<HitRecord> closest_hit;
        int hit_index = -1;

        for ( int i = 0; i < num_objects; i++ ) {
            auto hit = objects[i].hit(ray_in_out, 0.001f, closest_hit ? closest_hit->t : 1e30f);
            if ( hit ) {
                closest_hit = hit;
                hit_index = i;
            }
        }

        // If no object was hit the ray escapes to the void
        if ( !closest_hit ) {
            return {0, 0, 0};
        }

        // If the hit object emits light, return the attenuated emission
        float3 emitted = objects[hit_index].emit(*closest_hit);
        if ( emitted.x != 0 || emitted.y != 0 || emitted.z != 0 ) {
            return mul(attenuation, emitted);
        }

        // Scatter the ray; if the material absorbs it, terminate the path
        auto scattered = objects[hit_index].scatter(ray_in_out, *closest_hit, rng);
        if ( !scattered ) {
            return {0, 0, 0};
        }

        // Update the attenuation and continue with the scattered ray
        attenuation = mul(attenuation, scattered->attenuation);
        ray_in_out = scattered->scattered;
    }
    return {0, 0, 0};
}

// ── Render entry point ─────────────────────────────────────────────────

template <typename BgFn>
void render_main(sycl::queue *queue,
                 int width,
                 int height,
                 const float *params,
                 float *accum_buffer,
                 int sample_index,
                 const Object *scene_objects,
                 int num_objects,
                 BgFn &&background_fn) {
    // Read standard camera and render parameters from the params buffer
    int samples_per_frame = (int)params[RT_SPP_FRAME];
    int max_bounces = (int)params[RT_MAX_BOUNCES];

    float3 camera_eye, camera_at, camera_up;
    memcpy(&camera_eye, params + RT_CAM_EYE, 12);
    memcpy(&camera_at, params + RT_CAM_AT, 12);
    memcpy(&camera_up, params + RT_CAM_UP, 12);
    float field_of_view = params[RT_CAM_FOV];
    float aperture_size = params[RT_CAM_APERTURE];
    float aspect_ratio = (float)width / (float)height;

    // Build the camera frustum from the lookAt parameters
    Camera camera = lookat(camera_eye, camera_at, camera_up, field_of_view, aspect_ratio);

    // Launch a SYCL parallel_for over all pixels
    queue
        ->parallel_for(
            sycl::range<2> {(size_t)height, (size_t)width},
            [=](sycl::item<2> pixel) {
                int x = pixel[1], y = pixel[0], flat_index = y * width + x;

                for ( int sample = 0; sample < samples_per_frame; sample++ ) {
                    // Initialise the per-pixel, per-sample RNG from the pixel index and frame
                    // number
                    RNG rng {static_cast<uint32_t>(flat_index * 6364136223846793005ull +
                                                   (uint64_t)(sample_index * 2654435761u) +
                                                   sample)};

                    // Generate stratified pixel coordinates with random offsets
                    float u = (x + rng.next()) / (float)width;
                    float v = (y + rng.next()) / (float)height;

                    // Optionally jitter the ray origin for depth-of-field effects
                    float3 ray_origin = camera.origin;
                    if ( aperture_size > 0.f ) {
                        float3 jitter = scale(random_in_unit_sphere(rng), aperture_size * 0.5f);
                        ray_origin.x += jitter.x;
                        ray_origin.y += jitter.y;
                    }

                    // Construct the primary ray from the camera frustum
                    Ray ray;
                    ray.orig = ray_origin;
                    ray.dir = norm(sub(add(add(camera.lower_left, scale(camera.horizontal, u)),
                                           scale(camera.vertical, v)),
                                       ray_origin));

                    // Trace the ray and accumulate its colour
                    float3 colour = trace(ray, scene_objects, num_objects, max_bounces, rng);

                    // If the ray hit nothing (colour is black), use the background colour
                    if ( colour.x == 0.f && colour.y == 0.f && colour.z == 0.f ) {
                        colour = background_fn(ray);
                    }

                    // Accumulate the sample into the RGBA output buffer
                    int base = flat_index * 4;
                    accum_buffer[base + 0] += colour.x;
                    accum_buffer[base + 1] += colour.y;
                    accum_buffer[base + 2] += colour.z;
                    accum_buffer[base + 3] += 1.0f;
                }
            })
        .wait();
}

} // namespace rt
