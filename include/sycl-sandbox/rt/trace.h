#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/profiler_device.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/scene/camera.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/kernel/execution_context.h>

namespace rt {

constexpr const char *KEY_SPP_FRAME    = "spp_frame";
constexpr const char *KEY_MAX_BOUNCES  = "max_bounces";
constexpr const char *KEY_CAM_EYE      = "cam_eye";
constexpr const char *KEY_CAM_AT       = "cam_at";
constexpr const char *KEY_CAM_UP       = "cam_up";
constexpr const char *KEY_CAM_FOV      = "cam_fov";
constexpr const char *KEY_CAM_APERTURE = "cam_aperture";
constexpr const char *KEY_TICK         = "tick";
constexpr const char *KEY_TIME         = "time";

// ── Path tracing ───────────────────────────────────────────────────────

/// Trace a ray through the scene and return the accumulated colour.
/// Uses per-handle dispatch to hit, scatter, and emit on the correct
/// per-type arrays.
inline float3 trace(const Ray &ray, const SceneView &scene, int max_bounces, RNG &rng) {
    PROFILER_ZONE("trace_path");
    // Initialise the path throughput (attenuation) and the working ray
    float3 attenuation = {1, 1, 1};
    Ray ray_in_out = ray;

    // Trace the ray through successive bounces
    for ( int bounce = 0; bounce < max_bounces; bounce++ ) {
        // Find the closest object hit by the ray within [0.001, ∞)
        optional<HitRecord> closest_hit;
        Handle hit_handle = {};

        for ( int i = 0; i < scene.num_handles; i++ ) {
            auto hit = handle_hit(scene.handles[i],
                                  ray_in_out,
                                  0.001f,
                                  closest_hit ? closest_hit->t : 1e30f,
                                  scene);
            if ( hit ) {
                closest_hit = hit;
                hit_handle = scene.handles[i];
            }
        }

        // If no object was hit the ray escapes to the void
        if ( !closest_hit ) {
            return {0, 0, 0};
        }

        // If the hit object emits light, return the attenuated emission
        float3 emitted = handle_emit(hit_handle, *closest_hit, scene);
        if ( emitted.x != 0 || emitted.y != 0 || emitted.z != 0 ) {
            return mul(attenuation, emitted);
        }

        // Scatter the ray; if the material absorbs it, terminate the path
        auto scattered = handle_scatter(hit_handle, ray_in_out, *closest_hit, rng, scene);
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

/// KernelName is an explicit SYCL kernel name tag, unique per calling
/// kernel .so — see rt::Runtime::foreach_pixel() for why this is required
/// (AdaptiveCpp's kernel identity is derived from the mangled name of the
/// enclosing function, which collides across .so's that reuse the same
/// function name, e.g. every kernel's `render_kernel` entry point).
template <typename KernelName, typename BgFn>
void render_main(const Runtime &rt,
                 int width,
                 int height,
                 const ParamLookup &reader,
                 float *accum_buffer,
                 uint32_t spp_total,
                 const SceneView &scene,
                 BgFn &&background_fn,
                 profiler::DeviceRing prof = {}) {
    PROFILER_FUNCTION();
    // Read all parameters by name using the type-safe ParamLookup API.
    // These reads happen on the host (before the pixel loop), so there
    // is zero per-pixel lookup overhead.
    int samples_per_frame = reader.read<int>(KEY_SPP_FRAME);
    int max_bounces = reader.read<int>(KEY_MAX_BOUNCES);

    float3 camera_eye   = reader.read_vec3<float3>(KEY_CAM_EYE);
    float3 camera_at    = reader.read_vec3<float3>(KEY_CAM_AT);
    float3 camera_up    = reader.read_vec3<float3>(KEY_CAM_UP);
    float field_of_view = reader.read<float>(KEY_CAM_FOV);
    float aperture_size = reader.read<float>(KEY_CAM_APERTURE);
    float aspect_ratio  = (float)width / (float)height;

    // Animation time for the frame; textures/materials use it for
    // time-varying sampling (e.g. animated procedural textures).
    float frame_time = reader.read<float>(KEY_TIME);

    // Build the camera frustum from the lookAt parameters
    Camera camera = lookat(camera_eye, camera_at, camera_up, field_of_view, aspect_ratio);

    // Iterate over every pixel through the Runtime abstraction.
    // In SYCL mode this uses queue->parallel_for; in software mode it
    // uses plain nested for-loops — no OpenMP, no SYCL in the kernel.
    rt.foreach_pixel<KernelName>(width, height, [=](int x, int y, int flat_index) {
        PROFILER_DEVICE_ZONE(prof, "trace_px", flat_index);
        for ( int sample = 0; sample < samples_per_frame; sample++ ) {
            // Initialise the per-pixel, per-sample RNG from the pixel index and
            // total accumulated samples for stream decorrelation.
            RNG rng {static_cast<uint32_t>(flat_index * 6364136223846793005ull +
                                           (uint64_t)(spp_total * 2654435761u) +
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
            ray.time = frame_time;

            // Trace the ray and accumulate its colour
            float3 colour = trace(ray, scene, max_bounces, rng);

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
    });
}

} // namespace rt
