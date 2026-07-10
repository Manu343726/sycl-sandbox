#pragma once
#include <rt/math.h>
#include <rt/types_fwd.h>

/// Camera utility: computes ray origin, horizontal, vertical, and lower-left
/// corner from lookAt parameters.  Used by rt::render_main().
namespace rt {

/// Camera frustum parameters computed by lookat().
struct Camera {
    float3 origin;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
};

/// Build a Camera from standard lookAt parameters.
inline Camera
lookat(float3 from, float3 at, float3 up, float vfov_deg, float aspect, float focus_dist = 1.f) {
    // Convert vertical FOV from degrees to radians and compute viewport dimensions
    float theta = vfov_deg * 3.14159265f / 180.f;
    float half_height = sycl::tan(theta / 2.f);
    float viewport_height = 2.f * half_height;
    float viewport_width = aspect * viewport_height;

    // Build orthonormal camera basis: forward (ww), right (uu), up (vv)
    float3 forward = norm(sub(from, at));
    float3 right = norm(cross(up, forward));
    float3 cam_up = cross(forward, right);

    // Compute the frustum corners from the basis and focus distance
    Camera camera;
    camera.origin = from;
    camera.horizontal = scale(right, viewport_width * focus_dist);
    camera.vertical = scale(cam_up, viewport_height * focus_dist);
    camera.lower_left =
        sub(sub(sub(from, scale(camera.horizontal, 0.5f)), scale(camera.vertical, 0.5f)),
            scale(forward, focus_dist));
    return camera;
}

} // namespace rt
