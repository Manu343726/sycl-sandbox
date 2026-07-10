#pragma once
#include "math.h"
#include "types_fwd.h"

/// Camera utility: computes ray origin, horizontal, vertical, and lower-left
/// corner from lookAt parameters.  Used by rt::render_main().
namespace rt {

/// Camera frustum parameters computed by lookat().
struct Camera {
    float3 origin;     ///< Camera position in world space.
    float3 lower_left; ///< World-space position of the viewport's bottom-left corner.
    float3 horizontal; ///< Width vector of the viewport (right direction * viewport width).
    float3 vertical;   ///< Height vector of the viewport (up direction * viewport height).
};

/// Builds a Camera from standard lookAt parameters.
/// @param from       Camera position (eye).
/// @param at         Look-at target point.
/// @param up         Up vector (usually {0,1,0}).
/// @param vfov_deg   Vertical field of view in degrees.
/// @param aspect     Width / height ratio of the output image.
/// @param focus_dist Distance from the camera to the focal plane (default 1).
inline Camera lookat(float3 from, float3 at, float3 up,
                     float vfov_deg, float aspect, float focus_dist = 1.f) {
    float theta = vfov_deg * 3.14159265f / 180.f;
    float h = sycl::tan(theta / 2.f);
    float vh = 2.f * h;
    float vw = aspect * vh;

    float3 ww = norm(sub(from, at));
    float3 uu = norm(cross(up, ww));
    float3 vv = cross(ww, uu);

    Camera c;
    c.origin     = from;
    c.horizontal = scale(uu, vw * focus_dist);
    c.vertical   = scale(vv, vh * focus_dist);
    c.lower_left = sub(sub(sub(from, scale(c.horizontal, 0.5f)),
                            scale(c.vertical, 0.5f)), scale(ww, focus_dist));
    return c;
}

} // namespace rt
