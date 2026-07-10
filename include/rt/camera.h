#pragma once
#include "types.h"

namespace rt {

struct Camera {
    float3 origin;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
};

inline Camera lookat(float3 from, float3 at, float3 up, float vfov_deg,
                     float aspect, float focus_dist = 1.f) {
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
