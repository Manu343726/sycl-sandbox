#pragma once
#include "math.h"

namespace rt {

struct Ray { float3 orig, dir; };
struct HitRecord {
    float3   p;
    float3   normal;
    float    t;
    bool     front_face;
};

} // namespace rt
