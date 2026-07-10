#pragma once
#include "math.h"

namespace rt {

enum class MatType : uint8_t { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

struct Ray {
    float3 orig;
    float3 dir;
};

struct HitRecord {
    float3   p;
    float3   normal;
    float    t;
    float3   albedo;
    float3   emit;
    MatType  mat_type;
    float    fuzz;
    float    ir;
    bool     front_face;
};

struct Quad {
    float3  a, b, c;
    float3  normal;
    MatType mat_type;
    float3  albedo;
    float3  emit;
};

struct Sphere {
    float3  center;
    float   radius;
    MatType mat_type;
    float3  albedo;
    float   fuzz;
    float   ir;
    float3  emit;
};

} // namespace rt
