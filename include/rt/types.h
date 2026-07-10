#pragma once
#include "math.h"

namespace rt {

// ── Ray ────────────────────────────────────────────────────────────────
struct Ray { float3 orig, dir; };

// ── Material parameter struct (type-erased, stored per-object inline) ──
// Valid fields depend on mat_type:
//   LAMBERTIAN:   albedo
//   METAL:        albedo, fuzz
//   DIELECTRIC:   ir
//   DIFFUSE_LIGHT: emit
enum MatType : uint8_t { MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC, MAT_DIFFUSE_LIGHT };

struct Material {
    MatType type;
    float3  albedo;
    float   fuzz;
    float   ir;
    float3  emit;
};

// ── Geometry parameter struct (type-erased, stored per-object inline) ──
enum GeomType : uint8_t { GEOM_SPHERE, GEOM_QUAD };

struct Geometry {
    GeomType type;
    float3   center;    // sphere
    float    radius;    // sphere
    float3   a, b, c;   // quad (3 corners)
    float3   normal;    // quad (precomputed)
};

// ── Object = geometry + material pair (inline storage, no pointers) ────
struct Object {
    Geometry geom;
    Material mat;
};

// ── Hit record ─────────────────────────────────────────────────────────
struct HitRecord {
    float3      p;
    float3      normal;
    float       t;
    bool        front_face;
    Material    mat;
};

} // namespace rt
