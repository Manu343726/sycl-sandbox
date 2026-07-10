#pragma once
#include "math.h"
#include "types_fwd.h"
#include "variant.h"
#include "hittables/sphere.h"
#include "hittables/quad.h"
#include "materials/lambertian.h"
#include "materials/metal.h"
#include "materials/dielectric.h"
#include "materials/diffuse_light.h"
#include <variant>

namespace rt {

// ── Hittable variant (any geometry type) ───────────────────────────────
using Hittable = std::variant<
    hittables::Sphere,
    hittables::Quad
>;

// ── Material variant (any material type) ───────────────────────────────
using Material = std::variant<
    materials::Lambertian,
    materials::Metal,
    materials::Dielectric,
    materials::DiffuseLight
>;

// ── Object ─────────────────────────────────────────────────────────────
class Object {
public:
    Hittable hittable;
    Material material;

    Object() = default;
    Object(Hittable h, Material m) : hittable(std::move(h)), material(std::move(m)) {}

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const;
    float3 emit(const HitRecord& rec) const;
};

} // namespace rt
