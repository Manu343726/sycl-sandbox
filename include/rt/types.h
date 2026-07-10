#pragma once
#include "math.h"
#include "types_fwd.h"
#include <variant>
#include "variant.h"
#include "hittables/sphere.h"
#include "hittables/triangle.h"
#include "hittables/quad.h"
#include "hittables/box.h"
#include "materials/lambertian.h"
#include "materials/metal.h"
#include "materials/dielectric.h"
#include "materials/diffuse_light.h"

/// Raised to the top-level `rt` namespace for convenience.
namespace rt {

using Hittable =
    std::variant<hittables::Sphere, hittables::Triangle, hittables::Quad, hittables::Box>;

using Material = std::variant<materials::Lambertian,
                              materials::Metal,
                              materials::Dielectric,
                              materials::DiffuseLight>;

/// A scene Object is a geometry + material pair, stored inline (no pointers).
/// Dispatch to the active variant type is handled by visit_rt() — a
/// compile-time if/else-if chain that works on all SYCL backends.
class Object {
public:
    Hittable hittable;
    Material material;

    Object() = default;
    Object(Hittable h, Material m) : hittable(std::move(h)), material(std::move(m)) {
    }

    std::optional<HitRecord> hit(const Ray &r, float t_min, float t_max) const;
    std::optional<ScatterRecord> scatter(const Ray &in, const HitRecord &rec, RNG &rng) const;
    float3 emit(const HitRecord &rec) const;
};

} // namespace rt
