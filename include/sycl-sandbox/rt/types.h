#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <variant>
#include <sycl-sandbox/rt/variant.h>
#include <sycl-sandbox/rt/hittables/sphere.h>
#include <sycl-sandbox/rt/hittables/triangle.h>
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/rt/hittables/box.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/metal.h>
#include <sycl-sandbox/rt/materials/dielectric.h>
#include <sycl-sandbox/rt/materials/diffuse_light.h>

/// Raised to the top-level `rt` namespace for convenience.
namespace rt {

using Hittable =
    std::variant<hittables::Sphere, hittables::Triangle, hittables::Quad, hittables::Box>;

using Material = std::variant<materials::Lambertian,
                              materials::Metal,
                              materials::Dielectric,
                              materials::DiffuseLight>;

/// A scene Object is a geometry + material pair, stored inline (no pointers).
/// Dispatch to the active variant type is handled by visit() — a
/// compile-time if/else-if chain that works on all SYCL backends.
class Object {
public:
    Hittable hittable;
    Material material;

    Object() = default;
    Object(Hittable h, Material m) : hittable(std::move(h)), material(std::move(m)) {
    }

    Optional<HitRecord> hit(const Ray &r, float t_min, float t_max) const;
    Optional<ScatterRecord> scatter(const Ray &in, const HitRecord &rec, RNG &rng) const;
    float3 emit(const HitRecord &rec) const;
};

} // namespace rt
