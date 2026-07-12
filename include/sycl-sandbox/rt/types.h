#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <variant>
#include <sycl-sandbox/variant.h>
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
/// Used as a convenience type for SceneBuilder::add() — dispatch is now
/// handled by Handle + per-type arrays in the data-oriented layout.
class Object {
public:
    Hittable hittable;
    Material material;

    Object() = default;
    Object(Hittable h, Material m) : hittable(std::move(h)), material(std::move(m)) {
    }
};

} // namespace rt
