#pragma once
#include "math.h"
#include "types_fwd.h"
#include <variant>
#include "variant.h"
#include "hittables/sphere.h"
#include "hittables/quad.h"
#include "materials/lambertian.h"
#include "materials/metal.h"
#include "materials/dielectric.h"
#include "materials/diffuse_light.h"

/// Raised to the top-level `rt` namespace for convenience.  Users typically
/// write `using namespace rt;` in their kernel code, then:
///
///   Object obj = {sphere({0,0,0}, 1), lambertian({0.8,0.2,0.2})};
///
namespace rt {

/// Variant type that can hold any supported geometry primitive.
using Hittable = std::variant<
    hittables::Sphere,
    hittables::Quad
>;

/// Variant type that can hold any supported material.
using Material = std::variant<
    materials::Lambertian,
    materials::Metal,
    materials::Dielectric,
    materials::DiffuseLight
>;

/// A scene Object is a geometry + material pair, stored inline (no pointers).
/// Dispatch to the active variant type is handled by visit_rt() — a
/// compile-time if/else-if chain that works on all SYCL backends.
class Object {
public:
    Hittable hittable; ///< The geometry (Sphere, Quad, …).
    Material material; ///< The surface material (Lambertian, Metal, …).

    Object() = default;
    Object(Hittable h, Material m) : hittable(std::move(h)), material(std::move(m)) {}

    /// Ray intersection test: dispatches to the active geometry's hit() method.
    /// @returns true if the ray hits the geometry within [t_min, t_max].
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;

    /// Scatter ray off the surface: dispatches to the active material's
    /// scatter() method.
    /// @returns true if the ray scatters (false for light sources).
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const;

    /// Emission colour of the material (non-zero only for DiffuseLight).
    float3 emit(const HitRecord& rec) const;
};

} // namespace rt
