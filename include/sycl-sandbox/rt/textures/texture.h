#pragma once

/// @file
/// Raytracing texture interface.
///
/// A texture maps surface parametric coordinates plus time to a colour:
/// @code
///   float3 sample(float u, float v, float time, RNG &rng) const;
/// @endcode
/// `u`/`v` are the hit point's parametric coordinates on the primitive
/// (quad/triangle: affine/barycentric coordinates, sphere: spherical
/// mapping, box: the hit face's quad coordinates) — see `HitRecord::u/v`.
/// Out-of-range UVs are NOT clamped by the pipeline: every texture
/// implements its own clamp / round / wrap behavior.  This is what
/// allows both procedural textures (infinite tiling, noise, ...)
/// and file-loaded image textures (clamped or wrapped sampling).
///
/// The sampler also receives the path's `RNG`, so stochastic textures
/// (noise, jittered filtering, ...) can draw from it without owning
/// their own state.  Deterministic textures simply ignore it.
///
/// Textures are plain procedural structs — no base class, no virtual
/// dispatch (SYCL device code cannot use vtables).  The polymorphic
/// `Texture` variant dispatches to the concrete texture via compile-time
/// `visit()`, exactly like `Hittable`/`Material` do.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/variant.h>
#include <sycl-sandbox/rt/textures/solid_color.h>
#include <sycl-sandbox/rt/textures/colorchecker.h>
#include <sycl-sandbox/rt/textures/text.h>
#include <sycl-sandbox/rt/textures/blend.h>

namespace rt::textures {

/// Polymorphic texture: variant of all procedural texture types.
/// Trivially copyable when all alternatives are, so it can be stored
/// in material arrays and uploaded to device memory by value.
using Texture = std::variant<SolidColor, ColorChecker, Text, Blend>;

/// Sample any texture at surface coordinates (u, v) and time `time`,
/// passing the path's RNG through to stochastic textures.
/// Dispatches to the concrete texture's `sample()` via visit().
/// UV out-of-range behavior is defined by each texture kind.
///
/// \param ctx per-call kernel context: forwarded to the concrete sampler
///        (it records a "sample_*" profiler zone and reports the sample
///        through the trace collector).
inline float3 sample(const Texture &t, float u, float v, float time, RNG &rng,
                     const Context &ctx = Context{}) {
    float3 result = {0, 0, 0};
    visit(t, [&](const auto &tex) {
        result = tex.sample(u, v, time, rng, ctx);
    });
    return result;
}

} // namespace rt::textures
