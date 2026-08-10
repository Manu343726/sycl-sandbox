#pragma once

/// @file
/// Solid-colour texture: the simplest procedural texture — returns the
/// same colour for every (u, v, time).  Canonical example of how a
/// texture can simply ignore its coordinates.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/math.h>

namespace rt::textures {

/// Texture that ignores (u, v, time) entirely and always returns one
/// colour.  Useful as a constant input for other textures, or as the
/// simplest possible texture.
struct SolidColor {
    float3 color = {0, 0, 0};

    SolidColor() = default;

    explicit SolidColor(float3 c) : color(c) {
    }

    /// Returns the constant colour; (u, v, time, rng) are ignored.
    float3 sample(float u, float v, float time, RNG &rng,
                  const Context &ctx = Context{}) const {
        (void)u;
        (void)v;
        (void)time;
        (void)rng;
        PROFILER_ZONE("sample_solid_color");
        ctx.collector.on_texture_sample(0);
        return color;
    }
};

} // namespace rt::textures
