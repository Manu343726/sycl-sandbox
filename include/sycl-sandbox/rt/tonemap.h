#pragma once

/// @file
/// Per-pixel tone-map: reads accumulated float4 from ctx.accum, applies
/// exposure + operator + gamma, packs RGBA8 to ctx.output with a Y-flip
/// for OpenGL's bottom-left texture origin.
///
/// Host/device-neutral — pure float/uint8 math, no SYCL includes.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/tonemap_ops.h>
#include <sycl-sandbox/math.h>
#include <cstdint>

namespace rt {

inline uint8_t float_to_u8(float v) {
    return (uint8_t)(v < 0.f ? 0 : (v > 1.f ? 255 : (int)(v * 255.99f)));
}

/// Compute the output byte offset for flat_index, Y-flipped.
inline int output_offset(const Context &ctx, int flat_index) {
    int x = flat_index % ctx.width;
    int y = flat_index / ctx.width;
    int fy = ctx.height - 1 - y;
    return (fy * ctx.width + x) * 4;
}

/// Disabled: normalise + expose + clamp, no operator, no gamma.
inline void tonemap_pixel_disabled(const Context &ctx, int flat_index,
                                   float exposure) {
    int b = flat_index * 4;
    float n = ctx.accum[b + 3];
    float inv = n > 0.5f ? 1.0f / n : 1.0f;
    float r = math::clamp(ctx.accum[b + 0] * inv * exposure, 0.f, 1.f);
    float g = math::clamp(ctx.accum[b + 1] * inv * exposure, 0.f, 1.f);
    float bl = math::clamp(ctx.accum[b + 2] * inv * exposure, 0.f, 1.f);
    int out = output_offset(ctx, flat_index);
    ctx.output[out + 0] = float_to_u8(r);
    ctx.output[out + 1] = float_to_u8(g);
    ctx.output[out + 2] = float_to_u8(bl);
    ctx.output[out + 3] = 255;
}

/// Tone-map with a compile-time operator id, then gamma-correct.
template <int Op>
inline void tonemap_pixel_op(const Context &ctx, int flat_index,
                             float exposure, float gamma) {
    int b = flat_index * 4;
    float n = ctx.accum[b + 3];
    float inv = n > 0.5f ? 1.0f / n : 1.0f;
    float r = ctx.accum[b + 0] * inv * exposure;
    float g = ctx.accum[b + 1] * inv * exposure;
    float bl = ctx.accum[b + 2] * inv * exposure;
    r = tonemap::apply_operator(r, Op);
    g = tonemap::apply_operator(g, Op);
    bl = tonemap::apply_operator(bl, Op);
    float ginv = 1.0f / gamma;
    int out = output_offset(ctx, flat_index);
    ctx.output[out + 0] = float_to_u8(math::pow(math::clamp(r, 0.f, 1.f), ginv));
    ctx.output[out + 1] = float_to_u8(math::pow(math::clamp(g, 0.f, 1.f), ginv));
    ctx.output[out + 2] = float_to_u8(math::pow(math::clamp(bl, 0.f, 1.f), ginv));
    ctx.output[out + 3] = 255;
}

/// Per-pixel dispatch: reads ctx.accum, writes ctx.output (Y-flipped).
inline void tonemap_pixel(const Context &ctx, int flat_index,
                          bool enabled, int tm_op,
                          float exposure, float gamma) {
    if (!enabled) {
        tonemap_pixel_disabled(ctx, flat_index, exposure);
        return;
    }
    switch (tm_op) {
    case tonemap::Operator::Reinhard:
        tonemap_pixel_op<tonemap::Operator::Reinhard>(ctx, flat_index, exposure, gamma);
        break;
    case tonemap::Operator::ACES:
        tonemap_pixel_op<tonemap::Operator::ACES>(ctx, flat_index, exposure, gamma);
        break;
    case tonemap::Operator::Filmic:
        tonemap_pixel_op<tonemap::Operator::Filmic>(ctx, flat_index, exposure, gamma);
        break;
    default:
        tonemap_pixel_op<tonemap::Operator::Reinhard>(ctx, flat_index, exposure, gamma);
        break;
    }
}

} // namespace rt
