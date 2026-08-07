#pragma once

/// @file
/// Standard 24-patch ColorChecker chart (classic) reference colors and
/// helpers to render it through the sandbox display pipeline.
///
/// The chart is a 6×4 grid of patches; every patch has a canonical sRGB
/// reference value (BabelColor table, 0–255).  Two debug modes consume
/// these values:
///   - Accum fill: writes *inverse tone-mapped* values into the
///     accumulation buffer so the standard pipeline tone-map step
///     (Reinhard + gamma 1/2.2) reproduces the reference sRGB colors
///     exactly — the kernel is bypassed, the rest of the pipeline runs
///     unchanged.
///   - Raw: writes the reference sRGB values directly as RGBA8,
///     bypassing the tone-map entirely.
/// See https://en.wikipedia.org/wiki/ColorChecker

#include <cmath>

namespace colorchecker {

/// Chart layout: 6 columns × 4 rows of patches.
constexpr int COLS = 6;
constexpr int ROWS = 4;
constexpr int PATCH_COUNT = COLS * ROWS;

/// Reference sRGB values (0–255) of the 24 classic patches, row-major
/// (row 0 = top of the chart).  Source: BabelColor ColorChecker sRGB
/// table.  Plain POD so it is usable from SYCL device code.
constexpr float SRGB_REF[PATCH_COUNT][3] = {
    // Row 0 — top
    {115, 82, 68},   // Dark Skin
    {194, 150, 130}, // Light Skin
    {98, 122, 157},  // Blue Sky
    {87, 108, 67},   // Foliage
    {133, 128, 177}, // Blue Flower
    {103, 189, 170}, // Bluish Green
    // Row 1
    {214, 126, 44},  // Orange
    {80, 91, 166},   // Purplish Blue
    {193, 90, 99},   // Moderate Red
    {94, 60, 108},   // Purple
    {157, 188, 64},  // Yellow Green
    {224, 163, 46},  // Orange Yellow
    // Row 2
    {56, 61, 150},   // Blue
    {70, 148, 73},   // Green
    {175, 54, 60},   // Red
    {231, 199, 31},  // Yellow
    {187, 86, 149},  // Magenta
    {8, 133, 161},   // Cyan
    // Row 3 — bottom
    {243, 243, 242}, // White
    {200, 200, 200}, // Neutral 8
    {160, 160, 160}, // Neutral 6.5
    {122, 122, 121}, // Neutral 5
    {85, 85, 85},    // Neutral 3.5
    {52, 52, 52},    // Black
};

/// Patch index (0–23) covering pixel (x, y) of a w×h frame.  The chart is
/// stretched to fill the whole frame, so every patch is always visible.
inline int patch_index(int x, int y, int w, int h) {
    return (y * ROWS / h) * COLS + (x * COLS / w);
}

/// Convert a reference sRGB component (0–255) into the "linear" value the
/// accumulation buffer must hold so the pipeline tone-map (Reinhard
/// `v/(1+v)` followed by gamma `^1/2.2`) reproduces the reference color
/// exactly:
///   out = (s^2.2) / (1 - s^2.2)
/// Host-side helper (uses std::pow).  Device kernels inline the same math
/// with sycl::pow — see render/tonemap_kernel.cpp.
inline float srgb_to_accum_linear(float s) {
    float y = std::pow(s / 255.f, 2.2f);
    return y / (1.f - y);
}

} // namespace colorchecker
