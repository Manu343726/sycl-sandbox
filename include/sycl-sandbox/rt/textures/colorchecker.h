#pragma once

/// @file
/// Procedural ColorChecker texture.
///
/// Renders the classic 24-patch X-Rite ColorChecker chart (reference
/// sRGB values from <sycl-sandbox/colorchecker.h>) as an infinite
/// tiling: if u and/or v overflow [0,1] the chart simply continues —
/// the texture never clamps, so any surface UV maps onto an endless
/// grid of charts.

#include <sycl-sandbox/colorchecker.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/math.h>

namespace rt::textures {

/// Procedural texture: the 24-patch ColorChecker chart, tiling
/// infinitely over (u, v).
struct ColorChecker {
    /// Chart repetitions per UV unit (1 = one full chart per [0,1]).
    float scale_u = 1.f;
    float scale_v = 1.f;

    /// The 24 reference sRGB colours in [0,1], row-major 6×4
    /// (row 0 = top of the chart), copied from colorchecker::SRGB_REF.
    float3 patches[24] = {};

    ColorChecker() {
        fill_patches();
    }

    explicit ColorChecker(float su, float sv) : scale_u(su), scale_v(sv) {
        fill_patches();
    }

    /// Sample the chart at (u, v); out-of-range UVs wrap around, so the
    /// chart repeats forever in both directions.  `time` and `rng` are
    /// ignored (the chart is deterministic).
    float3 sample(float u, float v, float time, RNG &rng) const {
        (void)time;
        (void)rng;
        // Wrap (u, v) to [0,1) per axis — floor handles negatives too,
        // so on UV overflow the chart just continues.
        u = u * scale_u - math::floor(u * scale_u);
        v = v * scale_v - math::floor(v * scale_v);
        constexpr int COLS = 6, ROWS = 4;
        int col = (int)(u * COLS);
        int row = (int)(v * ROWS);
        if ( col >= COLS ) col = COLS - 1;
        if ( row >= ROWS ) row = ROWS - 1;
        return patches[row * COLS + col];
    }

private:
    void fill_patches() {
        for ( int i = 0; i < 24; i++ ) {
            patches[i] = {colorchecker::SRGB_REF[i][0] / 255.f,
                          colorchecker::SRGB_REF[i][1] / 255.f,
                          colorchecker::SRGB_REF[i][2] / 255.f};
        }
    }
};

} // namespace rt::textures
