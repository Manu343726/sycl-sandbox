#pragma once

/// @file
/// Tone-map operator curves shared by the framebuffer display pipeline
/// (tonemap.h) and the scene-debug overlay (renderer.h): the debug ray
/// colours are displayed with the SAME operator the framebuffer uses,
/// so the traced ray matches what the rendered image shows.
///
/// Host/device-neutral (pure float math) — no SYCL includes.

namespace tonemap {

// ── Tone-map operators ────────────────────────────────────────────────
/// Indexes into the `tonemap_operator` standard param; keep in sync with
/// the enum_options declared in loader.cpp.
enum Operator : int {
    Reinhard = 0,   ///< classic x/(1+x)
    ACES     = 1,   ///< ACES fitted (Narkowicz 2015)
    Filmic   = 2,   ///< Hable / Uncharted 2 curve
};

/// Apply the selected tone-map operator to one linear channel.
/// No clamping inside — callers clamp before the final pack.
inline float apply_operator(float x, int op) {
    switch ( op ) {
        case Operator::Reinhard:
            return x / (1.0f + x);
        case Operator::ACES: {
            // Narkowicz 2015 ACES fitted curve.
            const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
            return (x * (a * x + b)) / (x * (c * x + d) + e);
        }
        case Operator::Filmic: {
            // Hable's Uncharted 2 filmic curve.
            const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
        }
        default:
            return x / (1.0f + x);
    }
}

} // namespace tonemap
