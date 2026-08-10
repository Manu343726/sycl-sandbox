#pragma once

/// @file
/// Alpha-blend texture.
///
/// Layers N given textures and alpha-blends them in order (source-over
/// compositing): the result of each layer is mixed over the previous
/// result with its alpha:
/// @code
///   result = lerp(result, layer[i], alpha[i])   for i = 0..N-1
/// @endcode
///
/// Layers are stored in a fixed-size array of the *leaf* texture types
/// (everything except `Blend` itself) — `Blend` is part of the
/// polymorphic `Texture` variant, so storing `Texture` members here
/// would make the variant recursively infinite.  All data is plain
/// POD, so the texture stays trivially copyable and device-uploadable.

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/textures/solid_color.h>
#include <sycl-sandbox/rt/textures/colorchecker.h>
#include <sycl-sandbox/rt/textures/text.h>
#include <sycl-sandbox/variant.h>

namespace rt::textures {

/// Leaf textures that can be layered (no recursion into `Blend`).
using BlendLayer = std::variant<SolidColor, ColorChecker, Text>;

/// Alpha-blend texture: blends up to `MAX_LAYERS` textures in order.
struct Blend {
    static constexpr int MAX_LAYERS = 4;

    BlendLayer layers[MAX_LAYERS] = {};   ///< The layered textures.
    float alphas[MAX_LAYERS] = {};        ///< Per-layer alpha [0..1].
    int num_layers = 0;                   ///< Number of active layers.

    Blend() = default;

    /// Append a texture with the given alpha; layers past MAX_LAYERS
    /// are silently ignored.
    void add_layer(BlendLayer layer, float alpha) {
        if ( num_layers < MAX_LAYERS ) {
            layers[num_layers] = layer;
            alphas[num_layers] = alpha;
            num_layers++;
        }
    }

    /// Alpha-blend all layers at (u, v, time): source-over compositing.
    /// The path RNG is passed through to stochastic layers.
    float3 sample(float u, float v, float time, RNG &rng,
                  const Context &ctx = Context{}) const {
        PROFILER_ZONE("sample_blend");
        ctx.collector.on_texture_sample(3);
        float3 result = {0, 0, 0};
        for ( int i = 0; i < num_layers && i < MAX_LAYERS; i++ ) {
            float3 layer_color = {0, 0, 0};
            visit(layers[i],
                  [&](const auto &tex) { layer_color = tex.sample(u, v, time, rng, ctx); });
            result = lerp(result, layer_color, alphas[i]);
        }
        return result;
    }
};

} // namespace rt::textures
