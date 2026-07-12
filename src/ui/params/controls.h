#pragma once
#include <sycl-sandbox/scene_loader.h>
#include <cstdint>

/// Render procedural parameter controls from a SceneDescriptor.
///
/// Iterates all params in order (Render → Camera → Kernel) and renders
/// the appropriate ImGui widget for each based on type + range:
///   - INT with range    → SliderInt
///   - INT without range → InputInt (step 1)
///   - FLOAT with range  → SliderFloat
///   - FLOAT without range → InputFloat
///   - COLOR_RGB         → ColorEdit3
///   - VEC3              → DragFloat3 / InputFloat3
///   - BOOL              → Checkbox
///
/// Each widget shows a tooltip with the parameter's description (doc).
///
/// @param desc    Scene descriptor (contains param metadata + buffer layout).
/// @param params  Live float buffer (scene_desc.current_buffer.data()).
/// @param read_only  If true, all controls are disabled (display-only).
/// @param category Optional — if non-null, only render params in this category.
/// @return true if any value changed.
bool render_param_controls(const scene_loader::SceneDescriptor &desc,
                            float *params,
                            bool read_only = false,
                            scene_loader::ParamCategory category =
                                static_cast<scene_loader::ParamCategory>(-1));
