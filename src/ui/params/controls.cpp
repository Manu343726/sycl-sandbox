#include "controls.h"
#include <imgui.h>
#include <cstdio>
#include <cmath>

bool render_param_controls(const scene_loader::SceneDescriptor &desc,
                            float *params,
                            bool read_only,
                            scene_loader::ParamCategory category) {
    bool changed = false;

    for ( const auto &pd : desc.params ) {
        // Filter by category if specified
        if ( category != static_cast<scene_loader::ParamCategory>(-1) &&
             pd.category != category ) continue;

        // Build a typed ParamRef instead of raw pointer arithmetic.
        scene_loader::ParamRef ref = desc.find_param_ref(pd.name);
        if ( !ref.valid() ) continue;

        ImGui::PushID(pd.name.c_str());

        switch ( pd.type ) {
        case ParamType::INT: {
            int v = ref.as_int();
            if ( pd.has_range ) {
                // INT params store their range in the _i fields; the
                // _f fields are only populated for FLOAT params.
                if ( ImGui::SliderInt(pd.name.c_str(), &v,
                                      pd.range_min_i, pd.range_max_i) ) {
                    ref.set(v); changed = true;
                }
            } else {
                if ( ImGui::InputInt(pd.name.c_str(), &v, 1, 10) ) {
                    ref.set(v); changed = true;
                }
            }
            break;
        }
        case ParamType::FLOAT: {
            if ( pd.has_range ) {
                if ( ImGui::SliderFloat(pd.name.c_str(), ref.ptr(), pd.range_min_f, pd.range_max_f) ) {
                    changed = true;
                }
            } else {
                if ( ImGui::InputFloat(pd.name.c_str(), ref.ptr(), 0.1f, 1.0f, "%.3f") ) {
                    changed = true;
                }
            }
            break;
        }
        case ParamType::COLOR_RGB: {
            if ( ImGui::ColorEdit3(pd.name.c_str(), ref.ptr(), ImGuiColorEditFlags_NoInputs) ) {
                changed = true;
            }
            break;
        }
        case ParamType::COLOR_RGBA: {
            if ( ImGui::ColorEdit4(pd.name.c_str(), ref.ptr(), ImGuiColorEditFlags_NoInputs) ) {
                changed = true;
            }
            break;
        }
        case ParamType::VEC3: {
            if ( ImGui::DragFloat3(pd.name.c_str(), ref.ptr(), 0.01f) ) {
                changed = true;
            }
            break;
        }
        case ParamType::BOOL: {
            bool b = ref.as_bool();
            if ( ImGui::Checkbox(pd.name.c_str(), &b) ) {
                ref.set(b);
                changed = true;
            }
            break;
        }
        case ParamType::ENUM: {
            int v = ref.as_int();
            if ( !pd.enum_options.empty() ) {
                // Named choices → combo box.  Clamp out-of-range values
                // (e.g. a YAML default beyond the option list).
                if ( v < 0 || v >= (int)pd.enum_options.size() ) {
                    v = 0;
                    ref.set(v);
                    changed = true;
                }
                const char *label = pd.enum_options[(size_t)v].c_str();
                if ( ImGui::BeginCombo(pd.name.c_str(), label) ) {
                    for ( size_t i = 0; i < pd.enum_options.size(); ++i ) {
                        bool selected = ((size_t)v == i);
                        if ( ImGui::Selectable(pd.enum_options[i].c_str(),
                                               selected) ) {
                            ref.set((int)i);
                            changed = true;
                        }
                        if ( selected ) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            } else {
                if ( ImGui::InputInt(pd.name.c_str(), &v, 1, 10) ) {
                    ref.set(v); changed = true;
                }
            }
            break;
        }
        }

        // Tooltip with description
        if ( ImGui::IsItemHovered() && !pd.description.empty() ) {
            ImGui::SetTooltip("%s", pd.description.c_str());
        }

        ImGui::PopID();
    }

    return changed;
}
