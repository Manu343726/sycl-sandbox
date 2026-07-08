#include "param_ui.h"
#include "imgui.h"
#include <cstdio>
#include <cstring>

static bool render_one(const ParamMeta& p, float* data) {
    bool changed = false;

    switch (p.type) {
    case ParamType::FLOAT: {
        auto* v = reinterpret_cast<float*>(data);
        if (p.range.f.min_f <= p.range.f.max_f) {
            changed = ImGui::SliderFloat(p.name, v,
                                         p.range.f.min_f, p.range.f.max_f);
        } else {
            float step  = p.range.f.step_f > 0.f ? p.range.f.step_f : 0.1f;
            float step2 = step * 10.f;
            changed = ImGui::InputFloat(p.name, v, step, step2, "%.4f");
        }
        break;
    }
    case ParamType::INT: {
        auto* v = reinterpret_cast<int32_t*>(data);
        if (p.range.i.min_i <= p.range.i.max_i) {
            changed = ImGui::SliderInt(p.name, v,
                                       p.range.i.min_i, p.range.i.max_i);
        } else {
            changed = ImGui::InputInt(p.name, v,
                                      p.range.i.step_i ? p.range.i.step_i : 1);
        }
        break;
    }
    case ParamType::COLOR_RGB:
        changed = ImGui::ColorEdit3(p.name, data,
                                    ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_PickerHueWheel);
        break;
    case ParamType::COLOR_RGBA:
        changed = ImGui::ColorEdit4(p.name, data,
                                    ImGuiColorEditFlags_NoInputs |
                                    ImGuiColorEditFlags_PickerHueWheel);
        break;
    case ParamType::VEC3:
        changed = ImGui::InputFloat3(p.name, data, "%.3f");
        break;
    case ParamType::BOOL:
        changed = ImGui::Checkbox(p.name, reinterpret_cast<bool*>(data));
        break;
    case ParamType::ENUM: {
        auto* v = reinterpret_cast<int32_t*>(data);
        if (p.enum_labels && p.enum_count > 0)
            changed = ImGui::Combo(p.name, v, p.enum_labels, p.enum_count);
        break;
    }
    }

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
        ImGui::SetTooltip("%s", p.description ? p.description : "");
    return changed;
}

bool render_param_controls(const KernelDesc& desc, float* params, bool read_only) {
    bool any_changed = false;
    for (int i = 0; i < desc.param_count; i++) {
        const auto& p = desc.params[i];

        if (read_only) {
            ImGui::BeginDisabled();
            render_one(p, params + p.buffer_offset / sizeof(float));
            ImGui::EndDisabled();
        } else {
            if (render_one(p, params + p.buffer_offset / sizeof(float)))
                any_changed = true;
        }
    }
    return any_changed;
}
