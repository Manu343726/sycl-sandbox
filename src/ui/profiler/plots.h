#pragma once

#include "kernel/profiler_host.h"
#include <imgui.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

// ── Kernel plots ─────────────────────────────────────────────────────

static void render_kernel_plots(KernelProfiler& profiler) {
    auto plots = profiler.plots();
    if (plots.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "No kernel plot data.");
        return;
    }

    std::unordered_map<std::string, std::vector<float>> grouped;
    std::vector<std::string> names;
    for (auto& p : plots) {
        if (grouped.find(p.name) == grouped.end())
            names.push_back(p.name);
        grouped[p.name].push_back(p.value);
    }

    for (auto& n : names) {
        auto& vals = grouped[n];
        if (vals.empty()) continue;
        float min_v = *std::min_element(vals.begin(), vals.end());
        float max_v = *std::max_element(vals.begin(), vals.end());
        float range = max_v - min_v;
        if (range == 0) { min_v *= 0.9f; max_v *= 1.1f; range = max_v - min_v; }
        ImGui::SeparatorText(n.c_str());
        ImGui::PlotLines("##vals", vals.data(), (int)vals.size(),
                         0, nullptr, min_v - range * 0.1f, max_v + range * 0.1f,
                         ImVec2(ImGui::GetContentRegionAvail().x, 80));
        ImGui::Text("%d samples | min: %.4g  max: %.4g  last: %.4g",
                    (int)vals.size(), min_v, max_v, vals.back());
    }
}
