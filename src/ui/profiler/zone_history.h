#pragma once

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "common.h"
#include "kernel/profiler_host.h"
#include <imgui.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

// ── Zone details: Timeline of durations ──────────────────────────────

static void render_zone_duration_timeline(KernelProfiler& profiler,
                                           const ZoneAggregate* agg,
                                           const std::string& name,
                                           int selected_idx) {
    if (!agg || agg->sample_count == 0) return;

    auto samples = profiler.samples();
    std::vector<float> vals;
    for (auto& s : samples) {
        if (s.name == name)
            vals.push_back((float)s.duration());
    }
    if (vals.empty()) return;

    // Draw with ImPlotLines-style using ImDrawList
    float w = ImGui::GetContentRegionAvail().x;
    float h = 100.0f;
    ImVec2 orig = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    dl->AddRectFilled(orig, ImVec2(orig.x + w, orig.y + h), IM_COL32(25, 25, 25, 200));
    dl->AddRect(orig, ImVec2(orig.x + w, orig.y + h), IM_COL32(60, 60, 60, 200));

    float min_v = (float)agg->min_duration;
    float max_v = (float)agg->max_duration;
    float range_v = max_v - min_v;
    if (range_v < 1) range_v = 1;

    int n = (int)vals.size();

    // Draw line
    if (n > 1) {
        for (int i = 1; i < n; i++) {
            float x0 = orig.x + (i - 1) * w / (n - 1);
            float x1 = orig.x + i * w / (n - 1);
            float y0 = orig.y + h - 2 - (vals[i - 1] - min_v) / range_v * (h - 4);
            float y1 = orig.y + h - 2 - (vals[i] - min_v) / range_v * (h - 4);
            dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(100, 200, 255, 200), 1.5f);
        }
    }

    // Draw dots
    for (int i = 0; i < n; i++) {
        float x = orig.x + (i * w / std::max(n - 1, 1));
        float y = orig.y + h - 2 - (vals[i] - min_v) / range_v * (h - 4);
        bool is_selected = false;
        // Check if this value corresponds to the selected sample
        if (selected_idx >= 0 && (size_t)selected_idx < samples.size() &&
            samples[selected_idx].name == name) {
            int match_count = 0;
            int target_match = -1;
            for (size_t si = 0; si <= (size_t)selected_idx && si < samples.size(); si++) {
                if (samples[si].name == name) {
                    if ((int)si == selected_idx) {
                        target_match = match_count;
                        break;
                    }
                    match_count++;
                }
            }
            is_selected = (i == target_match);
        }
        ImU32 dot_col = is_selected ? IM_COL32(255, 255, 80, 255) : IM_COL32(200, 200, 200, 150);
        dl->AddCircleFilled(ImVec2(x, y), 2.5f, dot_col);
    }

    ImGui::SetCursorScreenPos(ImVec2(orig.x, orig.y + h + 2));
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "max: %s", cycle_to_string(agg->max_duration));
    ImGui::Text("Duration history  |  min: %s  |  %s  |  %lld samples",
                cycle_to_string(agg->min_duration), tmp, (long long)agg->sample_count);
}
