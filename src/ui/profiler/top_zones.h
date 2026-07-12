#pragma once

#include "common.h"
#include "kernel/profiler_host.h"
#include <imgui.h>
#include <cstdio>

// ── Top zones table ─────────────────────────────────────────────────

static void render_top_zones(KernelProfiler& profiler, TimelineState& ts) {
    auto top = profiler.top_zones(50);
    if (top.empty()) { ImGui::TextUnformatted("No zone data"); return; }

    ImGui::Columns(6, "##topzones", false);
    ImGui::Text("Zone"); ImGui::NextColumn();
    ImGui::Text("Samples"); ImGui::NextColumn();
    ImGui::Text("Total"); ImGui::NextColumn();
    ImGui::Text("Avg"); ImGui::NextColumn();
    ImGui::Text("Max"); ImGui::NextColumn();
    ImGui::Text("% Total"); ImGui::NextColumn();
    ImGui::Separator();

    uint64_t grand_total = 0;
    for (auto* a : top) grand_total += a->total_duration;

    for (auto* a : top) {
        if (ImGui::Selectable(a->name.c_str(), a->name == ts.selected_name,
                              ImGuiSelectableFlags_SpanAllColumns)) {
            ts.selected_name = a->name;
            ts.selected_idx = -1;
            auto samples = profiler.samples();
            for (size_t i = 0; i < samples.size(); i++) {
                if (samples[i].name == a->name) {
                    ts.selected_idx = (int)i;
                    break;
                }
            }
        }
        ImGui::NextColumn();
        ImGui::Text("%lld", (long long)a->sample_count); ImGui::NextColumn();
        ImGui::Text("%s", cycle_to_string(a->total_duration)); ImGui::NextColumn();
        ImGui::Text("%s", cycle_to_string((uint64_t)a->avg_duration())); ImGui::NextColumn();
        ImGui::Text("%s", cycle_to_string(a->max_duration)); ImGui::NextColumn();
        if (grand_total > 0) {
            double pct = 100.0 * a->total_duration / grand_total;
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f%%", pct);
            ImGui::TextUnformatted(buf);
        } else {
            ImGui::Text("-");
        }
        ImGui::NextColumn();
    }
    ImGui::Columns(1);
}
