#pragma once

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "common.h"
#include "distribution.h"
#include "zone_history.h"
#include "kernel/profiler_host.h"
#include <imgui.h>
#include <cstring>
#include <vector>

// ── Zone details popup (Tracy-style separate window) ─────────────────

static void render_zone_detail_popup(KernelProfiler& profiler,
                                      const std::string& name,
                                      int selected_idx) {
    if (name.empty() || selected_idx < 0) return;

    auto* agg = profiler.zone_stats(name);
    if (!agg || agg->sample_count == 0) return;

    ImGui::SetNextWindowSize(ImVec2(550, 500), ImGuiCond_FirstUseEver);
    bool open = true;
    if (!ImGui::Begin("Zone Info", &open, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::End();
        return;
    }

    // ── Header ────────────────────────────────────────────────────
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextUnformatted(name.c_str());
    ImGui::PopFont();

    ImGui::Separator();

    // ── Stats ─────────────────────────────────────────────────────
    const float ty = ImGui::GetTextLineHeight();
    ImGui::Text("Samples:  %lld", (long long)agg->sample_count);
    ImGui::SameLine(); ImGui::Text("   Total:  %s", cycle_to_string(agg->total_duration));
    ImGui::Text("Average:  %s", cycle_to_string((uint64_t)agg->avg_duration()));
    ImGui::SameLine(); ImGui::Text("   Min:  %s", cycle_to_string(agg->min_duration));
    ImGui::SameLine(); ImGui::Text("   Max:  %s", cycle_to_string(agg->max_duration));
    ImGui::Text("Last:     %s", cycle_to_string(agg->last_duration));

    ImGui::Separator();

    // ── Self time vs children (parent-relative bar) ───────────────
    auto samples = profiler.samples();
    if ((size_t)selected_idx < samples.size()) {
        auto& sel = samples[selected_idx];
        uint64_t child_total = 0;
        for (auto& s : samples) {
            if (s.frame_index == sel.frame_index &&
                s.depth == sel.depth + 1 &&
                s.start_cycles >= sel.start_cycles &&
                s.end_cycles <= sel.end_cycles) {
                child_total += s.duration();
            }
        }
        uint64_t self_time = sel.duration();
        if (child_total < self_time) self_time -= child_total;
        else self_time = 0;

        if (sel.duration() > 0) {
            ImGui::Text("Parent-relative breakdown:");
            char pbuf[128];
            double self_pct = 100.0 * self_time / sel.duration();
            snprintf(pbuf, sizeof(pbuf), "Self: %s (%.1f%%)",
                     cycle_to_string(self_time), self_pct);
            ImGui::ProgressBar((float)(self_pct / 100.0), ImVec2(-1, ty), pbuf);

            if (child_total > 0) {
                double child_pct = 100.0 * child_total / sel.duration();
                snprintf(pbuf, sizeof(pbuf), "Children: %s (%.1f%%)",
                         cycle_to_string(child_total), child_pct);
                ImGui::ProgressBar((float)(child_pct / 100.0), ImVec2(-1, ty), pbuf);
            }

            // Call stack (zone hierarchy): path from root to this zone
            std::vector<const ZoneSample*> call_stack;
            int target_depth = sel.depth;
            for (auto& s : samples) {
                if (s.frame_index != sel.frame_index) continue;
                if (s.depth == 0 && call_stack.empty()) {
                    call_stack.push_back(&s);
                } else if (!call_stack.empty() && s.depth == call_stack.back()->depth + 1 &&
                           s.start_cycles >= call_stack.back()->start_cycles &&
                           s.end_cycles <= call_stack.back()->end_cycles) {
                    call_stack.push_back(&s);
                    if (&s == &sel) break;
                }
                if (&s == &sel) break;
            }

            if (!call_stack.empty()) {
                ImGui::Separator();
                if (ImGui::TreeNode("Call Stack (Zone Hierarchy)")) {
                    for (size_t si = 0; si < call_stack.size(); si++) {
                        char indent[64];
                        memset(indent, ' ', si * 2);
                        indent[si * 2] = '\0';
                        ImGui::Text("%s%s  \xe2\x94\x82  %s", indent,
                                    call_stack[si]->name.c_str(),
                                    cycle_to_string(call_stack[si]->duration()));
                    }
                    ImGui::TreePop();
                }
            }
        }
    }

    ImGui::Separator();

    // ── Interchangeable graph: Timeline vs Distribution ────────────
    static int graph_mode = 0;
    ImGui::RadioButton("Duration Timeline", &graph_mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Distribution Histogram", &graph_mode, 1);

    if (graph_mode == 0) {
        render_zone_duration_timeline(profiler, agg, name, selected_idx);
    } else {
        render_zone_distribution(profiler, agg, name, selected_idx);
    }

    ImGui::End();
    if (!open) {
        const_cast<std::string&>(name) = std::string();
    }
}
