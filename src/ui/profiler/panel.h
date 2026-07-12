#pragma once

#include "common.h"
#include "timeline.h"
#include "top_zones.h"
#include "messages.h"
#include "plots.h"
#include "zone_detail.h"
#include "kernel/profiler_host.h"
#include <imgui.h>

// ── Main profiler panel ──────────────────────────────────────────────

inline void render_profiler_panel(KernelProfiler& profiler) {
    if (!profiler.is_enabled()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "Kernel profiler is disabled.");
        if (ImGui::Button("Enable Profiler")) profiler.set_enabled(true);
        return;
    }

    bool e = profiler.is_enabled();
    bool paused = profiler.is_paused();

    // ── Tracy-style toolbar ──────────────────────────────────────────
    if (paused) {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.33f, 0.6f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.33f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.33f, 0.8f, 0.8f));
        if (ImGui::Button("> Resume")) profiler.set_paused(false);
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.6f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.0f, 0.8f, 0.8f));
        if (ImGui::Button("|| Pause")) profiler.set_paused(true);
        ImGui::PopStyleColor(3);
    }
    ImGui::SameLine();
    ImGui::Spacing(); ImGui::SameLine();

    if (ImGui::Button("X Clear")) {
        profiler.clear_data();
    }
    ImGui::SameLine();
    ImGui::Spacing(); ImGui::SameLine();

    ImGui::Checkbox("Enabled", &e);
    if (e != profiler.is_enabled()) profiler.set_enabled(e);

    ImGui::SameLine();
    ImGui::Spacing(); ImGui::SameLine();
    if (paused) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "!! PAUSED");
    } else {
        int64_t n = profiler.total_frames();
        ImGui::TextDisabled("%lld frames", (long long)n);
    }

    static TimelineState ts;
    const char* tabs[] = { "Timeline", "Top Zones", "Messages", "Plots" };

    if (ImGui::BeginTabBar("##prof_tabs")) {
        if (ImGui::BeginTabItem("Timeline")) {
            render_timeline(profiler, ts);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Top Zones")) {
            render_top_zones(profiler, ts);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Messages")) {
            render_kernel_messages(profiler);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Plots")) {
            render_kernel_plots(profiler);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // ── Zone detail popup (shown when a zone is selected) ──────────
    if (!ts.selected_name.empty() && ts.selected_idx >= 0) {
        render_zone_detail_popup(profiler, ts.selected_name, ts.selected_idx);
    }
}
