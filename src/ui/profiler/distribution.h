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

// ── Zone details: Distribution histogram ──────────────────────────────

static void render_zone_distribution(KernelProfiler& profiler,
                                      const ZoneAggregate* agg,
                                      const std::string& name,
                                      int selected_idx) {
    if (!agg || agg->sample_count == 0) return;

    // Collect durations for this zone
    auto samples = profiler.samples();
    std::vector<uint64_t> durations;
    for (auto& s : samples) {
        if (s.name == name)
            durations.push_back(s.duration());
    }
    if (durations.empty()) return;

    // Bucket into histogram
    int num_bins = 50;
    uint64_t dmin = agg->min_duration;
    uint64_t dmax = agg->max_duration;
    if (dmax <= dmin) dmax = dmin + 1;
    double bin_w = (double)(dmax - dmin) / num_bins;
    if (bin_w <= 0) bin_w = 1;

    std::vector<int> bins(num_bins, 0);
    int selected_bin = -1;
    uint64_t selected_start = 0;
    if (selected_idx >= 0 && (size_t)selected_idx < samples.size()) {
        selected_start = samples[selected_idx].duration();
    }

    int max_count = 0;
    for (auto d : durations) {
        int b = (int)((d - dmin) / bin_w);
        if (b >= num_bins) b = num_bins - 1;
        bins[b]++;
        if (bins[b] > max_count) max_count = bins[b];
    }
    if (max_count < 1) max_count = 1;

    // Determine selected bin
    if (selected_idx >= 0) {
        selected_bin = (int)((selected_start - dmin) / bin_w);
        if (selected_bin >= num_bins) selected_bin = num_bins - 1;
    }

    // ── Draw histogram bars with ImDrawList (more control) ────────
    float w = ImGui::GetContentRegionAvail().x;
    float h = 120.0f;
    ImVec2 orig = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    dl->AddRectFilled(orig, ImVec2(orig.x + w, orig.y + h), IM_COL32(25, 25, 25, 200));
    dl->AddRect(orig, ImVec2(orig.x + w, orig.y + h), IM_COL32(60, 60, 60, 200));

    float bar_w = w / (float)num_bins;
    for (int i = 0; i < num_bins; i++) {
        float bh = ((float)bins[i] / (float)max_count) * (h - 4);
        ImVec2 p0(orig.x + i * bar_w + 1, orig.y + h - 2 - bh);
        ImVec2 p1(p0.x + bar_w - 1, orig.y + h - 2);

        ImU32 col;
        if (i == selected_bin) {
            col = IM_COL32(255, 255, 80, 220);
        } else {
            float t = (float)i / num_bins;
            col = ImColor::HSV(t * 0.6f, 0.5f, 0.6f);
        }
        dl->AddRectFilled(p0, p1, col);
        dl->AddRect(p0, p1, IM_COL32(0, 0, 0, 80));

        // Hover tooltip on bin
        if (ImGui::IsMouseHoveringRect(p0, p1)) {
            uint64_t bin_lo = dmin + (uint64_t)(i * bin_w);
            uint64_t bin_hi = dmin + (uint64_t)((i + 1) * bin_w);
            ImGui::BeginTooltip();
            ImGui::Text("Range: %s \xe2\x80\x93 %s", cycle_to_string(bin_lo), cycle_to_string(bin_hi));
            ImGui::Text("Count: %d", bins[i]);
            ImGui::EndTooltip();
        }
    }

    // ── Label the selected zone on the histogram ──────────────────
    if (selected_bin >= 0 && selected_idx >= 0) {
        auto all_samp = profiler.samples();
        if ((size_t)selected_idx < all_samp.size()) {
            auto& sel = all_samp[selected_idx];
            float sel_x = orig.x + (float)((sel.duration() - dmin) * (w - 4) / (dmax - dmin));
            dl->AddLine(ImVec2(sel_x, orig.y), ImVec2(sel_x, orig.y + h),
                        IM_COL32(255, 255, 255, 180), 1.5f);
        }
    }

    // Label under graph
    ImGui::SetCursorScreenPos(ImVec2(orig.x, orig.y + h + 2));
    ImGui::Text("Distribution: %s \xe2\x80\x93 %s  (%d bins)",
                cycle_to_string(dmin), cycle_to_string(dmax), num_bins);
}
