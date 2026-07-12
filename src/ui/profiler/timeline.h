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

// ── Draw timeline header ruler (Tracy-style) ─────────────────────────

static void draw_timeline_ruler(ImDrawList* dl, const ImVec2& wpos, float w,
                                 uint64_t t_start, uint64_t t_range,
                                 double view_lo, double view_hi) {
    const float ty = ImGui::GetTextLineHeight();
    const float ty05 = roundf(ty * 0.5f);
    const float ty025 = roundf(ty * 0.25f);
    const float ty0375 = roundf(ty * 0.375f);
    const float dpos_x = wpos.x + 0.5f;
    const float dpos_y = wpos.y + 0.5f;

    double pxns = w / (view_hi - view_lo);
    double nspx = 1.0 / pxns;

    // Major tick spacing (log10-based, like Tracy)
    double scale = std::max(0.0, round(log10(nspx) + 2.0));
    double step = pow(10.0, scale);

    // Ensure at least ~5 major ticks
    while (step * 5 < (view_hi - view_lo)) step *= 10.0;
    while (step * 20 > (view_hi - view_lo)) step /= 10.0;

    // Background
    dl->AddRectFilled(wpos, ImVec2(wpos.x + w, wpos.y + ty * 1.5f), IM_COL32(30, 30, 30, 200));

    double x = 0;
    int64_t tt = (int64_t)(ceil(view_lo / step) * step);
    while (x < w) {
        double t = tt;
        double px = (t - view_lo) * pxns;
        if (px > w) break;

        if (px >= 0) {
            // Major tick
            dl->AddLine(ImVec2(dpos_x + (float)px, dpos_y), ImVec2(dpos_x + (float)px, dpos_y + ty05), IM_COL32(120, 120, 120, 200));
            // Label
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "%s", cycle_to_string((uint64_t)t));
            dl->AddText(ImVec2(wpos.x + (float)px + 2, wpos.y + 1), IM_COL32(180, 180, 180, 240), lbl);

            // Minor ticks
            for (int i = 1; i < 10; i++) {
                double mpx = px + i * step * 0.1 * pxns;
                if (mpx > w) break;
                float mh = (i == 5) ? ty0375 : ty025;
                dl->AddLine(ImVec2(dpos_x + (float)mpx, dpos_y), ImVec2(dpos_x + (float)mpx, dpos_y + mh), IM_COL32(80, 80, 80, 120));
            }
        }
        x += step * pxns;
        tt += (int64_t)step;
    }
}

// ── Render the timeline flamegraph ────────────────────────────────────

static void render_timeline(KernelProfiler& profiler, TimelineState& ts) {
    auto samples = profiler.samples();
    if (samples.empty()) { ImGui::TextUnformatted("No samples"); return; }

    uint64_t t_start = profiler.timeline_start();
    uint64_t t_end   = profiler.timeline_end();
    if (t_end <= t_start) return;

    uint64_t t_range = t_end - t_start;
    double view_lo = t_start + ts.view_start * t_range;
    double view_hi = t_start + ts.view_end   * t_range;
    double view_range = view_hi - view_lo;

    float avail_w = ImGui::GetContentRegionAvail().x;
    if (avail_w < 10) return;

    // ── Ruler header (Tracy-style) ─────────────────────────────────
    float ruler_h = ImGui::GetTextLineHeight() * 1.5f + 4;
    ImVec2 ruler_origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Reserve ruler space
    ImGui::InvisibleButton("##ruler", ImVec2(avail_w, ruler_h));
    draw_timeline_ruler(dl, ruler_origin, avail_w, t_start, t_range, view_lo, view_hi);

    // ── Frame overview bar (thin bar showing all zones) ────────────
    float overview_h = 10.0f;
    ImVec2 ov_origin = ImGui::GetCursorScreenPos();
    dl->AddRectFilled(ov_origin, ImVec2(ov_origin.x + avail_w, ov_origin.y + overview_h), IM_COL32(20, 20, 20, 200));
    float scale_x = avail_w / (float)t_range;
    for (auto& s : samples) {
        float sx = (float)(s.start_cycles - t_start) * scale_x;
        float sw = std::max(1.0f, (float)(s.duration()) * scale_x);
        ImU32 col = flame_color(s.name.c_str(), s.depth);
        dl->AddRectFilled(ImVec2(ov_origin.x + sx, ov_origin.y),
                          ImVec2(ov_origin.x + sx + sw, ov_origin.y + overview_h), col);
    }
    // Viewport rect on overview
    float ov_vs = (float)(ts.view_start * avail_w);
    float ov_ve = (float)(ts.view_end   * avail_w);
    dl->AddRect(ImVec2(ov_origin.x + ov_vs, ov_origin.y),
                ImVec2(ov_origin.x + ov_ve, ov_origin.y + overview_h),
                IM_COL32(255, 255, 100, 200));

    ImGui::SetCursorScreenPos(ImVec2(ov_origin.x, ov_origin.y + overview_h + 2));

    // ── Zone flamegraph ────────────────────────────────────────────
    float zone_h = ImGui::GetContentRegionAvail().y - 40;
    if (zone_h < 50) zone_h = 50;
    ImVec2 canvas_size = ImVec2(avail_w, zone_h);
    ImGui::BeginChild("##timeline_body", canvas_size, true,
                      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
    dl = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##tl_canvas", canvas_size);
    bool hovered = ImGui::IsItemHovered();

    // ── Zoom / Pan ────────────────────────────────────────────────
    if (hovered) {
        float mw = ImGui::GetIO().MouseWheel;
        if (mw != 0.0f) {
            double cx = (ImGui::GetMousePos().x - origin.x) / canvas_size.x;
            if (mw > 0) ts.zoom_in(cx);
            else        ts.zoom_out(cx);
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 3.0f)) {
            auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle, 3.0f);
            ts.pan(d.x / canvas_size.x);
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
        }
        // Right-click to reset view
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
            ts.reset();
    }

    // ── Update zoom from view changes ─────────────────────────────
    view_lo = t_start + ts.view_start * t_range;
    view_hi = t_start + ts.view_end   * t_range;
    view_range = view_hi - view_lo;
    scale_x = canvas_size.x / (float)view_range;

    // ── Build visible samples + compute depth ──────────────────────
    struct VisSample {
        int index;
        const ZoneSample* s;
        float x, w;
    };
    std::vector<VisSample> visible;
    visible.reserve(samples.size());
    int idx = 0;
    int max_depth = 0;
    for (auto& s : samples) {
        if (s.end_cycles < view_lo || s.start_cycles > view_hi) { idx++; continue; }
        float px = (float)((double)(s.start_cycles - view_lo) * scale_x);
        float pw = std::max(1.0f, (float)((double)(s.duration()) * scale_x));
        visible.push_back({idx, &s, px, pw});
        max_depth = std::max(max_depth, s.depth);
        idx++;
    }

    if (visible.empty()) { ImGui::EndChild(); return; }

    // ── Render blocks (Tracy-style depth layout) ───────────────────
    float ostep = ts.row_h + 1;
    float y_base = origin.y + 2;

    for (auto& v : visible) {
        float y = y_base + v.s->depth * ostep;
        ImU32 col = flame_color(v.s->name.c_str(), v.s->depth);
        ImVec2 p0(origin.x + v.x, y);
        ImVec2 p1(origin.x + v.x + v.w, y + ts.row_h - 1);

        dl->AddRectFilled(p0, p1, col);
        dl->AddRect(p0, p1, darken_color(col));

        // ── Zone text: only if zone is wide enough ────────────────
        ImVec2 tsz = ImGui::CalcTextSize(v.s->name.c_str());
        float zsz = std::max(v.w, 1.0f); // visible zone width in pixels

        if (tsz.x < zsz - 4) {
            // Zone wide enough — center text
            float tx = v.x + (v.w - tsz.x) * 0.5f;
            if (tx < 0) tx = 2;
            if (tx + tsz.x > canvas_size.x) tx = canvas_size.x - tsz.x - 2;
            dl->AddText(ImVec2(origin.x + tx, p0.y + 1), IM_COL32(255, 255, 255, 240), v.s->name.c_str());
        } else if (zsz > 10) {
            // Zone partially visible — draw clipped text
            ImGui::PushClipRect(p0, ImVec2(p1.x, p0.y + ts.row_h), true);
            dl->AddText(ImVec2(p0.x + 2, p0.y + 1), IM_COL32(200, 200, 200, 180), v.s->name.c_str());
            ImGui::PopClipRect();
        }

        // ── Click to select (opens popup) ─────────────────────────
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            ImGui::IsMouseHoveringRect(p0, p1)) {
            ts.selected_idx = v.index;
            ts.selected_name = v.s->name;
        }

        // ── Hover tooltip ─────────────────────────────────────────
        if (ImGui::IsMouseHoveringRect(p0, p1)) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", v.s->name.c_str());
            ImGui::Separator();
            ImGui::Text("Duration: %s", cycle_to_string(v.s->duration()));
            ImGui::Text("Depth: %d", v.s->depth);
            ImGui::Text("Frame: %lld", (long long)v.s->frame_index);
            ImGui::EndTooltip();
        }
    }

    // ── Highlight selected ────────────────────────────────────────
    if (ts.selected_idx >= 0) {
        for (auto& v : visible) {
            if (v.index == ts.selected_idx) {
                float y = y_base + v.s->depth * ostep;
                ImVec2 p0(origin.x + v.x, y);
                ImVec2 p1(origin.x + v.x + v.w, y + ts.row_h - 1);
                dl->AddRect(p0, p1, IM_COL32(255, 255, 100, 255), 0.0f, ImDrawFlags_None, 3.0f);
                break;
            }
        }
    }

    // ── Stats bar ─────────────────────────────────────────────────
    float total_h = (max_depth + 2) * ostep;
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + total_h + 8));
    ImGui::Text("Range: %s  |  view: [%s, %s]  |  %zu samples  |  %lld frames",
                cycle_to_string(t_range),
                cycle_to_string((uint64_t)view_lo),
                cycle_to_string((uint64_t)view_hi),
                samples.size(), (long long)profiler.total_frames());

    ImGui::EndChild();
}
