#pragma once
#include <sycl-sandbox/scene_loader.h>
#include <imgui.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <deque>

// ── StatHistory ────────────────────────────────────────────────────────

/// Rolling window of statistic samples for time-series graphs.
struct StatHistory {
    static constexpr int MAX_SAMPLES = 300; // ~5 seconds at 60 FPS

    std::string name;
    float min_val = 0.0f;
    float max_val = 1.0f;
    float current = 0.0f;
    std::deque<float> samples;

    void push(float value) {
        current = value;
        samples.push_back(value);
        if (samples.size() > MAX_SAMPLES)
            samples.pop_front();
        // Update min/max from entire history
        min_val = *std::min_element(samples.begin(), samples.end());
        max_val = *std::max_element(samples.begin(), samples.end());
        // Ensure min/max are not equal (avoids division by zero in graph)
        if (max_val - min_val < 1e-6f) {
            max_val = min_val + 1.0f;
        }
    }

    void reset() {
        samples.clear();
        current = 0.0f;
        min_val = 0.0f;
        max_val = 1.0f;
    }
};

// ── StatStore ──────────────────────────────────────────────────────────

/// Host-side collection of statistic histories, keyed by name.
/// The host reads the current stat buffer after each frame and pushes
/// values into the corresponding StatHistory for graphing.
struct StatStore {
    std::vector<StatHistory> histories;

    /// Ensure a history entry exists for each stat descriptor.
    void init_from(const std::vector<scene_loader::StatDescriptor> &stats) {
        for (auto &sd : stats) {
            if (find(sd.name)) continue;
            StatHistory h;
            h.name = sd.name;
            histories.push_back(std::move(h));
        }
    }

    StatHistory *find(const std::string &name) {
        for (auto &h : histories)
            if (h.name == name) return &h;
        return nullptr;
    }

    const StatHistory *find(const std::string &name) const {
        for (auto &h : histories)
            if (h.name == name) return &h;
        return nullptr;
    }

    /// Read a value from the current stat buffer and push it into history.
    void ingest(const scene_loader::SceneDescriptor &desc) {
        for (auto &h : histories) {
            scene_loader::StatRef ref = desc.find_stat_ref(h.name);
            if (ref.valid()) h.push(ref.as_float());
        }
    }

    void reset() {
        for (auto &h : histories) h.reset();
    }
};

// ── Helpers ───────────────────────────────────────────────────────────

/// Map a float to a colour for gauge/flag display.
inline ImU32 gauge_color(float t) {
    // t in [0,1]: green → yellow → red
    if (t < 0.5f) {
        float s = t * 2.0f;
        return IM_COL32((int)(255 * s), 255, 0, 255);
    } else {
        float s = (t - 0.5f) * 2.0f;
        return IM_COL32(255, (int)(255 * (1.0f - s)), 0, 255);
    }
}

/// Draw a single statistic value with its history graph.
inline void draw_stat(const StatHistory &h,
                       const scene_loader::StatDescriptor &sd) {
    // Build a label with tooltip
    ImGui::PushID(sd.name.c_str());

    ImU32 col = IM_COL32(200, 200, 200, 255);
    switch (sd.viz) {
        case scene_loader::VisualizationHint::Flag: {
            bool ok = h.current != 0.0f;
            col = ok ? IM_COL32(80, 220, 80, 255) : IM_COL32(220, 80, 80, 255);
            ImGui::TextColored(ImVec4(ImColor(col)), "%s", sd.name.c_str());
            ImGui::SameLine();
            ImGui::TextUnformatted(ok ? "● OK" : "○ FAIL");
            break;
        }
        case scene_loader::VisualizationHint::Gauge: {
            float t = h.current;
            if (sd.has_range && (sd.range_max_f - sd.range_min_f) > 1e-6f)
                t = (h.current - sd.range_min_f) / (sd.range_max_f - sd.range_min_f);
            ImGui::Text("%s", sd.name.c_str());
            ImGui::SameLine();
            char val_buf[32];
            snprintf(val_buf, sizeof(val_buf), "%.2f", h.current);
            ImGui::ProgressBar(t, ImVec2(ImGui::GetContentRegionAvail().x, 0), val_buf);
            break;
        }
        case scene_loader::VisualizationHint::Graph:
        default: {
            // Graph — copy deque to vector for contiguous storage
            ImGui::Text("%s  [%.2f]", sd.name.c_str(), h.current);
            std::vector<float> svec(h.samples.begin(), h.samples.end());
            if ( !svec.empty() ) {
                ImGui::PlotLines("##hist", svec.data(), (int)svec.size(),
                                 0, nullptr, h.min_val, h.max_val,
                                 ImVec2(ImGui::GetContentRegionAvail().x, 40));
            }
            break;
        }
    }

    // Tooltip with description
    if ( ImGui::IsItemHovered() && !sd.description.empty() ) {
        ImGui::SetTooltip("%s", sd.description.c_str());
    }

    ImGui::PopID();
}

/// Render all statistics as a vertical list of labeled values + graphs.
inline void render_stat_panel(const StatStore &store,
                               const scene_loader::SceneDescriptor &desc) {
    for (auto &h : store.histories) {
        // Find matching descriptor
        const scene_loader::StatDescriptor *sd = nullptr;
        for (auto &s : desc.stats) {
            if (s.name == h.name) { sd = &s; break; }
        }
        if (!sd) continue;
        draw_stat(h, *sd);
    }
}
