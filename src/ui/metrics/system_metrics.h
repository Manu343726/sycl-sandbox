#pragma once

#include <imgui.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

// ── MetricsHistory ────────────────────────────────────────────────────

/// Rolling window of metric samples for time-series graphs.
struct MetricsHistory {
    static constexpr int MAX_SAMPLES = 120; // ~60 s at 2 Hz

    float min_val = 0.0f;
    float max_val = 1.0f;
    float current = 0.0f;
    std::deque<float> samples;

    void push(float value) {
        current = value;
        samples.push_back(value);
        if (samples.size() > MAX_SAMPLES) samples.pop_front();
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

// ── SystemMetrics ─────────────────────────────────────────────────────

/// Host-side collector of system resource usage (CPU / RAM / GPU) with
/// rolling histories for time-series graphs.
///
/// Sampling is platform-specific (see `system_metrics.cpp`):
///   - Linux   : `/proc/stat` + `/proc/meminfo`, GPU via NVIDIA NVML
///               (`libnvidia-ml.so.1`, dlopen) or amdgpu sysfs
///               (`gpu_busy_percent`, `mem_info_vram_*`).
///   - Windows : `GetSystemTimes` + `GlobalMemoryStatusEx`, GPU via
///               NVIDIA NVML (`nvml.dll`, LoadLibrary).
///   - Other   : no-op — all metrics stay at 0 / unavailable.
///
/// NVML is resolved at runtime so the app runs fine without an NVIDIA
/// GPU or driver; the GPU section then reports "unavailable".
class SystemMetrics {
public:
    /// Minimum interval between samples (sampling is throttled to this).
    static constexpr std::chrono::milliseconds SAMPLE_PERIOD{500};

    /// Gather a fresh snapshot. Internally throttled to SAMPLE_PERIOD,
    /// safe to call every frame from the UI thread.
    void sample();

    /// Clear all history and reset the throttle timer.
    void reset();

    // ── Current snapshot ───────────────────────────────────────────
    float cpu_percent() const { return cpu_.current; }
    float ram_percent() const { return ram_.current; }
    float ram_used_gb() const { return ram_used_gb_; }
    float ram_total_gb() const { return ram_total_gb_; }
    float gpu_percent() const { return gpu_.current; }
    float vram_used_gb() const { return vram_used_gb_; }
    float vram_total_gb() const { return vram_total_gb_; }
    bool gpu_available() const { return gpu_available_; }
    const std::string &gpu_name() const { return gpu_name_; }

    /// Render the graphs inside an already-open ImGui window.
    void draw() const;

private:
    MetricsHistory cpu_;   // [0..100]
    MetricsHistory ram_;   // [0..100]
    MetricsHistory gpu_;   // [0..100]
    float ram_used_gb_ = 0.0f;
    float ram_total_gb_ = 0.0f;
    float vram_used_gb_ = 0.0f;
    float vram_total_gb_ = 0.0f;
    bool gpu_available_ = false;
    std::string gpu_name_;
    std::chrono::steady_clock::time_point last_sample_{};
};

// ── ImGui rendering ───────────────────────────────────────────────────

/// Draw one labelled history graph row (label + current value + plot).
inline void draw_metric_graph(const char *label, const MetricsHistory &h,
                              const char *value_text) {
    ImGui::TextUnformatted(label);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.6f, 1.0f), "%s", value_text);
    ImGui::PushID(label);
    if (!h.samples.empty()) {
        // Copy deque to contiguous storage for PlotLines
        std::vector<float> svec(h.samples.begin(), h.samples.end());
        ImGui::PlotLines("##hist", svec.data(), (int)svec.size(),
                         0, nullptr, h.min_val, h.max_val,
                         ImVec2(ImGui::GetContentRegionAvail().x, 36));
    } else {
        ImGui::TextDisabled("collecting\u2026");
    }
    ImGui::PopID();
}

/// Render the full metrics panel content (CPU / RAM / GPU graphs).
inline void SystemMetrics::draw() const {
    char buf[128];

    snprintf(buf, sizeof(buf), "%.1f %%", cpu_percent());
    draw_metric_graph("CPU", cpu_, buf);

    snprintf(buf, sizeof(buf), "%.1f / %.1f GiB  (%.0f %%)",
             ram_used_gb_, ram_total_gb_, ram_percent());
    draw_metric_graph("System RAM", ram_, buf);

    if (gpu_available_) {
        snprintf(buf, sizeof(buf), "%.0f %%  %s",
                 gpu_percent(), gpu_name_.empty() ? "GPU" : gpu_name_.c_str());
        draw_metric_graph("GPU", gpu_, buf);
        snprintf(buf, sizeof(buf), "VRAM %.1f / %.1f GiB",
                 vram_used_gb_, vram_total_gb_);
        ImGui::TextUnformatted(buf);
    } else {
        ImGui::TextDisabled("GPU usage: unavailable (no GPU metrics source)");
    }

#if defined(_WIN32)
    ImGui::TextDisabled("Sources: GetSystemTimes, GlobalMemoryStatusEx, NVML");
#elif defined(__linux__)
    ImGui::TextDisabled("Sources: /proc/stat, /proc/meminfo, NVML / amdgpu sysfs");
#else
    ImGui::TextDisabled("System metrics are supported on Linux and Windows only");
#endif
}
