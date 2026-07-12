#pragma once

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include <imgui.h>
#include <cstdio>
#include <cmath>
#include <string>

// ── Tracy-style time formatting for cycles ────────────────────────────

/// Format cycles in human-readable units (like Tracy's TimeToString).
/// Uses snprintf for robustness — no IntTable100 static data needed.
static inline const char* cycle_to_string(uint64_t cycles) {
    constexpr size_t Pool = 8;
    static char bufpool[Pool][48];
    static int bufsel = 0;
    char* buf = bufpool[bufsel];
    bufsel = (bufsel + 1) % Pool;

    if (cycles < 1000) {
        snprintf(buf, 48, "%llu cyc", (unsigned long long)cycles);
    } else if (cycles < 1000000) {
        snprintf(buf, 48, "%.2f Kcyc", cycles / 1000.0);
    } else if (cycles < 1000000000ull) {
        snprintf(buf, 48, "%.2f Mcyc", cycles / 1000000.0);
    } else {
        snprintf(buf, 48, "%.2f Gcyc", cycles / 1000000000.0);
    }
    return buf;
}

// ── Colour helpers ───────────────────────────────────────────────────

inline ImU32 flame_color(const char* name, int depth) {
    uint32_t h = 2166136261u;
    for (const char* p = name; *p; ++p) { h ^= (uint8_t)*p; h *= 16777619u; }
    float hue = ((float)(h % 360) + depth * 37.0f);
    return ImColor::HSV(fmodf(hue, 360.0f) / 360.0f, 0.55f, 0.80f);
}

inline ImU32 darken_color(ImU32 col) {
    return (col & 0xFEFEFEFE) >> 1;
}

// ── Timeline state ───────────────────────────────────────────────────
struct TimelineState {
    double view_start = 0.0;
    double view_end   = 1.0;
    float  row_h      = 18.0f;
    int    selected_idx = -1;
    std::string selected_name;

    void reset() { view_start = 0.0; view_end = 1.0; selected_idx = -1; }

    void zoom_in(double cx) {
        double range = view_end - view_start;
        double new_range = range * 0.5;
        double c = view_start + cx * range;
        view_start = c - new_range * cx;
        view_end   = view_start + new_range;
        clamp();
    }
    void zoom_out(double cx) {
        double range = view_end - view_start;
        double new_range = range * 1.5;
        double c = view_start + cx * range;
        view_start = c - new_range * cx;
        view_end   = view_start + new_range;
        clamp();
    }
    void pan(double dx) {
        double range = view_end - view_start;
        view_start -= dx * range;
        view_end   -= dx * range;
        clamp();
    }
    void zoom_to(double s, double e) {
        view_start = s; view_end = e; clamp();
    }
    void clamp() {
        if (view_start < 0.0) { view_end -= view_start; view_start = 0.0; }
        if (view_end   > 1.0) { view_start -= (view_end - 1.0); view_end = 1.0; }
        if (view_start < 0.0) view_start = 0.0;
        if (view_end - view_start < 1e-9) view_end = view_start + 1e-9;
    }
};
