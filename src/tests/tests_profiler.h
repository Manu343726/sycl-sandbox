// sycl-sandbox — Kernel Profiler Panel UI Tests
#pragma once

#include "app_state.h"
#include "ui/profiler/panel.h"
#include "kernel/profiler_host.h"
#include "imgui_test_engine/imgui_te_context.h"

extern AppState* g_test_state;

inline void RegisterProfilerTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    // ── Profiler shows disabled state when inactive ───────────────
    t = IM_REGISTER_TEST(e, "profiler", "disabled_state");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_profiler = true;
        s->kernel_profiler.set_enabled(false);
        ctx->Yield();
        ctx->SetRef("Kernel Profiler");
        IM_CHECK(true);
    };

    // ── Enable Profiler button works ──────────────────────────────
    t = IM_REGISTER_TEST(e, "profiler", "enable_button");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_profiler = true;
        s->kernel_profiler.set_enabled(false);
        ctx->Yield();
        ctx->SetRef("Kernel Profiler");
        ctx->ItemClick("Enable Profiler");
        ctx->Yield();
        IM_CHECK(s->kernel_profiler.is_enabled());
    };

    // ── Profiler controls appear when enabled ─────────────────────
    t = IM_REGISTER_TEST(e, "profiler", "controls_visible_when_enabled");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_profiler = true;
        s->kernel_profiler.set_enabled(true);
        s->kernel_profiler.set_paused(false);
        ctx->Yield();
        ctx->SetRef("Kernel Profiler");
        ctx->ItemClick("|| Pause");
        ctx->Yield();
        ctx->ItemClick("> Resume");
        ctx->Yield();
    };

    // ── Clear data button ─────────────────────────────────────────
    t = IM_REGISTER_TEST(e, "profiler", "clear_data");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_profiler = true;
        s->kernel_profiler.set_enabled(true);
        s->kernel_profiler.set_paused(false);
        ctx->Yield();
        ctx->SetRef("Kernel Profiler");
        ctx->ItemClick("X Clear");
        ctx->Yield();
        IM_CHECK(true);
    };

    // ── Disable profiler via checkbox ─────────────────────────────
    t = IM_REGISTER_TEST(e, "profiler", "disable_via_checkbox");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_profiler = true;
        s->kernel_profiler.set_enabled(true);
        ctx->Yield();
        ctx->SetRef("Kernel Profiler");
        ctx->ItemUncheck("Enabled");
        ctx->Yield();
        IM_CHECK(!s->kernel_profiler.is_enabled());
        ctx->ItemCheck("Enabled");
        ctx->Yield();
        IM_CHECK(s->kernel_profiler.is_enabled());
    };
}
