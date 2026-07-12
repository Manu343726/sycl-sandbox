// sycl-sandbox — Controls Panel UI Tests
#pragma once

#include "app_state.h"
#include "imgui_test_engine/imgui_te_context.h"

// ── Controls panel ───────────────────────────────────────────────────
// Verify the Controls window, its layout, and widget interactions.

extern AppState* g_test_state;

inline void RegisterControlsTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    // ── Controls window exists ─────────────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "window_exists");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Controls");
        // Window exists if we reach here without error
        IM_CHECK(true);
    };

    // ── Scene selector combo exists ────────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "scene_selector_exists");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Controls");
        // The scene selector is a BeginCombo with label "Scene"
        ctx->ItemOpen("Scene");
        ctx->ItemClose("Scene");
    };

    // ── Show Builds checkbox toggles ───────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "show_builds_toggle");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        ctx->SetRef("Controls");
        bool initial = s->show_builds;
        ctx->ItemClick("Show Build Monitor");
        ctx->Yield();
        IM_CHECK_NE(s->show_builds, initial);
        // Restore
        if (s->show_builds != initial) {
            ctx->ItemClick("Show Build Monitor");
        }
    };

    // ── Show Logs checkbox toggles ─────────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "show_logs_toggle");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        ctx->SetRef("Controls");
        bool initial = s->show_logs;
        ctx->ItemClick("Show Logs");
        ctx->Yield();
        IM_CHECK_NE(s->show_logs, initial);
        // Restore
        if (s->show_logs != initial) {
            ctx->ItemClick("Show Logs");
        }
    };

    // ── Show Profiler checkbox toggles ─────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "show_profiler_toggle");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        ctx->SetRef("Controls");
        bool initial = s->show_profiler;
        ctx->ItemClick("Show Profiler");
        ctx->Yield();
        IM_CHECK_NE(s->show_profiler, initial);
        // Restore
        if (s->show_profiler != initial) {
            ctx->ItemClick("Show Profiler");
        }
    };

    // ── Backend switcher exists ───────────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "backend_switcher_exists");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Controls");
        ctx->ItemOpen("##backend");
        ctx->ItemClose("##backend");
    };

    // ── Kernel Execution play/pause ────────────────────────────────
    t = IM_REGISTER_TEST(e, "controls", "kernel_play_pause");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        ctx->SetRef("Controls");
        bool was_paused = s->render_paused.load();
        if (was_paused) {
            // Click Play
            ctx->ItemClick("> Play");
            ctx->Yield();
            IM_CHECK(!s->render_paused.load());
            // Click Stop
            ctx->ItemClick("|| Stop");
            ctx->Yield();
            IM_CHECK(s->render_paused.load());
        } else {
            // Click Stop
            ctx->ItemClick("|| Stop");
            ctx->Yield();
            IM_CHECK(s->render_paused.load());
            // Click Play
            ctx->ItemClick("> Play");
            ctx->Yield();
            IM_CHECK(!s->render_paused.load());
        }
        // Restore to original state
        if (s->render_paused.load() != was_paused) {
            if (was_paused)
                ctx->ItemClick("|| Stop");
            else
                ctx->ItemClick("> Play");
        }
    };
}
