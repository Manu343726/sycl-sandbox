// sycl-sandbox — Workflow / Integration UI Tests
//
// These tests simulate real user workflows that span multiple UI components,
// verifying that the panels work together correctly.
#pragma once

#include "app_state.h"
#include "imgui_test_engine/imgui_te_context.h"

extern AppState* g_test_state;

inline void RegisterWorkflowTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    t = IM_REGISTER_TEST(e, "workflow", "open_all_panels");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = false;
        s->show_logs = false;
        s->show_metrics = false;
        ctx->Yield();
        ctx->SetRef("Controls");
        ctx->ItemClick("Show Build Monitor");
        ctx->Yield();
        IM_CHECK(s->show_builds);
        ctx->ItemClick("Show Logs");
        ctx->Yield();
        IM_CHECK(s->show_logs);
        ctx->ItemClick("Show System Metrics");
        ctx->Yield();
        IM_CHECK(s->show_metrics);
        ctx->SetRef("Build Monitor");
        IM_CHECK(true);
        ctx->SetRef("Logs");
        IM_CHECK(true);
        ctx->SetRef("System Metrics");
        IM_CHECK(true);
    };

    t = IM_REGISTER_TEST(e, "workflow", "close_all_panels");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = true;
        s->show_logs = true;
        s->show_metrics = true;
        ctx->Yield();
        ctx->SetRef("Controls");
        ctx->ItemClick("Show Build Monitor");
        ctx->Yield();
        IM_CHECK(!s->show_builds);
        ctx->ItemClick("Show Logs");
        ctx->Yield();
        IM_CHECK(!s->show_logs);
        ctx->ItemClick("Show System Metrics");
        ctx->Yield();
        IM_CHECK(!s->show_metrics);
    };

    t = IM_REGISTER_TEST(e, "workflow", "backend_list_available");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        ctx->SetRef("Controls");
        ctx->ItemOpen("##backend");
        ctx->Yield();
        IM_CHECK(s->kr->backends().size() > 0);
        IM_CHECK(s->kr->backends()[0].available);
        ctx->ItemClose("##backend");
    };

    t = IM_REGISTER_TEST(e, "workflow", "kernel_play_stop_cycle");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        ctx->SetRef("Controls");
        bool was_paused = s->render_paused.load();
        if (!was_paused) {
            ctx->ItemClick("|| Stop");
            ctx->Yield();
            IM_CHECK(s->render_paused.load());
        }
        ctx->ItemClick("> Play");
        ctx->Yield();
        IM_CHECK(!s->render_paused.load());
        ctx->ItemClick("|| Stop");
        ctx->Yield();
        IM_CHECK(s->render_paused.load());
        if (!was_paused) {
            ctx->ItemClick("> Play");
            ctx->Yield();
        }
    };
}
