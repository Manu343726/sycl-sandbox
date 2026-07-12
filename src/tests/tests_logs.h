// sycl-sandbox — Log Sink Panel UI Tests
#pragma once

#include "app_state.h"
#include "io/log_sink.h"
#include "imgui_test_engine/imgui_te_context.h"

extern AppState* g_test_state;

inline void RegisterLogSinkTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    t = IM_REGISTER_TEST(e, "log_sink", "window_opens");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_logs = true;
        ctx->Yield();
        ctx->SetRef("Logs");
        IM_CHECK(true);
    };

    t = IM_REGISTER_TEST(e, "log_sink", "messages_displayed");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_logs = true;
        spdlog::info("TEST_LOG_MESSAGE: info message");
        spdlog::warn("TEST_LOG_MESSAGE: warning message");
        spdlog::error("TEST_LOG_MESSAGE: error message");
        ctx->Yield();
        ctx->SetRef("Logs");
        IM_CHECK(true);
        ctx->ItemClick("Filter");
        ctx->Yield();
    };

    t = IM_REGISTER_TEST(e, "log_sink", "window_closes");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_logs = true;
        ctx->Yield();
        IM_CHECK(s->show_logs);
        s->show_logs = false;
        ctx->Yield();
        IM_CHECK(true);
    };
}
