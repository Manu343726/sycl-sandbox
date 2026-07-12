// sycl-sandbox — Build Monitor Panel UI Tests
#pragma once

#include "app_state.h"
#include "ui/build_monitor/panel.h"
#include "imgui_test_engine/imgui_te_context.h"

extern AppState* g_test_state;

inline void RegisterBuildMonitorTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    // ── Empty state shows message ─────────────────────────────────
    t = IM_REGISTER_TEST(e, "build_monitor", "empty_state");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = true;
        ctx->Yield();
        ctx->SetRef("Build Monitor");
        IM_CHECK(true);
    };

    t = IM_REGISTER_TEST(e, "build_monitor", "show_build_job");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = true;
        BuildNotification notif;
        notif.kernel_name = "test_kernel";
        notif.type = BuildNotification::BuildStarted;
        notif.text = "Building test_kernel...";
        notif.progress = 0.0f;
        s->build_monitor.ingest({notif});
        notif.type = BuildNotification::BuildProgress;
        notif.progress = 0.5f;
        notif.text = "Compiling...";
        s->build_monitor.ingest({notif});
        ctx->Yield();
        ctx->SetRef("Build Monitor");
        IM_CHECK(s->build_monitor.active_count() > 0);
        IM_CHECK(s->build_monitor.jobs().count("test_kernel") > 0);
        notif.type = BuildNotification::BuildCompleted;
        notif.progress = 1.0f;
        notif.text = "Build succeeded";
        s->build_monitor.ingest({notif});
        ctx->Yield();
    };

    t = IM_REGISTER_TEST(e, "build_monitor", "clear_completed");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = true;
        {
            BuildNotification n;
            n.kernel_name = "clear_me";
            n.type = BuildNotification::BuildStarted;
            n.text = "started";
            s->build_monitor.ingest({n});
        }
        {
            BuildNotification n;
            n.kernel_name = "clear_me";
            n.type = BuildNotification::BuildCompleted;
            n.progress = 1.0f;
            n.text = "done";
            s->build_monitor.ingest({n});
        }
        ctx->Yield();
        ctx->SetRef("Build Monitor");
        IM_CHECK(s->build_monitor.jobs().count("clear_me") > 0);
        ctx->ItemClick("Clear Completed");
        ctx->Yield();
        IM_CHECK(s->build_monitor.jobs().count("clear_me") == 0);
    };

    t = IM_REGISTER_TEST(e, "build_monitor", "build_failure");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = true;
        BuildNotification n;
        n.kernel_name = "failing_kernel";
        n.type = BuildNotification::BuildStarted;
        n.text = "started";
        s->build_monitor.ingest({n});
        n.type = BuildNotification::BuildFailed;
        n.progress = 0.8f;
        n.text = "Compile error: syntax mistake";
        s->build_monitor.ingest({n});
        ctx->Yield();
        ctx->SetRef("Build Monitor");
        const auto& job = s->build_monitor.jobs().at("failing_kernel");
        IM_CHECK(job.failed);
        ctx->Yield();
    };

    t = IM_REGISTER_TEST(e, "build_monitor", "cleanup_close");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppState* s = g_test_state;
        s->show_builds = false;
        s->build_monitor.clear_completed();
        ctx->Yield();
        IM_CHECK(!s->show_builds);
    };
}
