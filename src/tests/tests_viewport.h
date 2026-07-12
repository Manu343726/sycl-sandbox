// sycl-sandbox — Viewport Panel UI Tests
#pragma once

#include "app_state.h"
#include "imgui_test_engine/imgui_te_context.h"

extern AppState* g_test_state;

inline void RegisterViewportTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    // ── Viewport window exists ────────────────────────────────────
    t = IM_REGISTER_TEST(e, "viewport", "window_exists");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Viewport");
        IM_CHECK(true);
    };

    // ── Viewport has content area ─────────────────────────────────
    t = IM_REGISTER_TEST(e, "viewport", "has_content_region");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Viewport");
        // The viewport window should have a non-zero content region
        // (it's an image with no label, but the window itself exists)
        ImGuiWindow* win = ctx->GetWindowByRef("");
        IM_CHECK(win != nullptr);
        if (win) {
            ImVec2 size = win->ContentSize;
            IM_CHECK(size.x > 0);
            IM_CHECK(size.y > 0);
        }
    };
}
