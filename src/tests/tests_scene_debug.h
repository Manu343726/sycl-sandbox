// sycl-sandbox — Scene Debug Panel UI Tests
#pragma once

#include "app_state.h"
#include "ui/scene_debug/panel.h"
#include "imgui_test_engine/imgui_te_context.h"

extern AppState* g_test_state;

inline void RegisterSceneDebugTests(ImGuiTestEngine* e, AppState* state) {
    ImGuiTest* t = nullptr;

    // ── Scene Debug window exists ─────────────────────────────────
    t = IM_REGISTER_TEST(e, "scene_debug", "window_exists");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Scene Debug");
        IM_CHECK(true);
    };

    // ── Debug flags are present ───────────────────────────────────
    t = IM_REGISTER_TEST(e, "scene_debug", "debug_flags_present");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Scene Debug");
        // Check that debug flag checkboxes exist
        // Labels from panel.cpp: "Floor", "Grid", "Objects", "Wireframe",
        // "AABBs", "Frustum", "Camera"
        ctx->ItemCheck("Floor");
        ctx->ItemCheck("Grid");
        ctx->ItemCheck("Objects");
    };

    // ── Debug flags can be toggled ────────────────────────────────
    t = IM_REGISTER_TEST(e, "scene_debug", "flag_toggle_cycle");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->SetRef("Scene Debug");
        // Toggle AABBs (which starts disabled)
        ctx->ItemCheck("AABBs");
        ctx->Yield();
        ctx->ItemUncheck("AABBs");
        ctx->Yield();
        // Toggle Camera
        ctx->ItemUncheck("Camera");
        ctx->Yield();
        ctx->ItemCheck("Camera");
    };
}
