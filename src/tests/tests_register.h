// sycl-sandbox — Unified test registration
// Included from main.cpp to register all UI tests with the test engine.
#pragma once

#include "imgui_test_engine/imgui_te_engine.h"

// Each component test registration is in its own header.
// All use `g_test_state` (defined in main.cpp) to access AppState.
#include "tests_controls.h"
#include "tests_viewport.h"
#include "tests_scene_debug.h"
#include "tests_build_monitor.h"
#include "tests_logs.h"
#include "tests_workflow.h"

/// Register all sandbox UI tests with the given test engine instance.
inline void RegisterSandboxTests(ImGuiTestEngine* engine) {
    RegisterControlsTests(engine, g_test_state);
    RegisterViewportTests(engine, g_test_state);
    RegisterSceneDebugTests(engine, g_test_state);
    RegisterBuildMonitorTests(engine, g_test_state);
    RegisterLogSinkTests(engine, g_test_state);
    RegisterWorkflowTests(engine, g_test_state);
}
