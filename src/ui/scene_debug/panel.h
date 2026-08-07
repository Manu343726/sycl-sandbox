#pragma once

#include <sycl-sandbox/sandbox_api.h>
#include "imgui.h"

/// Toggle flags for the 3D debug scene view.
struct DebugViewFlags {
    bool show_floor = true;
    bool show_grid = true;
    bool show_objects = true;
    bool show_wireframe = true;
    bool show_aabbs = false;
    bool show_frustum = true;
    bool show_camera = true;
    /// Shade scene geometry by what the scene camera can actually see:
    /// objects (or box faces) visible from the camera eye are drawn in
    /// their material colour, occluded ones are darkened.
    bool show_visibility = false;
};

void init_scene_debug();
void shutdown_scene_debug();

/// Render the "Scene Debug" window with a proper 3D OpenGL viewport.
///
/// Draws a reference floor grid, scene objects (spheres, boxes, quads) with
/// their real form and scale, and optional overlays (AABBs, camera frustum).
/// Interactive controls let the user toggle which debug info is shown.
///
/// Must be called within an active ImGui frame.  Returns the ImGui window
/// content area size (for mouse interaction tracking).
///
/// @param dbg        Scene debug info (nullptr if unavailable)
/// @param cam_eye    Camera eye position (3 floats)
/// @param cam_at     Camera look-at target (3 floats)
/// @param cam_up     Camera up vector (3 floats)
/// @param cam_fov    Vertical FOV in degrees
/// @param aspect     Viewport aspect ratio (width / height)
/// @param flags      Toggle flags (persistent across frames)
void render_scene_debug(const SceneDebugInfo *dbg,
                        const float *cam_eye,
                        const float *cam_at,
                        const float *cam_up,
                        float cam_fov,
                        float aspect,
                        DebugViewFlags &flags);
