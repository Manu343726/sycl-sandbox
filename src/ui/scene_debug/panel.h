#pragma once

#include "scene/host_scene.h"
#include "ui/scene_debug/renderer.h"

/// Initialize the scene debug renderer: creates the hidden GLFW window
/// (sharing the main window's GL context) and starts the background
/// render thread.  `share_window` must be the main application window.
/// Safe to call once; the main window's GL context must be current.
void init_scene_debug(GLFWwindow *share_window);

/// Stop the debug render thread and tear down its GL context/window.
/// Blocks until the thread has exited.  Must run before glfwTerminate().
void shutdown_scene_debug();

/// If the debug camera has a pending "Set Scene Camera" request,
/// apply it to the given non-const camera pointers.  Returns true if
/// applied (caller should then publish through the ParamStore).
bool debug_apply_scene_camera(float *cam_eye, float *cam_at, float *cam_up);

/// Render the "Scene Debug" window (ImGui only — the 3D view itself is
/// rendered offscreen by the background thread and displayed here as a
/// texture via ImGui::Image).
///
/// Must be called within an active ImGui frame.  The window owns its
/// own orbit camera, initialized by default at the scene camera's
/// position (and restored there by "Reset View"), after which it is
/// freely orbitable; the scene camera parameters are also used for the
/// frustum / visibility overlays.
///
/// @param host_scene  Host-side scene state (the debug render thread
///                    reads a host copy of it — read-only, never the
///                    device buffers).  May be null/empty for kernels
///                    without a YAML scene.
/// @param cam_eye     Scene camera eye (3 floats, may be null if the
///                    scene has no camera params)
/// @param cam_at      Scene camera look-at target (may be null)
/// @param cam_up      Scene camera up vector (may be null)
/// @param cam_fov     Scene camera vertical FOV in degrees
/// @param cam_aspect  Render-target aspect (width/height) of the
///                    raytraced image — the framebuffer rectangle in
///                    the debug view is drawn at these proportions.
/// @param fb_w, fb_h  Raytraced framebuffer size in pixels (camera-
///                    visibility depth pass + picked-pixel markers).
/// @param render      Render parameters mirrored from the kernel: the
///                    debug ray is traced with the same background /
///                    max_bounces / transparent_backfaces the kernel
///                    uses, and overlay colours are displayed through
///                    the same tone-map operator + gamma as the
///                    framebuffer (see SceneRenderParams).
/// @param flags       Toggle flags (persistent across frames)
void render_scene_debug(const HostScene *host_scene,
                        const float *cam_eye,
                        const float *cam_at,
                        const float *cam_up,
                        float cam_fov,
                        float cam_aspect,
                        int fb_w,
                        int fb_h,
                        const SceneRenderParams &render,
                        DebugViewFlags &flags);

