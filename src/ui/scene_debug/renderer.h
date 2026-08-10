#pragma once

// ── SceneDebugRenderer ────────────────────────────────────────────────
/// Offscreen 3D scene debug renderer, hosted on a background thread.
///
/// The "Scene Debug" window is composed by the UI from a plain OpenGL
/// texture (ImGui::Image).  That texture is produced here, on a dedicated
/// background thread that owns its own GL context (a hidden GLFW window
/// sharing the main window's context) and runs a render + swap loop:
///
///   ┌──────────────┐   shared state    ┌──────────────────────────┐
///   │  UI thread   │  (mutex + atomics)│  debug render thread     │
///   │  ImGui panel │ ─────────────────▶│  orbit camera, flags,    │
///   │  camera input│                   │  scene snapshot          │
///   └──────────────┘                   │  ┌─────────────────────┐ │
///         │  ImGui::Image(texture)     │  │ GL 3.3 core render  │ │
///         │  ◀─────────── latest ready │  │ (FBO -> slot tex)   │ │
///         └────────────────────────────│  └─────────────────────┘ │
///                                      └──────────────────────────┘
///
/// Triple-buffered slot hand-off (same pattern as DisplayTarget):
///   - render thread renders into a FREE slot, then glFinish + marks it
///     READY (release/acquire via std::atomic)
///   - UI thread presents the newest READY slot (marks it PRESENTING),
///     ImGui samples it during ImGui::Render later in the same frame,
///     and the slot is released back to FREE at the start of the next
///     present() call — the render thread never overwrites a texture
///     the UI may still be sampling.
///
/// The render thread has READ-ONLY access to a host copy of the loaded
/// scene (HostScene::debug_scene — a SceneDebugScene shared_ptr).  The
/// full scene lives on the host (hittables, materials, handles, AABBs,
/// BVH nodes, lights), so the renderer can trace rays against it (BVH
/// traversal, visibility tests, hit points, portal recursion — the whole
/// rt runtime is available, it is all host-compilable header code).
/// The shared_ptr hand-off guarantees the render thread can never read
/// freed memory: a scene rebuild swaps in a new instance and the old
/// one is freed only when the last reference (renderer included) drops.
///
/// Camera: orbit camera (yaw/pitch/distance/target).  UI input (drag /
/// wheel / pan deltas) is applied to the shared state by the UI thread;
/// the render thread reads it each frame.  By default (first scene and
/// "Reset View") the orbit camera is placed at the scene camera's exact
/// position and look-at point (the scene camera eye/at/up/fov is pushed
/// every UI frame); it then stays put and never follows the scene
/// camera.  Scenes without a camera fall back to framing the scene
/// bounds (`autoframe`).  The scene camera data is also used for the
/// frustum / visibility overlays.

#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <sycl-sandbox/tonemap_ops.h>

#include "scene/host_scene.h"
#include "ui/scene_debug/ray_trace.h"

/// Render parameters the scene-debug view mirrors from the raytracer:
/// the debug ray is traced with the SAME parameters the kernel uses
/// (bounce depth, backface transparency, background colour), and the
/// overlay colours are displayed with the SAME tone-map operator +
/// gamma as the framebuffer — so the traced ray matches the rendered
/// image, not just the linear radiance values.
struct SceneRenderParams {
    float background[3] = {0, 0, 0};
    int max_bounces = 10;
    bool transparent_backfaces = false;
    bool tonemap_enabled = false;
    int tonemap_operator = tonemap::Operator::Reinhard;
    float tonemap_exposure = 1.f;
    float tonemap_gamma = 2.2f;
};

/// Display colour of a linear radiance, exactly as the framebuffer
/// shows it (mirrors tonemap::map_pixel: exposure → operator → clamp →
/// gamma; hard clamp when tone-mapping is disabled).
inline rt::float3 scene_debug_display_color(rt::float3 c,
                                            const SceneRenderParams &p) {
    auto map = [&p](float v) {
        float x = v * p.tonemap_exposure;
        if ( !p.tonemap_enabled ) {
            return std::clamp(x, 0.f, 1.f);
        }
        x = tonemap::apply_operator(x, p.tonemap_operator);
        x = std::clamp(x, 0.f, 1.f);
        return std::pow(x, 1.f / p.tonemap_gamma);
    };
    return rt::float3{map(c.x), map(c.y), map(c.z)};
}

/// Human-readable names for rt handle/material tags (shared by the
/// render thread's trace summary and the UI hit inspector).
inline const char *hittable_name(rt::HittableType t) {
    switch ( t ) {
        case rt::HittableType::Sphere: return "sphere";
        case rt::HittableType::Triangle: return "triangle";
        case rt::HittableType::Quad: return "quad";
        case rt::HittableType::Box: return "box";
        case rt::HittableType::Portal: return "portal";
        case rt::HittableType::Mesh: return "mesh";
    }
    return "?";
}

inline const char *material_name(rt::MaterialType t) {
    switch ( t ) {
        case rt::MaterialType::Lambertian: return "lambertian";
        case rt::MaterialType::Metal: return "metal";
        case rt::MaterialType::Dielectric: return "dielectric";
        case rt::MaterialType::DiffuseLight: return "emissive";
        case rt::MaterialType::TexturedLambertian: return "textured";
    }
    return "?";
}

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
    /// Draw the scene's BVH node AABBs as a wireframe tree (depth
    /// gradient; nodes entered by the traced ray are highlighted).
    bool show_bvh = false;
    /// BVH tree depth to draw (1 = root only).
    int bvh_depth = 4;
    /// Trace a single ray through the scene (segments, hit points,
    /// BVH hits) and show it in the view.
    bool show_ray = false;
    /// Ray trace recursion depth (number of bounces).
    int ray_bounces = 4;
};

class SceneDebugRenderer {
public:
    SceneDebugRenderer() = default;
    ~SceneDebugRenderer() = default;
    SceneDebugRenderer(const SceneDebugRenderer &) = delete;
    SceneDebugRenderer &operator=(const SceneDebugRenderer &) = delete;

    // ── Main-thread lifecycle ─────────────────────────────────────
    /// Create the hidden GLFW window (sharing `share_window`'s context)
    /// and start the background render thread.  `share_window` must be
    /// the main application window; its context must be current on the
    /// calling thread (it is released during creation and restored).
    /// Returns false (with a logged error) if context creation fails.
    bool init(GLFWwindow *share_window);

    /// Stop the render thread and tear down the context/window.
    /// Blocks until the thread has exited.  Main thread only.
    void shutdown();

    // ── Shared state (UI thread writes, render thread reads) ──────
    struct State {
        std::mutex mutex;

        // Scene snapshot (shared_ptr hand-off — see header docs)
        std::shared_ptr<SceneDebugScene> scene;

        // Scene camera (for the frustum overlay + visibility tests)
        bool scene_cam_valid = false;
        float scene_cam_eye[3] = {0, 0, 0};
        float scene_cam_at[3] = {0, 0, 0};
        float scene_cam_up[3] = {0, 1, 0};
        float scene_cam_fov = 45.f;
        /// Render-target aspect (width/height) of the raytraced image —
        /// used to place the scene framebuffer rectangle at the right
        /// proportions in world space.
        float scene_cam_aspect = 1.f;

        // Requested render resolution (ImGui window content region)
        int size_w = 512;
        int size_h = 512;

        DebugViewFlags flags{};

        // Orbit camera — UI thread applies input deltas, render thread
        // reads (and may reset via autoframe on scene change).
        float orbit_yaw = -45.f;
        float orbit_pitch = 30.f;   // rad convention matches the render math
        float orbit_dist = 12.f;
        float orbit_eye[3] = {0, 0, 0};
        bool camera_user_interacted = false;
        /// UI requests an autoframe (Reset View button).
        bool reset_view = false;
        /// UI requests setting the scene camera to the debug camera.
        bool set_scene_camera = false;
        float set_scene_eye[3] = {};
        float set_scene_at[3] = {};
        float set_scene_up[3] = {};

        /// False while the ImGui window is collapsed/closed — the render
        /// thread skips frames (and the slots stay untouched).
        bool window_open = true;

        // ── Ray trace overlay (UI thread picks, render thread traces) ──
        /// UI clicked in the view: pick a ray from the debug orbit camera
        /// through the clicked pixel (NDC coords, y up).
        bool pick_ray_request = false;
        float pick_ndc_x = 0.f;
        float pick_ndc_y = 0.f;
        /// A picked ray is active (render thread owns these once set);
        /// when false the default ray (scene camera center, or orbit
        /// camera through its target) is traced instead.
        bool ray_picked = false;
        float ray_origin[3] = {0, 0, 0};
        float ray_dir[3] = {0, 0, -1};
        /// UI asks to drop the picked ray and go back to the default.
        bool reset_ray = false;
        /// Human-readable trace summary, written by the render thread
        /// every frame while the ray overlay is on (mutex-protected).
        std::string ray_trace_text;
        /// Full ray trace (one step per bounce), written by the render
        /// thread every frame while the ray overlay is on — the UI
        /// reads it for the hits treeview + details inspector.
        rt::RayTraceResult ray_trace;
        /// The picked ray was selected ON the scene-camera framebuffer
        /// rectangle (a framebuffer pixel pick); `picked_rect_u/v` is
        /// the picked pixel in kernel (u,v) convention (v=0 = the
        /// rectangle's lower-left corner, matching the raytracer).
        bool ray_from_rect = false;
        float picked_rect_u = 0.f;
        float picked_rect_v = 0.f;
        /// Raytraced framebuffer size in pixels (camera-visibility
        /// depth pass resolution + pixel-size pick markers).
        int fb_w = 0;
        int fb_h = 0;
        /// Render parameters mirrored from the kernel (background,
        /// max_bounces, transparent_backfaces, tone-map stage): the
        /// debug trace uses them, and overlay colours are displayed
        /// through the same tone-map as the framebuffer.
        SceneRenderParams render;
    };
    State &state() { return state_; }

    // ── Presentation (main thread, once per UI frame) ─────────────
    /// Release the previously presented slot, then return the texture id
    /// of the newest READY frame.  When no new frame is ready (the UI
    /// outran the render thread) the previously presented texture is
    /// returned again — a presented slot is never overwritten until the
    /// next successful present(), so it is always safe to sample with
    /// ImGui::Image.  Returns 0 only when the renderer is inactive.
    GLuint present();

    /// Texture id of a slot (valid only while the slot is READY/PRESENTING).
    GLuint slot_texture(int slot) const { return slots_[slot].tex; }

private:
    struct Slot {
        GLuint fbo = 0;
        GLuint tex = 0;
        /// Depth renderbuffer — FBOs have NO default depth attachment,
        /// without this the depth test is a no-op and later draws
        /// (grid/wireframe lines) paint over earlier geometry.
        GLuint depth = 0;
        /// 0 = free, 1 = ready, 2 = presenting (UI).
        std::atomic<int> state{0};
        uint64_t generation = 0;
    };

    void thread_main();
    bool gl_init();
    void gl_shutdown();
    void resize_slots(int w, int h);
    /// (Re)size the camera-visibility depth pass buffers to the
    /// framebuffer's pixel resolution (no-op when unchanged).
    void resize_cam_fbo(int w, int h);

    // ── Render-thread members ──────────────────────────────────────
    GLFWwindow *window_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    Slot slots_[3];
    int num_slots_ = 3;
    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    // Camera-visibility pass: the scene rendered from the scene
    // camera's POV into an offscreen depth buffer — the z-test result
    // (the "stencil" of what the framebuffer sees) — then the orbit
    // view's solid pass shades every fragment against it, per-pixel.
    GLuint cam_fbo_ = 0;
    GLuint cam_color_tex_ = 0;
    GLuint cam_depth_tex_ = 0;
    int cam_w_ = 0;
    int cam_h_ = 0;
    GLuint vis_program_ = 0;
    GLint vis_u_mvp_ = -1;
    GLint vis_u_cam_vp_ = -1;
    GLint vis_u_depth_ = -1;
    GLint vis_u_eps_ = -1;
    int size_w_ = 0;            ///< current render resolution
    int size_h_ = 0;
    uint64_t generation_ = 0;   ///< incremented per published frame
    uint64_t last_scene_version_ = 0;
    bool camera_framed_ = false;

    // ── Main-thread members ────────────────────────────────────────
    int presented_slot_ = -1;

    State state_;
};
