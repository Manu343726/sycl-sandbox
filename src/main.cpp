#include <sycl-sandbox/profiling.h>
#include <sycl-sandbox/sandbox_api.h>
#include "watcher.h"
#include "kernel_library.h"
#include "scene_registry.h"
#include "param_ui.h"
#include <sycl-sandbox/rt/params.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <args.hxx>

#include <sycl/sycl.hpp>
#include <yaml-cpp/yaml.h>

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

// ── configuration ──────────────────────────────────────────────────────
static int WIDTH = 1280;
static int HEIGHT = 720;

static void recreate_render_buffers(sycl::queue &q,
                                    GLuint &tex,
                                    float *&d_accum,
                                    float *&h_accum,
                                    size_t &pixel_count,
                                    int w,
                                    int h);

// ── forward declarations ───────────────────────────────────────────────
static void
call_init_kernel(void *handle, sycl::queue &q, int w, int h, const void *params, size_t sz);
static void call_render_kernel(void *handle,
                               sycl::queue &q,
                               int w,
                               int h,
                               const void *params,
                               void *accum,
                               int sample);
static float *find_param(float *params, const KernelDesc &desc, const char *name);
static void init_std_params(float *buf, size_t buf_size);
static void apply_yaml_std_params(const SceneDef &scene, float *buf, size_t buf_size);

// ── GLFW error callback ────────────────────────────────────────────────
static void glfw_error_cb(int error, const char *desc) {
    spdlog::error("GLFW {}: {}", error, desc);
}

// ── orbit camera (for 3D scenes) ──────────────────────────────────────
// Spherical coordinate convention:
//   theta = azimuth (horizontal angle around the Y axis)
//   phi   = elevation (vertical angle from the XZ plane)
//   dist  = distance from the camera to the look-at target
//   roll  = rotation of the up vector around the forward (lookat) axis
struct OrbitCam {
    float theta = 1.35f;
    float phi = 1.05f;
    float dist = 12.f;
    float roll = 0.f;
    float target[3] = {0.f, 0.f, 0.f};
};

/// Convert spherical orbit coordinates to a Cartesian eye position.
///
///   eye = target + dist · (cos(phi)·sin(theta),  sin(phi),  cos(phi)·cos(theta))
///
/// This places the camera on a sphere of radius `dist` centred on the target,
/// with `theta` as azimuth and `phi` as elevation.
static void orbit_to_eye(const OrbitCam &orbit, float eye[3]) {
    float cos_theta = cosf(orbit.theta), sin_theta = sinf(orbit.theta);
    float cos_phi = cosf(orbit.phi), sin_phi = sinf(orbit.phi);
    eye[0] = orbit.target[0] + orbit.dist * cos_phi * sin_theta;
    eye[1] = orbit.target[1] + orbit.dist * sin_phi;
    eye[2] = orbit.target[2] + orbit.dist * cos_phi * cos_theta;
}

/// Rotate the default up vector {0,1,0} around the forward (lookat) axis
/// by the camera's roll angle, using Rodrigues' rotation formula.
///
///   up_rotated = up · cos(roll) + (forward × up) · sin(roll)
///                + forward · (forward · up) · (1 − cos(roll))
///
/// The forward direction points from the camera toward the target
/// (in the orbit frame: forward = -lookat_direction_in_camera_frame).
static void orbit_up(const OrbitCam &orbit, float up[3]) {
    float cos_theta = cosf(orbit.theta), sin_theta = sinf(orbit.theta);
    float cos_phi = cosf(orbit.phi), sin_phi = sinf(orbit.phi);
    float forward[3] = {-cos_phi * sin_theta, -sin_phi, -cos_phi * cos_theta};
    float default_up[3] = {0.f, 1.f, 0.f};
    float cos_roll = cosf(orbit.roll), sin_roll = sinf(orbit.roll);
    float dot_product =
        forward[0] * default_up[0] + forward[1] * default_up[1] + forward[2] * default_up[2];
    // Rodrigues' rotation:  up_rotated = up * cos(r) + (forward × up) * sin(r) + forward *
    // (forward·up) * (1-cos(r))
    up[0] = default_up[0] * cos_roll +
            (forward[1] * default_up[2] - forward[2] * default_up[1]) * sin_roll +
            forward[0] * dot_product * (1 - cos_roll);
    up[1] = default_up[1] * cos_roll +
            (forward[2] * default_up[0] - forward[0] * default_up[2]) * sin_roll +
            forward[1] * dot_product * (1 - cos_roll);
    up[2] = default_up[2] * cos_roll +
            (forward[0] * default_up[1] - forward[1] * default_up[0]) * sin_roll +
            forward[2] * dot_product * (1 - cos_roll);
}

// ── OpenGL texture helpers ─────────────────────────────────────────────
static GLuint create_render_texture(int w, int h) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// ── main ───────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    // ---- CLI args ----
    args::ArgumentParser parser("sycl-sandbox");
    args::ValueFlag<std::string> backend_arg(parser, "cpu|gpu", "SYCL backend", {'b', "backend"});
    args::ValueFlag<std::string> log_level_arg(parser, "trace|debug|info|warn|error",
                                               "spdlog log level", {'l', "log-level"});
    try {
        parser.ParseCLI(argc, argv);
    } catch ( const args::Help & ) {
        std::cerr << parser;
        return 0;
    } catch ( const args::ParseError &e ) {
        spdlog::error("{}", e.what());
        std::cerr << parser;
        return 1;
    }

    std::string backend = backend_arg ? backend_arg.Get() : "gpu";
    if ( backend != "cpu" && backend != "gpu" ) {
        spdlog::error("backend must be 'cpu' or 'gpu', got '{}'", backend);
        return 1;
    }

    std::string log_level = log_level_arg ? log_level_arg.Get() : "info";
    auto sl = spdlog::level::from_str(log_level);
    if ( sl == spdlog::level::off && log_level != "off" ) {
        spdlog::error("invalid log level '{}'", log_level);
        return 1;
    }
    spdlog::set_level(sl);
    spdlog::info("log level set to {}", log_level);

    // ---- GLFW window ----
    glfwSetErrorCallback(glfw_error_cb);
    if ( !glfwInit() ) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "sycl-sandbox", nullptr, nullptr);
    if ( !window ) {
        spdlog::error("glfwCreateWindow failed");
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Use actual framebuffer size
    glfwGetFramebufferSize(window, &WIDTH, &HEIGHT);

    // ---- ImGui ----
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ---- SYCL queue ----
    spdlog::info("[startup] creating queue (backend={})", backend);
    sycl::queue q = sycl::queue(sycl::default_selector_v);
    if ( backend == "gpu" ) {
        try {
            sycl::queue gq(sycl::gpu_selector_v);
            q = std::move(gq);
        } catch ( const std::exception &e ) {
            spdlog::warn("[startup] gpu selector failed ({}), keeping default", e.what());
        }
    }
    spdlog::info("[startup] device: {}",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- Render buffers ----
    spdlog::debug("[startup] creating render buffers...");
    GLuint tex = 0;
    size_t pixel_count = 0;
    float *d_accum = nullptr;
    float *h_accum = nullptr;
    recreate_render_buffers(q, tex, d_accum, h_accum, pixel_count, WIDTH, HEIGHT);

    // ---- Scene system ----
    spdlog::debug("[startup] loading scenes...");
    SceneRegistry scenes("scenes");
    KernelLibrary kernels("build");
    SourceWatcher watcher;
    spdlog::info("[startup] {} scenes loaded", scenes.all().size());

    for ( auto &s : scenes.all() ) {
        spdlog::debug("[startup] watch kernel: {}", s.kernel);
        watcher.watch_kernel(s.kernel, std::string("kernels/") + s.kernel);
    }

    // ---- Active state ----
    const SceneDef *active_scene = nullptr;
    KernelHandle *active_kernel = nullptr;
    float *h_params = nullptr;
    float *d_params = nullptr;
    int current_spp = 0;
    int target_spp = 1;

    // Clear accumulator to black
    q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();

    // Auto-load first scene
    spdlog::info("[startup] scenes loaded: {}", scenes.all().size());
    if ( !scenes.all().empty() ) {
        active_scene = &scenes.all().front();
        spdlog::info("[startup] first scene: name='{}' kernel='{}' yaml='{}'",
                     active_scene->name, active_scene->kernel, active_scene->yaml_path);
        spdlog::info("[startup] loading kernel: {}", active_scene->kernel);
        active_kernel = kernels.load(active_scene->kernel);
        spdlog::info("[startup] kernel loaded, target_spp={}, sz={}",
                     active_kernel ? active_kernel->desc.max_spp : 0,
                     active_kernel ? active_kernel->desc.params_buffer_size : 0);
        if ( active_kernel ) {
            spdlog::info("[startup] kernel '{}' has {} params, params_buffer={}",
                         active_kernel->desc.name,
                         active_kernel->desc.param_count,
                         active_kernel->desc.params_buffer_size);
            target_spp = active_kernel->desc.max_spp;
            auto sz = active_kernel->desc.params_buffer_size;
            h_params = (float *)calloc(sz, 1);
            spdlog::info("[startup] apply_params...");
            init_std_params(h_params, sz);
            apply_yaml_std_params(*active_scene, h_params, sz);
            scenes.apply_params(*active_scene, active_kernel->desc, h_params, sz);
            spdlog::info("[startup] alloc d_params...");
            d_params = sycl::malloc_host<float>(sz / 4, q);
            spdlog::info("[startup] upload params...");
            q.memcpy(d_params, h_params, sz).wait();
            spdlog::info("[startup] init_kernel...");
            call_init_kernel(active_kernel->handle, q, WIDTH, HEIGHT, d_params, sz);
            spdlog::info("[startup] init_kernel done");
        }
    }

    // ---- Main loop ----
    while ( !glfwWindowShouldClose(window) ) {
        glfwPollEvents();

        // 0. Check for window resize
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        if ( fb_w != WIDTH || fb_h != HEIGHT ) {
            spdlog::info("[resize] {} x {} -> {} x {}", WIDTH, HEIGHT, fb_w, fb_h);
            WIDTH = fb_w;
            HEIGHT = fb_h;
            recreate_render_buffers(q, tex, d_accum, h_accum, pixel_count, WIDTH, HEIGHT);
            q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
            current_spp = 0;
            if ( active_kernel && d_params ) {
                call_init_kernel(active_kernel->handle,
                                 q,
                                 WIDTH,
                                 HEIGHT,
                                 d_params,
                                 active_kernel->desc.params_buffer_size);
            }
        }

        // 1. Check source changes → rebuild + reload
        for ( auto &dirty_name : watcher.poll() ) {
            spdlog::info("[watcher] {} changed, rebuilding...", dirty_name);
            if ( kernels.rebuild(dirty_name) ) {
                auto *new_kh = kernels.load(dirty_name);
                if ( new_kh && active_scene && active_kernel &&
                     active_kernel->name == dirty_name ) {
                    free(h_params);
                    auto sz = new_kh->desc.params_buffer_size;
                    h_params = (float *)calloc(sz, 1);
                    init_std_params(h_params, sz);
                    apply_yaml_std_params(*active_scene, h_params, sz);
                    scenes.apply_params(*active_scene, new_kh->desc, h_params, sz);
                    if ( d_params ) {
                        sycl::free(d_params, q);
                    }
                    d_params = sycl::malloc_host<float>(sz / 4, q);
                    q.memcpy(d_params, h_params, sz).wait();
                    call_init_kernel(new_kh->handle, q, WIDTH, HEIGHT, d_params, sz);
                    q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                    active_kernel = new_kh;
                    current_spp = 0;
                }
            }
        }

        // 2. ImGui frame
        PROF_ZONE_SCOPED_N("ImGui frame");
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 3. Dockspace (full-window, always-on background)
        {
            ImGuiViewport *viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
            ImGui::Begin("##dockspace",
                         nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
                             ImGuiWindowFlags_NoMouseInputs);
            ImGui::PopStyleVar(3);
            ImGui::DockSpace(ImGui::GetID("dockspace"), ImVec2(0.f, 0.f),
                             ImGuiDockNodeFlags_PassthruCentralNode);
            ImGui::End();
        }

        // 4. Render full-window background image (behind all docks)
        {
            auto disp = ImGui::GetIO().DisplaySize;
            ImGui::GetBackgroundDrawList()->AddImage((ImTextureID)(intptr_t)tex,
                                                     ImVec2(0, 0),
                                                     disp,
                                                     ImVec2(0, 1),
                                                     ImVec2(1, 0));
        }

        // ---- Scene / kernel selector ----
        ImGui::Begin("Controls");
        std::string cur = active_scene ? active_scene->name : "\u2014";
        if ( ImGui::BeginCombo("Scene", cur.c_str()) ) {
            for ( auto &s : scenes.all() ) {
                bool sel = active_scene && active_scene->name == s.name;
                if ( ImGui::Selectable(s.name.c_str(), sel) ) {
                    if ( !sel ) {
                        active_scene = &s;
                        active_kernel = kernels.load(s.kernel);
                        if ( active_kernel ) {
                            free(h_params);
                            auto sz = active_kernel->desc.params_buffer_size;
                            h_params = (float *)calloc(sz, 1);
                            init_std_params(h_params, sz);
                            apply_yaml_std_params(s, h_params, sz);
                            scenes.apply_params(s, active_kernel->desc, h_params, sz);
                            if ( d_params ) {
                                sycl::free(d_params, q);
                            }
                            d_params = sycl::malloc_host<float>(sz / 4, q);
                            q.memcpy(d_params, h_params, sz).wait();
                            call_init_kernel(active_kernel->handle, q, WIDTH, HEIGHT, d_params, sz);
                            q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                            current_spp = 0;
                        }
                    }
                }
            }
            ImGui::EndCombo();
        }

        // ---- param controls ----
        if ( active_kernel && h_params ) {
            ImGui::SeparatorText("Parameters");
            if ( render_param_controls(active_kernel->desc, h_params, false) ) {
                spdlog::info("[param] param changed, re-init kernel and reset accum");
                q.memcpy(d_params, h_params, active_kernel->desc.params_buffer_size).wait();
                try {
                    call_init_kernel(active_kernel->handle,
                                     q,
                                     WIDTH,
                                     HEIGHT,
                                     d_params,
                                     active_kernel->desc.params_buffer_size);
                } catch ( const std::exception &e ) {
                    spdlog::error("[sycl] init error: {}", e.what());
                }
                q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                current_spp = 0;
            }
        }

        // ---- stats ----
        if ( active_kernel ) {
            ImGui::SeparatorText("Stats");
            ImGui::Text("%s  (gen %d)", active_kernel->desc.name, active_kernel->generation);
            ImGui::Text("SPP %d / %d", current_spp, target_spp);
            ImGui::Text("%d x %d", WIDTH, HEIGHT);
            int kernel_max = active_kernel->desc.max_spp;
            if ( ImGui::SliderInt("Target SPP", &target_spp, 1, kernel_max) ) {
                current_spp = std::min(current_spp, target_spp);
            }
            if ( ImGui::Button("Reset") ) {
                q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                current_spp = 0;
            }
        }

        // ── camera info & controls ──────────────────────────────────
        bool has_std_params =
            active_kernel && h_params &&
            active_kernel->desc.params_buffer_size >= RT_NUM_STD_PARAMS * sizeof(float);
        float *camera_eye_ptr = has_std_params ? h_params + RT_CAM_EYE : nullptr;
        float *camera_at_ptr = has_std_params ? h_params + RT_CAM_AT : nullptr;
        float *fov_ptr = has_std_params ? h_params + RT_CAM_FOV : nullptr;
        float *aperture_ptr = has_std_params ? h_params + RT_CAM_APERTURE : nullptr;
        float *camera_up_ptr = has_std_params ? h_params + RT_CAM_UP : nullptr;
        bool has_2d = false;

        if ( active_kernel && h_params ) {
            ImGui::SeparatorText("Camera");
            float *center_x_ptr = find_param(h_params, active_kernel->desc, "center_x");
            float *center_y_ptr = find_param(h_params, active_kernel->desc, "center_y");
            float *zoom_ptr = find_param(h_params, active_kernel->desc, "zoom");

            if ( center_x_ptr && center_y_ptr && zoom_ptr ) {
                has_2d = true;
                ImGui::Text("LMB drag = pan  |  scroll = zoom  |  arrows = pan");
                ImGui::Text("Center: (%.4f, %.4f)  Zoom: %.4f",
                            *center_x_ptr,
                            *center_y_ptr,
                            *zoom_ptr);
            } else if ( camera_eye_ptr && camera_at_ptr && fov_ptr ) {
                ImGui::Text("LMB drag = orbit  |  scroll = zoom");
                ImGui::Text("Ctrl+scroll = aperture  |  Ctrl+Shift+scroll = FOV");
                ImGui::Text("Ctrl+Alt+scroll = roll  |  WASD = move camera");
                ImGui::Text("Arrows = orbit  |  Shift+arrows = pan target");
                ImGui::Text("Q/E = up/down");
                ImGui::Text("Eye: (%.2f, %.2f, %.2f)",
                            camera_eye_ptr[0],
                            camera_eye_ptr[1],
                            camera_eye_ptr[2]);
                ImGui::Text("FOV: %.1f\u00b0", *fov_ptr);
                ImGui::Text("Aperture: %.3f", *aperture_ptr);
            }
        }

        ImGui::End();

        // ---- camera controls (2D pan / 3D orbit) ----
        if ( active_kernel && h_params ) {
            float *center_x_ptr_ctrl =
                has_2d ? find_param(h_params, active_kernel->desc, "center_x") : nullptr;
            float *center_y_ptr_ctrl =
                has_2d ? find_param(h_params, active_kernel->desc, "center_y") : nullptr;
            float *zoom_ptr_ctrl =
                has_2d ? find_param(h_params, active_kernel->desc, "zoom") : nullptr;

            bool changed = false;

            // ── 2D camera (center + zoom) ──────────────────────────
            if ( center_x_ptr_ctrl && center_y_ptr_ctrl && zoom_ptr_ctrl ) {
                if ( !io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
                    auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    *center_x_ptr_ctrl -=
                        delta.x / WIDTH * *zoom_ptr_ctrl * ((float)WIDTH / HEIGHT);
                    *center_y_ptr_ctrl += delta.y / HEIGHT * *zoom_ptr_ctrl;
                    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                    changed = true;
                }
                if ( !io.WantCaptureMouse && io.MouseWheel != 0.f ) {
                    *zoom_ptr_ctrl *= (io.MouseWheel > 0.f) ? 0.9f : 1.1f;
                    *zoom_ptr_ctrl = (*zoom_ptr_ctrl < 0.001f)
                                         ? 0.001f
                                         : (*zoom_ptr_ctrl > 1000.f ? 1000.f : *zoom_ptr_ctrl);
                    changed = true;
                }
            }

            // ── 3D orbit camera ────────────────────────────────────
            if ( camera_eye_ptr && camera_at_ptr && fov_ptr ) {
                static OrbitCam orbit;
                static bool orbit_init = false;
                if ( !orbit_init ) {
                    float dx = camera_eye_ptr[0] - camera_at_ptr[0],
                          dy = camera_eye_ptr[1] - camera_at_ptr[1],
                          dz = camera_eye_ptr[2] - camera_at_ptr[2];
                    orbit.dist = sqrtf(dx * dx + dy * dy + dz * dz);
                    orbit.theta = atan2f(dx, dz);
                    orbit.phi = asinf(dy / orbit.dist);
                    orbit.target[0] = camera_at_ptr[0];
                    orbit.target[1] = camera_at_ptr[1];
                    orbit.target[2] = camera_at_ptr[2];
                    orbit.theta = std::fmod(orbit.theta, 2.f * 3.14159265f);
                    orbit.phi = std::max(-1.5f, std::min(1.5f, orbit.phi));
                    orbit_init = true;
                }

                if ( !io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
                    auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    orbit.theta -= d.x * 0.005f;
                    orbit.phi += d.y * 0.005f;
                    orbit.phi = std::max(-1.5f, std::min(1.5f, orbit.phi));
                    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                    changed = true;
                }

                float mouse_wheel = io.MouseWheel;
                if ( !io.WantCaptureMouse && mouse_wheel != 0.f ) {
                    bool ctrl = io.KeyCtrl;
                    bool shift = io.KeyShift;
                    bool alt = io.KeyAlt;
                    if ( ctrl && alt ) {
                        orbit.roll += mouse_wheel * 0.05f;
                        changed = true;
                    } else if ( ctrl && shift && fov_ptr ) {
                        *fov_ptr += mouse_wheel * 2.f;
                        *fov_ptr = std::max(1.f, std::min(120.f, *fov_ptr));
                        changed = true;
                    } else if ( ctrl && aperture_ptr ) {
                        *aperture_ptr += mouse_wheel * 0.02f;
                        *aperture_ptr = std::max(0.f, std::min(1.f, *aperture_ptr));
                        changed = true;
                    } else {
                        orbit.dist *= (mouse_wheel > 0.f) ? 0.9f : 1.1f;
                        orbit.dist = std::max(1.f, std::min(100.f, orbit.dist));
                        changed = true;
                    }
                }

                if ( !io.WantCaptureKeyboard ) {
                    float kspeed = 0.04f;
                    bool shift = io.KeyShift;

                    // Orbit
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_LeftArrow) ) {
                        orbit.theta -= kspeed;
                        changed = true;
                    }
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_RightArrow) ) {
                        orbit.theta += kspeed;
                        changed = true;
                    }
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_UpArrow) ) {
                        orbit.phi += kspeed;
                        orbit.phi = std::min(1.5f, orbit.phi);
                        changed = true;
                    }
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_DownArrow) ) {
                        orbit.phi -= kspeed;
                        orbit.phi = std::max(-1.5f, orbit.phi);
                        changed = true;
                    }

                    // Pan target
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_LeftArrow) ) {
                        orbit.target[0] -= 0.2f * cosf(orbit.theta);
                        orbit.target[2] -= 0.2f * sinf(orbit.theta);
                        changed = true;
                    }
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_RightArrow) ) {
                        orbit.target[0] += 0.2f * cosf(orbit.theta);
                        orbit.target[2] += 0.2f * sinf(orbit.theta);
                        changed = true;
                    }
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_UpArrow) ) {
                        orbit.target[1] += 0.2f;
                        changed = true;
                    }
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_DownArrow) ) {
                        orbit.target[1] -= 0.2f;
                        changed = true;
                    }

                    // WASD translate camera (move eye + target together)
                    float ct = cosf(orbit.theta), st = sinf(orbit.theta);
                    float cp = cosf(orbit.phi), sp = sinf(orbit.phi);
                    float forward[3] = {cp * st, sp, cp * ct};
                    float right[3] = {ct, 0.f, -st};
                    float tspeed = 0.3f;

                    if ( ImGui::IsKeyDown(ImGuiKey_W) ) {
                        orbit.target[0] += forward[0] * tspeed;
                        orbit.target[1] += forward[1] * tspeed;
                        orbit.target[2] += forward[2] * tspeed;
                        changed = true;
                    }
                    if ( ImGui::IsKeyDown(ImGuiKey_S) ) {
                        orbit.target[0] -= forward[0] * tspeed;
                        orbit.target[1] -= forward[1] * tspeed;
                        orbit.target[2] -= forward[2] * tspeed;
                        changed = true;
                    }
                    if ( ImGui::IsKeyDown(ImGuiKey_A) ) {
                        orbit.target[0] -= right[0] * tspeed;
                        orbit.target[1] -= right[1] * tspeed;
                        orbit.target[2] -= right[2] * tspeed;
                        changed = true;
                    }
                    if ( ImGui::IsKeyDown(ImGuiKey_D) ) {
                        orbit.target[0] += right[0] * tspeed;
                        orbit.target[1] += right[1] * tspeed;
                        orbit.target[2] += right[2] * tspeed;
                        changed = true;
                    }
                    if ( ImGui::IsKeyDown(ImGuiKey_Q) ) {
                        orbit.target[1] -= 0.3f;
                        changed = true;
                    }
                    if ( ImGui::IsKeyDown(ImGuiKey_E) ) {
                        orbit.target[1] += 0.3f;
                        changed = true;
                    }
                }

                if ( changed ) {
                    float eye[3], up[3];
                    orbit_to_eye(orbit, eye);
                    orbit_up(orbit, up);
                    camera_eye_ptr[0] = eye[0];
                    camera_eye_ptr[1] = eye[1];
                    camera_eye_ptr[2] = eye[2];
                    camera_at_ptr[0] = orbit.target[0];
                    camera_at_ptr[1] = orbit.target[1];
                    camera_at_ptr[2] = orbit.target[2];
                    if ( camera_up_ptr ) {
                        camera_up_ptr[0] = up[0];
                        camera_up_ptr[1] = up[1];
                        camera_up_ptr[2] = up[2];
                    }
                }
            }

            if ( changed ) {
                q.memcpy(d_params, h_params, active_kernel->desc.params_buffer_size).wait();
                try {
                    call_init_kernel(active_kernel->handle,
                                     q,
                                     WIDTH,
                                     HEIGHT,
                                     d_params,
                                     active_kernel->desc.params_buffer_size);
                } catch ( const std::exception &e ) {
                    spdlog::error("[sycl] init error: {}", e.what());
                }
                q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                current_spp = 0;
            }
        }

        // ---- render ----
        bool rendered = false;
        spdlog::trace("[frame] spp={}/{} kernel={}", current_spp, target_spp, active_kernel ? active_kernel->name : "null");
        if ( active_kernel && current_spp < target_spp ) {
            spdlog::trace("[frame] calling render_kernel sample={}", current_spp);
            try {
                call_render_kernel(active_kernel->handle,
                                   q,
                                   WIDTH,
                                   HEIGHT,
                                   d_params,
                                   d_accum,
                                   current_spp);
                current_spp++;
                rendered = true;
                spdlog::trace("[frame] render_kernel OK, spp now {}", current_spp);
            } catch ( const std::exception &e ) {
                spdlog::error("[sycl] render error: {}", e.what());
                current_spp = target_spp;
            }
        } else {
            spdlog::trace("[frame] skip render: kernel={} spp={}/{}",
                          active_kernel ? active_kernel->name : "null",
                          current_spp, target_spp);
        }

        // Only update display when rendering happened or on first frame
        if ( rendered || current_spp == 0 ) {
            spdlog::trace("[frame] display upload (rendered={}, spp={})", rendered, current_spp);
            PROF_ZONE_SCOPED_N("Display upload");
            try {
                q.memcpy(h_accum, d_accum, pixel_count * 4 * sizeof(float)).wait();
            } catch ( const std::exception &e ) {
                spdlog::error("[sycl] memcpy error: {}", e.what());
                continue;
            }

            std::vector<uint8_t> display(pixel_count * 4);
            float inv_spp = 1.f / std::max(current_spp, 1);
            for ( size_t i = 0; i < pixel_count; i++ ) {
                size_t src = i * 4, dst = i * 4;
                // Divide accumulated colour by the number of samples (average)
                float red = h_accum[src + 0] * inv_spp;
                float green = h_accum[src + 1] * inv_spp;
                float blue = h_accum[src + 2] * inv_spp;
                // Reinhard tone mapping:  maps [0, ∞) to [0, 1)
                red = red / (1.f + red);
                green = green / (1.f + green);
                blue = blue / (1.f + blue);
                // Gamma correction for sRGB output (gamma ≈ 2.2)
                red = powf(red, 1.f / 2.2f);
                green = powf(green, 1.f / 2.2f);
                blue = powf(blue, 1.f / 2.2f);
                display[dst + 0] = (uint8_t)(red * 255.f);
                display[dst + 1] = (uint8_t)(green * 255.f);
                display[dst + 2] = (uint8_t)(blue * 255.f);
                display[dst + 3] = 255;
            }

            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D,
                            0,
                            0,
                            0,
                            WIDTH,
                            HEIGHT,
                            GL_RGBA,
                            GL_UNSIGNED_BYTE,
                            display.data());
        }

        // ---- render ImGui ----
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.0f, 0.0f, 0.0f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        PROF_FRAME_MARK;
    }

    // ---- cleanup ----
    sycl::free(d_accum, q);
    if ( d_params ) {
        sycl::free(d_params, q);
    }
    delete[] h_accum;
    free(h_params);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// ── kernel dispatch helpers ────────────────────────────────────────────
static void
call_init_kernel(void *handle, sycl::queue &q, int w, int h, const void *params, size_t sz) {
    PROF_ZONE_SCOPED;
    using fn_t = void (*)(sycl::queue *, int, int, const void *, size_t);
    auto *fn = reinterpret_cast<fn_t>(dlsym(handle, "init_kernel"));
    if ( fn ) {
        fn(&q, w, h, params, sz);
    } else {
        spdlog::error("[kernel] init_kernel not found");
    }
}

static void call_render_kernel(void *handle,
                               sycl::queue &q,
                               int w,
                               int h,
                               const void *params,
                               void *accum,
                               int sample) {
    PROF_ZONE_SCOPED;
    using fn_t = void (*)(sycl::queue *, int, int, const void *, void *, int);
    auto *fn = reinterpret_cast<fn_t>(dlsym(handle, "render_kernel"));
    if ( fn ) {
        fn(&q, w, h, params, accum, sample);
    } else {
        spdlog::error("[kernel] render_kernel not found");
    }
}

/// Fill the standard parameter area of a params buffer with defaults.
/// Kernels that support standard params have
/// `buffer_size >= RT_NUM_STD_PARAMS * sizeof(float)`.
static void init_std_params(float *buf, size_t buf_size) {
    if ( buf_size < RT_NUM_STD_PARAMS * sizeof(float) ) {
        return;
    }
    buf[RT_SPP_FRAME] = 1;
    buf[RT_MAX_BOUNCES] = 10;
    buf[RT_CAM_EYE + 0] = 0;
    buf[RT_CAM_EYE + 1] = 1.5f;
    buf[RT_CAM_EYE + 2] = 4.5f;
    buf[RT_CAM_AT + 0] = 0;
    buf[RT_CAM_AT + 1] = 1.2f;
    buf[RT_CAM_AT + 2] = 0;
    buf[RT_CAM_FOV] = 35;
    buf[RT_CAM_APERTURE] = 0;
    buf[RT_CAM_UP + 0] = 0;
    buf[RT_CAM_UP + 1] = 1;
    buf[RT_CAM_UP + 2] = 0;
}

/// Apply YAML overrides for standard params (cam_eye, cam_at, cam_fov, etc.)
static void apply_yaml_std_params(const SceneDef &scene, float *buf, size_t buf_size) {
    if ( buf_size < RT_NUM_STD_PARAMS * sizeof(float) || !scene.has_overrides ) {
        return;
    }
    try {
        YAML::Node root = YAML::LoadFile(scene.yaml_path);
        auto params = root["params"];
        if ( !params ) {
            return;
        }
        auto vec3 = [&](const char *key, int base) {
            auto n = params[key];
            if ( n && n.size() >= 3 ) {
                buf[base + 0] = n[0].as<float>();
                buf[base + 1] = n[1].as<float>();
                buf[base + 2] = n[2].as<float>();
            }
        };
        auto scalar = [&](const char *key, int idx) {
            auto n = params[key];
            if ( n ) {
                buf[idx] = n.as<float>();
            }
        };
        scalar("spp_frame", RT_SPP_FRAME);
        scalar("max_bounces", RT_MAX_BOUNCES);
        vec3("cam_eye", RT_CAM_EYE);
        vec3("cam_at", RT_CAM_AT);
        scalar("cam_fov", RT_CAM_FOV);
        scalar("cam_aperture", RT_CAM_APERTURE);
        vec3("cam_up", RT_CAM_UP);
    } catch ( const std::exception &e ) {
        spdlog::error("[scenes] error applying YAML std params: {}", e.what());
    }
}

static void recreate_render_buffers(sycl::queue &q,
                                    GLuint &tex,
                                    float *&d_accum,
                                    float *&h_accum,
                                    size_t &pixel_count,
                                    int w,
                                     int h) {
    PROF_ZONE_SCOPED;
    spdlog::debug("[buf] recreate {}x{}", w, h);
    if ( tex ) {
        spdlog::trace("[buf] delete tex");
        glDeleteTextures(1, &tex);
    }
    if ( d_accum ) {
        spdlog::trace("[buf] free d_accum");
        sycl::free(d_accum, q);
    }
    delete[] h_accum;

    pixel_count = (size_t)w * h;
    spdlog::debug("[buf] create tex");
    tex = create_render_texture(w, h);
    spdlog::debug("[buf] alloc {} floats", pixel_count * 4);
    d_accum = sycl::malloc_device<float>(pixel_count * 4, q);
    spdlog::trace("[buf] alloc h_accum");
    h_accum = new float[pixel_count * 4]();
    spdlog::trace("[buf] done");
}

static float *find_param(float *params, const KernelDesc &desc, const char *name) {
    for ( int i = 0; i < desc.param_count; i++ ) {
        if ( std::strcmp(desc.params[i].name, name) == 0 ) {
            return params + desc.params[i].buffer_offset / sizeof(float);
        }
    }
    return nullptr;
}
