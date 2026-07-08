#include "sandbox_api.h"
#include "watcher.h"
#include "kernel_library.h"
#include "scene_registry.h"
#include "param_ui.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <sycl/sycl.hpp>

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// ── configuration ──────────────────────────────────────────────────────
static constexpr int  WIDTH    = 1280;
static constexpr int  HEIGHT   = 720;
static constexpr int  MAX_SPP  = 4096;

// ── forward declarations for kernel dispatch helpers ───────────────────
static void call_init_kernel(void* handle, sycl::queue& q,
                              int w, int h, const void* params, size_t sz);
static void call_render_kernel(void* handle, sycl::queue& q,
                                int w, int h, const void* params,
                                void* accum, int sample);

// ── GLFW error callback ────────────────────────────────────────────────
static void glfw_error_cb(int error, const char* desc) {
    fprintf(stderr, "GLFW %d: %s\n", error, desc);
}

// ── OpenGL texture helpers ─────────────────────────────────────────────
static GLuint create_render_texture(int w, int h) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// ── orbit camera state ─────────────────────────────────────────────────
struct OrbitCamera {
    float theta   = 1.35f;
    float phi     = 1.05f;
    float dist    = 12.f;
    float fov     = 20.f;
    float target[3] = { 0.f, 0.f, 0.f };
    float eye[3]  = { 0.f, 0.f, 0.f };
    bool  dirty   = true;

    void update_eye() {
        float ct = cosf(theta), st = sinf(theta);
        float cp = cosf(phi),   sp = sinf(phi);
        eye[0] = target[0] + dist * cp * st;
        eye[1] = target[1] + dist * sp;
        eye[2] = target[2] + dist * cp * ct;
        dirty = false;
    }
};

// ── YAML scene overrides for camera (optional) ─────────────────────────
struct SceneCameraOverrides {
    float pos[3]  = { 13.f, 2.f, 3.f };
    float look_at[3] = { 0.f, 0.f, 0.f };
    float vfov    = 20.f;
    float defocus = 0.f;
};

// ── main ───────────────────────────────────────────────────────────────
int main() {
    // ---- GLFW window ----
    glfwSetErrorCallback(glfw_error_cb);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT,
                                          "sycl-sandbox", nullptr, nullptr);
    if (!window) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ---- ImGui ----
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ---- Render texture ----
    GLuint tex = create_render_texture(WIDTH, HEIGHT);

    // ---- Scene system ----
    SceneRegistry  scenes("scenes");
    KernelLibrary  kernels("build");
    SourceWatcher  watcher;

    for (auto& s : scenes.all())
        watcher.watch_kernel(s.kernel,
                             std::string("kernels/") + s.kernel);

    // ---- SYCL queue ----
    sycl::queue q;
    fprintf(stderr, "[sycl] device: %s\n",
            q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- Device buffers ----
    const size_t pixel_count = WIDTH * HEIGHT;
    float* d_accum = sycl::malloc_device<float>(pixel_count * 4, q);
    float* h_accum = new float[pixel_count * 4];

    // ---- Active state ----
    const SceneDef*     active_scene  = nullptr;
    KernelHandle*       active_kernel = nullptr;
    float*              h_params      = nullptr;
    float*              d_params      = nullptr;
    int                 current_spp   = 0;

    OrbitCamera         cam;

    // ---- Main loop ----
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 1. Check source changes → rebuild + reload
        for (auto& dirty_name : watcher.poll()) {
            fprintf(stderr, "[watcher] %s changed, rebuilding...\n",
                    dirty_name.c_str());
            if (kernels.rebuild(dirty_name)) {
                auto* new_kh = kernels.load(dirty_name);
                if (new_kh && active_scene &&
                    active_kernel && active_kernel->name == dirty_name) {
                    free(h_params);
                    auto sz = new_kh->desc.params_buffer_size;
                    h_params = (float*)calloc(sz, 1);
                    scenes.apply_params(*active_scene, new_kh->desc,
                                        h_params, sz);
                    if (d_params) sycl::free(d_params, q);
                    d_params = sycl::malloc_device<float>(sz / 4, q);
                    q.memcpy(d_params, h_params, sz).wait();
                    call_init_kernel(new_kh->handle, q,
                                     WIDTH, HEIGHT, d_params, sz);
                    active_kernel = new_kh;
                    current_spp = 0;
                }
            }
        }

        // 2. ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ---- Scene / kernel selector ----
        ImGui::Begin("Controls");
        std::string cur = active_scene ? active_scene->name : "\u2014";
        if (ImGui::BeginCombo("Scene", cur.c_str())) {
            for (auto& s : scenes.all()) {
                bool sel = active_scene && active_scene->name == s.name;
                if (ImGui::Selectable(s.name.c_str(), sel)) {
                    if (!sel) {
                        active_scene = &s;
                        active_kernel = kernels.load(s.kernel);
                        if (active_kernel) {
                            free(h_params);
                            auto sz = active_kernel->desc.params_buffer_size;
                            h_params = (float*)calloc(sz, 1);
                            scenes.apply_params(s, active_kernel->desc,
                                                h_params, sz);
                            if (d_params) sycl::free(d_params, q);
                            d_params = sycl::malloc_device<float>(sz / 4, q);
                            q.memcpy(d_params, h_params, sz).wait();
                            call_init_kernel(active_kernel->handle, q,
                                             WIDTH, HEIGHT, d_params, sz);
                            q.memset(d_accum, 0, pixel_count * 4 * 4).wait();
                            current_spp = 0;
                        }
                    }
                }
            }
            ImGui::EndCombo();
        }

        // ---- param controls ----
        if (active_kernel && h_params) {
            ImGui::SeparatorText("Parameters");
            if (render_param_controls(active_kernel->desc, h_params, false)) {
                q.memcpy(d_params, h_params,
                         active_kernel->desc.params_buffer_size).wait();
                q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                current_spp = 0;
            }
        }

        // ---- stats ----
        if (active_kernel) {
            ImGui::SeparatorText("Stats");
            ImGui::Text("%s  (gen %d)", active_kernel->desc.name,
                        active_kernel->generation);
            ImGui::Text("SPP %d / %d", current_spp, MAX_SPP);
            ImGui::Text("%d x %d", WIDTH, HEIGHT);
            if (ImGui::Button("Reset Accumulation")) {
                q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                current_spp = 0;
            }
        }
        ImGui::End();

        // ---- Camera ----
        ImGui::Begin("Camera");
        ImGui::Text("LMB drag = orbit  |  scroll = zoom");
        if (ImGui::SliderFloat("fov", &cam.fov, 5.f, 90.f, "%.1f\u00b0"))
            cam.dirty = true;
        ImGui::End();

        // ---- mouse camera ----
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
            cam.theta -= d.x * 0.005f;
            cam.phi   += d.y * 0.005f;
            cam.phi    = std::max(-1.5f, std::min(1.5f, cam.phi));
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            cam.dirty = true;
        }
        if (io.MouseWheel != 0.f) {
            cam.dist *= (io.MouseWheel > 0.f) ? 0.9f : 1.1f;
            cam.dist  = std::max(1.f, std::min(100.f, cam.dist));
            cam.dirty = true;
        }

        // ---- render ----
        if (active_kernel && current_spp < MAX_SPP) {
            if (cam.dirty) {
                cam.update_eye();
                q.memset(d_accum, 0, pixel_count * 4 * sizeof(float)).wait();
                current_spp = 0;
            }
            call_render_kernel(active_kernel->handle, q,
                               WIDTH, HEIGHT, d_params, d_accum, current_spp);
            current_spp++;
        }

        // ---- tonemap + display ----
        q.memcpy(h_accum, d_accum, pixel_count * 4 * sizeof(float)).wait();

        std::vector<float> display(pixel_count * 3);
        float inv_spp = 1.f / std::max(current_spp, 1);
        for (size_t i = 0; i < pixel_count; i++) {
            float r = h_accum[i * 4 + 0] * inv_spp;
            float g = h_accum[i * 4 + 1] * inv_spp;
            float b = h_accum[i * 4 + 2] * inv_spp;
            r = r / (1.f + r);
            g = g / (1.f + g);
            b = b / (1.f + b);
            r = powf(r, 1.f / 2.2f);
            g = powf(g, 1.f / 2.2f);
            b = powf(b, 1.f / 2.2f);
            display[i * 3 + 0] = r;
            display[i * 3 + 1] = g;
            display[i * 3 + 2] = b;
        }

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT,
                        GL_RGB, GL_FLOAT, display.data());

        // ---- viewport ----
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Viewport");
        ImVec2 vp = ImGui::GetContentRegionAvail();
        ImGui::Image((ImTextureID)(intptr_t)tex, vp);
        ImGui::End();
        ImGui::PopStyleVar();

        // ---- render ImGui ----
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ---- cleanup ----
    sycl::free(d_accum, q);
    if (d_params) sycl::free(d_params, q);
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
static void call_init_kernel(void* handle, sycl::queue& q,
                              int w, int h,
                              const void* params, size_t sz) {
    using fn_t = void(*)(sycl::queue*, int, int, const void*, size_t);
    auto* fn = reinterpret_cast<fn_t>(dlsym(handle, "init_kernel"));
    if (fn) fn(&q, w, h, params, sz);
    else fprintf(stderr, "[kernel] init_kernel not found\n");
}

static void call_render_kernel(void* handle, sycl::queue& q,
                                int w, int h,
                                const void* params,
                                void* accum, int sample) {
    using fn_t = void(*)(sycl::queue*, int, int, const void*, void*, int);
    auto* fn = reinterpret_cast<fn_t>(dlsym(handle, "render_kernel"));
    if (fn) fn(&q, w, h, params, accum, sample);
    else fprintf(stderr, "[kernel] render_kernel not found\n");
}
