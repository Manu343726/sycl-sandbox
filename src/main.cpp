#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include "app_state.h"
#include "frame_loop.h"
#include "io/log_sink.h"
#include "io/acpp_spdlog_stream.h"
#include "diags/diags.h"
#include "tests/tests_register.h"

// AdaptiveCpp patched logging — must be included after the SYCL chain
// so hipsycl::common::output_stream is visible.
#include <hipSYCL/common/debug.hpp>

#include "imgui_test_engine/imgui_te_engine.h"
#include "imgui_test_engine/imgui_te_ui.h"
#include "imgui_test_engine/imgui_te_exporters.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>

#include <sycl/sycl.hpp>
#include <memory>
#include <pthread.h>

// ── GLFW error callback ────────────────────────────────────────────────
static void glfw_error_cb(int error, const char *desc) {
    spdlog::error("GLFW {}: {}", error, desc);
}

// ── Test engine global state pointer ──────────────────────────────────
AppState* g_test_state = nullptr;
ImGuiTestEngine* g_test_engine = nullptr;

// ── main ───────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    // ---- CLI args ----------------
    argparse::ArgumentParser program("sycl-sandbox", "0.1.0",
                                     argparse::default_arguments::help);
    program.add_argument("-b", "--backend")
        .default_value(std::string("auto"))
        .help("SYCL backend: auto|software|cpu|gpu (auto: GPU SYCL > CPU SYCL > CPU Software)");
    program.add_argument("-l", "--log-level")
        .default_value(std::string("info"))
        .help("spdlog log level: trace|debug|info|warn|error");
    program.add_argument("--log-file")
        .default_value(std::string("sycl-sandbox.log"))
        .help("rotating log file path");
    program.add_argument("--launch-tracy")
        .help("Launch Tracy profiler GUI on startup (auto-connects to the "
              "in-process client)")
        .default_value(false)
        .implicit_value(true);
    register_diag_subcommands(program);

    try {
        program.parse_args(argc, argv);
    } catch ( const std::exception &e ) {
        spdlog::error("{}", e.what());
        std::cerr << program;
        return 1;
    }

    // Diagnostic subcommands run standalone — before any GLFW/ImGui/app
    // state is created (each exits with its own code).
    if ( program.is_subcommand_used("diag") ) {
        return run_diag_subcommand(program);
    }

    pthread_setname_np(pthread_self(), "sycl-ui");

    std::string preferred_backend = program.get<std::string>("--backend");
    if ( preferred_backend != "auto" && preferred_backend != "software" && preferred_backend != "cpu" && preferred_backend != "gpu" ) {
        spdlog::error("backend must be 'auto', 'software', 'cpu', or 'gpu', got '{}'", preferred_backend);
        return 1;
    }

    std::string log_level = program.get<std::string>("--log-level");
    auto sl = spdlog::level::from_str(log_level);
    if ( sl == spdlog::level::off && log_level != "off" ) {
        spdlog::error("invalid log level '{}'", log_level);
        return 1;
    }
    spdlog::set_level(sl);
    spdlog::info("log level set to {}", log_level);

    namespace fs = std::filesystem;

    // ── AdaptiveCpp log redirect ────────────────────────────────────
    // Redirect AdaptiveCpp's output_stream (used by HIPSYCL_DEBUG_*)
    // to spdlog. This must happen before any SYCL queue operations
    // so ACPP's runtime startup messages are captured.
    // The patch in patches/adaptivecpp-spdlog-logging.patch adds
    // set_stream() to hipsycl::common::output_stream.
    {
        static AcppSpdlogStream acpp_stream;
        hipsycl::common::output_stream::get().set_stream(acpp_stream.stream());
        spdlog::info("AdaptiveCpp logging redirected to spdlog");
    }

    // ── Rotating file sink ──────────────────────────────────────────
    // 5 MiB per file, keep up to 3 rotated logs.
    // Flush immediately so log output survives crashes.
    std::string log_path = program.get<std::string>("--log-file");
    try {
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_path, 5 * 1024 * 1024, 3);
        file_sink->set_level(spdlog::level::trace);
        spdlog::default_logger()->sinks().push_back(file_sink);
        spdlog::default_logger()->flush_on(spdlog::level::trace);
        spdlog::info("logging to rotating file '{}' (5 MiB x 3)", log_path);
    } catch (const std::exception &e) {
        spdlog::error("failed to open log file '{}': {}", log_path, e.what());
    }

    // ---- AppState ----
    AppState state;

    // ── ImGui log sink ───────────────────────────────────────────────
    state.log_sink = std::make_shared<LogSink>();
    spdlog::default_logger()->sinks().push_back(state.log_sink);

    // ---- GLFW window ----
    glfwSetErrorCallback(glfw_error_cb);
    if ( !glfwInit() ) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    state.window = glfwCreateWindow(state.width, state.height, "sycl-sandbox", nullptr, nullptr);
    if ( !state.window ) {
        spdlog::error("glfwCreateWindow failed");
        return 1;
    }
    glfwMakeContextCurrent(state.window);
    glfwSwapInterval(1);

    // Use actual framebuffer size
    glfwGetFramebufferSize(state.window, &state.width, &state.height);

    // ---- ImGui ----
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(state.window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Scene debug renderer: hidden window sharing the main GL context +
    // background render thread (offscreen 3D view for the debug window).
    init_scene_debug(state.window);

    // ---- Test Engine (UI automation/testing) ----
    g_test_state = &state;
    g_test_engine = ImGuiTestEngine_CreateContext();
    {
        ImGuiTestEngineIO& test_io = ImGuiTestEngine_GetIO(g_test_engine);
        test_io.ConfigVerboseLevel = ImGuiTestVerboseLevel_Warning;
        test_io.ConfigVerboseLevelOnError = ImGuiTestVerboseLevel_Debug;
        test_io.ConfigRunSpeed = ImGuiTestRunSpeed_Fast;
        test_io.ConfigLogToTTY = false;
    }
    ImGuiTestEngine_Start(g_test_engine, ImGui::GetCurrentContext());
    ImGuiTestEngine_InstallDefaultCrashHandler();
    RegisterSandboxTests(g_test_engine);

    // ---- Probe available SYCL backends and create KernelRuntime ----
    // Derive the build directory from the executable path so it works
    // for both debug (build_debug/) and release (build/) builds.
    // When argv[0] is relative (e.g. just "sycl-sandbox" under the debugger),
    // fall back to probing common build directories.
    std::string build_dir = "build"; // fallback
    {
        std::string exe(argv[0]);
        auto last_slash = exe.rfind('/');
        if (last_slash != std::string::npos) {
            exe.resize(last_slash); // strip executable name
            auto prev_slash = exe.rfind('/');
            if (prev_slash != std::string::npos)
                exe.resize(prev_slash); // strip "src" subdir
            build_dir = exe;
        } else {
            // argv[0] has no path — check which build directory exists
            if (fs::exists(fs::path("build_debug/kernels/minimal/libminimal.so")))
                build_dir = "build_debug";
            else if (fs::exists(fs::path("build/kernels/minimal/libminimal.so")))
                build_dir = "build";
        }
        spdlog::info("[startup] using build directory: {}", build_dir);
    }
    state.build_dir = build_dir;
    // Derive project root from build directory
    fs::path project_root = fs::canonical(fs::path(build_dir) / "..");
    spdlog::info("[startup] project root: {}", project_root.string());

    state.kr = std::make_unique<KernelRuntime>(build_dir, project_root.string(),
                                                preferred_backend);

    // Profiler compile-time flag for on-the-fly kernel builds — start the
    // UI checkbox from the CMake cache so it matches reality.
    state.profiler_enabled = state.kr->profiler_enabled();

    // ── Display target (triple-buffered frame pipeline) ──────────────
    // The factory picks CUDA-GL zero-copy on NVIDIA (after a successful
    // interop self-test) or the portable staging path, and returns it
    // fully initialized.  Requires the GL context to be current.
    state.display_target = std::shared_ptr<DisplayTarget>(
        create_display_target(state.kr->queue_ptr(),
                              state.width, state.height));
    state.tex = state.display_target->texture();
    spdlog::info("[startup] display target: {}", state.display_target->name());

    // ── Render buffers at the actual framebuffer size ────────────────
    state.recreate_buffers(state.width, state.height);

    // ── Scene system ─────────────────────────────────────────────────
    spdlog::debug("[startup] loading scenes...");
    state.scenes = std::make_unique<SceneRegistry>("scenes");
    spdlog::info("[startup] {} scenes loaded", state.scenes->all().size());

    // ── Register all kernels ─────────────────────────────────────────
    state.kr->setup_all_kernels(*state.scenes);

    // ── Device profiler ring ─────────────────────────────────────
    state.init_profiler_buffers(&state.kr->runtime(),
                                state.kr->queue_ptr());

    // Device zone-name registry: the build-time extractor (libclang,
    // CMake dependency) compiles every profiler-macro string literal from
    // the kernel + host sources into a constexpr perfect-hash table.
#ifndef SANDBOX_HAVE_GENERATED_ZONE_NAMES
    profiler::scan_zone_names_from_sources((project_root / "src").string());
    profiler::scan_zone_names_from_sources((project_root / "include").string());
#endif

    // ── Tracy profiler client ────────────────────────────────────
    state.tracy_bridge.init();

#ifdef SANDBOX_ENABLE_TRACY
    if (program.get<bool>("--launch-tracy")) {
        tracy_launcher::launch_profiler(state.build_dir, "127.0.0.1", 8086);
    }
#endif

    // Clear accumulator
    state.kr->clear_accum();

    // ── Auto-load first scene ─────────────────────────────────────────
    spdlog::info("[startup] scenes loaded: {}", state.scenes->all().size());
    if ( !state.scenes->all().empty() ) {
        state.kr->switch_scene(state.scenes->all().front(),
                                state.width, state.height);
        state.current_spp = 0;
        state.target_spp = state.kr->scene() ? state.kr->scene()->max_spp : 1;
        state.tick.store(0);
        state.scene_start_time = std::chrono::steady_clock::now();
        state.on_scene_changed();
        state.kernel_ready.store(true);
        state.render_paused.store(false);
    }

    // ── Start the render thread ──────────────────────────────────────
    spdlog::debug("[render] thread starting");
    state.render_running.store(true);
    state.render_thread = std::thread(render_thread_func, std::ref(state));
    spdlog::info("[render] execution loop started on background thread");

    // ---- Main loop (delegates to subsystem calls) ----
    frame_loop(state);

    // ---- Shutdown (tears down everything) ----
    shutdown_sandbox(state);
    return 0;
}


