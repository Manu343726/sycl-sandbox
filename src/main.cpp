#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include "app_state.h"
#include "frame_loop.h"
#include "io/log_sink.h"
#include "io/acpp_spdlog_stream.h"
#include "tests/tests_register.h"

// AdaptiveCpp patched logging — must be included after the SYCL chain
// so hipsycl::common::output_stream is visible.
#include <hipSYCL/common/debug.hpp>

#include "imgui_test_engine/imgui_te_engine.h"
#include "imgui_test_engine/imgui_te_ui.h"
#include "imgui_test_engine/imgui_te_exporters.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <args.hxx>
#include <filesystem>

#include <sycl/sycl.hpp>
#include <memory>

// ── GLFW error callback ────────────────────────────────────────────────
static void glfw_error_cb(int error, const char *desc) {
    spdlog::error("GLFW {}: {}", error, desc);
}

// ── Test engine global state pointer ──────────────────────────────────
AppState* g_test_state = nullptr;
ImGuiTestEngine* g_test_engine = nullptr;

// ── main ───────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    // ---- CLI args ----
    args::ArgumentParser parser("sycl-sandbox");
    args::ValueFlag<std::string> backend_arg(parser, "software|cpu|gpu", "SYCL backend", {'b', "backend"});
    args::ValueFlag<std::string> log_level_arg(parser,
                                               "trace|debug|info|warn|error",
                                               "spdlog log level",
                                               {'l', "log-level"});
    args::ValueFlag<std::string> log_file_arg(parser, "path",
                                              "rotating log file path (default: sandbox.log)",
                                              {"log-file"});
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

    std::string preferred_backend = backend_arg ? backend_arg.Get() : "gpu";
    if ( preferred_backend != "software" && preferred_backend != "cpu" && preferred_backend != "gpu" ) {
        spdlog::error("backend must be 'software', 'cpu', or 'gpu', got '{}'", preferred_backend);
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
    std::string log_path = log_file_arg ? log_file_arg.Get() : "sycl-sandbox.log";
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
    init_scene_debug();

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
    // Derive project root from build directory
    fs::path project_root = fs::canonical(fs::path(build_dir) / "..");
    spdlog::info("[startup] project root: {}", project_root.string());

    state.kr = std::make_unique<KernelRuntime>(build_dir, project_root.string(),
                                                preferred_backend);

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

    // ── Profiler (host ring + device ring) ───────────────────────────
    state.kernel_profiler.init(&state.kr->runtime());
    state.kernel_profiler.init_device_ring(state.kr->queue_ptr());

    // Clear accumulator
    state.kr->clear_accum();

    // ── Auto-load first scene ─────────────────────────────────────────
    spdlog::info("[startup] scenes loaded: {}", state.scenes->all().size());
    if ( !state.scenes->all().empty() ) {
        state.kr->switch_scene(state.scenes->all().front(),
                                state.width, state.height);
        state.current_spp = 0;
        state.target_spp = state.kr->kernel() ? state.kr->kernel()->desc.max_spp : 1;
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


