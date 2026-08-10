#ifdef SANDBOX_ENABLE_TRACY

#include "tracy/tracy_launcher.h"

#include <spdlog/spdlog.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tracy_launcher {
namespace {

/// Locate the standalone profiler executable.  The app's CMake sets the
/// tracy-profiler RUNTIME_OUTPUT_DIRECTORY to <build>/profiler/bin, so
/// the binary sits next to the app's own build directory.
std::string find_profiler(const std::string &build_dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> candidates;
    if (!build_dir.empty()) {
        candidates.push_back(fs::path(build_dir) / "profiler" / "bin" /
                             "tracy-profiler");
        // Debug builds live in build_debug/, release in build/ — also try
        // the sibling build dir in case the profiler was only built there.
        if (build_dir == "build")
            candidates.push_back(fs::path("build_debug") / "profiler" / "bin" /
                                 "tracy-profiler");
        else
            candidates.push_back(fs::path("build") / "profiler" / "bin" /
                                 "tracy-profiler");
    }
    // Fallback: relative to the current working directory.
    candidates.push_back(fs::path("profiler") / "bin" / "tracy-profiler");

    for (const auto &c : candidates) {
        std::error_code ec;
        const fs::path p(c);
        if (fs::exists(p, ec) && !ec)
            return fs::absolute(p).string();
    }
    return {};
}

} // namespace

bool launch_profiler(const std::string &build_dir,
                     const std::string &addr,
                     uint16_t port) {
    const std::string exe = find_profiler(build_dir);
    if (exe.empty()) {
        spdlog::warn(
            "[tracy] tracy-profiler not found — build the tracy-profiler "
            "target first (cmake --build <build> --target tracy-profiler)");
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        spdlog::error("[tracy] fork() failed: {}", strerror(errno));
        return false;
    }

    if (pid == 0) {
        // First child: fork again so the grandchild is re-parented to
        // init when the first child exits.  The parent (app process)
        // waitpid()s the first child — no zombie, no blocking.
        const pid_t pid2 = fork();
        if (pid2 < 0) _exit(1);
        if (pid2 > 0) _exit(0);   // first child exits immediately

        // Grandchild: fully detached from the app process.
        setsid();
        const int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) close(devnull);
        }
        // Close any other inherited fds to fully sever the connection.
        // Starting from 3 (after stdin/out/err) up to a reasonable limit.
        for (int fd = 3; fd < 64; ++fd) close(fd);

        if (!addr.empty()) {
            const std::string port_str = std::to_string(port);
            execl(exe.c_str(), exe.c_str(),
                  "-a", addr.c_str(),
                  "-p", port_str.c_str(),
                  static_cast<char *>(nullptr));
        } else {
            execl(exe.c_str(), exe.c_str(), static_cast<char *>(nullptr));
        }
        _exit(127);
    }

    // Parent: reap the first child (exits immediately after the second
    // fork), then return — the grandchild is now owned by init (PID 1).
    int wstatus;
    while (waitpid(pid, &wstatus, 0) < 0) {
        if (errno != EINTR) break;
    }

    if (!addr.empty())
        spdlog::info("[tracy] launched profiler (detached) at {} -a {} -p {}",
                     exe, addr, port);
    else
        spdlog::info("[tracy] launched profiler (detached) at {}", exe);
    return true;
}

} // namespace tracy_launcher

#endif // SANDBOX_ENABLE_TRACY
