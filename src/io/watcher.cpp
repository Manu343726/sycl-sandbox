#include "watcher.h"
#include <sycl-sandbox/profiler.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

// ── Constructor / Destructor ──────────────────────────────────────────

SourceWatcher::SourceWatcher(std::string build_dir, std::string project_root)
    : build_dir_(std::move(build_dir))
    , project_root_(std::move(project_root)) {

    inotify_fd_ = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
    if ( inotify_fd_ < 0 ) {
        spdlog::error("[watcher] inotify_init1 failed: {}", strerror(errno));
    }
}

SourceWatcher::~SourceWatcher() {
    if ( inotify_fd_ >= 0 ) {
        // Remove all watches (closing the fd also removes them, but
        // doing it explicitly is cleaner).
        for (auto &[wd, kernel_name] : wd_to_kernel_) {
            inotify_rm_watch(inotify_fd_, wd);
        }
        close(inotify_fd_);
    }
}

// ── Project-file filter ──────────────────────────────────────────────

bool SourceWatcher::is_project_file(const fs::path &p) const {
    if (project_root_.empty()) return true;
    // Resolve the path to a canonical form if it exists.
    fs::path canonical;
    try {
        canonical = fs::exists(p) ? fs::canonical(p) : fs::absolute(p);
    } catch (...) {
        return false;
    }
    // The project root itself and all its subdirectories are "project files".
    auto rel = fs::relative(canonical, fs::absolute(project_root_));
    return rel.native()[0] != '.'; // no ".." → inside project
}

// ── Dependency file parsing ──────────────────────────────────────────

std::vector<std::string> SourceWatcher::parse_dep_file(const fs::path &dep_path) const {
    std::vector<std::string> deps;
    std::ifstream file(dep_path);
    if (!file.is_open()) return deps;

    std::string line;
    while (std::getline(file, line)) {
        // Handle continuation lines (backslash at end of line).
        while (!line.empty() && line.back() == '\\') {
            line.pop_back();
            std::string next;
            if (std::getline(file, next)) line += next;
        }

        // Find the colon separator (target.o: dep1 dep2 ...)
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;

        // Parse space-separated dependency paths after the colon.
        std::string deps_str = line.substr(colon + 1);
        std::istringstream iss(deps_str);
        std::string dep;
        while (iss >> dep) {
            if (dep.empty()) continue;
            // Remove trailing semicolons (GNU Make .d format sometimes
            // includes stray semicolons from shell expansion).
            while (!dep.empty() && dep.back() == ';') dep.pop_back();
            if (dep.empty()) continue;

            // Canonicalize and store.
            try {
                fs::path p = dep;
                if (p.is_relative()) {
                    // .d file paths are typically relative to the build dir,
                    // or absolute.  Try resolving against the build dir.
                    p = fs::absolute(fs::path(build_dir_) / dep);
                }
                if (fs::exists(p)) {
                    deps.push_back(fs::canonical(p).string());
                }
            } catch (...) {
                // Skip paths that can't be resolved
            }
        }
    }
    return deps;
}

// ── Dependency resolution ────────────────────────────────────────────

std::unordered_set<std::string> SourceWatcher::resolve_dependencies(
    const std::string &kernel_name,
    const std::string &build_target) const {

    std::unordered_set<std::string> resolved;

    // Look for compiler-generated .d files in:
    //   <build_dir>/CMakeFiles/<target>.dir/
    fs::path dep_dir = fs::path(build_dir_) / "CMakeFiles" / (build_target + ".dir");
    if (!fs::is_directory(dep_dir)) {
        spdlog::debug("[watcher] no dep directory '{}' for '{}'",
                     dep_dir.string(), kernel_name);
        return resolved;
    }

    for (auto &entry : fs::directory_iterator(dep_dir)) {
        if (entry.path().extension() != ".d") continue;

        auto deps = parse_dep_file(entry.path());
        for (auto &d : deps) {
            if (is_project_file(d)) {
                resolved.insert(d);
            }
        }
    }

    // Always include the kernel's own source directory as a fallback.
    auto src_it = kernels_.find(kernel_name);
    if (src_it != kernels_.end() && fs::is_directory(src_it->second.source_dir)) {
        auto src_dir = fs::canonical(src_it->second.source_dir);
        resolved.insert(src_dir.string());
        for (auto &entry : fs::recursive_directory_iterator(src_dir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension();
            if (ext == ".cpp" || ext == ".h" || ext == ".hpp" || ext == ".sycl") {
                resolved.insert(fs::canonical(entry.path()).string());
            }
        }
    }

    return resolved;
}

// ── Watch management ─────────────────────────────────────────────────

std::unordered_map<int, std::string> SourceWatcher::add_watches(
    const std::unordered_set<std::string> &files) {

    std::unordered_map<int, std::string> result;
    for (auto &f : files) {
        int wd = inotify_add_watch(inotify_fd_, f.c_str(),
                                   IN_CLOSE_WRITE | IN_MOVED_TO);
        if (wd >= 0) {
            result[wd] = f;
        } else {
            spdlog::trace("[watcher] cannot watch '{}': {}", f, strerror(errno));
        }
    }
    return result;
}

void SourceWatcher::remove_watches(
    const std::unordered_map<int, std::string> &watches) {

    for (auto &[wd, path] : watches) {
        inotify_rm_watch(inotify_fd_, wd);
    }
}

// ── Public API ───────────────────────────────────────────────────────

void SourceWatcher::watch_kernel(const std::string &kernel_name,
                                  const std::string &source_dir,
                                  const std::string &build_target) {

    // Determine the target name — default to the kernel name if not specified.
    std::string target = build_target.empty() ? kernel_name : build_target;

    // Resolve source dir to absolute.
    std::string abs_source_dir;
    try {
        fs::path sd = source_dir;
        abs_source_dir = fs::canonical(sd).string();
    } catch (...) {
        abs_source_dir = fs::absolute(source_dir).string();
    }

    spdlog::info("[watcher] watching kernel '{}' (target '{}', src='{}')",
                kernel_name, target, abs_source_dir);

    // Create or update kernel state.
    auto &ks = kernels_[kernel_name];
    ks.kernel_name = kernel_name;
    ks.source_dir = abs_source_dir;
    ks.build_target = target;
    ks.last_dep_scan = std::chrono::steady_clock::time_point{}; // force initial scan

    // Remove old watches for this kernel.
    if (!ks.file_watches.empty()) {
        remove_watches(ks.file_watches);
        for (auto &[wd, _] : ks.file_watches) {
            wd_to_kernel_.erase(wd);
        }
        ks.file_watches.clear();
    }

    // Resolve and watch all dependency files.
    ks.dependency_files = resolve_dependencies(kernel_name, target);
    ks.file_watches = add_watches(ks.dependency_files);

    // Register the reverse mapping.
    for (auto &[wd, path] : ks.file_watches) {
        wd_to_kernel_[wd] = kernel_name;
    }

    spdlog::debug("[watcher] '{}': watching {} files", kernel_name,
                 ks.file_watches.size());
}

std::vector<std::string> SourceWatcher::poll() {
    PROFILER_FUNCTION();
    std::vector<std::string> dirty;

    // ── Read inotify events ─────────────────────────────────────────
    char buf[65536];
    ssize_t len = read(inotify_fd_, buf, sizeof(buf));
    if (len > 0) {
        const char *ptr = buf;
        while (ptr < buf + len) {
            auto *event = reinterpret_cast<struct inotify_event *>(
                const_cast<char *>(ptr));

            auto it = wd_to_kernel_.find(event->wd);
            if (it != wd_to_kernel_.end()) {
                auto &kernel_name = it->second;
                spdlog::trace("[watcher] change detected for '{}' (wd={})",
                            kernel_name, event->wd);
                dirty.push_back(kernel_name);
            }

            ptr += sizeof(struct inotify_event) + event->len;
        }
    }

    // ── Re-scan .d files periodically to detect new dependencies ────
    auto now = std::chrono::steady_clock::now();
    for (auto &[kernel_name, ks] : kernels_) {
        bool re_scan = false;

        // Force a scan if this kernel was just marked dirty.
        if (std::find(dirty.begin(), dirty.end(), kernel_name) != dirty.end()) {
            re_scan = true;
            // Update the scan timestamp so we don't continuously re-scan.
            ks.last_dep_scan = now;
        }

        // Also re-scan periodically regardless of changes.
        if (now - ks.last_dep_scan >= kDepRescanInterval) {
            re_scan = true;
            ks.last_dep_scan = now;
        }

        if (!re_scan) continue;

        // Re-resolve dependencies from .d files.
        auto new_deps = resolve_dependencies(kernel_name, ks.build_target);

        // Check if the dependency set has changed (new includes added).
        if (new_deps != ks.dependency_files) {
            spdlog::debug("[watcher] '{}': dependency set changed ({} files → {})",
                        kernel_name, ks.dependency_files.size(), new_deps.size());

            // Add watches for new files.
            std::unordered_set<std::string> added;
            for (auto &d : new_deps) {
                if (ks.dependency_files.find(d) == ks.dependency_files.end()) {
                    added.insert(d);
                }
            }
            auto new_watches = add_watches(added);
            for (auto &[wd, path] : new_watches) {
                ks.file_watches[wd] = path;
                wd_to_kernel_[wd] = kernel_name;
            }

            // Remove watches for files that are no longer dependencies.
            std::unordered_set<std::string> removed;
            for (auto &d : ks.dependency_files) {
                if (new_deps.find(d) == new_deps.end()) {
                    removed.insert(d);
                }
            }
            // Collect wds to remove.
            std::unordered_map<int, std::string> old_watches;
            for (auto &[wd, path] : ks.file_watches) {
                if (removed.find(path) != removed.end()) {
                    old_watches[wd] = path;
                }
            }
            remove_watches(old_watches);
            for (auto &[wd, _] : old_watches) {
                ks.file_watches.erase(wd);
                wd_to_kernel_.erase(wd);
            }

            ks.dependency_files = std::move(new_deps);
        }
    }

    // Remove duplicates from dirty list (a kernel may be marked
    // multiple times in a single poll cycle).
    std::sort(dirty.begin(), dirty.end());
    dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());

    return dirty;
}
