#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <filesystem>

/// Watches kernel source files for changes using inotify.
///
/// Unlike the old implementation that only watched a single source
/// directory per kernel, this version discovers all source dependencies
/// — including headers — by parsing the compiler-generated .d dependency
/// files that CMake/Make/Ninja produce during the build
/// (e.g. build/CMakeFiles/<target>.dir/kernel.cpp.d).
///
/// Dependencies are resolved from the build directory's CMakeFiles
/// structure so we accurately track every file that influences the
/// kernel binary — without hard-coding include paths.
///
/// Files outside the project root are ignored so system/Conan headers
/// don't trigger rebuilds.
class SourceWatcher {
public:
    /// @param build_dir    Absolute path to the CMake build directory
    ///                     (e.g. /home/user/project/build).
    /// @param project_root Absolute path to the project root.  Dependency
    ///                     files outside this path are ignored.
    SourceWatcher(std::string build_dir, std::string project_root);
    ~SourceWatcher();

    SourceWatcher(const SourceWatcher&) = delete;
    SourceWatcher& operator=(const SourceWatcher&) = delete;

    /// Register a kernel for watching.  The first call also populates
    /// the file watch list from the build system's dependency files.
    /// @param kernel_name  Logical kernel name (e.g. "minimal").
    /// @param source_dir   Source directory for this kernel's source files
    ///                     (relative or absolute, used as fallback).
    /// @param build_target CMake target name (e.g. "minimal" or "minimal_native").
    void watch_kernel(const std::string &kernel_name,
                      const std::string &source_dir,
                      const std::string &build_target = "");

    /// Poll for file changes.  Returns a list of kernel names whose
    /// source files have been modified since the last poll.
    std::vector<std::string> poll();

private:
    // ── Per-kernel state ───────────────────────────────────────────
    struct KernelState {
        std::string kernel_name;
        std::string source_dir;      ///< Primary source directory
        std::string build_target;    ///< CMake target name

        /// All watched filesystem paths for this kernel.
        /// Maps inotify watch descriptor → absolute file path.
        std::unordered_map<int, std::string> file_watches;

        /// Set of all resolved absolute dependency file paths.
        std::unordered_set<std::string> dependency_files;

        /// When we last re-parsed the .d files.
        std::chrono::steady_clock::time_point last_dep_scan;
    };

    // ── Helpers ────────────────────────────────────────────────────
    /// Resolve the full dependency list for a kernel by parsing
    /// compiler-generated .d files from the build directory.
    /// Returns the set of absolute paths (project files only).
    std::unordered_set<std::string> resolve_dependencies(
        const std::string &kernel_name,
        const std::string &build_target) const;

    /// Parse a single compiler .d file and return the dependency paths.
    std::vector<std::string> parse_dep_file(
        const std::filesystem::path &dep_path) const;

    /// Add inotify watches for all files in the given set.
    /// Returns a map of wd → file_path for newly added watches.
    std::unordered_map<int, std::string> add_watches(
        const std::unordered_set<std::string> &files);

    /// Remove inotify watches for all entries in the given map.
    void remove_watches(
        const std::unordered_map<int, std::string> &watches);

    /// Check if a path is within the project root.
    bool is_project_file(const std::filesystem::path &p) const;

    // ── Members ────────────────────────────────────────────────────
    int inotify_fd_ = -1;
    std::string build_dir_;
    std::string project_root_;

    /// Kernel-name → KernelState
    std::unordered_map<std::string, KernelState> kernels_;

    /// Reverse map: watch descriptor → kernel name (for poll())
    std::unordered_map<int, std::string> wd_to_kernel_;

    /// Re-parse .d files every 5 seconds after the last change.
    static constexpr auto kDepRescanInterval = std::chrono::seconds(5);
};
