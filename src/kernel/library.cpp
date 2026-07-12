#include "library.h"
#include <sycl-sandbox/profiler.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

KernelLibrary::KernelLibrary(std::string build_dir, const std::string &native_suffix)
    : build_dir_(std::move(build_dir)), native_suffix_(native_suffix) {
}

KernelLibrary::~KernelLibrary() {
    // Do NOT dlclose — SYCL device images registered via global ctors
    // would cause duplicate registrations on reload. Process exit cleans up.
    handles_.clear();
    active_.clear();
}

// ── Path helpers ───────────────────────────────────────────────────────

std::string KernelLibrary::so_path(const std::string &kernel_name) const {
    std::string so_name = "lib" + kernel_name + native_suffix_ + ".so";
    return (fs::path(build_dir_) / "kernels" / kernel_name / so_name).string();
}

bool KernelLibrary::so_exists(const std::string &kernel_name) const {
    return fs::exists(so_path(kernel_name));
}

// ── Dependency tracking ───────────────────────────────────────────────

void KernelLibrary::add_kernel_source_dir(const std::string &kernel_name,
                                          const std::string &source_dir) {
    kernel_source_dirs_[kernel_name] = source_dir;
}

bool KernelLibrary::is_up_to_date(const std::string &kernel_name) const {
    PROFILER_FUNCTION();

    fs::path so = so_path(kernel_name);
    if (!fs::exists(so)) return false;

    auto so_time = fs::last_write_time(so);

    // Check the CMakeFiles/<target>.dir/ for dependency (.d) files.
    // The compiler generates .d files listing all headers included
    // during compilation.  We compare the .so timestamp against each
    // dependency listed in those files.
    //
    // Kernel targets build in a CMake subdirectory, so their dependency
    // files live under <build_dir>/kernels/CMakeFiles/<target>.dir/.
    // Probe both that location and the top-level one.
    std::string target = kernel_name + native_suffix_;

    bool deps_checked = false;
    std::vector<fs::path> candidates = {
        fs::path(build_dir_) / "kernels" / "CMakeFiles" / (target + ".dir"),
        fs::path(build_dir_) / "CMakeFiles" / (target + ".dir"),
    };
    for (const fs::path &dep_root : candidates) {
        if (!fs::is_directory(dep_root)) continue;
        // .d files may be nested (e.g. <target>.dir/<target>/kernel.cpp.o.d),
        // so scan recursively.
        for (auto &entry : fs::recursive_directory_iterator(dep_root)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".d") {
                deps_checked = true;
                std::ifstream dep_file(entry.path());
                std::string line;
                // .d files have the format: target.o: dep1 dep2 ...
                // Skip the first token (target.o:) then check each dependency.
                while (std::getline(dep_file, line)) {
                    // Handle continuation lines (backslash-terminated)
                    // by appending the next line.
                    while (!line.empty() && line.back() == '\\') {
                        line.pop_back();
                        std::string next;
                        if (std::getline(dep_file, next)) line += next;
                    }

                    // Find the colon separator
                    auto colon = line.find(':');
                    if (colon == std::string::npos) continue;

                    // Parse dependency paths (space-separated, after the colon)
                    std::string deps_str = line.substr(colon + 1);
                    std::istringstream iss(deps_str);
                    std::string dep_path;
                    while (iss >> dep_path) {
                        if (dep_path.empty()) continue;
                        // Skip the target itself (first token before colon)
                        // and relative paths that don't exist.
                        fs::path dp(dep_path);
                        if (!dp.is_absolute()) continue;
                        if (!fs::exists(dp)) continue;
                        auto dep_time = fs::last_write_time(dp);
                        if (dep_time > so_time) {
                            spdlog::debug("[kernel] '{}' out of date: {} newer than .so",
                                         kernel_name, dep_path);
                            return false;
                        }
                    }
                }
            }
        }
    }

    // If no .d files were found, fall back to checking source directory
    // modification times.
    if (!deps_checked) {
        auto src_it = kernel_source_dirs_.find(kernel_name);
        if (src_it != kernel_source_dirs_.end() && fs::is_directory(src_it->second)) {
            for (auto &entry : fs::recursive_directory_iterator(src_it->second)) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension();
                if (ext == ".cpp" || ext == ".h" || ext == ".hpp" || ext == ".sycl") {
                    auto file_time = entry.last_write_time();
                    if (file_time > so_time) {
                        spdlog::debug("[kernel] '{}' out of date: {} newer than .so",
                                     kernel_name, entry.path().string());
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

// ── rebuild via cmake ──────────────────────────────────────────────────
bool KernelLibrary::rebuild(const std::string &kernel_name) {
    PROFILER_FUNCTION();
    // When native_suffix_ is "_native", rebuild the native variant target
    std::string target = kernel_name + native_suffix_;
    spdlog::debug("[kernel] rebuilding target '{}' (native_suffix='{}')", target, native_suffix_);
    std::string cmd = "cmake --build " + build_dir_ + " --target " + target + " -j4 2>&1";
    int ret = std::system(cmd.c_str());
    if ( ret == 0 ) {
        spdlog::info("[kernel] rebuild succeeded: target '{}'", target);
    } else {
        spdlog::error("[kernel] rebuild FAILED (exit {}) for target '{}'", ret, target);
    }
    return ret == 0;
}

// ── load (or reload) a kernel .so ──────────────────────────────────────
KernelHandle *KernelLibrary::load(const std::string &kernel_name) {
    PROFILER_FUNCTION();
    std::string so_name = "lib" + kernel_name + native_suffix_ + ".so";
    fs::path src_so = fs::path(build_dir_) / "kernels" / kernel_name / so_name;
    if ( !fs::exists(src_so) ) {
        spdlog::error("[kernel] {} not found at {} — binary does not exist, build required",
                     kernel_name, src_so.string());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    // Skip the reload entirely if this kernel is already active and its
    // .so hasn't changed on disk since we loaded it.  Scene switches call
    // load() unconditionally even when the new scene reuses the same
    // kernel (e.g. two raytracer YAML scenes) — re-dlopen()'ing a
    // byte-identical copy makes AdaptiveCpp's HCF cache see a duplicate
    // "hcf object id collision", which leaves the kernel cache in a state
    // where a later launch can hit a stale/dangling module (observed as
    // CUDA illegal-address errors after a scene switch).  Only re-dlopen
    // when the binary genuinely changed (a real hot-reload rebuild).
    auto so_time = fs::last_write_time(src_so);
    {
        auto it = active_.find(kernel_name);
        if ( it != active_.end() && it->second->src_mtime == so_time ) {
            spdlog::debug("[kernel] '{}' already active and unchanged, "
                          "skipping reload (gen {})",
                          kernel_name, it->second->generation);
            return it->second;
        }
    }

    // Copy to a versioned path so dlopen gives us a fresh handle
    int gen = next_generation_++;
    std::string ver_name = "lib" + kernel_name + ".v" + std::to_string(gen) + ".so";
    fs::path ver_so = src_so.parent_path() / ver_name;

    std::error_code ec;
    fs::copy_file(src_so, ver_so, fs::copy_options::overwrite_existing, ec);
    if ( ec ) {
        spdlog::error("[kernel] copy failed: {}", ec.message());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    // dlopen the versioned copy
    void *handle = dlopen(ver_so.c_str(), RTLD_NOW);
    if ( !handle ) {
        spdlog::error("[kernel] dlopen {} failed: {}", ver_so.string(), dlerror());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    // Resolve get_kernel_desc
    using desc_fn_t = KernelDesc *(*)();
    auto desc_fn = reinterpret_cast<desc_fn_t>(dlsym(handle, "get_kernel_desc"));
    if ( !desc_fn ) {
        spdlog::error("[kernel] get_kernel_desc not found in {}", ver_so.string());
        dlclose(handle);
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }
    KernelDesc *desc = desc_fn();
    if ( !desc ) {
        spdlog::error("[kernel] get_kernel_desc returned null in {}", ver_so.string());
        dlclose(handle);
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    // Create (or update) the KernelHandle
    auto it = handles_.find(kernel_name);
    if ( it == handles_.end() ) {
        auto kh = std::make_unique<KernelHandle>();
        kh->name = kernel_name;
        kh->path = ver_so.string();
        kh->generation = gen;
        kh->handle = handle;
        kh->desc = *desc;
        kh->src_mtime = so_time;
        auto *raw = kh.get();
        handles_[kernel_name] = std::move(kh);
        active_[kernel_name] = raw;
        spdlog::info("[kernel] loaded '{}' (gen {})", kernel_name, gen);
        return raw;
    } else {
        // Reload: replace handle + desc, keep generation bump
        auto &kh = it->second;
        // Keep the old handle alive (never dlclose — SYCL static state)
        kh->path = ver_so.string();
        kh->generation = gen;
        kh->handle = handle;
        kh->desc = *desc;
        kh->src_mtime = so_time;
        active_[kernel_name] = kh.get();
        spdlog::info("[kernel] reloaded '{}' (gen {})", kernel_name, gen);
        return kh.get();
    }
}

// ── unload ─────────────────────────────────────────────────────────────
void KernelLibrary::unload(KernelHandle *kh) {
    if ( !kh ) return;
    active_.erase(kh->name);
    handles_.erase(kh->name);
}

// ── active ─────────────────────────────────────────────────────────────
KernelHandle *KernelLibrary::active(const std::string &kernel_name) const {
    auto it = active_.find(kernel_name);
    return it != active_.end() ? it->second : nullptr;
}
