#include "kernel_library.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

KernelLibrary::KernelLibrary(std::string build_dir) : build_dir_(std::move(build_dir)) {
}

KernelLibrary::~KernelLibrary() {
    // Do NOT dlclose — SYCL device images registered via global ctors
    // would cause duplicate registrations on reload. Process exit cleans up.
    handles_.clear();
    active_.clear();
}

// ── rebuild via cmake ──────────────────────────────────────────────────
bool KernelLibrary::rebuild(const std::string &kernel_name) {
    std::string cmd = "cmake --build " + build_dir_ + " --target " + kernel_name + " -j 2>&1";
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

// ── load (or reload) a kernel .so ──────────────────────────────────────
KernelHandle *KernelLibrary::load(const std::string &kernel_name) {
    fs::path src_so =
        fs::path(build_dir_) / "kernels" / kernel_name / ("lib" + kernel_name + ".so");
    if ( !fs::exists(src_so) ) {
        spdlog::error("[kernel] {} not found at {}", kernel_name, src_so.string());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
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

    void *h = dlopen(ver_so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if ( !h ) {
        spdlog::error("[kernel] dlopen {}: {}", ver_so.string(), dlerror());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    auto *get_desc = reinterpret_cast<KernelDesc *(*)()>(dlsym(h, "get_kernel_desc"));
    if ( !get_desc ) {
        spdlog::error("[kernel] dlsym get_kernel_desc: {}", dlerror());
        // keep old handle active
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    auto kh = std::make_unique<KernelHandle>();
    kh->handle = h;
    kh->name = kernel_name;
    kh->generation = gen;
    kh->path = ver_so;
    kh->desc = *get_desc();

    // buffer_offset, buffer_size, and params_buffer_size are already set by
    // the kernel's get_kernel_desc() — trust them as-is.

    // Store the handle, activate it
    auto *ptr = kh.get();
    handles_[ver_so] = std::move(kh);
    active_[kernel_name] = ptr;

    spdlog::info("[kernel] loaded {} (gen {}, {} params, {} bytes)",
                 kernel_name,
                 gen,
                 ptr->desc.param_count,
                 ptr->desc.params_buffer_size);
    return ptr;
}

void KernelLibrary::unload(KernelHandle *kh) {
    if ( !kh ) {
        return;
    }
    // Remove from active, but KEEP the handle alive (never dlclose)
    for ( auto it = active_.begin(); it != active_.end(); ) {
        if ( it->second == kh ) {
            it = active_.erase(it);
        } else {
            ++it;
        }
    }
}

KernelHandle *KernelLibrary::active(const std::string &kernel_name) const {
    auto it = active_.find(kernel_name);
    return it != active_.end() ? it->second : nullptr;
}
