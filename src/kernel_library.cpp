#include "kernel_library.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

KernelLibrary::KernelLibrary(std::string build_dir)
    : build_dir_(std::move(build_dir)) {}

KernelLibrary::~KernelLibrary() {
    // Do NOT dlclose — SYCL device images registered via global ctors
    // would cause duplicate registrations on reload. Process exit cleans up.
    handles_.clear();
    active_.clear();
}

// ── rebuild via cmake ──────────────────────────────────────────────────
bool KernelLibrary::rebuild(const std::string& kernel_name) {
    std::string cmd = "cmake --build " + build_dir_
                    + " --target " + kernel_name + " -j 2>&1";
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

// ── load (or reload) a kernel .so ──────────────────────────────────────
KernelHandle* KernelLibrary::load(const std::string& kernel_name) {
    fs::path src_so = fs::path(build_dir_) / "kernels" / kernel_name / ("lib" + kernel_name + ".so");
    if (!fs::exists(src_so)) {
        fprintf(stderr, "[kernel] %s not found at %s\n",
                kernel_name.c_str(), src_so.c_str());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    // Copy to a versioned path so dlopen gives us a fresh handle
    int gen = next_generation_++;
    std::string ver_name = "lib" + kernel_name + ".v" + std::to_string(gen) + ".so";
    fs::path ver_so = src_so.parent_path() / ver_name;

    std::error_code ec;
    fs::copy_file(src_so, ver_so, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        fprintf(stderr, "[kernel] copy failed: %s\n", ec.message().c_str());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    void* h = dlopen(ver_so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "[kernel] dlopen %s: %s\n", ver_so.c_str(), dlerror());
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    auto* get_desc = reinterpret_cast<KernelDesc*(*)()>(dlsym(h, "get_kernel_desc"));
    if (!get_desc) {
        fprintf(stderr, "[kernel] dlsym get_kernel_desc: %s\n", dlerror());
        // keep old handle active
        return active_.count(kernel_name) ? active_[kernel_name] : nullptr;
    }

    auto kh = std::make_unique<KernelHandle>();
    kh->handle     = h;
    kh->name       = kernel_name;
    kh->generation = gen;
    kh->path       = ver_so;
    kh->desc       = *get_desc();

    // Compute buffer offsets for each param
    uint32_t offset = 0;
    for (int i = 0; i < kh->desc.param_count; i++) {
        auto& p = const_cast<ParamMeta&>(kh->desc.params[i]);
        p.buffer_offset = offset;
        p.buffer_size   = param_buffer_size(p);
        offset += p.buffer_size;
    }
    kh->desc.params_buffer_size = offset;

    // Store the handle, activate it
    auto* ptr = kh.get();
    handles_[ver_so] = std::move(kh);
    active_[kernel_name] = ptr;

    fprintf(stderr, "[kernel] loaded %s (gen %d, %d params, %zu bytes)\n",
            kernel_name.c_str(), gen, ptr->desc.param_count, offset);
    return ptr;
}

void KernelLibrary::unload(KernelHandle* kh) {
    if (!kh) return;
    // Remove from active, but KEEP the handle alive (never dlclose)
    for (auto it = active_.begin(); it != active_.end(); ) {
        if (it->second == kh) it = active_.erase(it);
        else ++it;
    }
}

KernelHandle* KernelLibrary::active(const std::string& kernel_name) const {
    auto it = active_.find(kernel_name);
    return it != active_.end() ? it->second : nullptr;
}
