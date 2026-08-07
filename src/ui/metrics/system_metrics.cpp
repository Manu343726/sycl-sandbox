// System resource sampling for the metrics panel — Linux + Windows.
//
//   CPU  : /proc/stat deltas (Linux)  |  GetSystemTimes (Windows)
//   RAM  : /proc/meminfo (Linux)      |  GlobalMemoryStatusEx (Windows)
//   GPU  : NVIDIA NVML loaded at runtime (both platforms), falling back
//          to amdgpu sysfs on Linux.
//
// NVML is dynamically loaded (dlopen / LoadLibrary) so this file links
// against nothing GPU-specific and the app works without an NVIDIA GPU.

#include "ui/metrics/system_metrics.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <dirent.h>
#include <dlfcn.h>
#endif

namespace {

// ── Minimal NVML (NVIDIA Management Library) binding ─────────────────
// Only the handful of entry points needed for utilization + memory are
// declared; symbol names follow the official nvml.h.

typedef void *nvml_device_t;
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvml_utilization_t;
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvml_memory_t;
typedef enum { NVML_SUCCESS = 0 } nvml_return_t;

struct NvmlApi {
    void *handle = nullptr;
    nvml_return_t (*init_v2)() = nullptr;
    nvml_return_t (*shutdown)() = nullptr;
    nvml_return_t (*device_get_handle_by_index_v2)(unsigned int,
                                                   nvml_device_t *) = nullptr;
    nvml_return_t (*device_get_utilization_rates)(nvml_device_t,
                                                  nvml_utilization_t *) = nullptr;
    nvml_return_t (*device_get_memory_info)(nvml_device_t,
                                            nvml_memory_t *) = nullptr;
    nvml_return_t (*device_get_name)(nvml_device_t, char *,
                                     unsigned int) = nullptr;
};

void nvml_unload(NvmlApi &api) {
    if (!api.handle) return;
#if defined(_WIN32)
    FreeLibrary((HMODULE)api.handle);
#elif defined(__linux__)
    dlclose(api.handle);
#endif
    api = NvmlApi{};
}

bool nvml_load(NvmlApi &api) {
#if defined(_WIN32)
    api.handle = (void *)LoadLibraryA("nvml.dll");
    if (!api.handle) return false;
    auto sym = [&api](const char *name) -> void * {
        return (void *)GetProcAddress((HMODULE)api.handle, name);
    };
#elif defined(__linux__)
    api.handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!api.handle) return false;
    auto sym = [&api](const char *name) -> void * {
        return dlsym(api.handle, name);
    };
#else
    (void)api;
    return false;
#endif

    api.init_v2 =
        (nvml_return_t(*)())sym("nvmlInit_v2");
    api.shutdown =
        (nvml_return_t(*)())sym("nvmlShutdown");
    api.device_get_handle_by_index_v2 =
        (nvml_return_t(*)(unsigned int, nvml_device_t *))sym(
            "nvmlDeviceGetHandleByIndex_v2");
    api.device_get_utilization_rates =
        (nvml_return_t(*)(nvml_device_t, nvml_utilization_t *))sym(
            "nvmlDeviceGetUtilizationRates");
    api.device_get_memory_info =
        (nvml_return_t(*)(nvml_device_t, nvml_memory_t *))sym(
            "nvmlDeviceGetMemoryInfo");
    api.device_get_name =
        (nvml_return_t(*)(nvml_device_t, char *, unsigned int))sym(
            "nvmlDeviceGetName_v2");
    if (!api.device_get_name) {
        api.device_get_name =
            (nvml_return_t(*)(nvml_device_t, char *, unsigned int))sym(
                "nvmlDeviceGetName");
    }

    const bool complete = api.init_v2 && api.shutdown &&
                          api.device_get_handle_by_index_v2 &&
                          api.device_get_utilization_rates &&
                          api.device_get_memory_info && api.device_get_name;
    if (!complete) {
        nvml_unload(api);
        return false;
    }
    return true;
}

/// Process-wide NVML state (lazy init on first sample).
struct NvmlBackend {
    NvmlApi api;
    nvml_device_t device = nullptr;
    bool tried = false;
    bool ready = false;
    std::string name;
};
NvmlBackend g_nvml;

bool nvml_ensure() {
    if (g_nvml.tried) return g_nvml.ready;
    g_nvml.tried = true;

    if (!nvml_load(g_nvml.api)) {
        spdlog::debug("[metrics] NVML unavailable \u2014 NVIDIA GPU metrics disabled");
        return false;
    }
    if (g_nvml.api.init_v2() != NVML_SUCCESS) {
        spdlog::debug("[metrics] NVML init failed");
        nvml_unload(g_nvml.api);
        return false;
    }
    if (g_nvml.api.device_get_handle_by_index_v2(0, &g_nvml.device) !=
        NVML_SUCCESS) {
        spdlog::debug("[metrics] no NVML device found");
        g_nvml.api.shutdown();
        nvml_unload(g_nvml.api);
        return false;
    }

    char name[128] = {0};
    if (g_nvml.api.device_get_name(g_nvml.device, name, sizeof(name)) ==
        NVML_SUCCESS) {
        g_nvml.name = name;
    }
    g_nvml.ready = true;
    spdlog::info("[metrics] GPU metrics via NVML: {}", g_nvml.name);
    return true;
}

// ── GPU sampling ──────────────────────────────────────────────────────

struct GpuSample {
    float utilization = 0.0f;
    float vram_used_gb = 0.0f;
    float vram_total_gb = 0.0f;
    bool valid = false;
};

GpuSample sample_gpu_nvml() {
    GpuSample out;
    if (!nvml_ensure()) return out;

    nvml_utilization_t rates{};
    if (g_nvml.api.device_get_utilization_rates(g_nvml.device, &rates) ==
        NVML_SUCCESS) {
        out.utilization = (float)rates.gpu;
    }
    nvml_memory_t mem{};
    if (g_nvml.api.device_get_memory_info(g_nvml.device, &mem) ==
        NVML_SUCCESS) {
        constexpr double GB = 1024.0 * 1024.0 * 1024.0;
        out.vram_used_gb = (float)((double)mem.used / GB);
        out.vram_total_gb = (float)((double)mem.total / GB);
    }
    out.valid = true;
    return out;
}

#if defined(__linux__)
/// amdgpu fallback: read gpu_busy_percent + mem_info_vram_* sysfs files
/// of the first DRM card that exposes them.
GpuSample sample_gpu_amdgpu() {
    GpuSample out;
    DIR *dir = opendir("/sys/class/drm");
    if (!dir) return out;

    char path[512];
    struct dirent *ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        // Only "cardN" nodes (skip "cardN-*" connector nodes) and only
        // cards that expose amdgpu's gpu_busy_percent.
        if (std::strncmp(ent->d_name, "card", 4) != 0) continue;
        if (std::strchr(ent->d_name + 4, '-') != nullptr) continue;

        snprintf(path, sizeof(path),
                 "/sys/class/drm/%s/device/gpu_busy_percent", ent->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        unsigned int busy = 0;
        const bool ok = fscanf(f, "%u", &busy) == 1;
        fclose(f);
        if (!ok) continue;

        out.utilization = (float)busy;

        unsigned long long used = 0, total = 0;
        snprintf(path, sizeof(path),
                 "/sys/class/drm/%s/device/mem_info_vram_used", ent->d_name);
        FILE *fu = fopen(path, "r");
        if (fu) {
            if (fscanf(fu, "%llu", &used) != 1) used = 0;
            fclose(fu);
        }
        snprintf(path, sizeof(path),
                 "/sys/class/drm/%s/device/mem_info_vram_total", ent->d_name);
        FILE *ft = fopen(path, "r");
        if (ft) {
            if (fscanf(ft, "%llu", &total) != 1) total = 0;
            fclose(ft);
        }
        constexpr double GB = 1024.0 * 1024.0 * 1024.0;
        out.vram_used_gb = (float)((double)used / GB);
        out.vram_total_gb = (float)((double)total / GB);
        out.valid = true;
        break;
    }
    closedir(dir);
    return out;
}
#endif // __linux__

GpuSample sample_gpu() {
    GpuSample out = sample_gpu_nvml();
    if (out.valid) return out;
#if defined(__linux__)
    out = sample_gpu_amdgpu();
#endif
    return out;
}

// ── CPU sampling ──────────────────────────────────────────────────────

struct CpuTimes {
    unsigned long long user;
    unsigned long long nice;
    unsigned long long system;
    unsigned long long idle;
    unsigned long long iowait;
    unsigned long long irq;
    unsigned long long softirq;
    unsigned long long steal;
};

#if defined(__linux__)
bool read_cpu_times(CpuTimes &t) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return false;
    const bool ok = fscanf(f,
                           "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                           &t.user, &t.nice, &t.system, &t.idle,
                           &t.iowait, &t.irq, &t.softirq, &t.steal) == 8;
    fclose(f);
    return ok;
}
#elif defined(_WIN32)
bool read_cpu_times(CpuTimes &t) {
    FILETIME idle_ft, kernel_ft, user_ft;
    if (!GetSystemTimes(&idle_ft, &kernel_ft, &user_ft)) return false;
    const auto to_ull = [](const FILETIME &ft) {
        return ((unsigned long long)ft.dwHighDateTime << 32) |
               (unsigned long long)ft.dwLowDateTime;
    };
    t.idle = to_ull(idle_ft);
    // GetSystemTimes' kernel time includes idle time.
    t.system = to_ull(kernel_ft) - t.idle;
    t.user = to_ull(user_ft);
    t.nice = t.iowait = t.irq = t.softirq = t.steal = 0;
    return true;
}
#endif

/// CPU utilisation over the [prev, cur] interval, in percent [0..100].
float compute_cpu_percent(const CpuTimes &prev, const CpuTimes &cur) {
    const auto total = [](const CpuTimes &t) {
        return (long long)t.user + (long long)t.nice + (long long)t.system +
               (long long)t.idle + (long long)t.iowait + (long long)t.irq +
               (long long)t.softirq + (long long)t.steal;
    };
    const long long d_total = total(cur) - total(prev);
    if (d_total <= 0) return 0.0f;
    const long long d_idle =
        (long long)(cur.idle + cur.iowait) - (long long)(prev.idle + prev.iowait);
    const long long d_busy = d_total - std::max<long long>(d_idle, 0);
    if (d_busy <= 0) return 0.0f;
    return 100.0f * (float)d_busy / (float)d_total;
}

CpuTimes g_prev_cpu{};
bool g_prev_cpu_valid = false;

// ── RAM sampling ──────────────────────────────────────────────────────

#if defined(__linux__)
bool read_meminfo(unsigned long long &total_kb,
                  unsigned long long &avail_kb) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return false;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char key[64] = {0};
        unsigned long long val = 0;
        if (sscanf(line, "%63s %llu", key, &val) == 2) {
            if (std::strcmp(key, "MemTotal:") == 0) total_kb = val;
            else if (std::strcmp(key, "MemAvailable:") == 0) avail_kb = val;
        }
    }
    fclose(f);
    return total_kb > 0 && avail_kb > 0;
}
#endif

} // anonymous namespace

// ── SystemMetrics implementation ──────────────────────────────────────

void SystemMetrics::sample() {
    const auto now = std::chrono::steady_clock::now();
    if (last_sample_.time_since_epoch() != std::chrono::steady_clock::duration::zero() &&
        (now - last_sample_) < SAMPLE_PERIOD) {
        return; // throttled
    }
    last_sample_ = now;

    // CPU (platform-independent delta math over platform samples)
#if defined(__linux__) || defined(_WIN32)
    CpuTimes cur{};
    if (read_cpu_times(cur)) {
        if (g_prev_cpu_valid) {
            cpu_.push(compute_cpu_percent(g_prev_cpu, cur));
        }
        g_prev_cpu = cur;
        g_prev_cpu_valid = true;
    }
#endif

    // RAM
    {
        double total_gb = 0.0, used_gb = 0.0;
#if defined(__linux__)
        unsigned long long total_kb = 0, avail_kb = 0;
        if (read_meminfo(total_kb, avail_kb)) {
            constexpr double MB = 1024.0 * 1024.0;
            total_gb = (double)total_kb / MB;
            used_gb = (double)(total_kb - avail_kb) / MB;
        }
#elif defined(_WIN32)
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            constexpr double GB = 1024.0 * 1024.0 * 1024.0;
            total_gb = (double)ms.ullTotalPhys / GB;
            used_gb = (double)(ms.ullTotalPhys - ms.ullAvailPhys) / GB;
        }
#endif
        if (total_gb > 0.0) {
            ram_total_gb_ = (float)total_gb;
            ram_used_gb_ = (float)used_gb;
            ram_.push(100.0f * (float)(used_gb / total_gb));
        }
    }

    // GPU
    {
        const GpuSample gpu = sample_gpu();
        if (gpu.valid) {
            gpu_available_ = true;
            gpu_.push(gpu.utilization);
            vram_used_gb_ = gpu.vram_used_gb;
            vram_total_gb_ = gpu.vram_total_gb;
            if (!g_nvml.name.empty()) gpu_name_ = g_nvml.name;
        }
    }
}

void SystemMetrics::reset() {
    cpu_.reset();
    ram_.reset();
    gpu_.reset();
    ram_used_gb_ = 0.0f;
    ram_total_gb_ = 0.0f;
    vram_used_gb_ = 0.0f;
    vram_total_gb_ = 0.0f;
    gpu_available_ = false;
    gpu_name_.clear();
    last_sample_ = {};
}
