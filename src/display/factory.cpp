#include "display_target.h"
#include "staging_target.h"
#include "cuda_gl_target.h"

#include <spdlog/spdlog.h>
#include <cstdlib>
#include <string>

// Pick and INITIALIZE the display target.  All backend detection stays
// here; callers never touch CUDA.  Selection:
//   SYCL_SANDBOX_DISPLAY=staging  → always portable staging
//   SYCL_SANDBOX_DISPLAY=cuda    → require CUDA-GL (falls back with warning)
//   unset / auto                 → CUDA-GL on NVIDIA GPUs when the interop
//                                  self-test passes, staging otherwise
DisplayTarget *create_display_target(sycl::queue *q, int w, int h, int slots) {
    const char *env = std::getenv("SYCL_SANDBOX_DISPLAY");
    std::string mode = env ? env : "auto";

    bool try_cuda = false;
    if (mode == "cuda") {
        try_cuda = true;
    } else if (mode != "staging" && q) {
        try {
            auto dev = q->get_device();
            try_cuda = dev.is_gpu() &&
                       dev.get_info<sycl::info::device::vendor>().find(
                           "NVIDIA") != std::string::npos;
        } catch (const std::exception &e) {
            spdlog::debug("[display] device query failed: {}", e.what());
        }
    }

    if (try_cuda) {
        auto *cuda = new CudaGLDisplayTarget();
        if (cuda->init(q, w, h, slots)) return cuda;
        delete cuda;
        if (mode == "cuda")
            spdlog::warn("[display] SYCL_SANDBOX_DISPLAY=cuda requested but "
                         "interop init failed — using staging");
    }

    auto *staging = new StagingDisplayTarget();
    staging->init(q, w, h, slots);
    return staging;
}
