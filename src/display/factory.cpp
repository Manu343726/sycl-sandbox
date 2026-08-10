#include "display_target.h"
#include "staging_target.h"
#include "cuda_gl_target.h"

#include <spdlog/spdlog.h>
#include <cstdlib>
#include <string>

// Pick and INITIALIZE the display target.  All backend detection stays
// here; callers never touch CUDA.  Selection:
//   SYCL_SANDBOX_DISPLAY=staging  → always portable staging (default)
//   SYCL_SANDBOX_DISPLAY=cuda    → CUDA-GL zero-copy (requires idle GPU —
//                                  conflicts with background render thread)
//   unset / auto                 → staging (CUDA-GL blocks cuGraphicsMapResources
//                                  on in-flight kernels, starving the pipeline)
DisplayTarget *create_display_target(sycl::queue *q, int w, int h, int slots) {
    const char *env = std::getenv("SYCL_SANDBOX_DISPLAY");
    std::string mode = env ? env : "auto";

    if (mode == "cuda" && q) {
        auto *cuda = new CudaGLDisplayTarget();
        if (cuda->init(q, w, h, slots)) return cuda;
        delete cuda;
        spdlog::warn("[display] SYCL_SANDBOX_DISPLAY=cuda requested but "
                     "interop init failed — using staging");
    }

    auto *staging = new StagingDisplayTarget();
    staging->init(q, w, h, slots);
    return staging;
}
