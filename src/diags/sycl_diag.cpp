// `diag sycl` — SYCL runtime probe.
//
// NOTE: this binary is compiled with g++ (host-only).  Kernel launchers
// are only registered by the acpp compiler's device pass (used for the
// kernel .so), so host-compiled code CANNOT dispatch kernels — any launch
// attempt fails and aborts the process at exit.  This probe therefore
// verifies what the host binary can:
//   1. platform/device enumeration with info queries
//   2. in-order GPU queue creation via gpu_selector (runtime sees the
//      CUDA backend and initializes it)
//   3. device info queries on the GPU queue
// GPU kernel execution itself is exercised end-to-end by `diag gpu`,
// which dlopens the acpp-compiled kernel .so exactly like the app does.

#include "diags.h"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <utility>

namespace {

int run_sycl_diag() {
    // 1. Platform / device enumeration -------------------------------------
    const auto platforms = sycl::platform::get_platforms();
    std::printf("platforms: %zu\n", platforms.size());
    int n_gpus = 0;
    for (const auto &p : platforms) {
        const auto devices = p.get_devices();
        std::printf("  platform '%s': %zu devices\n",
                    p.get_info<sycl::info::platform::name>().c_str(),
                    devices.size());
        for (const auto &d : devices) {
            const bool is_gpu = d.is_gpu();
            n_gpus += is_gpu ? 1 : 0;
            std::printf("    device '%s' [%s]\n",
                        d.get_info<sycl::info::device::name>().c_str(),
                        is_gpu ? "gpu" : (d.is_cpu() ? "cpu" : "other"));
        }
    }
    if (n_gpus == 0) {
        std::printf("FAIL: no GPU device found\n");
        return 1;
    }

    // 2. In-order GPU queue (validates CUDA backend initialization) --------
    try {
        sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        std::printf("gpu queue: ok\n");
        std::printf("  device: %s\n",
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        std::printf("  platform: %s\n",
                    q.get_device()
                        .get_platform()
                        .get_info<sycl::info::platform::name>()
                        .c_str());
        std::printf("  max work-group size: %zu\n",
                    q.get_device()
                        .get_info<sycl::info::device::max_work_group_size>());
        std::printf("  max compute units: %u\n",
                    q.get_device()
                        .get_info<sycl::info::device::max_compute_units>());
    } catch (const sycl::exception &e) {
        std::printf("FAIL: gpu queue creation: %s\n", e.what());
        return 1;
    }

    std::printf("note: kernel dispatch is exercised by `diag gpu` "
                "(acpp-compiled kernel .so, as used by the app)\n");
    std::printf("OK\n");
    return 0;
}

} // namespace

void register_sycl_diag(argparse::ArgumentParser &diag,
                        std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "sycl";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("sycl");
    cmd.run = [](argparse::ArgumentParser &) { return run_sycl_diag(); };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
