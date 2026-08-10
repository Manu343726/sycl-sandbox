// `diag gpu` — drive the REAL raytracer kernel .so on the SYCL GPU
// backend, exactly like the app.
//
// Ported from the /tmp/mesh_so_diag.cpp harness (the tool that caught
// the degenerate-triangle NaN bug on the RTX 5080).  Loads
// build/kernels/raytracer/libraytracer.so via dlopen, resolves the
// kernel ABI entry points, and renders N frames of a YAML scene with
// per-frame accumulation, reporting NaN/Inf pixel counts and average
// luminance every 25 frames — the regression guard for GPU fast-math
// NaN leakage.

#include "diags.h"

#include <sycl/sycl.hpp>
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/kernel/stats.h>

#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int run_gpu_diag(const std::string &yaml, const std::string &so_path,
                 int width, int height, int frames, bool bench = false) {
    const int W = width, H = height, FRAMES = frames;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fflush(stdout);

    auto config = scene_loader::load_and_resolve(yaml);
    rt::Runtime rt;
    rt.queue = &q;
    rt::SceneBuilder builder;
    scene_loader::build_scene(builder, config);
    builder.build_bvh();
    builder.build_mesh_bvhs();
    rt::SceneData data = builder.build(&rt);
    rt::SceneView v = data.view();

    auto desc = scene_loader::load_scene_descriptor(yaml);
    desc.build_layout();
    size_t pbytes = desc.buffer_size();
    // NOTE: params MUST be host-visible (alloc_host) — the kernel .so's
    // init_kernel reads them from host code.
    float *d_params = rt.alloc_host<float>(pbytes / sizeof(float));
    rt.copy_to_device(d_params, desc.current_buffer.data(), pbytes);
    rt::ParamLookup lookup = desc.make_lookup();
    lookup.set_buffer(d_params);

    float *d_accum = rt.alloc_device<float>((size_t)W * H * 4);
    rt.fill(d_accum, 0, (size_t)W * H * 4 * sizeof(float));
    uint8_t *d_output = rt.alloc_device<uint8_t>((size_t)W * H * 4);
    std::vector<float> h_accum((size_t)W * H * 4);

    void *so = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::printf("dlopen failed (%s): %s\n", so_path.c_str(), dlerror());
        return 2;
    }

    auto fn_entry =
        (void (*)(const rt::Context *))dlsym(so, "kernel_entry");
    if (!fn_entry) {
        std::printf("dlsym kernel_entry failed: %s\n", dlerror());
        return 3;
    }

    // Single entry point: render straight away with the full per-frame
    // context (profiler ring / stat writer left null — the diag has
    // neither).  No setup, no desc — the caller owns everything.
    if (bench) {
        // Warm-up: 2 frames to stabilise JIT / GPU clocks.
        for (int f = -2; f < 0; f++) {
            rt::Context ctx;
            ctx.runtime = &rt;
            ctx.cancel_flag = rt.cancel_flag;
            ctx.params = &lookup;
            ctx.scene = &v;
            ctx.width = W;
            ctx.height = H;
            ctx.accum = d_accum;
            ctx.output = d_output;
            ctx.spp_frame = 1;
            ctx.spp_total = 0;
            ctx.frame_index = 0;
            ctx.prof = profiler::DeviceRing{};
            fn_entry(&ctx);
            q.wait();
        }
        double sum = 0, mn = 1e9, mx = 0;
        for (int f = 0; f < FRAMES; f++) {
            rt::Context ctx;
            ctx.runtime = &rt;
            ctx.cancel_flag = rt.cancel_flag;
            ctx.params = &lookup;
            ctx.scene = &v;
            ctx.width = W;
            ctx.height = H;
            ctx.accum = d_accum;
            ctx.output = d_output;
            ctx.spp_frame = 1;
            ctx.spp_total = (uint32_t)f;
            ctx.frame_index = (uint64_t)f;
            ctx.prof = profiler::DeviceRing{};
            auto t0 = std::chrono::steady_clock::now();
            fn_entry(&ctx);
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum += ms;
            mn = std::min(mn, ms);
            mx = std::max(mx, ms);
        }
        double avg = sum / FRAMES;
        std::printf("bench %dx%d  spp=1  frames=%d  "
                    "avg=%.2fms  min=%.2f  max=%.2f  FPS=%.1f  (%.2f MRays/s)\n",
                    W, H, FRAMES, avg, mn, mx, 1000.0 / avg,
                    (double)W * H / 1e6 / (avg / 1000.0));
        std::fflush(stdout);
    } else {
        for (int f = 0; f < FRAMES; f++) {
            rt::Context ctx;
            ctx.runtime = &rt;
            ctx.cancel_flag = rt.cancel_flag;
            ctx.params = &lookup;
            ctx.scene = &v;
            ctx.width = W;
            ctx.height = H;
            ctx.accum = d_accum;
            ctx.output = d_output;
            ctx.spp_frame = 1;
            ctx.spp_total = (uint32_t)f;
            ctx.frame_index = (uint64_t)f;
            ctx.prof = profiler::DeviceRing{};
            fn_entry(&ctx);
            q.wait();
            rt.copy_to_host(h_accum.data(), d_accum, h_accum.size() * sizeof(float));

            long nan_pixels = 0;
            double lum = 0, count = 0;
            for (size_t p = 0; p < (size_t)W * H; p++) {
                float n = h_accum[p * 4 + 3];
                if (n <= 0) continue;
                float r = h_accum[p * 4 + 0] / n;
                float g = h_accum[p * 4 + 1] / n;
                float b = h_accum[p * 4 + 2] / n;
                if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
                    nan_pixels++;
                lum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
                count++;
            }
            if (f % 25 == 0 || f == FRAMES - 1)
                std::printf("frame %4d: nan/inf=%ld avg_lum=%.4f\n", f,
                            nan_pixels, count ? lum / count : 0.0);
            std::fflush(stdout);
        }
    }

    data.free(&rt);
    rt.dealloc(d_params);
    rt.dealloc(d_accum);
    rt.dealloc(d_output);
    dlclose(so);
    return 0;
}

} // namespace

void register_gpu_diag(argparse::ArgumentParser &diag,
                       std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "gpu";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("gpu");
    // NOTE: argparse v3.2 stores CLI-supplied values as std::string (bad
    // any_cast for get<int>), so convert at parse time via .action().
    auto int_action = [](const std::string &s) { return std::stoi(s); };
    cmd.parser->add_argument("yaml")
        .nargs(argparse::nargs_pattern::optional)
        .default_value(std::string("scenes/mesh_demo.yaml"))
        .help("scene YAML file");
    cmd.parser->add_argument("--so")
        .default_value(std::string("build/kernels/raytracer/libraytracer.so"))
        .help("path to the raytracer kernel .so to dlopen");
    cmd.parser->add_argument("--width")
        .default_value(128)
        .action(int_action)
        .help("render width in pixels");
    cmd.parser->add_argument("--height")
        .default_value(80)
        .action(int_action)
        .help("render height in pixels");
    cmd.parser->add_argument("--frames")
        .default_value(300)
        .action(int_action)
        .help("number of accumulation frames");
    cmd.parser->add_argument("--bench")
        .default_value(false)
        .implicit_value(true)
        .help("benchmark mode: time kernel_entry+q.wait(), report FPS (no nan check)");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_gpu_diag(p.get<std::string>("yaml"),
                            p.get<std::string>("--so"),
                            p.get<int>("--width"), p.get<int>("--height"),
                            p.get<int>("--frames"),
                            p.get<bool>("--bench"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
