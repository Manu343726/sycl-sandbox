#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/kernel/stats.h>

using namespace rt;

extern "C" void kernel_entry(const rt::Context *ctx) {

    const auto bg = ctx->params->read<float3>("background");
    const auto &scene = *ctx->scene;
    const auto background_fn = [bg](const Ray &ray) -> float3 {
        const float t = 0.5f * (ray.dir.y + 1.0f);
        return lerp({1.f, 1.f, 1.f}, bg, t);
    };

    rt::render<class RaytracerPixelKernel>(*ctx, background_fn);

    if (ctx->stats) {
        ctx->stats->write<int>("num_objects", scene.num_handles);
        ctx->stats->write<int>("num_bvh_nodes", scene.num_bvh_nodes);
        ctx->stats->write<int>("num_lights", scene.num_lights);
    }
}
