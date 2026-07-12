# SYCL Sandbox: Device-Resident Rendering Pipeline Redesign

**Status**: This is the original pre-implementation design document — read it
as a plan, not a status report. For the actual, verified current state (what
was built, what was fixed, and the known unresolved GPU-backend crash), see
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md). Note the MCP server
described in a few places below (`get_viewport`, capture CV) was removed
from the project entirely on 2026-07-21 and no longer exists.

**Date**: July 2026 | **Scope**: Full redesign to fix raytracing bugs, enable device-resident pipeline, correct synchronization.

---

## Executive Summary

The current sycl-sandbox has three critical bugs preventing correct progressive raytracing:

1. **Sample-splitting**: Ping-pong accumulation buffers are swapped every frame while kernels accumulate with `+=`, splitting even/odd samples across two buffers. Display normalizes by host frame counter (not per-pixel sample count), making images ~2× too dark and flickering.
2. **Normalization bug**: Display divides by total SPP count, ignoring the true per-pixel sample counter kernels maintain in alpha channel. Wrong by ~`spp_frame`× when rendering multiple samples per frame.
3. **Data race**: `swap_accum()` is a plain pointer swap while the UI thread may be mid-copy. Stats and param buffers also have unsynchronized cross-thread access.

Additionally, the current design blocks on every frame (CPU tone-map on UI thread, full device→host readback), wastes 64 MiB on a pinned profiler ring that can't access device code, and requires device-side profiler zones to live in .so globals (causing CUDA error 700 on GPU).

**Solution**: Single persistent accumulation buffer, GPU tone-mapping, device-resident profiler ring passed as a kernel argument, triple-buffered display with GL interop (CUDA) or staging fallback, drain-based synchronization for all structural mutations, in-order queue eliminating event webs.

---

## Architecture Overview

### Core Principles

1. **One in-order SYCL queue** — Behaves like a single CUDA stream; enqueue order = execution order. No event graphs, no cross-device races. Enables AdaptiveCpp instant submission.

2. **Render thread is sole queue submitter** — All structural mutations (hot-reload, resize, backend switch, accum clear) go through a **drain protocol**: main thread posts DRAIN → render thread finishes frame + `queue.wait()` → main thread mutates with exclusive ownership → RESUME. Replaces ad-hoc `render_paused`/`kernel_ready` flags.

3. **Enqueue-only kernels** — `render_kernel` submits `parallel_for` and returns without waiting. In-order queue chains tone-map + display-copy after it. `host_task` (runs host code after GPU work completes) replaces all `.wait()`+flag patterns.

4. **Single persistent accumulation buffer** — Kernels add samples (`rgb += sample; alpha += spp_frame`) to one device buffer. No swap, no host involvement, no clearing between frames (cleared only on scene/param/resize/camera changes via drain protocol).

### Frame Lifecycle (Render Thread, Steady-State)

```
1. Acquire display slot (of 3)
   └─ May block on previous slot's GPU work + GL fence
   
2. Snapshot params (under lock, generation-based)
   └─ If generation changed or target_spp==1: enqueue memset(accum)
   
3. Enqueue render_kernel(RenderContext)
   └─ Kernel adds spp_frame samples to persistent accum
   └─ Returns immediately (no wait)
   
4. Enqueue GPU tone-map kernel
   └─ Reads accum.rgb / max(accum.alpha, 1), Reinhard, γ2.2
   └─ Writes RGBA8 to display staging buffer
   └─ Y-flips for OpenGL origin (bottom-left)
   
5. Enqueue display publish
   ├─ CUDA path: custom-op maps PBO, memcpyAsync D2D, unmaps
   └─ Staging path: memcpy to pinned host slot
   
6. Enqueue profiler ring readback (if enabled)
   └─ ~1 MiB memcpy to pinned staging
   
7. Submit host_task (runs when all prior ops complete)
   ├─ Parse profiler ring staging → KernelProfiler
   ├─ Publish stats via seqlock (UI never blocks)
   └─ Set slot state = READY (release-store atomic)

8. Loop (in-order queue ensures all GPU work before next frame)
```

### UI Thread (Per VSync)

```
Check if new frame is READY (acquire-load atomic)
  ├─ Yes: pick newest ready slot
  │   ├─ present() = bind PBO → glTexSubImage2D → glFenceSync
  │   └─ Slot transitions FREE once fence signals
  └─ No: reuse previous texture
```

### Display Target (Triple-Buffered)

| Method | Thread | Purpose |
|--------|--------|---------|
| `acquire()` | Render | Reserve a slot; may block on previous GPU work/GL fence |
| `staging_ptr(slot)` | Render | Get device-writable RGBA8 pointer for this slot |
| `publish(slot, info)` | Render | Enqueue copy into presentable surface; set READY when done |
| `latest_ready(slot, info)` | Main | Check if a new frame is available (non-blocking) |
| `present(slot)` | Main | Upload to GL texture; bind for ImGui; GL fence per slot |

Two implementations:
- **StagingDisplayTarget**: Portable. Tone-map writes device staging → memcpy to pinned USM slot → glTexSubImage2D. Works all backends.
- **CudaGLDisplayTarget** (M3): Zero-copy. Tone-map writes device staging → CUDA custom-op maps GL PBO → memcpyAsync D2D on native stream → unmap. CUDA runtime loaded via dlopen (no hard link).

---

## Synchronization Design

### Rule: One Lock, One Queue, One Atomic Per Structure

| Shared Structure | Writer | Reader | Mechanism |
|---|---|---|---|
| **Params** (`current_buffer` + frame snapshot) | UI/render | Render | `ParamStore::snapshot()` — lock, copy, unlock; frame snapshot immutable after snapshot() returns |
| **Accum buffer** (device) | GPU kernel | GPU tone-map | In-order queue — device-only, no host access |
| **Display slots** | Render GPU + main GL | Main GL | Per-slot `atomic<State>` (FREE→INFLIGHT→READY→PRESENTED), GL fence gates reuse |
| **Stats** (per-frame) | Kernel via `StatWriter` | UI thread | Seqlock in `PublishedStats` — kernel increments seq (odd), writes block, increments again (even); UI retries until even and unchanged |
| **Profiler ring** (device) | GPU via `atomic_ref` | Nobody until copied | Queue-ordered memcpy to pinned staging; host_task parses staging, never live ring |
| **Scene data, kernel .so, queue** | Main thread | Render thread | Drain protocol — main posts DRAIN, render waits, main mutates, render resumes |

### Drain Protocol (Structural Mutation)

Used for: hot-reload, resize, backend switch, scene load, param change triggering clear.

```
main thread:
  ├─ Acquire command lock
  ├─ Set command = DRAIN
  ├─ Post drain_cv (wake render thread)
  └─ Wait on drained_cv (render thread signals when idle)

render thread (in frame loop):
  ├─ Check command (every iteration)
  ├─ If DRAIN:
  │  ├─ Finish current frame (or skip if mid-frame)
  │  ├─ queue.wait() (device idle)
  │  ├─ Set state = DRAINED
  │  └─ Wait on drain_cv (wake main thread)
  └─ Continue

main thread (after drain_cv wakes):
  ├─ Perform mutation (reload, clear, resize, etc.)
  ├─ Set command = RESUME
  ├─ Post drain_cv (wake render thread)
  └─ Release command lock

render thread (after drain_cv wakes):
  ├─ Resume rendering loop
  └─ Check command next iteration
```

---

## Implementation Status

### Completed (M1 + Partial M2)

#### Headers & Core Infrastructure
- ✅ `include/sycl-sandbox/profiler_device.h` — Device-side ring (16B records), compile-time zone hashing, DeviceZone RAII, globaltimer via JIT reflection
- ✅ `include/sycl-sandbox/sandbox_api.h` (v2) — `RenderContext` POD struct, enqueue-only `render_kernel`, removed `set_profiler_buffer`
- ✅ `include/sycl-sandbox/kernel/execution_context.h` — Async `foreach_pixel` (no `.wait()`), sync variant for init
- ✅ `src/kernel/dispatch.h` — Updated `call_render_kernel(handle, queue, RenderContext)`
- ✅ `src/kernel/runtime.h/cpp` — Single persistent `d_accum_`, removed ping-pong, simplified alloc/free
- ✅ `src/kernel/runtime.cpp::make_queue()` — In-order queue with async_handler

#### Display & Tone-Mapping
- ✅ `src/display/display_target.h` — Triple-buffered abstract interface
- ✅ `src/display/staging_target.h` — Portable implementation (pinned USM or host)
- ✅ `src/render/tonemap.h` — GPU kernel + CPU fallback, normalizes by `accum.alpha`
- ✅ `src/render/param_store.h` — Thread-safe per-frame snapshot
- ✅ `src/render/stat_publish.h` — Seqlock for stats publication

#### Kernel ABI Migration
- ✅ `kernels/raytracer/kernel.cpp` — Updated to v2, RenderContext, per-frame stats
- ✅ `kernels/mandelbrot/kernel.cpp` — Updated to v2, RenderContext
- ✅ `include/sycl-sandbox/rt/trace.h` — `render_main(spp_total)` instead of `sample_index`

### TODO (M2 Completion)

#### Kernel Migration (11 remaining)
- [ ] `kernels/seascape/kernel.cpp` — Update to v2 ABI
- [ ] `kernels/creation/kernel.cpp`
- [ ] `kernels/cyber_fuji/kernel.cpp`
- [ ] `kernels/lover/kernel.cpp`
- [ ] `kernels/octagrams/kernel.cpp`
- [ ] `kernels/fractal_pyramid/kernel.cpp`
- [ ] `kernels/rainforest/kernel.cpp`
- [ ] `kernels/cinelava/kernel.cpp`
- [ ] `kernels/basewarp/kernel.cpp`
- [ ] `kernels/minimal/kernel.cpp`
- [ ] `kernels/raymarching_primitives/kernel.cpp`

Each: Replace old `render_kernel(queue, w, h, params, accum, sample_index)` with `render_kernel(queue, RenderContext*)`.

#### Frame Pipeline
- [ ] `src/render/pipeline.h` — `FramePipeline` class, command queue, drain protocol
- [ ] `src/render/pipeline.cpp` — Implement frame chain orchestration, drain handler

#### App Integration
- [ ] `src/app_state.h` — Add `DisplayTarget` member, `FramePipeline` member
- [ ] `src/render_loop.h` — Rewrite `render_thread_func()` to use FramePipeline
- [ ] `src/frame_loop.h` — Replace `upload_display()` with `display->latest_ready() / present()`
- [ ] `src/ui/viewport/panel.h` — Remove `d_accum_front()`/`d_accum_back()` references

#### Build & Test
- [ ] CMakeLists updates: add `src/render/*.h` to include, link DisplayTarget impls
- [ ] Build on GPU SYCL backend
- [ ] Build on CPU SYCL backend
- [ ] Build on native (KERNEL_NATIVE) backend
- [ ] Test raytracer convergence (should be 2× brighter, no flicker)
- [ ] Test mandelbrot, seascape, other kernels

### TODO (M3–M5)

#### M3: CUDA-GL Zero-Copy
- [ ] `src/display/cuda_api.h` — dlopen libcudart, function pointers
- [ ] `src/display/cuda_gl_target.h/cpp` — Register PBOs, custom-op memcpy, interop
- [ ] Selection logic: CUDA backend + dlopen success → use CudaGLDisplayTarget

#### M4: Device Profiler
- [ ] Ring buffer per-frame readback in FramePipeline
- [ ] Calibration kernel (map device ns ↔ host timeline)
- [ ] `KernelProfiler::ingest_samples()` — parse device ring, build ZoneSample stream
- [ ] Zone name registry in RT (declare_zones in init_kernel)
- [ ] Right-size rings: device 1 MiB default, host ~4 MiB (was 64 MiB)

#### M5: Cleanup & Verification
- [ ] Delete old `display_upload()` path
- [ ] Remove `h_accum_` mirror buffer completely
- [ ] Remove per-frame `std::vector<uint8_t>` allocation in display
- [ ] Profiler UI integration (should be transparent)
- [ ] Memory profiling: RSS before/after (target −64 MiB)
- [ ] Convergence tests (mean luminance stability) — was planned via the
      now-removed MCP `get_viewport`; needs a different verification path
- [ ] TSAN on native backend (race-detection)

---

## Technical Details

### RenderContext (ABI v2)

```cpp
struct RenderContext {
    int32_t width, height;
    const void *params;           // host-visible, immutable for frame
    float *accum;                 // device float4[w*h], PERSISTENT
    uint32_t spp_frame;           // samples added this call
    uint32_t spp_total;           // samples before this call (RNG seed)
    uint64_t frame_index;         // animation tick
    profiler::DeviceRing prof;    // POD, captured by value
    const rt::StatWriter *stats;  // per-frame stat block (host-side writes)
};
```

### Device Profiler Ring

```cpp
struct DeviceRing {
    DeviceRingHeader *header;     // device malloc_device, atomic write_pos
    DeviceRecord *records;        // device malloc_device, 16B each
    uint32_t capacity;            // power of two
    uint32_t sample_interval;     // decimation (1/N work items)
    
    void push(zone_id, type, linear_id);  // atomic claim + write
    bool want(linear_id);                 // decimation filter
};

// In kernel:
PROFILER_DEVICE_ZONE(ctx.prof, "trace", linear_id);
```

Zone IDs are compile-time FNV-1a hashes; resolved against host registry in `ingest_samples()`.

### Tone-Mapping Kernel

```cpp
// Device kernel (enqueued, no wait):
for each (x, y):
    n = accum[i*4+3]
    inv = n > 1 ? 1/n : 1
    c = accum[i*4:3] * inv
    c = c / (1 + c)  // Reinhard
    c = pow(c, 1/2.2)  // gamma
    dst_y = h - 1 - y  // Y-flip for OpenGL
    out[dst_y*w+x] = RGBA8(c)
```

Correctness: per-pixel sample count (`accum.alpha`) is ground truth; no frame counter needed.

### ParamStore (Snapshot)

```cpp
ParamStore::Snapshot snapshot = store.snapshot(current_buffer.data(), bytes, gen);
// snapshot.buffer is immutable copy for this frame
// snapshot.generation tracks if params changed this frame
// snapshot.clear_requested is atomic bool consumed once per frame
```

UI thread bumps generation counter on param change; render thread checks it to decide whether to clear accum.

### Seqlock (Stats)

```cpp
// Kernel (or host_task after kernel):
writer.set_data(src, bytes);  // seq++ (odd) → copy → seq++ (even)

// UI thread:
while (!published.try_read(dst)) { /* retry */ }
```

Readers never block; writers never wait for readers. Bounded retry count for readers to avoid starvation.

---

## Key Files Reference

### Infrastructure
- `include/sycl-sandbox/profiler_device.h` — Device profiler ring, zone macros, globaltimer
- `include/sycl-sandbox/sandbox_api.h` — RenderContext, new kernel ABI
- `src/kernel/dispatch.h` — Kernel dispatch helpers
- `src/kernel/runtime.h/cpp` — SYCL queue, buffer allocation, in-order setup

### Display & Rendering
- `src/display/display_target.h` — Abstract triple-buffered interface
- `src/display/staging_target.h` — Portable implementation
- `src/display/texture.h` — GL texture helpers (existing)
- `src/render/tonemap.h` — GPU tone-map + CPU fallback
- `src/render/param_store.h` — Thread-safe param snapshot
- `src/render/stat_publish.h` — Seqlock-protected stats

### Main Loop (To Update)
- `src/render_loop.h` — Render thread; will use FramePipeline
- `src/frame_loop.h` — UI thread; will call DisplayTarget
- `src/app_state.h` — Add DisplayTarget, FramePipeline members
- `src/ui/viewport/panel.h` — Remove old accum buffer references

### Kernels
- `kernels/raytracer/kernel.cpp` — ✅ Migrated to v2
- `kernels/mandelbrot/kernel.cpp` — ✅ Migrated to v2
- `kernels/[11 others]/*.cpp` — TODO: migrate to v2

---

## Verified Assumptions (AdaptiveCpp 25.10.0)

- ✅ `SYCL_EXT_HIPSYCL_BACKEND_CUDA` defined; `librt-backend-cuda.so` present
- ✅ `sycl::AdaptiveCpp_jit::compile_if_else` with `compiler_backend::ptx` JIT-time constant available
- ✅ `sycl::interop_handle::get_native_queue<backend::cuda>()` returns `CUstream`
- ✅ In-order queue property: `sycl::property::queue::in_order{}`
- ✅ `sycl::atomic_ref` available for device-side atomics
- ✅ `sycl::host_task` (SYCL 2020) for running host code after GPU work

---

## Implementation Progress: M2 Completion (July 19, 2026)

### Completed This Session

✅ **ABI v2 Migration** (all 13 kernels)
- raytracer, mandelbrot, seascape, creation, cyber_fuji, lover, octagrams, fractal_pyramid, rainforest, cinelava, basewarp, minimal, raymarching_primitives
- Changed from `render_kernel(queue*, w, h, params, accum, spp)` to `render_kernel(queue*, RenderContext*)`
- All kernels now use `ctx->width`, `ctx->height`, `ctx->params`, `ctx->accum`, `ctx->spp_total`

✅ **Core Infrastructure** (frameworks in place)
- `profiler_device.h`: Device-side ring with timestamp(), compile_if_else() for PTX globaltimer fallback to rdtsc
- `sandbox_api.h` v2: RenderContext struct with width, height, params (const void*), accum (float*), spp_frame, spp_total, frame_index, prof (DeviceRing), stats (StatWriter*)
- `execution_context.h`: foreach_pixel async (no .wait() on parallel_for)
- `dispatch.h`: Updated call_render_kernel(handle, queue, ctx)
- `runtime.h/cpp`: Single persistent d_accum_ (no ping-pong), clear_accum(), resize()
- `tonemap.h`: Device kernel enqueue() (Reinhard + gamma + Y-flip) + CPU fallback
- `param_store.h`: ParamSnapshot with generation counter
- `stat_publish.h`: Seqlock for thread-safe stats (no blocking on render thread)

✅ **Display Pipeline** (triple-buffered)
- `display_target.h`: Abstract interface (init, destroy, resize, acquire, staging_ptr, publish, latest_ready, present, texture)
- `staging_target.h`: Portable implementation (pinned USM slots or host malloc)
- `factory.cpp`: create_display_target() factory (staging fallback; M3 will add CUDA-GL)

✅ **Orchestration** (FramePipeline skeleton)
- `render/pipeline.h/cpp`: FramePipeline class (render, set_display_target, drain protocol placeholders, frame state tracking)
- Integrated into CMakeLists.txt

✅ **API Updates** (Compatibility layer)
- `render_loop.h`: Updated to use new d_accum() API, RenderContext dispatch, removed swap_accum()
- `viewport/panel.h`: Updated to clear single d_accum() instead of d_accum_front/back

### Build Status
- **Kernels**: All 13 compiling successfully (SYCL + native backends)
- **Main app**: In progress (currently compiling src/main.cpp, scene loaders, UI panels)
- **Issues fixed**: rdtsc fallback for x86_64 profiler timestamp in native mode

### Next Steps

1. ✅ **Verify build completes** (in progress, ~10-20 min for full SYCL compilation)
2. **Test basic rendering** (raytracer convergence, no flicker)
3. **M3: CUDA-GL interop** (dlopen libcudart, PBO custom-op)
4. **M4: Device profiler** (per-frame ring readback, calibration, ingest)
5. **M5: Polish & verify** (TSAN clean, frame rate stable, memory reduction validated)

---

## Verification Checklist

- [ ] App builds on all three backends (GPU SYCL, CPU SYCL, native)
- [ ] Raytracer converges (no more ping-pong flicker, correct luminance)
- [ ] Other kernels render correctly
- [ ] Screenshots show stable output (no cross-frame variation) — was planned
      via the now-removed MCP `get_viewport`; needs a different verification path
- [ ] Profiler shows device zones (timing, nesting)
- [ ] Memory RSS down by ~23 MiB (h_accum gone) + ~60 MiB (64 MiB profiler ring shrunk)
- [ ] No data races detected by TSAN (native build)
- [ ] Frame rate stable (no stuttering from GL fences or accum clears)

---

## Risks & Open Questions

1. **CUDA context ownership**: Verify `cudaSetDevice` called from main thread sets up the same primary context that AdaptiveCpp's CUDA backend uses.
2. **Custom-op completion semantics**: Confirm that `host_task` after a custom-op truly waits for the op's async native work (the `cudaMemcpyAsync`).
3. **Inline PTX under generic SSCP**: Test that `compile_if_else` with PTX branch actually dead-code-eliminates on non-PTX backends.
4. **GL fence starvation**: If UI thread stalls, all PBO slots can end up fence-pending. Bounded (3 frames), self-healing, but log warnings.
5. **Per-lane zone matching**: Device ring decimation means sparse per-pixel zones; enter/exit matching approximate. May need non-nested zones if timeline looks corrupted.

---

## References

- AdaptiveCpp docs: https://github.com/AdaptiveCpp/AdaptiveCpp (vendor interop, JIT reflection)
- SYCL 2020 spec: `sycl::host_task`, `sycl::atomic_ref`, in-order queues
- CUDA interop: `cudaGraphicsGLRegisterBuffer`, `cudaGraphicsMapResources`, `cudaMemcpyAsync`
- Architecture design notes: `/home/manu343726/.claude/plans/analyze-the-whole-project-quiet-allen.md`
