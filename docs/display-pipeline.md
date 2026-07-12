# GPU Display Pipeline — Design Document

## 1. Purpose and Scope

This document specifies a **GPU-side display pipeline** that replaces the
current CPU-based tone-mapping and texture upload path in sycl-sandbox.
The goal is to eliminate the only blocking PCIe transfer in the render loop
(the accumulation buffer readback) by moving tone-mapping to a SYCL compute
kernel and uploading the result asynchronously through an OpenGL PBO
(Pixel Buffer Object).

**What it does:**

- **GPU tone-mapping.** A SYCL kernel that reads the floating-point
  accumulation buffer (`d_accum_front`) and applies Reinhard tone-mapping
  plus sRGB gamma correction, writing uint8 RGBA output to a device buffer
  (`d_display`). The math is identical to the current CPU path — no visual
  change.

- **PBO-based texture upload.** The `d_display` buffer is transferred to an
  OpenGL PBO, then blitted to the OpenGL texture via `glTexSubImage2D` with
  the PBO bound. On GPUs with unified memory (NVIDIA CUDA, Intel) this is
  effectively zero-copy; on others it is a single async DMA transfer rather
  than a blocking host round-trip.

- **Async capture for MCP.** When an MCP agent requests a viewport snapshot,
  a pinned host buffer receives the GPU tone-mapped frame via an async copy
  (no stall on the critical path). The capture is delivered once the copy
  completes.

**What it does NOT do:**

- It does not change the kernel parameter system (`d_params`), statistics
  (`StatWriter`), or profiler ring buffer — those already use USM shared
  memory or plain host allocations with zero PCIe overhead per frame.
- It does not change the tone-mapping math — the Reinhard + sRGB equations
  are identical to the current CPU path.
- It does not add HDR output, multi-GPU support, or CUDA-OpenGL interop
  (the latter is documented as a future Phase 3 enhancement).
- It does not change any kernel `.so` interface or scene descriptor format.

---

## 2. Motivation and PCIe Transfer Analysis

### 2.1. Current Render Pipeline

```
Frame N:

Render thread:
  call_render_kernel(...)
    → GPU raytraces → writes d_accum_back (device memory)
  std::swap(d_accum_front, d_accum_back)
  kernel_profiler.collect(...)          ← reads USM buffer (zero copy)

Main thread (display_upload):
  krt.copy_to_host(h_accum, d_accum_front, 64MB)   ← ⚠️ BLOCKING PCIe
  CPU tone-map loop (powf × 3 per pixel)
  glTexSubImage2D(tex, display_data)                ← ⚠️ Another host→GPU xfer
  if MCP capture requested: save display_data
```

### 2.2. Per-Frame Data Flow Audit

Every frame, the following data must be synchronised between host and device:

| Data | Allocation | Host→Device | Device→Host | Per-frame? | Transfer cost |
|------|-----------|-------------|-------------|------------|---------------|
| `d_params` (tick/time) | `sycl::malloc_host` (USM) | Coherent via USM | — | ✅ Yes | **Zero** — USM shared memory |
| `current_stat_buffer` | `std::vector<float>` (host) | — | Direct host writes (StatWriter) | ✅ Yes | **Zero** — host-only |
| Profiler ring buffer | `sycl::malloc_host` (USM) | — | Coherent via USM | ✅ Yes | **Zero** — USM shared memory |
| **Accum buffer** | `sycl::malloc_device` | — | `queue->memcpy().wait()` | ✅ Yes | **16 MB blocking readback** |

**The three data streams the user identified are already zero-cost.** The
accum buffer readback is the sole bottleneck.

### 2.3. Why PCIe Matters

At 1920×1080:
- Accum buffer = 1920 × 1080 × 4 floats × 4 bytes = **33 MB** (two buffers,
  16.5 MB each for front/back)
- One readback per frame = **16.5 MB across PCIe Gen3 x16** (~16 GB/s
  bidirectional) ≈ **~1 ms** minimum, plus sync overhead
- Typical measured `copy_to_host` + tone-map = **2–5 ms** per frame
- At 60 FPS, this is 12–30% of the frame budget wasted on data movement

### 2.4. Why PBO Eliminates This

An OpenGL PBO (`GL_PIXEL_UNPACK_BUFFER`) is a GPU-resident buffer.
`glTexSubImage2D` with a PBO bound copies from PBO → texture **entirely
on the GPU** (DMA engine, no PCIe round-trip). The host only needs to:

1. Map the PBO (get a pointer to GPU-resident memory — may be host-visible
   or require a sync).
2. Write pixel data into it (from the GPU tone-map output — zero copy if
   CUDA interop, or a single device→device copy if not).
3. Unmap and trigger the texture blit.

Net result: **the blocking PCIe readback is replaced by an async GPU-internal
DMA operation.**

---

## 3. Architecture

### 3.1. Layer Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  Kernel (GPU)                                                     │
│  call_render_kernel() → d_accum_front (float4, device memory)     │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPU Tone-Map (new)                                               │
│  tone_map::apply() → d_display (uint8_t RGBA, device memory)     │
│  Reinhard ÷ sRGB gamma, parallel over pixels                     │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PBO Transfer (new)                                               │
│  PBO::map_write() ← d_display → PBO::unmap()                     │
│  GPU-resident buffer (GL_PIXEL_UNPACK_BUFFER)                    │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  OpenGL Texture Blit                                              │
│  glTexSubImage2D(tex, nullptr)  ← DMA from PBO, GPU-internal     │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  ImGui Render → glfwSwapBuffers                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2. Threading Model

```
Render thread (background):
  ┌─────────────────────────────────────────────┐
  │ while (render_running) {                     │
  │   if paused/spp_done: sleep; continue;       │
  │                                              │
  │   // 1. Update tick/time in d_params (USM)   │
  │   state.tick++;  d_params[tick] = ...        │
  │                                              │
  │   // 2. Trace one sample                     │
  │   call_render_kernel(..., d_accum_back);     │
  │   std::swap(d_accum_front, d_accum_back);    │
  │                                              │
  │   // 3. GPU tone-map (NEW)                   │
  │   tone_map::apply(q, w, h,                   │
  │     d_accum_front, d_display, spp);          │
  │   q.wait();  ← sync tone-map                 │
  │                                              │
  │   // 4. Collect profiler (USM, zero-copy)    │
  │   kernel_profiler.collect(spp);              │
  │                                              │
  │   state.frame_done = true;                   │
  │ }                                            │
  └──────────────────────────────────────────────┘

Main thread (frame loop):
  ┌─────────────────────────────────────────────┐
  │ while (!glfwWindowShouldClose) {             │
  │   poll_events();                             │
  │   handle_kernel_rebuild();                   │
  │   update_frame_stats();                      │
  │   mcp_server->process_actions();             │
  │   render_ui();                               │
  │                                              │
  │   // display_upload (was CPU tone-map):      │
  │   if (frame_done) {                          │
  │     pbo.map_write();                         │
  │     memcpy(pbo_ptr, d_display, size);        │
  │     pbo.unmap();                             │
  │     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)│
  │     glTexSubImage2D(..., nullptr);           │
  │     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)  │
  │     // MCP capture from d_display_capture    │
  │   }                                          │
  │                                              │
  │   composite_frame();                         │
  │ }                                            │
  └──────────────────────────────────────────────┘
```

### 3.3. Data Flow Detail

```
render_thread_func():

  [d_params]          USM coherent         → device reads tick/time (no xfer)
  [call_render_kernel]                      → GPU writes d_accum_back
  [swap buffers]                            → d_accum_front ← d_accum_back
  [tone_map::apply]  d_accum_front → GPU    → GPU writes d_display (uint8)
                      tone-map kernel
  [q.wait()]                                → sync point
  [profiler.collect] USM coherent           → host reads profiler records
  [frame_done = true]

display_upload():

  → PBO::map_write()   Get GPU-resident pointer (or host-visible fallback)
  → memcpy(d_display    Write tone-mapped pixels into PBO
           → pbo_ptr)   (device→device if interop, host→device if fallback)
  → PBO::unmap()        Release mapping
  → glTexSubImage2D     DMA blit PBO → OpenGL texture (GPU-internal)
```

---

## 4. Core Types

### 4.1. `PBO` — Pixel Buffer Object Wrapper

```cpp
/// Manages an OpenGL PBO for async texture upload.
///
/// A PBO is a GPU-resident buffer bound to GL_PIXEL_UNPACK_BUFFER.
/// Writing pixel data into the PBO and then calling glTexSubImage2D
/// with the PBO bound causes the texture upload to happen entirely
/// on the GPU (DMA engine), without a blocking PCIe round-trip.
///
/// Lifecycle:
///   alloc(w, h)     → create or resize
///   map_write()     → get writable pointer (may block for sync)
///   unmap()         → release, ready for texture blit
///   upload(tex_id)  → glTexSubImage2D from PBO (convenience)
///   release()       → delete PBO
struct PBO {
    GLuint id = 0;           ///< OpenGL buffer name
    int width = 0;           ///< Current width (pixels)
    int height = 0;          ///< Current height (pixels)
    size_t byte_size = 0;    ///< width × height × 4 (RGBA8)

    /// Allocate (or reallocate) the PBO. Uses GL_STREAM_COPY usage
    /// hint: buffer data is written once (via map/memcpy) and copied
    /// from (via glTexSubImage2D). This is the optimal hint for our
    /// write-once-read-once pattern.
    ///
    /// Reallocation only happens when the resolution changes.
    void alloc(int w, int h);

    /// Release the PBO (glDeleteBuffers). Safe to call multiple times.
    void release();

    /// Map the PBO for write access. Returns a writable pointer to
    /// GPU-resident memory (may be host-visible on integrated GPUs
    /// or require a staging copy on discrete GPUs).
    ///
    /// On return: the PBO contains undefined data; caller must write
    /// exactly byte_size bytes before unmap().
    ///
    /// Returns nullptr on failure (logged via spdlog::error).
    uint8_t *map_write();

    /// Unmap the PBO after writing. The buffer is ready for texture
    /// upload after this call. Must be called before upload().
    void unmap();

    /// Blit the PBO contents into an OpenGL texture.
    /// Binds GL_PIXEL_UNPACK_BUFFER → glTexSubImage2D → unbinds.
    /// The texture must already have storage allocated matching w×h.
    void upload(GLuint tex_id);

    /// Check whether the PBO is allocated and matches the given size.
    bool matches(int w, int h) const;
};
```

**Usage hint rationale:** `GL_STREAM_COPY` tells the driver the buffer is
written once by the application (via map/memcpy) and read once by the GPU
(via texture upload). This allows the driver to choose the optimal placement
(device-local if possible) and avoid unnecessary synchronization.

### 4.2. `tone_map::apply()` — GPU Tone-Mapping Kernel

```cpp
namespace tone_map {

/// Apply Reinhard tone-mapping + sRGB gamma correction on the GPU.
///
/// Reads the floating-point accumulation buffer (RGBA float4 per pixel)
/// and writes quantised uint8 RGBA output suitable for display.
///
/// The transformation applied per pixel:
///   inv_spp = 1 / max(spp, 1)
///   r = accum.r * inv_spp;  g = accum.g * inv_spp;  b = accum.b * inv_spp
///   r = r / (1 + r)          ← Reinhard
///   g = g / (1 + g)
///   b = b / (1 + b)
///   r = pow(r, 1/2.2)        ← sRGB gamma
///   g = pow(g, 1/2.2)
///   b = pow(b, 1/2.2)
///   out = { uint8(r*255), uint8(g*255), uint8(b*255), 255 }
///
/// Parameters:
///   q         — SYCL queue on the compute device (same device as render)
///   w, h      — viewport dimensions in pixels
///   d_accum   — device pointer to float4[width × height] accumulation buffer
///   d_display — device pointer to uint8_t[width × height × 4] output buffer
///   spp       — current sample count (for normalisation; ≥ 1)
///
/// The kernel is synchronous from the caller's perspective: this function
/// calls q.wait() before returning. The caller (render thread) does not
/// proceed until tone-mapping is complete.
void apply(sycl::queue &q,
           int w, int h,
           const float *d_accum,
           uint8_t *d_display,
           int spp);

} // namespace tone_map
```

**Design decisions:**

- **Separate queue (recommended):** Tone-map on a separate SYCL queue
  sharing the same device. This allows the render kernel and tone-map
  kernel to overlap if the GPU supports concurrent execution (the tone-map
  can start once its input region is resolved). If a single queue is used,
  the tone-map is serialised after the render kernel, which is simpler but
  may leave the GPU idle between kernel launches.

- **`sycl::powr` precision:** SYCL 2020 guarantees `powr` for
  integer-exponent power functions. The gamma exponent `1/2.2 ≈ 0.4545` is
  a reciprocal, not an integer — use `sycl::pow` (floating-point exponent).
  This is available in all SYCL 2020 implementations (AdaptiveCpp,
  DPC++, hipSYCL).

- **No denormal flush:** Reinhard division by `(1 + colour)` means
  denominators are always ≥ 1 — no risk of denormals slowing GPU
  execution.

### 4.3. `AppState` Additions (new fields)

```cpp
struct AppState {
    // ── Existing fields (unchanged) ────────────────────────
    GLuint tex = 0;
    float *d_accum_front = nullptr;
    float *d_accum_back  = nullptr;
    // float *h_accum = nullptr;  ← REMOVED — no longer needed
    size_t accum_bytes = 0;
    // ...

    // ── New fields (Path A) ────────────────────────────────
    /// Device buffer for GPU tone-mapped output (uint8 RGBA).
    uint8_t *d_display = nullptr;

    /// OpenGL PBO for async texture upload from d_display.
    PBO display_pbo;

    /// SYCL queue for the tone-map kernel. Shares the same device
    /// as the main render queue (state.q). Using a separate queue
    /// may enable overlap of tone-map with the next render kernel.
    sycl::queue tone_map_queue;

    /// Host-pinned buffer for MCP viewport capture (async copy-back).
    /// Only allocated when an MCP agent subscribes to captures.
    uint8_t *d_display_capture = nullptr;
    std::atomic<bool> capture_pending{false};
};
```

### 4.4. Buffer Reallocation Contract

`AppState::recreate_buffers()` gains the following responsibilities:

1. `d_display` = `krt.alloc_device<uint8_t>(pixel_count * 4)`
2. `display_pbo.alloc(w, h)`
3. `tone_map_queue` = `sycl::queue(state.q->get_device())` (or recreate)
4. `d_display_capture` — not allocated here; lazy on first MCP capture
   request (to avoid wasting 16 MB on every frame when no agent is connected)

---

## 5. File Layout

### New files

```
src/display/
  tone_map.h       # Tone-map kernel declaration
  tone_map.cpp     # Tone-map SYCL kernel implementation
  pbo.h            # PBO class declaration
  pbo.cpp          # PBO class implementation (glMapBuffer, etc.)
```

### Modified files

```
src/gl_loader.h              # Add glMapBuffer, glUnmapBuffer, PBO constants
src/render_loop.h            # display_upload() rewrite, render_thread_func() changes
src/frame_loop.h             # h_accum removal, d_display cleanup
src/app_state.h              # New fields, remove h_accum
src/display/texture.h        # No change needed (create_render_texture stays)
```

### Unchanged files

```
include/sycl-sandbox/rt/param_reader.h   # USM — zero copy, no change
include/sycl-sandbox/rt/stat_writer.h    # Host-only, no change
include/sycl-sandbox/profiler.h          # USM — zero copy, no change
include/sycl-sandbox/runtime.h           # No change needed
src/kernel/dispatch.h                    # No change needed
src/kernel/profiler_host.h               # No change needed
src/scene/loader.cpp                     # No change needed
src/ui/controls/panel.h                  # No change needed
src/ui/stat/panel.h                      # No change needed
```

---

## 6. Implementation Steps

### Phase 1: OpenGL PBO Infrastructure

**Step 1.1 — `src/gl_loader.h` additions.**

Add the following function declarations and constants. The declarations are
needed because the project uses a custom `gl_loader.h` instead of a GL loader
library (no GLEW/glad).

Functions:
- `glMapBuffer(GLenum target, GLenum access) → void*`
- `glUnmapBuffer(GLenum target) → GLboolean`

Constants:
- `#define GL_PIXEL_UNPACK_BUFFER 0x8065`
- `#define GL_MAP_WRITE_BIT 0x0002`
- `#define GL_STREAM_COPY 0x88E2`
- `#define GL_WRITE_ONLY 0x88B9`

Note: `glGenBuffers`, `glBindBuffer`, `glBufferData`, `glDeleteBuffers` are
already declared (used by `scene_debug/panel.cpp`).

**Step 1.2 — `src/display/pbo.h` + `src/display/pbo.cpp`.**

Implement the `PBO` struct as specified in §4.1.

Key implementation details:
- `alloc()` calls `glGenBuffers(1, &id)`, `glBindBuffer`, `glBufferData`
  with `nullptr` data (reserve GPU memory, no initial data).
- `map_write()` calls `glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY)`.
- `unmap()` calls `glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)`.
- `upload()` binds PBO, calls `glTexSubImage2D` with `nullptr` for data
  (reads from PBO), unbinds.
- `matches()` checks `id != 0 && width == w && height == h`.

### Phase 2: GPU Tone-Mapping Kernel

**Step 2.1 — `src/display/tone_map.h` + `src/display/tone_map.cpp`.**

Implement the `tone_map::apply()` kernel as specified in §4.2.

Key implementation details:
- Submit a 2D `parallel_for` over `range<2>(h, w)`.
- Use `sycl::pow` for the gamma exponent.
- Call `q.wait()` at the end — the function is synchronous.
- The kernel captures `d_accum` and `d_display` by value (raw pointers,
  trivially copyable).

**Step 2.2 — Build system integration.**

Add `src/display/tone_map.cpp` and `src/display/pbo.cpp` to the
`sycl-sandbox` target in `CMakeLists.txt`. These are host-side SYCL
files (include `<sycl/sycl.hpp>`), compiled with AdaptiveCpp's host
compiler, not as kernels.

### Phase 3: Render Pipeline Integration

**Step 3.1 — `src/app_state.h` changes.**

- Remove `float *h_accum` field and its `delete[]` / `new float[]` in
  `recreate_buffers()`.
- Add `uint8_t *d_display`, `PBO display_pbo`, `sycl::queue tone_map_queue`,
  `uint8_t *d_display_capture`, `std::atomic<bool> capture_pending`.

**Step 3.2 — `src/render_loop.h` changes.**

In `render_thread_func()`:
- After `call_render_kernel(...)` and `std::swap`, insert:
  ```cpp
  if (state.d_display) {
      tone_map::apply(state.tone_map_queue,
                      state.width, state.height,
                      state.d_accum_front,
                      state.d_display,
                      state.current_spp.load());
  }
  ```

In `display_upload()`:
- Replace the entire function body (CPU copy + tone-map + `glTexSubImage2D`)
  with the PBO path:
  1. Map PBO → `memcpy` from `d_display` → unmap PBO
  2. `glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)`
  3. `glTexSubImage2D(... , nullptr)`
  4. `glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)`
- MCP capture: if `capture_requested`, issue `krt.copy_to_host` from
  `d_display` to `d_display_capture` (async, no `.wait()`), signal
  completion separately.

**Step 3.3 — `src/frame_loop.h` changes.**

In `shutdown_sandbox()`:
- Remove `delete[] state.h_accum`.
- Add `state.krt.dealloc(state.d_display)`.
- Add `state.display_pbo.release()`.
- Add `state.krt.dealloc(state.d_display_capture)`.

### Phase 4: Buffer Lifecycle

**Step 4.1 — Scene switch (`switch_scene()` in controls panel).**

On scene switch, the existing code already resets accum buffers via
`krt.fill(d_accum_front, 0, ...)`. `d_display` should also be cleared:
```cpp
state.krt.fill(state.d_display, 0, state.pixel_count * 4);
```

**Step 4.2 — Backend switch (`switch_backend()` in controls panel).**

On backend switch:
1. `krt.dealloc(state.d_display); state.d_display = nullptr;`
2. `state.display_pbo.release();`
3. Reallocate after new queue is created:
   ```cpp
   state.d_display = state.krt.alloc_device<uint8_t>(state.pixel_count * 4);
   state.display_pbo.alloc(state.width, state.height);
   state.tone_map_queue = sycl::queue(*state.q);
   ```

**Step 4.3 — Window resize (`recreate_buffers()`).**

Resolution changes are handled by the same code path as backend switch:
free + realloc.

### Phase 5: MCP Viewport Capture

**Step 5.1 — Async capture buffer.**

When `capture_requested` is set:
1. If `d_display_capture` doesn't exist yet (first capture this session),
   allocate via `krt.alloc_host<uint8_t>(pixel_count * 4)` (pinned, USM).
2. Issue async copy: `krt.copy_to_host(d_display_capture, d_display, byte_size)`
   without `.wait()`.
3. Set `capture_pending = true`.

**Step 5.2 — Capture completion.**

On the next frame, before the tone-map overwrites `d_display`, check
`capture_pending`. If the async copy is complete (check via an event or
queue status), copy data to `captured_viewport` and notify the MCP waiter.

If the copy is not yet complete, skip the capture and try again next
frame (the agent waits with a timeout, so this is acceptable).

---

## 7. CUDA-OpenGL Interop (Optional Phase 3)

### 7.1. Motivation

On NVIDIA GPUs with the CUDA SYCL backend, `d_display` can be mapped
directly into the PBO via `cudaGraphicsGLRegisterBuffer`. This makes the
tone-map kernel write **directly into the PBO's GPU memory**, eliminating
both the `d_display` allocation and the `memcpy` in the PBO map/unmap step.

### 7.2. Implementation Sketch

```cpp
// In PBO (or a CUDA interop helper):

#ifdef SYCL_BACKEND_CUDA
#include <cuda_gl_interop.h>

struct CudaInteropPBO {
    PBO pbo;
    cudaGraphicsResource *cuda_resource = nullptr;

    void alloc(int w, int h) {
        pbo.alloc(w, h);
        cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo.id,
                                      cudaGraphicsRegisterFlagsWriteDiscard);
    }

    uint8_t *map_for_kernel() {
        // Returns a CUDA device pointer directly into the PBO.
        // No memcpy needed — the tone-map kernel writes here.
        // ...
    }

    void unmap_for_upload() {
        // Synchronise before glTexSubImage2D uses the PBO.
        // ...
    }
};
#endif
```

### 7.3. When to Implement

CUDA interop is deferred because:
1. It requires compile-time detection of the CUDA backend.
2. It adds a dependency on `cuda_gl_interop.h`.
3. The basic PBO path already eliminates the blocking readback — the
   `memcpy` from `d_display` to the PBO is a device→device copy (fast,
   ~0.1 ms at 1080p) rather than a PCIe round-trip.

Implement Phase 3 only if profiling shows the device→device copy in the
basic PBO path is a measurable bottleneck.

---

## 8. Fallback Paths

### 8.1. CPU (Software) Backend

When the software backend is active (`state.is_software == true`):
- No GPU is available, so `tone_map::apply()` would be a no-op.
- PBO path is also unavailable (no OpenGL context issues, but PBO is
  pointless without a GPU).
- Fall back to a **CPU tone-map** identical to the current code, but
  running in `display_upload()` on the main thread.

Detection: `state.is_software` is already set in `switch_backend()`.
Check it in `display_upload()` and branch accordingly.

### 8.2. PBO Not Supported

On systems where `GL_PIXEL_UNPACK_BUFFER` is not available (OpenGL < 2.1
or software rasterisers): fall back to the current `glTexSubImage2D` with
host data pointer. The GPU tone-map output is not available either, so use
the CPU tone-map path as in §8.1.

Detection: Check `glGenBuffers` was successfully resolved (all GL function
pointers are loaded at startup). If not present, set a `bool pbo_available`
flag.

---

## 9. Verification

### 9.1. Correctness

| Test | Procedure | Expected result |
|------|-----------|-----------------|
| Visual comparison | Render a scene with Path A, take a screenshot, compare pixel-by-pixel with the same scene rendered with the old CPU path | All pixels identical (use `cv::norm` or similar diff) |
| SPP independence | Test at SPP 1, 10, 100, target | Tone-map looks correct at all accumulation levels |
| Scene switch | Switch between 3D raytracer and Shadertoy kernels | Display pipeline works for all kernel types |
| Backend switch | Cycle GPU → CPU → GPU | Display remains correct, no memory errors |
| Window resize | Drag window to resize | PBO and d_display are reallocated, no stale data |

### 9.2. Performance

| Metric | Tool | Expected improvement |
|--------|------|---------------------|
| `display_upload()` duration | Tracy profiler (PROFILER_ZONE) | 2–5 ms → <0.5 ms |
| Per-frame PCIe traffic | `nvidia-smi dmon` or `ncu` | 16 MB/frame → 0 MB/frame |
| Frame time (GPU, heavy scenes) | Tracy | Reduction proportional to tone-map time |
| Frame time (CPU, light scenes) | Tracy | Main thread unblocked, smoother input |

### 9.3. Stability

| Test | Procedure |
|------|-----------|
| Stress test | Render 10,000 frames with scene switches every 100 frames |
| MCP capture | Request 100 consecutive viewport captures, verify no crashes |
| Memory leak | Run for 5 minutes, monitor `nvidia-smi` memory usage |

---

## 10. Performance Characteristics (Expected)

| Metric | Current (CPU path) | After Path A | Notes |
|--------|--------------------|--------------|-------|
| PCIe readback | 16.5 MB/frame (blocking) | 0 MB/frame | Eliminated |
| PCIe upload (texture) | 8 MB/frame (blocking) | 8 MB/frame (async DMA) | Same data, async |
| Tone-map CPU time | 0.5–1.5 ms (× 3 powf per pixel) | 0 ms (offloaded) | GPU does it in parallel |
| Frame time saved | — | 2–5 ms | Reclaimed for rendering |
| Additional GPU time | 0 | ~0.05–0.1 ms at 1080p | Trivial — memory-bound kernel on 1000s of cores |
| Main thread latency | Blocked on copy | Unblocked | Input/UI more responsive |

---

## 11. Error Codes and Diagnostics

| Error | Cause | Recovery |
|-------|-------|----------|
| `PBO::map_write() == nullptr` | OpenGL error or buffer misconfiguration | Log via `spdlog::error`, fall back to CPU path for this frame |
| `tone_map::apply()` SYCL exception | Device lost or out of memory | Log error, fall back to CPU path, do not crash |
| `d_display == nullptr` | Buffer not yet allocated (during init race) | Skip tone-map for this frame, retry next frame |

All fallback paths are per-frame and self-healing: if the error condition
resolves (e.g., PBO mapping succeeds next frame), the GPU path is used
again automatically.

---

## 12. Exclusions and Future Work

### Excluded from this document:

| Feature | Reason |
|---------|--------|
| CUDA-OpenGL interop (§7) | Optional Phase 3; measurable gain is small |
| HDR / scRGB / PQ output | Out of scope — no display pipeline change |
| Intel Level Zero interop | No API available in AdaptiveCpp yet |
| Multi-GPU | All compute and display on same device |
| Bypass SYCL tone-map entirely (CUDA-only) | Would break portability to Intel/AMD |
| Dynamic PBO double/triple buffering | Not needed — render thread serialises frames anyway |

### Future investigations:

- Whether `glMapBufferRange` with `GL_MAP_UNSYNCHRONIZED_BIT` gives
  better performance than `glMapBuffer` (avoids driver stall).
- Whether using `GL_PIXEL_UNPACK_BUFFER` + `GL_STREAM_DRAW` (vs.
  `GL_STREAM_COPY`) affects performance on Intel/AMD.
- Whether AdaptiveCpp's `buffer`/`accessor` API (vs. USM pointers)
  would allow direct mapping into the PBO without `memcpy`.
