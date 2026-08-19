# AI context for this repository

Before making changes, read the following files in order:

1. **README.md**: Project overview, build instructions, and usage.
2. **docs/architecture.md**: Architecture — new features always follow this.
3. **docs/raytracing.md**: Raytracing lib, variant dispatch, scene building, primitives.
4. **docs/coding-guidelines.md**: Coding style, workflow rules.

After finishing implementation, go back and re-read the docs to verify guidelines are followed.

## Recent fixes applied (Chat session 2025-07-18)

### 1. CUDA error 700 (illegal memory access)
- **Root cause:** Profiler global variables (`g_records`, `g_write_pos`, etc.) in kernel `.so` data segment accessed from GPU threads inside SYCL `parallel_for` lambdas.
- **Fix:** `profiler.h` — wrapped all kernel-side profiler functions with `__SYCL_DEVICE_ONLY__` guards. When compiled for SYCL device code, profiler becomes no-op.
- **Files:** `include/sycl-sandbox/profiler.h`

### 2. CMake build driver
- **Changes:** `build_system.cpp` — rewritten with regex-based error/warning/progress parsing and spdlog routing. Added `build_sync()` for ordered shutdown/startup sequences.
- **Files:** `src/kernel/build_system.cpp`, `src/kernel/build_system.h`

### 3. Auto-build when binary missing
- **Changes:** `switch_scene()` and `switch_backend()` now check `so_exists()` and `is_up_to_date()` and trigger `build_sync()` if needed.
- **Files:** `src/ui/controls/panel.h`

### 4. Ordered backend/scene switching
- **Changes:** `switch_backend()` rewritten with explicit shutdown → rebuild → load → init sequence.
- **Files:** `src/ui/controls/panel.h`

### 5. `.d` file dependency tracking
- **Changes:** `SourceWatcher` parses CMake's compiler-generated dep files for dependency tracking. The target's `.dir` directory is located through CMake's `TargetDirectories.txt` index (CMake nests it according to the `add_library()` source location — kernel targets live in `<build>/kernels/CMakeFiles/<target>.dir`, not the flat `<build>/CMakeFiles/<target>.dir`). Dep files are scanned recursively and matched in either form — per-object `*.o.d` (older Makefile layout) or CMake's aggregated `compiler_depend.make` (modern CMake ≥3.20). Watches all resolved in-project headers (external/system files filtered by project root), re-parses dep files every 5s for new includes, and falls back to a recursive kernel source-dir watch. `KernelLibrary::is_up_to_date` mirrors the same recursive + dual-format scan for the startup/trigger rebuild check.
- **Files:** `src/io/watcher.h`, `src/io/watcher.cpp`, `src/kernel/library.cpp`

### 6. Tracy crash on UI connect (dangling string pointers)
- **Root cause:** `tracy_bridge` handed Tracy temporary `std::string` buffers and a cached compact srcloc id whose memory Tracy frees after transmission. When the Tracy server connects and queries those strings, the client's `HandleServerQuery()` dereferences the freed pointers (crash inside `strlen`). Three concrete defects:
  1. `___tracy_emit_messageL(zn.c_str())` — literal variant stores the raw pointer (no copy); server derefs later → UAF.
  2. `___tracy_emit_plot_float(zn.c_str())` — plots keyed by pointer identity on the server; `zn` was a temporary → dangling key.
  3. `___tracy_emit_gpu_zone_begin_serial({cached_srcloc,...})` — non-alloc variant expects a `SourceLocationData*` *struct*, but the cached id came from `___tracy_alloc_srcloc_name` (a *compact* block). The server misreads the compact block as a struct → garbage name/function/source pointers → `strlen` crash.
- **Fix:** `tracy_bridge.cpp` — DEV_MSG uses the copying `___tracy_emit_message(data, size, 0)`; DEV_PLOT emits via `plot_name_for()` (stable pointer owned by a `plot_names_` map, keyed by identity); GPU zones use `___tracy_emit_gpu_zone_begin_alloc_serial` with a **fresh** `make_srcloc()` per emit (the alloc variant frees the srcloc after transmission — never cache/reuse it). Per-helper locking on `mtx_`; `shutdown()` clears `plot_names_`.
- **Validation:** A/B harness (`src/tools/tracy_harness.cpp`, temporary — since removed) fed fabricated device records to `Bridge::submit_device_ring` under a real `tracy-profiler` connection: buggy build SIGSEGVs in `HandleServerQuery` → `strlen`; fixed build runs clean with `connected=1`.
- **Files:** `src/tracy/tracy_bridge.cpp`, `src/tracy/tracy_bridge.h`
- **Note:** Tracy server GUI needs an X display (e.g. `Xvfb :10`); client port overridden at runtime with `TRACY_PORT=8090`. `tracy-profiler` binary: `build/profiler/bin/tracy-profiler`.

### 7. Startup scene/kernel loading moved to the render thread
- **Changes:** `main.cpp` now starts the render thread BEFORE the first scene switch and posts the initial load through the same deferred `request_scene_switch()` protocol as runtime scene switches. The synchronous `KernelRuntime::switch_scene()` path and its `build_sync()` helper were REMOVED (`switch_scene()` was the only caller of `build_sync`). The viewport draws a loading overlay (`src/ui/viewport/panel.h`) while `kernel_ready` is false so the first (potentially multi-second) kernel build doesn't look like a hang.
- **Files:** `src/main.cpp`, `src/kernel/runtime.h`, `src/kernel/runtime.cpp`, `src/kernel/build_system.h`, `src/kernel/build_system.cpp`, `src/ui/viewport/panel.h`

## WIP — Memory system rewrite + profiler ring overhaul + diagnostic suite (2026-08-19)

### What was done (this session + prior)

#### 1. Memory system rewrite — `rt::MemoryPool` replaces per-kind allocators
- `include/sycl-sandbox/kernel/memory.h` (NEW): unified allocation registry.
  `MemoryPool` owns a registry of live allocations, per-kind byte counters
  and a drain-on-free protocol.  All kernel-usable buffers go through one
  pool.  `Buffer<T>` is the owning handle.  `MemoryKind` enum: Device, Host,
  HostImmutable, App (USM shared).  `MemoryPool::owns(ptr)` checks whether a
  pointer was allocated by the pool (used to guard stale-pointer access on
  backend switch).
- `include/sycl-sandbox/kernel/execution_context.h`: `Runtime::pool`
  replaces the old `device_memory_used`/`host_memory_used` atomics.  All
  alloc/dealloc/transfer methods now delegate to `pool->*`.  SYCL device
  pass gated with `__SYCL_DEVICE_ONLY__` (device code never allocates).
- Deleted: `include/sycl-sandbox/alloc/` (all 9 files) and
  `include/sycl-sandbox/containers/` (buffer.h, vector.h) — replaced by
  `memory.h`.

#### 2. Profiler ring overhaul — dynamic capacity, overflow tracking, interest-zone sampling
- `include/sycl-sandbox/profiler.h`: `DeviceRingHeader` gains `interest_pct`
  field; `DeviceRing` gains `sample_flags`/`sample_count` per-lane arrays,
  `dropped_lane(lid)` guard, and `full()` early-out (caps per-frame atomic
  traffic).  `push()` updated: `full()` check + `sample_flags` lane check.
- `src/app_state.h`: `ring_capacity` is now `std::atomic<uint32_t>` (default
  256K records = 4MB).  `init_profiler_buffers(rt, q, cap)` takes optional
  capacity param (0 = read from `ring_capacity`).  `ensure_profiler_sample_flags(n)`
  sizes the per-lane flags array.  `next_pow2()` utility.  `device_ring()`
  returns ring with live `sample_flags`/`sample_count`.
- `src/tracy/tracy_bridge.cpp` / `.h`: `submit_device_ring` signature
  simplified (no frame/timestamp args — timestamps now come from the bridge
  itself).  Added `last_write_pos_`, `last_overflow_`, `last_emitted_zones_`
  atomics for UI readback.  `zone_name_for()` hash→name cache with
  `zone_name_cache_`.  `make_srcloc` gains optional `function` param.
  `cumulative_records_` tracks record offset across frames for monotonic
  device-clock domain.
- `src/render_loop.h`: ring capacity read dynamically via `state.ring_capacity.load()`
  instead of old `RING_CAPACITY` constant.  `upload_profiler_pct()` called
  every frame.  Host timestamps (`host_t0`/`host_t1`) removed from render
  loop — bridge handles its own timing.
- `src/kernel/zone_names.h`: updated with `PROFILER_ZONE_IN` registry entry.

#### 3. UI ring-size selector — percent-of-memory slider
- `src/ui/controls/panel.h`: replaced fixed-option combo with log-scaled
  `ImGui::SliderFloat` (range 0.001%–100%).  Shows "Device ring: X% of
  Y GiB" + "= N records (N MiB)".  Resize applied on slider release via
  `request_ring_resize(target)`.  `ring_total_mem_bytes()` helper queries
  GPU `global_mem_size` or `sysinfo()` for system RAM.
- **Fix applied this session**: if `ring_total_mem_bytes()` returns 0
  (transient state), derive fallback `total_mem` from current capacity so
  the slider and resize logic always work.  Removed `has_total` guard from
  resize trigger — `target != cap` is now the only condition.

#### 4. Backend-switch UAF fix
- `include/sycl-sandbox/kernel/memory.h`: `MemoryPool::owns()` validates
  pointer ownership before freeing.
- `src/app_state.h`: `ensure_profiler_sample_flags` checks `pool->owns()`
  before touching stale buffer handles; `free_profiler_buffers` intentionally
  does NOT free `d_prof_sample_flags_` (survives ring resize; stale handle
  detected and re-allocated on next `ensure_profiler_sample_flags` call).
- `src/frame_loop.h`: backend-switch ack now calls `display_target->destroy()`
  before recreating on the new queue.

#### 5. Diagnostic suite
- `src/diags/profiler_diag.cpp` / `.h` (NEW): `diag profiler` — full
  profiler-system E2E diagnostic.  Drives the real kernel .so in-process
  with a live device ring + Tracy bridge, while a `tracy::Worker` captures
  over loopback.  Exercises all three backends (software, cpu, gpu),
  verifies GPU context + calibration, frame marks, GPU zones with name
  resolution, plots, messages, and pipeline integrity (bridge-emitted count
  ≈ worker-captured count).  Also measures per-frame record count and
  extrapolates ring size needed for 100% interest.
- `src/diags/tracy_diag.cpp` / `.h` (NEW): `diag tracy` — Tracy
  capture/receiver diagnostic.  Connects to a running sandbox instance via
  `tracy::Worker`, logs handshake/GPU contexts/zones/frames/plots/messages,
  can save raw `.tracy` trace.
- Both registered in `src/diags/diags.cpp`.  New files added to
  `src/CMakeLists.txt`.

#### 6. Misc
- `tools/zone_names_extractor/main.cpp`: updated with `PROFILER_ZONE_IN`.
- `scenes/mesh_demo.yaml`: minor adjustments.
- `docs/architecture.md`, `docs/raytracing.md`, `docs/coding-guidelines.md`:
  updated for memory system and profiler changes.
- `.clangd`, `.vscode/tasks.json`: tooling config updates.

### What remains (WIP)

#### A. `diag profiler` — zero GPU-zone capture (BLOCKED)
The diag runs end-to-end (connects, software frame OK, GPU context present +
calibrated, frame marks arrive, names resolve) but `gpu_zone_count=0` on the
GPU backend — the Worker sees the context and calibration but no zone records
arrive.  Root cause is NOT yet identified.  Key areas to investigate:
- `collect_gpu_zones` in `profiler_diag.cpp` — the `seen_zones` dedup set
  may be too aggressive, or the zone filter may exclude GPU zones.
- `tracy_bridge.cpp` `submit_device_ring` — GPU zone emission path may not
  be triggered for the specific record types the kernel produces.
- The kernel may not be entering `PROFILER_ZONE` regions on the GPU backend
  at all (check `sample_flags` / `interest_pct` state).

#### B. `diag profiler` — TEMP DEBUG instrumentation (TO REVERT)
- `src/frame_loop.h`: `[kernel-ready]` debug logging block (lines ~152-162)
  with `static uint64_t dbg_n` counter — REMOVE before release.
- `src/render_loop.h`: `spdlog::debug` lines for kernel dispatch/return
  timing — REMOVE before release.
- `src/diags/profiler_diag.cpp`: `PROFILER_DIAG_DEBUG` printf in capture
  thread — REMOVE before release.

#### C. Build verification
The build timed out during the last `make -j24 sycl-sandbox` (WSL2, RTX 5080,
AdaptiveCpp generic SSCP).  Build was clean before the panel.h slider fix.
Rebuild needed to confirm the fix compiles clean.

### Current thread model (UI vs render thread)

The UI thread NEVER blocks on the render pipeline. All gating device work
(scene switch, backend switch, resize, profiler toggle) runs on the render
thread via the deferred-op protocol:

- UI posts work with `AppState::post_cmd(closure)`; the render thread drains
  the mailbox at the top of its loop (`drain_cmds()`).
- Gating ops take `kernel_ready` down and bump `pending_device_ops`; the UI
  re-raises `kernel_ready` only when quiescent and the latest
  `scene_generation` has been processed.
- Render thread entry points live in `KernelRuntime` and are named
  `apply_*`: `apply_scene_switch`, `apply_backend_switch`, `apply_resize`,
  `apply_profiler_toggle`. These may block on a background build — that is
  allowed, the render thread is the one allowed to block.
- The render thread owns ALL device work, including the initial scene +
  kernel load: `main.cpp` starts the render thread BEFORE the first scene
  switch and posts the first load through the same `request_scene_switch()`
  deferred protocol used by runtime scene switches. The synchronous
  `KernelRuntime::switch_scene()` path is REMOVED — there is no
  pre-render-thread load anymore, do not reintroduce it.
  `reinit_kernel()` still runs on the main thread pre-startup via
  `recreate_buffers()`, but only touches non-gating state.
- The UI shows a loading overlay in the viewport (`src/ui/viewport/panel.h`)
  while `kernel_ready` is false (first build, scene/backend switch,
  profiler toggle), so the seconds-long first kernel build doesn't look
  like a hang.
- The old UI-thread `switch_backend()`/`set_profiler_enabled()`/
  `handle_kernel_rebuild()` are REMOVED — do not reintroduce them.
- Hot reload (`poll_hot_reload`) runs only on the render thread, at loop top.
- UI-side ack/coordination lives in `AppState` (`app_state.h`) and the
  frame-loop `process_frame_ops()` step (`src/frame_loop.h`). See
  `docs/architecture.md` and `docs/raytracing.md`.