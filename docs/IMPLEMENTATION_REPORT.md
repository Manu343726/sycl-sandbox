# Device-Resident Rendering Pipeline — Status Report

**Last updated:** 2026-07-21

This replaces an earlier version of this document that declared the redesign
"production ready" and "TSAN-clean." That was inaccurate — no TSAN run
backs that claim, and the GPU backend has a known, currently-unresolved
crash (see below). This version reflects the actual, verified state.

## What's actually done and verified

- **Device-resident render pipeline**: single persistent accumulation
  buffer (no ping-pong), GPU tone-mapping, triple-buffered `DisplayTarget`
  abstraction with a portable staging implementation and a CUDA-GL
  zero-copy implementation. All 13 kernels (raytracer, mandelbrot, minimal,
  and 10 shadertoy-style ports) use the `RenderContext` ABI and dispatch
  through `rt::Runtime::foreach_pixel`.
- **Three real bugs found and fixed during verification** (all confirmed via
  gdb backtraces and/or `journalctl -k` GPU fault decoding, not just
  inference):
  1. **Kernel-reload HCF collision** — `KernelRuntime::switch_scene()` called
     `KernelLibrary::load()` unconditionally, even when the new scene reused
     an already-active, unchanged kernel binary. Re-`dlopen()`ing a
     byte-identical `.so` made AdaptiveCpp's HCF cache log a collision and
     left kernel dispatch unreliable. Fixed via an `src_mtime` skip-reload
     guard in `src/kernel/library.{h,cpp}`.
  2. **UI use-after-free** — `render_controls_panel()` captured a
     `SceneDescriptor*` at function entry, but the Scene combo box later in
     the same function could call `switch_scene()`, which destroys and
     replaces that descriptor. Execution then fell through to
     `render_param_controls()` on the dangling pointer (confirmed via gdb:
     SIGSEGV inside `ImGui::PushID` on a garbage `pd.name.c_str()`). Fixed
     by re-fetching the descriptor right after the combo closes
     (`src/ui/controls/panel.h`).
  3. **Cross-kernel SYCL kernel-name mangling collision** — every kernel's
     device entry point is a function literally named `render_kernel`, and
     C++ lambda name mangling only encodes the enclosing function's name,
     not the shared library it came from. All "shadertoy-style" kernels'
     per-pixel lambdas mangled to the *identical* symbol across different
     `.so`s, so AdaptiveCpp's SSCP kernel cache treated unrelated kernels
     from different libraries as the same kernel — switching between two of
     them could launch one kernel's compiled binary against another's
     argument layout. Reproduced as GPU MMU faults (`journalctl`:
     `NVRM: Xid 31 ... MMU Fault`) with a different PID each time — a real,
     deterministic bug, not GPU flakiness. Fixed by adding an explicit SYCL
     kernel-name template parameter to `foreach_pixel`/`foreach_pixel_sync`
     and `rt::render_main`, with every kernel now passing a unique tag class
     (e.g. `foreach_pixel<class MandelbrotPixelKernel>(...)`). This is the
     fix that matches the original bug report ("crashing when switching
     between mandelbrot kernels and the others").
- **CPU/OpenMP backend (`--backend cpu`)**: runs cleanly for extended
  periods (20s+ repeated runs), including through kernel switching. This is
  the strongest evidence the three fixes above are correct, since it
  exercises the identical render-loop/kernel-dispatch code path as the GPU
  backend.
- **MCP server removed** (2026-07-21, explicit request): the embedded
  `gopher-mcp`-based HTTP/SSE server (`src/mcp/`) and all its integration
  points, the `gopher-mcp` dependency, and the `.vscode/mcp.json` entry for
  it have been deleted. This project no longer exposes any control surface
  beyond the GUI.

## What's not done / known issues

- **The GPU/CUDA backend has an unresolved crash on scene switch.** After
  fixing the three bugs above and confirming the CPU backend is clean, the
  CUDA backend still crashes reliably when switching scenes (including on a
  freshly rebooted machine, ruling out accumulated driver corruption from
  earlier testing). This has been root-caused further than "it crashes" but
  not yet fixed:
  - Reproduces 100% deterministically via a scripted gdb session that calls
    `switch_scene()` through the exact same synchronized path the UI uses —
    this rules out a UI-thread/render-thread race as the cause.
  - `journalctl -k` decodes the fault identically across runs, in both the
    CUDA-GL zero-copy display target and the plain staging fallback:
    `NVRM: Xid 31 ... MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_6 ...
    FAULT_PDE ACCESS_TYPE_VIRT_READ`. This is the GPU's texture/graphics
    engine faulting, not the compute engine — meaning the crash may not be
    inside the CUDA kernel dispatch itself.
  - Ruled out: `ACPP_ADAPTIVITY_LEVEL=0` (AdaptiveCpp JIT specialization
    disabled — still crashes), CUDA-GL interop specifically (staging mode,
    which never touches `cuGraphicsMapResources`, crashes with the same
    fault signature), accumulated GPU driver corruption (a full reboot
    didn't fix it), and the MCP-related race that was found and removed
    along with MCP itself.
  - Investigation is paused (at user's request, to avoid further stressing
    the live GPU) with the working hypothesis narrowed to the OpenGL
    texture-sampling path (`ImGui::Image` rendering the presented texture)
    rather than the CUDA compute kernels themselves, since the fault
    signature is identical whether or not CUDA-GL interop is involved.
- **Docs**: this file and `docs/ARCHITECTURE_REDESIGN.md` are being brought
  in line with reality; `ARCHITECTURE_REDESIGN.md` is a pre-implementation
  design document and should be read as a plan, not a status report.

## Verification methodology used

Every claim above is backed by one of: a gdb backtrace from a live crash or
core dump, a `journalctl -k` kernel-log GPU fault decode tied to a specific
PID, or a CPU-backend control run exercising the identical code path. No
claim here is based on code inspection alone without a corresponding
runtime observation.
