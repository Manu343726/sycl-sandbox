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
- **Changes:** `SourceWatcher` rewritten to parse `CMakeFiles/<target>.dir/*.d` files for dependency tracking. Watches all resolved headers, filters external files by project root, re-parses `.d` files every 5s for new includes. Falls back to recursive source dir watch.
- **Files:** `src/io/watcher.h`, `src/io/watcher.cpp`, `src/main.cpp`