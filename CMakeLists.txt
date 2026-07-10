cmake_minimum_required(VERSION 3.20)
project(sycl-sandbox LANGUAGES CXX)

# ── CUDA libdevice symlink ──────────────────────────────────────────────
# AdaptiveCpp's SSCP JIT needs libdevice.10.bc to compile kernels to PTX.
# If missing, run:
#   sudo ln -sf /opt/cuda/nvvm/libdevice/libdevice.10.bc /opt/cuda/lib64/libdevice.10.bc
# ─────────────────────────────────────────────────────────────────────────

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------ ASan
option(ENABLE_ASAN "Build sandbox with AddressSanitizer" OFF)
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

# ------------------------------------------------------------------ SYCL
find_package(AdaptiveCpp CONFIG REQUIRED)

# Default: compile kernels for CUDA + generic (CPU) fallback.
# Override with -DACPP_TARGETS="cuda:sm_89" for GPU-only.
if(NOT ACPP_TARGETS)
    set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp backend targets")
endif()

# ------------------------------------------------------------------ deps (conan-provided)
find_package(glm REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(OpenGL REQUIRED)

# ------------------------------------------------------------------ Tracy profiler
option(TRACY_PROFILER "Enable Tracy profiler integration (CPU zones + GPU timeline)" OFF)

if(TRACY_PROFILER)
    include(FetchContent)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    set(TRACY_ENABLE  ON  CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(tracy)

    set_target_properties(TracyClient PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    add_subdirectory(profiler)
endif()

# ------------------------------------------------------------------ targets
add_subdirectory(src)
add_subdirectory(kernels)
