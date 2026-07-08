cmake_minimum_required(VERSION 3.20)
project(sycl-sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------ ASan
option(ENABLE_ASAN "Build sandbox with AddressSanitizer" OFF)
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# ------------------------------------------------------------------ SYCL
find_package(AdaptiveCpp CONFIG REQUIRED)

# Default: compile kernels for CUDA + generic (CPU) fallback.
# Override with -DACPP_TARGETS="cuda:sm_89" for GPU-only.
if(NOT ACPP_TARGETS)
    set(ACPP_TARGETS "cuda:sm_89;generic" CACHE STRING "AdaptiveCpp backend targets")
endif()

# ------------------------------------------------------------------ deps (conan-provided)
find_package(glm REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(OpenGL REQUIRED)

# ------------------------------------------------------------------ targets
add_subdirectory(src)
add_subdirectory(kernels)
