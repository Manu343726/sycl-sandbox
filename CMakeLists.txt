cmake_minimum_required(VERSION 3.20)
project(sycl-sandbox LANGUAGES C CXX)

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
# Set ACPP_TARGETS *before* find_package so AdaptiveCpp's CMake module picks it
# up and passes --acpp-targets to the acpp compiler.  If we set it afterward,
# ACPP_EXTRA_ARGS (which carries --acpp-targets) is already computed and the
# acpp compiler falls back to its own default.
set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp backend targets")
find_package(AdaptiveCpp CONFIG REQUIRED)

# ------------------------------------------------------------------ deps (conan-provided)
find_package(glm REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)
find_package(OpenGL REQUIRED)

# ------------------------------------------------------------------ targets
add_subdirectory(src)
add_subdirectory(kernels)
