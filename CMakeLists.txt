cmake_minimum_required(VERSION 3.20)
project(sycl-sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------ SYCL
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
