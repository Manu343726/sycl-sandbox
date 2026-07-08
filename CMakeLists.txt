cmake_minimum_required(VERSION 3.20)
project(sycl-sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------ SYCL
find_package(AdaptiveCpp CONFIG REQUIRED)

# ------------------------------------------------------------------ deps
include(cmake/fetch_deps.cmake)

# ------------------------------------------------------------------ targets
add_subdirectory(src)
add_subdirectory(kernels)
