# ── Tracy Profiler Server (standalone UI) ───────────────────────────────────
#
# Builds the Tracy profiler executable as an isolated ExternalProject so its
# complex dependency tree (capstone, freetype, imgui, libcurl, …) does not
# interfere with the main sycl-sandbox build.
#
# Requirements
# ────────────
# The Conan profile must install the Tracy server dependencies so that
# Tracy's vendor.cmake finds them via pkg-config instead of downloading
# them again with CPM.  See conanfile.py for the full list.
#
# Usage
# ─────
#   cmake --build build --target tracy-server
#
# The binary lands in  build/profiler/bin/tracy-profiler.

include(ExternalProject)
include(ProcessorCount)

# ── Locate the Tracy source fetched by FetchContent in the root CMakeLists.txt.
#    FetchContent_Populate stores the source dir in <name>_SOURCE_DIR.
if(NOT tracy_SOURCE_DIR)
    message(FATAL_ERROR
        "tracy_SOURCE_DIR is not set.  Make sure FetchContent_Declare(tracy ...) "
        "runs before add_subdirectory(profiler).")
endif()

set(TRACY_PROFILER_BUILD_DIR "${CMAKE_BINARY_DIR}/profiler/build")
set(TRACY_PROFILER_INSTALL_DIR "${CMAKE_BINARY_DIR}/profiler/bin")

# ── Parallel build jobs ──────────────────────────────────────────────────────
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 4)
endif()

# ── Ensure pkg-config can find Conan packages ────────────────────────────────
# PkgConfigDeps writes .pc files into the generators folder.  We propagate
# that path so Tracy's vendor.cmake can resolve capstone, freetype, curl, etc.
set(_PC_PATH "")
if(DEFINED ENV{PKG_CONFIG_PATH})
    set(_PC_PATH "$ENV{PKG_CONFIG_PATH}")
endif()

foreach(_guess
    "${CMAKE_BINARY_DIR}/generators"
    "${CMAKE_BINARY_DIR}"
)
    if(EXISTS "${_guess}")
        if(_PC_PATH)
            set(_PC_PATH "${_guess}:${_PC_PATH}")
        else()
            set(_PC_PATH "${_guess}")
        endif()
    endif()
endforeach()

# ── ExternalProject ──────────────────────────────────────────────────────────
ExternalProject_Add(tracy_server
    SOURCE_DIR "${tracy_SOURCE_DIR}/profiler"
    BINARY_DIR "${TRACY_PROFILER_BUILD_DIR}"
    INSTALL_DIR "${TRACY_PROFILER_INSTALL_DIR}"

    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
        -DPKG_CONFIG_PATH=${_PC_PATH}
        # Use X11/GLFW backend (avoids Wayland deps)
        -DLEGACY=ON
        # Prevent CPM from re-downloading packages available via Conan
        -DDOWNLOAD_CAPSTONE=OFF
        -DDOWNLOAD_FREETYPE=OFF
        -DDOWNLOAD_GLFW=OFF
        -DDOWNLOAD_LIBCURL=OFF
        -DDOWNLOAD_PUGIXML=OFF
        # Tracy options
        -DTRACY_STATIC=OFF
        -DNO_FILESELECTOR=ON
        -DCMAKE_CXX_FLAGS=-DTRACY_NO_FILESELECTOR
        # Pass through compiler settings
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}

    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release -j${NPROC}
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config Release

    # Don't rebuild when the main project rebuilds
    BUILD_ALWAYS OFF
)

# ── Convenience alias so "cmake --build ... --target tracy-server" works ────
add_custom_target(tracy-server DEPENDS tracy_server
    COMMENT "Tracy profiler server -> ${TRACY_PROFILER_INSTALL_DIR}/tracy-profiler")
