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

# ------------------------------------------------------------------ Tracy profiler
# Tracy client + standalone profiler.  The sandbox links TracyClient and
# feeds kernel samples (device ring -> host -> Tracy) via the serial GPU-zone
# API (see src/tracy/tracy_bridge.* and docs/architecture.md).  The profiler
# UI is NOT embedded: it is Tracy's own standalone tracy-profiler GUI,
# built with Tracy's own CMake (add_subdirectory(profiler) below) and
# launched from the app's Controls panel (see src/tracy/tracy_launcher.*).
#
# Recording is on-demand: the client only captures data while a profiler is
# connected, and sampling / context-switch / crash-handler features are
# disabled — the sandbox only feeds explicit GPU zone timestamps.
option(SANDBOX_ENABLE_TRACY "Enable Tracy profiler integration (client + standalone profiler)" ON)
if(SANDBOX_ENABLE_TRACY)
    include(FetchContent)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        v0.13.1
        SOURCE_SUBDIR  NONE
    )
    FetchContent_GetProperties(tracy)
    if(NOT tracy_POPULATED)
        FetchContent_Populate(tracy)
    endif()

    # Client-side options — set before add_subdirectory so Tracy's own
    # option() calls pick them up (and so the PUBLIC compile definitions
    # propagate to every TU that links TracyClient).
    set(TRACY_ENABLE             ON  CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND          ON  CACHE BOOL "" FORCE)
    set(TRACY_ONLY_LOCALHOST     ON  CACHE BOOL "" FORCE)
    set(TRACY_NO_BROADCAST       ON  CACHE BOOL "" FORCE)
    set(TRACY_NO_SAMPLING        ON  CACHE BOOL "" FORCE)
    set(TRACY_NO_CONTEXT_SWITCH  ON  CACHE BOOL "" FORCE)
    set(TRACY_NO_CODE_TRANSFER   ON  CACHE BOOL "" FORCE)
    set(TRACY_NO_CRASH_HANDLER   ON  CACHE BOOL "" FORCE)
    # The profiler app's native file dialog (nfd) pulls in xdg-desktop-portal
    # deps (dbus) that are not present on every machine.  The embedded UI
    # does not need to open trace files, so disable it there as well.
    set(NO_FILESELECTOR ON CACHE BOOL "" FORCE)
    # The profiler app defaults to the Wayland backend on Linux, which
    # requires wayland-egl/wayland-cursor/xkbcommon dev packages.  Use the
    # X11 backend instead (x11 deps already provided by conan).
    set(LEGACY ON CACHE BOOL "" FORCE)

    # FetchContent_Populate with declared details requires CMP0169=OLD.
    # The terminal preset build uses cmake 3.28 (no CMP0169) while CMake
    # Tools uses 3.31 — guard so both can configure.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    add_subdirectory(${tracy_SOURCE_DIR} ${tracy_BINARY_DIR} EXCLUDE_FROM_ALL)
    # The standalone profiler app (also generates the embedded fonts /
    # manual / achievements data that the in-app UI links against).
    add_subdirectory(${tracy_SOURCE_DIR}/profiler ${tracy_BINARY_DIR}/profiler EXCLUDE_FROM_ALL)
    # Tracy's profiler CMake never turns NO_FILESELECTOR into a compile
    # definition (the sources gate nfd includes on TRACY_NO_FILESELECTOR),
    # so the standalone app would fail to find nfd.h.  Define it here.
    target_compile_definitions(tracy-profiler PRIVATE TRACY_NO_FILESELECTOR)
    # When NO_FILESELECTOR=ON, Tracy's vendor.cmake skips creating the
    # TracyNfd target, but the unconditional link in the profiler CMake
    # still references it.  Supply a dummy interface target.
    if(NO_FILESELECTOR AND NOT TARGET TracyNfd)
        add_library(TracyNfd INTERFACE)
    endif()
    # Match the launch path used by the tracy-server tasks in .vscode/tasks.json.
    set_target_properties(tracy-profiler PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/profiler/bin)
endif()

# ------------------------------------------------------------------ Profiler
# Master switch for profiler support.  When OFF, every PROFILER_* macro
# (include/sycl-sandbox/profiler.h) expands to nothing — no device ring,
# no Tracy zones, no plots — so kernels and host code carry zero
# profiling overhead.  The app can flip this at runtime for ON-THE-FLY
# kernel builds (Controls panel → "Enable profiler"): it reconfigures the
# CMake cache and rebuilds the active kernel .so with the new flag.  The
# app binary itself keeps whatever it was compiled with.
option(SANDBOX_ENABLE_PROFILER "Enable profiler support (device ring + Tracy). OFF makes all PROFILER_* macros no-ops for zero overhead." ON)
if(SANDBOX_ENABLE_PROFILER)
    # Directory property — inherited by src/ and kernels/ subdirectories,
    # so the app, the tone-map SYCL library and every kernel .so compile
    # with profiling enabled.
    add_compile_definitions(SANDBOX_ENABLE_PROFILER)
endif()

# ------------------------------------------------------------------ targets
add_subdirectory(src)
add_subdirectory(kernels)

# ------------------------------------------------------------------ zone-name extraction (libclang)
# Device-side profiler records carry only a hash of the zone name; the
# app resolves hashes → names through a registry.  This generator parses
# the kernel sources with libclang (exact AST extraction of
# PROFILER_ZONE literals) and emits the name table into a header
# the app includes.  When libclang is unavailable the app falls back to
# runtime source scanning (zone_names.h) — the generated header is
# guarded by SANDBOX_HAVE_GENERATED_ZONE_NAMES.  Only meaningful when
# profiler support is enabled (no-op macros produce no names to extract).
option(SANDBOX_ENABLE_ZONE_EXTRACTOR "Build the libclang-based zone-name extractor" ON)
if(SANDBOX_ENABLE_PROFILER AND SANDBOX_ENABLE_ZONE_EXTRACTOR)
    find_library(CLANG_LIBRARY NAMES clang libclang
        PATHS /usr/lib/llvm-18/lib /usr/lib/llvm-17/lib /usr/lib/llvm-16/lib
              /usr/lib/llvm-15/lib /usr/lib/llvm-14/lib)
    find_path(CLANG_INCLUDE_DIR clang-c/Index.h
        PATHS /usr/lib/llvm-18/include /usr/lib/llvm-17/include
              /usr/lib/llvm-16/include /usr/lib/llvm-15/include
              /usr/lib/llvm-14/include)
endif()
if(CLANG_LIBRARY AND CLANG_INCLUDE_DIR)
    add_executable(zone_names_extractor
        tools/zone_names_extractor/main.cpp)
    target_include_directories(zone_names_extractor PRIVATE ${CLANG_INCLUDE_DIR})
    target_link_libraries(zone_names_extractor PRIVATE ${CLANG_LIBRARY})

    file(GLOB_RECURSE SANDBOX_KERNEL_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*/*.cpp")
    file(GLOB_RECURSE SANDBOX_SHARED_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
    # Host source files that define or include profiler macros
    # (PROFILER_ZONE / PLOT / MSG / DEVICE_ZONE / FUNCTION).  The
    # extractor parses them as standalone TUs with -DKERNEL_BUILD
    # -DKERNEL_NATIVE; lexer-level macro-expansion tracking survives
    # semantic errors, so TUs that fail to find GLFW/ImGui/etc still
    # contribute their profiler-name expansions.
    file(GLOB_RECURSE SANDBOX_HOST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
    set(SANDBOX_ZONE_EXTRACTOR_SOURCES
        ${SANDBOX_KERNEL_SOURCES}
        ${SANDBOX_HOST_SOURCES})

    set(SANDBOX_ZONE_NAMES_H
        "${CMAKE_CURRENT_BINARY_DIR}/generated/zone_names.generated.h")
    add_custom_command(
        OUTPUT ${SANDBOX_ZONE_NAMES_H}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/zone_names_extractor
                --out ${SANDBOX_ZONE_NAMES_H}
                --include "${CMAKE_CURRENT_SOURCE_DIR}/include"
                -- ${SANDBOX_ZONE_EXTRACTOR_SOURCES}
        DEPENDS zone_names_extractor ${SANDBOX_ZONE_EXTRACTOR_SOURCES}
                ${SANDBOX_SHARED_HEADERS}
        COMMENT "Extracting profiler-macro names (libclang)"
        VERBATIM)
    add_custom_target(zone_names_generate DEPENDS ${SANDBOX_ZONE_NAMES_H})
    add_dependencies(sycl-sandbox zone_names_generate)
    target_include_directories(sycl-sandbox PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(sycl-sandbox PRIVATE
        SANDBOX_HAVE_GENERATED_ZONE_NAMES=1)
else()
    message(STATUS "libclang not found — zone names fall back to runtime source scanning")
endif()
