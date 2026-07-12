cmake_policy(SET CMP0169 OLD)

# ---- args (header-only CLI parser) ----
include(FetchContent)
FetchContent_Declare(args
    GIT_REPOSITORY  https://github.com/Taywee/args
    GIT_TAG         master
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   NONE
)
FetchContent_GetProperties(args)
if(NOT args_POPULATED)
    FetchContent_Populate(args)
endif()
add_library(args INTERFACE)
target_include_directories(args INTERFACE "${args_SOURCE_DIR}")

# ---- ImGui core + backends (with Docking support) ----
include(FetchContent)
FetchContent_Declare(imgui_src
    GIT_REPOSITORY  https://github.com/ocornut/imgui.git
    GIT_TAG         docking
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   NONE
)
FetchContent_GetProperties(imgui_src)
if(NOT imgui_src_POPULATED)
    FetchContent_Populate(imgui_src)
endif()

# Compile imgui core with Docking enabled
set(IMGUI_CORE_SRC
    "${imgui_src_SOURCE_DIR}/imgui.cpp"
    "${imgui_src_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_src_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_src_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_src_SOURCE_DIR}/imgui_widgets.cpp"
)
add_library(imgui_core STATIC ${IMGUI_CORE_SRC})
target_include_directories(imgui_core PUBLIC "${imgui_src_SOURCE_DIR}")
target_include_directories(imgui_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(imgui_core PUBLIC
    IMGUI_ENABLE_DOCKING
    IMGUI_ENABLE_TEST_ENGINE
)
target_compile_options(imgui_core PRIVATE -Wno-c99-designator)

# ImGui backends helper
set(IMGUI_BACKEND_SRC
    "${imgui_src_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_src_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
add_library(imgui_helper STATIC ${IMGUI_BACKEND_SRC})
target_include_directories(imgui_helper PUBLIC
    "${imgui_src_SOURCE_DIR}"
    "${imgui_src_SOURCE_DIR}/backends"
)
target_link_libraries(imgui_helper PUBLIC glfw imgui_core)

# ---- ImGuizmo (3D gizmo + view manipulation) ----
include(FetchContent)
FetchContent_Declare(imguizmo
    GIT_REPOSITORY  https://github.com/CedricGuillemet/ImGuizmo
    GIT_TAG         master
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   NONE
)
FetchContent_GetProperties(imguizmo)
if(NOT imguizmo_POPULATED)
    FetchContent_Populate(imguizmo)
endif()

set(IMGUIZMO_SRC "${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp")
add_library(imguizmo STATIC ${IMGUIZMO_SRC})
target_include_directories(imguizmo PUBLIC "${imguizmo_SOURCE_DIR}/src")
target_compile_definitions(imguizmo PRIVATE IMGUI_DEFINE_MATH_OPERATORS)
target_compile_options(imguizmo PRIVATE -Wno-c99-designator)
target_link_libraries(imguizmo PUBLIC imgui_core)

# ---- ImPlot3D (3D plotting in ImGui) ----
include(FetchContent)
FetchContent_Declare(implot3d
    GIT_REPOSITORY  https://github.com/brenocq/implot3d
    GIT_TAG         v0.4
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   NONE
)
FetchContent_GetProperties(implot3d)
if(NOT implot3d_POPULATED)
    FetchContent_Populate(implot3d)
endif()

set(IMPLOT3D_SRC
    "${implot3d_SOURCE_DIR}/implot3d.cpp"
    "${implot3d_SOURCE_DIR}/implot3d_items.cpp"
    "${implot3d_SOURCE_DIR}/implot3d_demo.cpp"
)
add_library(implot3d STATIC ${IMPLOT3D_SRC})
target_include_directories(implot3d PUBLIC "${implot3d_SOURCE_DIR}")
target_compile_options(implot3d PRIVATE -Wno-c99-designator)
target_link_libraries(implot3d PUBLIC imgui_core)

# ---- ImGui Test Engine (UI testing/automation) ----
include(FetchContent)
FetchContent_Declare(imgui_test_engine
    GIT_REPOSITORY  https://github.com/ocornut/imgui_test_engine.git
    GIT_TAG         main
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   NONE
)
FetchContent_GetProperties(imgui_test_engine)
if(NOT imgui_test_engine_POPULATED)
    FetchContent_Populate(imgui_test_engine)
endif()

set(IMGUI_TE_SRC
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_engine.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_context.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_coroutine.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_ui.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_utils.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_perftool.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_exporters.cpp"
    "${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_capture_tool.cpp"
)
add_library(imgui_test_engine STATIC ${IMGUI_TE_SRC})
target_include_directories(imgui_test_engine PUBLIC "${imgui_test_engine_SOURCE_DIR}")
target_compile_options(imgui_test_engine PRIVATE -Wno-c99-designator)
target_compile_definitions(imgui_test_engine PRIVATE
    IMGUI_TEST_ENGINE_ENABLE_COROUTINE_STDTHREAD_IMPL=1
)
target_link_libraries(imgui_test_engine PUBLIC imgui_core)

# ---- clangd config generator ------------------------------------------
# Generates ${CMAKE_SOURCE_DIR}/.clangd from the sycl-sandbox target's resolved
# include directories, compile definitions, and compile options — extracted
# from both the target itself and its transitive dependencies.
#
# This gives clangd the correct flags (-I, -isystem, -D) for headers in
# include/sycl-sandbox/ without hardcoding any Conan or system paths.
function(generate_clangd_config TARGET)
    set(_all_incs)
    set(_all_defs)
    set(_all_opts)

    # Direct includes / defs / options from the target itself
    get_target_property(_incs ${TARGET} INCLUDE_DIRECTORIES)
    if(_incs)
        list(APPEND _all_incs ${_incs})
    endif()
    get_target_property(_target_defs ${TARGET} COMPILE_DEFINITIONS)
    if(_target_defs)
        list(APPEND _all_defs ${_target_defs})
    endif()
    get_target_property(_target_opts ${TARGET} COMPILE_OPTIONS)
    if(_target_opts)
        list(APPEND _all_opts ${_target_opts})
    endif()

    # Walk linked libraries for transitive interface properties
    get_target_property(_link_libs ${TARGET} LINK_LIBRARIES)
    if(_link_libs)
        foreach(_lib IN LISTS _link_libs)
            if(TARGET ${_lib})
                get_target_property(_v ${_lib} INTERFACE_INCLUDE_DIRECTORIES)
                if(_v)
                    list(APPEND _all_incs ${_v})
                endif()
                get_target_property(_v ${_lib} INTERFACE_COMPILE_DEFINITIONS)
                if(_v)
                    list(APPEND _all_defs ${_v})
                endif()
                get_target_property(_v ${_lib} INTERFACE_COMPILE_OPTIONS)
                if(_v)
                    list(APPEND _all_opts ${_v})
                endif()
                # One level deeper for transitive deps of linked libraries
                get_target_property(_trans_libs ${_lib} INTERFACE_LINK_LIBRARIES)
                if(_trans_libs)
                    foreach(_tl IN LISTS _trans_libs)
                        if(TARGET ${_tl})
                            get_target_property(_v ${_tl} INTERFACE_INCLUDE_DIRECTORIES)
                            if(_v)
                                list(APPEND _all_incs ${_v})
                            endif()
                        endif()
                    endforeach()
                endif()
            endif()
        endforeach()
    endif()

    # Deduplicate and strip entries with generator expressions
    # (Conan targets use $<CONFIG:...> etc. which clangd cannot evaluate)
    if(_all_incs)
        list(REMOVE_DUPLICATES _all_incs)
        set(_clean)
        foreach(_e IN LISTS _all_incs)
            if(NOT _e MATCHES "\\$<")
                list(APPEND _clean "${_e}")
            endif()
        endforeach()
        set(_all_incs ${_clean})
    endif()
    if(_all_defs)
        list(REMOVE_DUPLICATES _all_defs)
        set(_clean)
        foreach(_e IN LISTS _all_defs)
            if(NOT _e MATCHES "\\$<")
                list(APPEND _clean "${_e}")
            endif()
        endforeach()
        set(_all_defs ${_clean})
    endif()
    if(_all_opts)
        list(REMOVE_DUPLICATES _all_opts)
        set(_clean)
        foreach(_e IN LISTS _all_opts)
            if(NOT _e MATCHES "\\$<")
                list(APPEND _clean "${_e}")
            endif()
        endforeach()
        set(_all_opts ${_clean})
    endif()

    # Separate system dirs (-isystem) from project dirs (-I).
    # Also unwrap simple generator expressions like $<$<CONFIG:Release>:/path>
    # that Conan targets commonly use.
    set(_project_incs)
    set(_system_incs)
    foreach(_dir IN LISTS _all_incs)
        if(_dir STREQUAL "")
            continue()
        endif()
        # Unwrap generator expressions:
        #   $<$<CONFIG:Release>:/path>   →  /path
        #   $<BUILD_INTERFACE:/path>     →  /path
        if(_dir MATCHES "^\\$<\\$<CONFIG:[^>]*>:(.*)>$")
            set(_dir "${CMAKE_MATCH_1}")
        elseif(_dir MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
            set(_dir "${CMAKE_MATCH_1}")
        endif()
        # Still has unresolved generator expression? Skip.
        if(_dir MATCHES "\\$<")
            continue()
        endif()
        # Skip /usr/include — clangd knows the system dirs
        if(_dir STREQUAL "/usr/include")
            continue()
        endif()
        if(_dir MATCHES "/\\.conan2/|^/usr/|^/usr/local/|^/opt/")
            list(APPEND _system_incs "${_dir}")
        else()
            list(APPEND _project_incs "${_dir}")
        endif()
    endforeach()

    # Write the .clangd YAML file
    set(_clangd_path "${CMAKE_SOURCE_DIR}/.clangd")
    file(WRITE "${_clangd_path}"
        "# Auto-generated by CMake — do not edit manually.\n"
        "# See src/CMakeLists.txt :: generate_clangd_config().\n"
        "# Re-run cmake configure to regenerate.\n"
        "\n"
        "CompileFlags:\n"
        "  CompilationDatabase: \"${CMAKE_BINARY_DIR}\"\n"
    )

    set(_add_lines)
    foreach(_dir IN LISTS _project_incs)
        list(APPEND _add_lines "-I${_dir}")
    endforeach()
    foreach(_dir IN LISTS _system_incs)
        list(APPEND _add_lines "-isystem" "${_dir}")
    endforeach()
    foreach(_d IN LISTS _all_defs)
        list(APPEND _add_lines "-D${_d}")
    endforeach()
    list(APPEND _add_lines ${_all_opts})

    # Strip trailing '>' from partially-unwrapped generator expressions
    set(_add_clean)
    foreach(_line IN LISTS _add_lines)
        if(_line STREQUAL "")
            continue()
        endif()
        string(REGEX REPLACE ">$" "" _line "${_line}")
        list(APPEND _add_clean "${_line}")
    endforeach()
    set(_add_lines ${_add_clean})

    # NOTE: We intentionally do NOT emit CompileFlags.Add here.
    # The compilation database already contains the complete flags from the
    # sycl-sandbox target (via CMake's compile_commands.json). Adding them
    # again via .clangd would duplicate flags in every preamble command,
    # wasting memory for no benefit.
    #
    # clangd walks the #include graph to find the owning .cpp file for each
    # header, so the correct flags are always inferred.

    file(APPEND "${_clangd_path}"
        "\n"
        "Index:\n"
        "  Background: Skip\n"
        "  External: None\n"
        "  StandardLibrary: false\n"
        "\n"
        "If:\n"
        "  PathExclude:\n"
        "    - build/\n"
        "    - build_debug/\n"
        "    - conan/\n"
        "    - profiles/\n"
        "    - scenes/\n"
    )

    list(LENGTH _all_incs _num)
    message(STATUS "[clangd] Generated ${_clangd_path} with ${_num} include directories")
endfunction()

# ---- Tone-map library (the app's only SYCL device code) ----
# The executable is a plain GCC build; SYCL kernels need the acpp compiler
# pass, so the tone-map lives in its own add_sycl_to_target library — the
# same split the hot-reloadable kernel .so's use.
add_library(sandbox_tonemap SHARED render/tonemap_kernel.cpp)
add_sycl_to_target(TARGET sandbox_tonemap SOURCES render/tonemap_kernel.cpp)
target_include_directories(sandbox_tonemap PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ---- sycl-sandbox executable (with integrated test engine) ----
add_executable(sycl-sandbox
    main.cpp
    io/watcher.cpp
    kernel/library.cpp
    kernel/build_system.cpp
    kernel/runtime.cpp
    scene/registry.cpp
    scene/loader.cpp
    ui/params/controls.cpp
    ui/scene_debug/panel.cpp
    display/factory.cpp
    display/cuda_api.cpp
    display/cuda_gl_target.cpp
)

target_compile_options(sycl-sandbox PRIVATE -Werror -Wno-c99-designator)

target_include_directories(sycl-sandbox PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${imguizmo_SOURCE_DIR}/src
)

find_package(spdlog REQUIRED)

target_link_libraries(sycl-sandbox PRIVATE
    AdaptiveCpp::acpp-rt
    sandbox_tonemap
    imgui_core
    imgui_helper
    imguizmo
    implot3d
    args
    glfw
    glm::glm-header-only
    yaml-cpp::yaml-cpp
    spdlog::spdlog
    OpenGL::GL
    ${CMAKE_DL_LIBS}
    imgui_test_engine
)

# Generate .clangd config from the fully-resolved sycl-sandbox target properties
generate_clangd_config(sycl-sandbox)
