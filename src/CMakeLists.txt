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
target_compile_definitions(imgui_core PUBLIC IMGUI_ENABLE_DOCKING)
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

# ---- sandbox executable ----
add_executable(sandbox
    main.cpp
    watcher.cpp
    kernel_library.cpp
    scene_registry.cpp
    param_ui.cpp
)

target_compile_options(sandbox PRIVATE -Werror -Wno-c99-designator)

target_include_directories(sandbox PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${imguizmo_SOURCE_DIR}/src
)

find_package(spdlog REQUIRED)

target_link_libraries(sandbox PRIVATE
    AdaptiveCpp::acpp-rt
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
)

if(TRACY_PROFILER)
    target_compile_definitions(sandbox PRIVATE TRACY_ENABLE)
    target_link_libraries(sandbox PRIVATE TracyClient)
endif()
