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
)

find_package(spdlog REQUIRED)

target_link_libraries(sandbox PRIVATE
    AdaptiveCpp::acpp-rt
    imgui_helper
    args
    glfw
    glm::glm-header-only
    yaml-cpp::yaml-cpp
    spdlog::spdlog
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)
