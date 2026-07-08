# ---- ImGui backends helper ----
# conan imgui provides core as a prebuilt static library (imgui::imgui).
# Backend .cpp files are fetched from GitHub since the conan package doesn't compile them.
include(FetchContent)
FetchContent_Declare(imgui_backends
    GIT_REPOSITORY  https://github.com/ocornut/imgui.git
    GIT_TAG         tags/v1.92.8
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   NONE
)
FetchContent_GetProperties(imgui_backends)
if(NOT imgui_backends_POPULATED)
    FetchContent_Populate(imgui_backends)
endif()

set(IMGUI_SRC
    "${imgui_backends_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_backends_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
add_library(imgui_helper STATIC ${IMGUI_SRC})
target_include_directories(imgui_helper PUBLIC
    "${imgui_backends_SOURCE_DIR}"
    "${imgui_backends_SOURCE_DIR}/backends"
)
target_link_libraries(imgui_helper PUBLIC glfw imgui::imgui)

# ---- sandbox executable ----
add_executable(sandbox
    main.cpp
    watcher.cpp
    kernel_library.cpp
    scene_registry.cpp
    param_ui.cpp
)

target_include_directories(sandbox PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sandbox PRIVATE
    AdaptiveCpp::acpp-rt
    imgui_helper
    glfw
    glm::glm-header-only
    yaml-cpp::yaml-cpp
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)
