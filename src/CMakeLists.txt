# ---- ImGui backends convenience library ----
# conan's imgui package is header-only (interface); we compile core + backends.
# get imgui's source directory from the conan-generated target's include path
get_target_property(_IMGUI_INCLUDE imgui::imgui INTERFACE_INCLUDE_DIRECTORIES)
# Pick the first matching include dir that contains imgui.h
set(IMGUI_DIR "")
foreach(_dir ${_IMGUI_INCLUDE})
    if(EXISTS "${_dir}/imgui.h")
        set(IMGUI_DIR "${_dir}")
        break()
    endif()
endforeach()
if(NOT IMGUI_DIR)
    message(FATAL_ERROR "Cannot find imgui.h from conan imgui package")
endif()

set(IMGUI_SRC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui_helper STATIC ${IMGUI_SRC})
target_include_directories(imgui_helper PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
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
    AdaptiveCpp::AdaptiveCpp
    imgui_helper
    glfw
    glm::glm-header-only
    yaml-cpp::yaml-cpp
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)
