# ---- ImGui convenience library ----
set(IMGUI_DIR  ${imgui_SOURCE_DIR})
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
target_link_libraries(imgui_helper PUBLIC glfw)

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

target_compile_definitions(sandbox PRIVATE SYCL_SANDBOX_HAVE_SYCL)

target_link_libraries(sandbox PRIVATE
    AdaptiveCpp::AdaptiveCpp
    imgui_helper
    glfw
    glm::glm
    yaml-cpp
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)
