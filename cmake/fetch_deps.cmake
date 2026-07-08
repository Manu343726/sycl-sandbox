include(FetchContent)

# ---- glm ----
FetchContent_Declare(glm
    GIT_REPOSITORY  https://github.com/g-truc/glm.git
    GIT_TAG         tags/1.0.1
    GIT_SHALLOW     TRUE
)
set(glm_BUILD_TESTS OFF CACHE BOOL "")

# ---- yaml-cpp ----
# Pin to a post-0.8.0 commit that compiles with C++20 (fixes missing <cstdint>).
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY  https://github.com/jbeder/yaml-cpp.git
    GIT_TAG         master
    GIT_SHALLOW     TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "")

# ---- GLFW ----
FetchContent_Declare(glfw
    GIT_REPOSITORY  https://github.com/glfw/glfw.git
    GIT_TAG         tags/3.4
    GIT_SHALLOW     TRUE
)
set(GLFW_BUILD_DOCS  OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")

# ---- Dear ImGui ----
FetchContent_Declare(imgui
    GIT_REPOSITORY  https://github.com/ocornut/imgui.git
    GIT_TAG         tags/v1.92.8
    GIT_SHALLOW     TRUE
)

FetchContent_MakeAvailable(glm yaml-cpp glfw imgui)

# Restore default policy
set(CMAKE_POLICY_VERSION_MINIMUM)
