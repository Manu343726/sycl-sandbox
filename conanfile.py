from conan import ConanFile

class SyclSandboxConan(ConanFile):
    name = "sycl-sandbox"
    version = "0.1.0"

    settings = "os", "compiler", "build_type", "arch"

    requires = [
        "glm/1.0.1",
        "yaml-cpp/0.8.0",
        "glfw/3.4",
        "imgui/1.92.8",
        "adaptivecpp/25.10.0",
        "spdlog/1.17.0",
    ]

    tool_requires = [
        "cmake/3.31.12",
        "ninja/1.12.1",
    ]

    generators = "CMakeToolchain", "CMakeDeps"
