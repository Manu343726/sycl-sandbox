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
        # adaptivecpp/25.10.0 is provided as a local recipe in conan/adaptivecpp/
        # Build it with: conan create conan/adaptivecpp -pr profiles/gcc
        # It requires system LLVM 22 and CUDA at /opt/cuda. Build takes ~45 min.
        # Uncomment and pre-build with conan create once built:
        # "adaptivecpp/25.10.0",
    ]

    tool_requires = [
        "cmake/3.31.12",
        "ninja/1.12.1",
    ]

    generators = "CMakeToolchain", "CMakeDeps"
