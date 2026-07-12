from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.files import apply_conandata_patches, get, copy, rmdir
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
import os

required_conan_version = ">=2.0"

class GopherMCPConan(ConanFile):
    name = "gopher-mcp"
    version = "0.1.1"
    description = "Model Context Protocol C++ SDK"
    license = "Apache-2.0"
    url = "https://github.com/GopherSecurity/gopher-mcp"
    homepage = "https://github.com/GopherSecurity/gopher-mcp"
    topics = ("mcp", "sdk", "ai", "model-context-protocol", "cpp")
    exports_sources = "patches/*"
    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        check_min_cppstd(self, "14")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("libevent/[>=2.1.12 <3]")
        self.requires("openssl/[>=3.0 <4]")
        self.requires("fmt/[>=10.0.0 <13]")
        self.requires("nlohmann_json/[>=3.11.0 <4]")
        self.requires("yaml-cpp/[>=0.8.0 <1]")
        self.requires("llhttp/[>=9.1.0 <10]")
        self.requires("nghttp2/[>=1.58.0 <2]")

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)
        apply_conandata_patches(self)
        config_dir = os.path.join(self.source_folder, "examples", "configs")
        os.makedirs(config_dir, exist_ok=True)
        for f in ["mcp_server_example.json", "mcp_server_config_example.json"]:
            p = os.path.join(config_dir, f)
            if not os.path.exists(p):
                with open(p, "w") as fh:
                    fh.write("{}\n")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.variables["BUILD_STATIC_LIBS"] = not self.options.shared
        tc.variables["BUILD_C_API"] = False
        tc.variables["BUILD_TESTS"] = False
        tc.variables["GOPHER_MCP_INSTALL"] = True
        tc.variables["BUILD_BINDINGS_EXAMPLES"] = False
        tc.variables["MCP_USE_STD_TYPES"] = True
        tc.variables["MCP_USE_LLHTTP"] = True
        tc.variables["MCP_USE_NGHTTP2"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.includedirs = ["include/gopher-mcp"]
        self.cpp_info.libs = ["gopher-mcp"]

        self.cpp_info.components["gopher-mcp-logging"].includedirs = ["include/gopher-mcp"]
        self.cpp_info.components["gopher-mcp-logging"].libs = ["gopher-mcp-logging"]
        self.cpp_info.components["gopher-mcp-logging"].requires = ["fmt::fmt"]

        self.cpp_info.components["gopher-mcp-event"].includedirs = ["include/gopher-mcp"]
        self.cpp_info.components["gopher-mcp-event"].libs = ["gopher-mcp-event"]
        self.cpp_info.components["gopher-mcp-event"].requires = ["libevent::libevent"]

        self.cpp_info.components["gopher-mcp-echo-advanced"].includedirs = ["include/gopher-mcp"]
        self.cpp_info.components["gopher-mcp-echo-advanced"].libs = ["gopher-mcp-echo-advanced"]
        self.cpp_info.components["gopher-mcp-echo-advanced"].requires = ["gopher-mcp-event"]

        self.cpp_info.components["gopher-mcp"].libs = ["gopher-mcp"]
        self.cpp_info.components["gopher-mcp"].includedirs = ["include/gopher-mcp"]
        self.cpp_info.components["gopher-mcp"].requires = [
            "gopher-mcp-logging",
            "gopher-mcp-event",
            "gopher-mcp-echo-advanced",
            "openssl::openssl",
            "nlohmann_json::nlohmann_json",
            "yaml-cpp::yaml-cpp",
            "llhttp::llhttp",
            "nghttp2::nghttp2",
        ]

        if self.settings.os == "Linux":
            self.cpp_info.system_libs = ["pthread", "dl", "rt", "c"]
