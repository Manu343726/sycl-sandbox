from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import get, copy
import os

required_conan_version = ">=2.0"

class Nghttp2Conan(ConanFile):
    name = "nghttp2"
    version = "1.58.0"
    description = "HTTP/2 C library"
    license = "MIT"
    url = "https://github.com/nghttp2/nghttp2"
    topics = ("http2", "c", "library")
    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)
        cmake = os.path.join(self.source_folder, "CMakeLists.txt")
        import conan.tools.files
        conan.tools.files.replace_in_file(self, cmake, "cmake_minimum_required(VERSION 3.0)", "cmake_minimum_required(VERSION 3.5)")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["ENABLE_LIB_ONLY"] = True
        tc.variables["ENABLE_STATIC_LIB"] = not self.options.shared
        tc.variables["ENABLE_SHARED_LIB"] = self.options.shared
        tc.variables["ENABLE_APP"] = False
        tc.variables["ENABLE_HPACK_TOOLS"] = False
        tc.variables["ENABLE_EXAMPLES"] = False
        tc.variables["ENABLE_PYTHON_BINDINGS"] = False
        tc.variables["ENABLE_FAILMALLOC"] = False
        tc.variables["ENABLE_THREADS"] = True
        tc.variables["ENABLE_WERROR"] = False
        tc.variables["ENABLE_DEBUG"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "nghttp2")
        self.cpp_info.set_property("cmake_target_name", "nghttp2::nghttp2")
        self.cpp_info.libs = ["nghttp2"]
        if self.settings.os == "Linux":
            self.cpp_info.system_libs = ["pthread", "c"]
