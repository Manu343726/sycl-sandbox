from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import get, copy, apply_conandata_patches
from conan.tools.build import check_min_cppstd
import os

class AdaptiveCppConan(ConanFile):
    name = "adaptivecpp"
    version = "25.10.0"
    description = "SYCL implementation for CPUs and GPUs"
    license = "BSD-2-Clause"
    url = "https://github.com/AdaptiveCpp/AdaptiveCpp"
    package_type = "library"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
        "with_openmp": [True, False],
        "experimental_llvm": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "with_cuda": True,
        "with_openmp": True,
        "experimental_llvm": True,
    }

    def export(self):
        copy(self, "patches/*.patch", src=self.recipe_folder, dst=self.export_folder)

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        check_min_cppstd(self, "17")

    def layout(self):
        cmake_layout(self)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)
        apply_conandata_patches(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["WITH_CUDA_BACKEND"] = self.options.with_cuda
        tc.variables["WITH_ROCM_BACKEND"] = False
        tc.variables["WITH_LEVEL_ZERO_BACKEND"] = False
        tc.variables["WITH_OPENMP_BACKEND"] = self.options.with_openmp
        tc.variables["ACPP_EXPERIMENTAL_LLVM"] = self.options.experimental_llvm

        if self.options.with_cuda:
            tc.variables["CUDA_TOOLKIT_ROOT_DIR"] = "/opt/cuda"
            tc.variables["CUDA_DEVICE_LIBS_PATH"] = "/opt/cuda/lib64"

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
        self.cpp_info.set_property("cmake_file_name", "AdaptiveCpp")
        self.cpp_info.set_property("cmake_target_name", "AdaptiveCpp::acpp-rt")
        self.cpp_info.libs = ["acpp-rt"]
        self.cpp_info.system_libs = ["dl", "rt", "pthread"]
        if self.settings.os == "Linux":
            self.cpp_info.system_libs.append("numa")
