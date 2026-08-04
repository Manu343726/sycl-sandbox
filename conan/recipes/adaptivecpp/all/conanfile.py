from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import apply_conandata_patches, get, copy, replace_in_file
from conan.tools.build import check_min_cppstd
import os

class AdaptiveCppConan(ConanFile):
    name = "adaptivecpp"
    version = "25.10.0"
    description = "SYCL implementation for CPUs and GPUs"
    license = "BSD-2-Clause"
    url = "https://github.com/AdaptiveCpp/AdaptiveCpp"
    exports_sources = "patches/*"
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

    settings = "os", "arch", "compiler", "build_type"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        check_min_cppstd(self, "17")

    def layout(self):
        cmake_layout(self)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

        cmake_lists = os.path.join(self.source_folder, "CMakeLists.txt")
        banner = (
            "cmake_minimum_required(VERSION 3.10)\n\n"
            "message(STATUS \"============================================================\")\n"
            "message(STATUS \"NOTE: This is a patched AdaptiveCpp build from sycl-sandbox\")\n"
            "message(STATUS \"This build includes configurable log output via output_stream::set_stream(std::ostream&)\")\n"
            "message(STATUS \"Patch: conan/recipes/adaptivecpp/all/patches/set_stream_logging.patch\")\n"
            "message(STATUS \"============================================================\")\n"
        )
        replace_in_file(self, cmake_lists, "cmake_minimum_required(VERSION 3.10)\n", banner)

        apply_conandata_patches(self)
        # Apply LLVM 22 compat patch inline
        for root, _, files in os.walk("."):
            for f in files:
                path = os.path.join(root, f)
                if f in ("AdaptiveCppLlvmPasses.cpp", "Emitter.cpp"):
                    with open(path) as fh:
                        content = fh.read()
                    content = content.replace(
                        '#include "llvm/Passes/PassPlugin.h"',
                        '#if LLVM_VERSION_MAJOR >= 22\n#include "llvm/Plugins/PassPlugin.h"\n#else\n#include "llvm/Passes/PassPlugin.h"\n#endif'
                    )
                    with open(path, "w") as fh:
                        fh.write(content)

    def generate(self):
        tc = CMakeToolchain(self)
        llvm_root = os.environ.get(
            "ACPP_LLVM_ROOT", "/usr/lib/llvm-22")
        llvm_dir = os.path.join(llvm_root, "lib", "cmake", "llvm")
        if os.path.isdir(llvm_dir):
            tc.variables["LLVM_ROOT"] = llvm_root
            tc.variables["LLVM_DIR"] = llvm_dir
        tc.variables["WITH_CUDA_BACKEND"] = self.options.with_cuda
        tc.variables["WITH_ROCM_BACKEND"] = False
        tc.variables["WITH_LEVEL_ZERO_BACKEND"] = False
        tc.variables["WITH_OPENMP_BACKEND"] = self.options.with_openmp
        tc.variables["ACPP_EXPERIMENTAL_LLVM"] = self.options.experimental_llvm

        if self.options.with_cuda:
            cuda_root = self._cuda_toolkit_root()
            if cuda_root is None:
                raise ConanException(
                    "with_cuda=True but no CUDA toolkit found. Set CUDA_HOME or install the "
                    "CUDA toolkit, or build with '-o adaptivecpp/*:with_cuda=False'."
                )
            tc.variables["CUDA_TOOLKIT_ROOT_DIR"] = cuda_root
            device_libs = os.path.join(cuda_root, "nvvm", "libdevice")
            if not os.path.isdir(device_libs):
                device_libs = os.path.join(cuda_root, "lib64")
            tc.variables["CUDA_DEVICE_LIBS_PATH"] = device_libs

        tc.generate()

    @staticmethod
    def _cuda_toolkit_root():
        for candidate in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                          "/usr/local/cuda", "/opt/cuda"):
            if candidate and os.path.isdir(os.path.join(candidate, "bin")):
                return candidate
        return None

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package_info(self):
        # Use the original adaptivecpp-config.cmake (with add_sycl_to_target)
        # instead of letting CMakeDeps generate a thin wrapper.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = [os.path.join("lib", "cmake", "AdaptiveCpp")]
        self.cpp_info.libs = ["acpp-rt"]
        self.cpp_info.system_libs = ["dl", "rt", "pthread"]
        if self.settings.os == "Linux":
            self.cpp_info.system_libs.append("numa")

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))
        # CMakeDeps generates a thin wrapper; the original config (with add_sycl_to_target)
        # lives in lib/cmake/AdaptiveCpp/. Expose it via builddirs above.
