# known kernel subdirectories — add new ones here
set(KERNEL_DIRS minimal raytracer rt_math_test rt_portal_test mandelbrot
    raymarching_primitives seascape creation cyber_fuji lover
    octagrams fractal_pyramid rainforest cinelava basewarp)

foreach(kdir ${KERNEL_DIRS})
    file(GLOB_RECURSE ksources "${kdir}/*.cpp")

    # ── SYCL-accelerated kernel (for GPU / SYCL CPU backends) ────
    add_library(${kdir} SHARED ${ksources})
    add_sycl_to_target(TARGET ${kdir} SOURCES ${ksources})

    target_compile_options(${kdir} PRIVATE -Werror -Wno-c99-designator)
    target_compile_definitions(${kdir} PRIVATE KERNEL_BUILD)

    target_include_directories(${kdir} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/${kdir}
    )
    target_link_libraries(${kdir} PRIVATE glm::glm)



    set_target_properties(${kdir} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/${kdir}"
    )

    # ── Pure C++ (native) kernel for CPU Software backend ────────
    # Compiled without add_sycl_to_target and with KERNEL_NATIVE,
    # producing a .so with zero SYCL runtime dependency.
    set(knative "${kdir}_native")
    add_library(${knative} SHARED ${ksources})

    target_compile_options(${knative} PRIVATE -Werror -Wno-c99-designator)
    target_compile_definitions(${knative} PRIVATE KERNEL_BUILD KERNEL_NATIVE)

    target_include_directories(${knative} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/${kdir}
    )
    target_link_libraries(${knative} PRIVATE glm::glm)

    set_target_properties(${knative} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/${kdir}"
        OUTPUT_NAME "${kdir}_native"
    )
endforeach()
