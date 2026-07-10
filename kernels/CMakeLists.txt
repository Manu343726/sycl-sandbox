# known kernel subdirectories — add new ones here
set(KERNEL_DIRS one_weekend mandelbrot cornell_box)

foreach(kdir ${KERNEL_DIRS})
    file(GLOB_RECURSE ksources "${kdir}/*.cpp")

    add_library(${kdir} SHARED ${ksources})
    add_sycl_to_target(TARGET ${kdir} SOURCES ${ksources})

    target_compile_options(${kdir} PRIVATE -Werror -Wno-c99-designator)

    target_include_directories(${kdir} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/${kdir}
    )
    target_link_libraries(${kdir} PRIVATE glm::glm)

    set_target_properties(${kdir} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/${kdir}"
    )
endforeach()
