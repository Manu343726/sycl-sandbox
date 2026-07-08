file(GLOB kernel_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)
foreach(kdir ${kernel_dirs})
    get_filename_component(kname ${kdir} NAME)
    file(GLOB_RECURSE ksources ${kdir}/*.cpp)

    add_library(${kname} SHARED ${ksources})
    add_sycl_to_target(TARGET ${kname} SOURCES ${ksources})

    target_include_directories(${kname} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/${kdir}
    )
    target_link_libraries(${kname} PRIVATE glm::glm)

    # Install to predictable location for the runtime to dlopen
    set_target_properties(${kname} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/${kname}"
    )
endforeach()
