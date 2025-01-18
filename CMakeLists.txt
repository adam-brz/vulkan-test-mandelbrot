cmake_minimum_required(VERSION 3.20)

project(vulkan_tests LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

set(SHADER_SUBDIR "shaders")
set(OUTPUT_FRAGMENT_SHADER "mandelbrot.frag.spv")
set(OUTPUT_VERTEX_SHADER "mandelbrot.vert.spv")
set(OUTPUT_SHADER_DIR "${CMAKE_BINARY_DIR}/${SHADER_SUBDIR}")

add_executable(${PROJECT_NAME}
    main.cpp
    ${OUTPUT_SHADER_DIR}/${OUTPUT_FRAGMENT_SHADER}
    ${OUTPUT_SHADER_DIR}/${OUTPUT_VERTEX_SHADER}
) 

find_program(GLSLC_PROGRAM "glslc")

add_custom_command(
    OUTPUT ${OUTPUT_SHADER_DIR}/${OUTPUT_FRAGMENT_SHADER} ${OUTPUT_SHADER_DIR}/${OUTPUT_VERTEX_SHADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_SHADER_DIR}
    COMMAND ${GLSLC_PROGRAM} -o ${OUTPUT_SHADER_DIR}/${OUTPUT_FRAGMENT_SHADER} ${CMAKE_SOURCE_DIR}/shaders/mandelbrot.frag
    COMMAND ${GLSLC_PROGRAM} -o ${OUTPUT_SHADER_DIR}/${OUTPUT_VERTEX_SHADER} ${CMAKE_SOURCE_DIR}/shaders/mandelbrot.vert
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/mandelbrot.frag ${CMAKE_SOURCE_DIR}/shaders/mandelbrot.vert 
    VERBATIM
)

target_compile_definitions(${PROJECT_NAME} PRIVATE 
    VK_MANDELBROT_FRAG_SHADER="${SHADER_SUBDIR}/${OUTPUT_FRAGMENT_SHADER}"
    VK_MANDELBROT_VERT_SHADER="${SHADER_SUBDIR}/${OUTPUT_VERTEX_SHADER}"
)

find_package(VulkanHeaders REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE vulkan-headers::vulkan-headers Vulkan::Vulkan glm::glm glfw)

