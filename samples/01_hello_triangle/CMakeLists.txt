add_executable(hello_triangle main.cpp)

target_link_libraries(hello_triangle PRIVATE
    vulcao
    glfw
)

get_filename_component(VULKAN_BIN_DIR "${Vulkan_GLSLC_EXECUTABLE}" DIRECTORY)
find_program(SLANGC_EXECUTABLE NAMES slangc HINTS "${VULKAN_BIN_DIR}")
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found; install the Vulkan SDK")
endif()

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(HELLO_TRIANGLE_SHADERS
    "triangle.slang|vertMain|vertex|triangle.vert.spv"
    "triangle.slang|fragMain|fragment|triangle.frag.spv"
)

set(HELLO_TRIANGLE_SPV)
foreach(shader IN LISTS HELLO_TRIANGLE_SHADERS)
    string(REPLACE "|" ";" fields "${shader}")
    list(GET fields 0 source)
    list(GET fields 1 entry)
    list(GET fields 2 stage)
    list(GET fields 3 output)

    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${output}"
        COMMAND "${SLANGC_EXECUTABLE}" "${SHADER_SOURCE_DIR}/${source}"
                -target spirv -entry "${entry}" -stage "${stage}"
                -fvk-use-entrypoint-name
                -o "${SHADER_OUTPUT_DIR}/${output}"
        DEPENDS "${SHADER_SOURCE_DIR}/${source}"
        COMMENT "Compiling ${source} (${entry})"
        VERBATIM
    )
    list(APPEND HELLO_TRIANGLE_SPV "${SHADER_OUTPUT_DIR}/${output}")
endforeach()

add_custom_target(hello_triangle_shaders DEPENDS ${HELLO_TRIANGLE_SPV})
add_dependencies(hello_triangle hello_triangle_shaders)

target_compile_definitions(hello_triangle PRIVATE
    "VULCAO_SAMPLE_SHADER_DIR=\"${SHADER_OUTPUT_DIR}\"")
