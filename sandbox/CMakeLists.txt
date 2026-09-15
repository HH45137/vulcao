# --- sandbox dependencies ----------------------------------------------------

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${VULCAO_THIRD_PARTY_DIR}/glfw" "${CMAKE_BINARY_DIR}/third_party/glfw")

# --- sandbox -----------------------------------------------------------------

add_executable(sandbox main.cpp)

target_link_libraries(sandbox PRIVATE
    vulcao
    glfw
)

# --- shaders -----------------------------------------------------------------

get_filename_component(VULKAN_BIN_DIR "${Vulkan_GLSLC_EXECUTABLE}" DIRECTORY)
find_program(SLANGC_EXECUTABLE NAMES slangc HINTS "${VULKAN_BIN_DIR}")
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found; install the Vulkan SDK")
endif()

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(SANDBOX_SHADERS
    "reduce.slang|compMain|compute|reduce.comp.spv"
    "triangle.slang|vertMain|vertex|triangle.vert.spv"
    "triangle.slang|fragMain|fragment|triangle.frag.spv"
)

set(SANDBOX_SPV)
foreach(shader IN LISTS SANDBOX_SHADERS)
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
    list(APPEND SANDBOX_SPV "${SHADER_OUTPUT_DIR}/${output}")
endforeach()

add_custom_target(sandbox_shaders DEPENDS ${SANDBOX_SPV})
add_dependencies(sandbox sandbox_shaders)

target_compile_definitions(sandbox PRIVATE "VULCAO_SHADER_DIR=\"${SHADER_OUTPUT_DIR}\"")
