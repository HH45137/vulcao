add_subdirectory(common)

# Compiles Slang shaders to SPIR-V for a sample and defines
# VULCAO_SAMPLE_SHADER_DIR to the generated directory.
#
# Each shader is "<source>|<entry>|<stage>|<output>".
function(vulcao_add_slang_shaders target)
    get_filename_component(vulkan_bin_dir "${Vulkan_GLSLC_EXECUTABLE}" DIRECTORY)
    find_program(SLANGC_EXECUTABLE NAMES slangc HINTS "${vulkan_bin_dir}")
    if(NOT SLANGC_EXECUTABLE)
        message(FATAL_ERROR "slangc not found; install the Vulkan SDK")
    endif()

    set(shader_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${shader_output_dir}")

    set(outputs)
    foreach(shader IN LISTS ARGN)
        string(REPLACE "|" ";" fields "${shader}")
        list(GET fields 0 source)
        list(GET fields 1 entry)
        list(GET fields 2 stage)
        list(GET fields 3 output)

        add_custom_command(
            OUTPUT "${shader_output_dir}/${output}"
            COMMAND "${SLANGC_EXECUTABLE}" "${shader_source_dir}/${source}"
                    -target spirv -entry "${entry}" -stage "${stage}"
                    -fvk-use-entrypoint-name
                    -o "${shader_output_dir}/${output}"
            DEPENDS "${shader_source_dir}/${source}"
            COMMENT "Compiling ${source} (${entry})"
            VERBATIM
        )
        list(APPEND outputs "${shader_output_dir}/${output}")
    endforeach()

    add_custom_target(${target}_shaders DEPENDS ${outputs})
    add_dependencies(${target} ${target}_shaders)

    target_compile_definitions(${target} PRIVATE
        "VULCAO_SAMPLE_SHADER_DIR=\"${shader_output_dir}\"")
endfunction()

add_subdirectory(01_hello_triangle)
add_subdirectory(02_uniforms)
