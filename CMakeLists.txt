cmake_minimum_required(VERSION 3.24)
project(vulcao VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VULCAO_BUILD_SANDBOX "Build the vulcao sandbox example" ON)

set(VULCAO_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

find_package(Vulkan REQUIRED)

# --- vulcao dependencies -----------------------------------------------------

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/vk-bootstrap)

set(VMA_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/VulkanMemoryAllocator)

set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/spirv-reflect)

# --- vulcao ------------------------------------------------------------------

add_library(vulcao STATIC
    src/allocator.cpp
    src/buffer.cpp
    src/command_buffer.cpp
    src/context.cpp
    src/descriptor_set.cpp
    src/fence.cpp
    src/image.cpp
    src/pipeline.cpp
    src/pipeline_layout.cpp
    src/query_pool.cpp
    src/reflection.cpp
    src/sampler.cpp
    src/semaphore.cpp
    src/shader_module.cpp
    src/vma_impl.cpp
)

target_include_directories(vulcao PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_compile_features(vulcao PUBLIC cxx_std_20)

target_link_libraries(vulcao PUBLIC
    Vulkan::Vulkan
    Vulkan::Headers
    vk-bootstrap::vk-bootstrap
    VulkanMemoryAllocator::Headers
    spirv-reflect-static
)

target_compile_definitions(vulcao PUBLIC
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)

# --- examples ----------------------------------------------------------------

if(VULCAO_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()
