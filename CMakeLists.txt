cmake_minimum_required(VERSION 3.24)
project(vulcao)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/vk-bootstrap)

set(VMA_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/VulkanMemoryAllocator)

set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glm)

add_library(vulcao STATIC
    src/allocator.cpp
    src/buffer.cpp
    src/command_buffer.cpp
    src/context.cpp
    src/fence.cpp
    src/image.cpp
    src/query_pool.cpp
    src/semaphore.cpp
    src/vma_impl.cpp
)

target_include_directories(vulcao PUBLIC include)

target_link_libraries(vulcao PUBLIC
    Vulkan::Vulkan
    Vulkan::Headers
    vk-bootstrap::vk-bootstrap
    VulkanMemoryAllocator::Headers
    glm::glm
)

target_compile_definitions(vulcao PUBLIC
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)

add_executable(sandbox sandbox/main.cpp)

target_link_libraries(sandbox PRIVATE
    vulcao
    glfw
)
