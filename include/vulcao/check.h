#pragma once

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>

namespace vulcao {

/// @brief Throws std::runtime_error if the Vulkan result is not success.
/// @param result Result to check.
/// @param what Description of the failed operation.
inline void check(vk::Result result, const char* what) {
    if (result != vk::Result::eSuccess)
        throw std::runtime_error(std::string(what) + " failed: " + vk::to_string(result));
}

/// @brief Returns the value of a VkBootstrap result, or throws std::runtime_error on failure.
/// @param result Result to check.
/// @param what Description of the failed operation.
/// @return The value held by the result.
template <typename T>
T check(vkb::Result<T> result, const char* what) {
    if (!result)
        throw std::runtime_error(std::string(what) + " failed: " + result.error().message());
    return result.value();
}

}
