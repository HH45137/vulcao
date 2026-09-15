#pragma once

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>

namespace vulcao {

inline void check(vk::Result result, const char* what) {
    if (result != vk::Result::eSuccess)
        throw std::runtime_error(std::string(what) + " failed: " + vk::to_string(result));
}

template <typename T>
T check(vkb::Result<T> result, const char* what) {
    if (!result)
        throw std::runtime_error(std::string(what) + " failed: " + result.error().message());
    return result.value();
}

}
