#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulcao/reflection.h"

namespace vulcao {

/// @brief Vertex input bindings and attributes.
struct VertexLayout {
    std::vector<vk::VertexInputBindingDescription> bindings;    ///< Vertex input bindings.
    std::vector<vk::VertexInputAttributeDescription> attributes; ///< Vertex input attributes.
};

/// @brief Builds a vertex layout from reflected attributes and member offsets.
/// @tparam Vertex Vertex struct type.
/// @param reflection Reflection data of the vertex shader.
/// @param member_offsets Byte offsets of the members, in attribute location order.
/// @param binding Vertex binding index.
/// @return Bindings and attributes matching the reflection.
/// @throws std::runtime_error if the attribute count does not match member_offsets.
template <typename Vertex>
VertexLayout make_vertex_layout(const ShaderReflection& reflection,
                                std::initializer_list<size_t> member_offsets,
                                uint32_t binding = 0) {
    std::vector<vk::VertexInputAttributeDescription> attributes = reflection.vertex_attributes;
    std::sort(attributes.begin(), attributes.end(),
              [](const vk::VertexInputAttributeDescription& a,
                 const vk::VertexInputAttributeDescription& b) { return a.location < b.location; });

    if (attributes.size() != member_offsets.size())
        throw std::runtime_error(
            "make_vertex_layout: attribute count does not match member offset count");

    size_t index = 0;
    for (size_t offset : member_offsets) {
        attributes[index].binding = binding;
        attributes[index].offset = offset;
        ++index;
    }

    VertexLayout layout;
    layout.bindings.push_back(vk::VertexInputBindingDescription{
        .binding = binding,
        .stride = static_cast<uint32_t>(sizeof(Vertex)),
        .inputRate = vk::VertexInputRate::eVertex,
    });
    layout.attributes = std::move(attributes);
    return layout;
}

}
