#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief A descriptor set and its bindings.
struct DescriptorSetLayoutInfo {
    uint32_t set = 0;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
};

/// @brief A specialization constant declared by a shader.
struct SpecializationConstantInfo {
    uint32_t constant_id = 0;
    uint32_t default_value = 0;
};

/// @brief Reflection data of a single shader stage.
struct ShaderReflection {
    vk::ShaderStageFlags stage;
    std::vector<DescriptorSetLayoutInfo> sets;
    std::vector<vk::PushConstantRange> push_constants;
    std::vector<vk::VertexInputAttributeDescription> vertex_attributes;
    std::vector<SpecializationConstantInfo> specialization_constants;

    /// @brief Returns the bindings of a descriptor set, or an empty range if the set is unused.
    /// @param set Descriptor set index.
    /// @return Bindings of the set.
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings_for_set(uint32_t set) const;
};

/// @brief Merged reflection data of several shader stages.
struct PipelineReflection {
    std::vector<DescriptorSetLayoutInfo> sets;
    std::vector<vk::PushConstantRange> push_constants;
};

/// @brief Reflects a SPIR-V module.
/// @param spirv SPIR-V code.
/// @return Reflection data of the module.
/// @throws std::runtime_error if the SPIR-V cannot be reflected.
ShaderReflection reflect_spirv(std::span<const uint32_t> spirv);

/// @brief Merges the reflection data of several shader stages.
/// @param reflections Reflection data to merge.
/// @return Merged descriptor sets and push constant ranges.
PipelineReflection merge_reflections(std::span<const ShaderReflection> reflections);

}
