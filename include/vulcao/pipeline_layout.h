#pragma once

#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulcao/descriptor_set.h"
#include "vulcao/reflection.h"

namespace vulcao {

/// @brief RAII wrapper around a Vulkan pipeline layout.
class PipelineLayout {
public:
    /// @brief Creates an empty pipeline layout.
    PipelineLayout() = default;

    /// @brief Destroys the pipeline layout and any owned set layouts.
    ~PipelineLayout();

    /// @brief Not copyable.
    PipelineLayout(const PipelineLayout&) = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;

    /// @brief Moves the pipeline layout, leaving the source empty.
    PipelineLayout(PipelineLayout&& other) noexcept;

    /// @brief Move assignment. Destroys the current layout first.
    PipelineLayout& operator=(PipelineLayout&& other) noexcept;

    /// @brief Creates a pipeline layout from descriptor set layouts and push constant ranges.
    /// @param device Device that creates the layout.
    /// @param set_layouts Descriptor set layouts used by the pipeline.
    /// @param push_constants Push constant ranges used by the pipeline.
    /// @return The created pipeline layout.
    static PipelineLayout create(vk::Device device,
                                 vk::ArrayProxy<const vk::DescriptorSetLayout> set_layouts,
                                 vk::ArrayProxy<const vk::PushConstantRange> push_constants = {});

    /// @brief Creates a pipeline layout by merging the reflection data of several stages.
    /// @param device Device that creates the layout.
    /// @param reflections Reflection data of the stages.
    /// @return The created pipeline layout, owning the merged set layouts.
    static PipelineLayout create_from_reflection(vk::Device device,
                                                 std::span<const ShaderReflection> reflections);

    /// @brief Creates a pipeline layout using descriptor set layouts from a cache.
    /// @param device Device that creates the layout.
    /// @param cache Cache that owns the descriptor set layouts; it must outlive the pipeline layout.
    /// @param reflections Reflection data of the stages.
    /// @return The created pipeline layout.
    static PipelineLayout create_from_reflection(vk::Device device,
                                                 DescriptorSetLayoutCache& cache,
                                                 std::span<const ShaderReflection> reflections);

    /// @brief Returns true if the layout holds a valid handle.
    bool valid() const { return static_cast<bool>(layout_); }

    /// @brief Returns true if the layout holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan pipeline layout handle.
    vk::PipelineLayout handle() const { return layout_; }

    /// @brief Returns the set layouts owned by this object, if created from reflection.
    const std::vector<DescriptorSetLayout>& set_layouts() const { return owned_set_layouts_; }

    /// @brief Returns the descriptor set layout handle at an index.
    /// @param set Descriptor set index.
    vk::DescriptorSetLayout set_layout(uint32_t set) const { return set_layout_handles_.at(set); }

    /// @brief Returns the number of descriptor set layouts.
    size_t set_count() const { return set_layout_handles_.size(); }

    /// @brief Destroys the layout and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::PipelineLayout layout_;
    std::vector<DescriptorSetLayout> owned_set_layouts_;
    std::vector<vk::DescriptorSetLayout> set_layout_handles_;
};

}
