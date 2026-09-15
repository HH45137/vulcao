#include "vulcao/pipeline_layout.h"

#include <utility>
#include <vector>

namespace vulcao {

PipelineLayout::~PipelineLayout() {
    destroy();
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      layout_(std::exchange(other.layout_, vk::PipelineLayout{})),
      owned_set_layouts_(std::move(other.owned_set_layouts_)) {}

PipelineLayout& PipelineLayout::operator=(PipelineLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        layout_ = std::exchange(other.layout_, vk::PipelineLayout{});
        owned_set_layouts_ = std::move(other.owned_set_layouts_);
    }
    return *this;
}

PipelineLayout PipelineLayout::create(vk::Device device,
                                      vk::ArrayProxy<const vk::DescriptorSetLayout> set_layouts,
                                      vk::ArrayProxy<const vk::PushConstantRange> push_constants) {
    PipelineLayout layout;
    layout.device_ = device;
    layout.layout_ = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = static_cast<uint32_t>(set_layouts.size()),
        .pSetLayouts = set_layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(push_constants.size()),
        .pPushConstantRanges = push_constants.data(),
    });
    return layout;
}

PipelineLayout PipelineLayout::create_from_reflection(vk::Device device,
                                                      std::span<const ShaderReflection> reflections) {
    const PipelineReflection merged = merge_reflections(reflections);

    std::vector<DescriptorSetLayout> set_layouts;
    set_layouts.reserve(merged.sets.size());
    for (const DescriptorSetLayoutInfo& set : merged.sets)
        set_layouts.push_back(DescriptorSetLayout::create(device, set.bindings));

    std::vector<vk::DescriptorSetLayout> raw_layouts;
    raw_layouts.reserve(set_layouts.size());
    for (const DescriptorSetLayout& layout : set_layouts)
        raw_layouts.push_back(layout.handle());

    PipelineLayout layout = create(device, raw_layouts, merged.push_constants);
    layout.owned_set_layouts_ = std::move(set_layouts);
    return layout;
}

void PipelineLayout::destroy() {
    if (layout_)
        device_.destroyPipelineLayout(layout_);

    owned_set_layouts_.clear();
    device_ = nullptr;
    layout_ = nullptr;
}

}
