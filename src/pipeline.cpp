#include "vulcao/pipeline.h"

#include "vulcao/check.h"
#include "vulcao/pipeline_layout.h"
#include "vulcao/shader_module.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace vulcao {

Pipeline::~Pipeline() {
    destroy();
}

Pipeline::Pipeline(Pipeline&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pipeline_(std::exchange(other.pipeline_, vk::Pipeline{})),
      bind_point_(std::exchange(other.bind_point_, vk::PipelineBindPoint::eGraphics)) {}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pipeline_ = std::exchange(other.pipeline_, vk::Pipeline{});
        bind_point_ = std::exchange(other.bind_point_, vk::PipelineBindPoint::eGraphics);
    }
    return *this;
}

Pipeline Pipeline::create_graphics(vk::Device device,
                                   const PipelineLayout& layout,
                                   const GraphicsPipelineInfo& info) {
    if (!layout.valid())
        throw std::runtime_error("Pipeline::create_graphics: invalid pipeline layout");
    if (!info.vertex_shader || !info.fragment_shader)
        throw std::runtime_error("Pipeline::create_graphics: missing shader module");

    const std::vector<vk::PipelineShaderStageCreateInfo> stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = info.vertex_shader,
            .pName = info.vertex_entry,
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = info.fragment_shader,
            .pName = info.fragment_entry,
        },
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input{
        .vertexBindingDescriptionCount = static_cast<uint32_t>(info.vertex_bindings.size()),
        .pVertexBindingDescriptions = info.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(info.vertex_attributes.size()),
        .pVertexAttributeDescriptions = info.vertex_attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = info.topology,
        .primitiveRestartEnable = VK_FALSE,
    };

    const vk::PipelineViewportStateCreateInfo viewport_state{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = info.polygon_mode,
        .cullMode = info.cull_mode,
        .frontFace = info.front_face,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = info.samples,
    };

    const bool depth_enabled = info.depth_test && info.depth_format != vk::Format::eUndefined;
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = depth_enabled ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = depth_enabled && info.depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp = info.depth_compare,
    };

    std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments;
    blend_attachments.reserve(info.color_formats.size());
    for (size_t i = 0; i < info.color_formats.size(); ++i) {
        blend_attachments.push_back(vk::PipelineColorBlendAttachmentState{
            .blendEnable = info.blend ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        });
    }

    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .attachmentCount = static_cast<uint32_t>(blend_attachments.size()),
        .pAttachments = blend_attachments.data(),
    };

    const std::vector<vk::DynamicState> dynamic_states{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    const vk::PipelineDynamicStateCreateInfo dynamic_state{
        .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = static_cast<uint32_t>(info.color_formats.size()),
        .pColorAttachmentFormats = info.color_formats.data(),
        .depthAttachmentFormat = info.depth_format,
    };

    const vk::GraphicsPipelineCreateInfo create_info{
        .pNext = &rendering,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
        .layout = layout.handle(),
    };

    Pipeline pipeline;
    pipeline.device_ = device;
    pipeline.bind_point_ = vk::PipelineBindPoint::eGraphics;
    const vk::ResultValue<vk::Pipeline> result = device.createGraphicsPipeline({}, create_info);
    check(result.result, "create graphics pipeline");
    pipeline.pipeline_ = result.value;
    return pipeline;
}

Pipeline Pipeline::create_compute(vk::Device device,
                                  const PipelineLayout& layout,
                                  const ShaderModule& shader,
                                  const char* entry) {
    if (!layout.valid())
        throw std::runtime_error("Pipeline::create_compute: invalid pipeline layout");
    if (!shader.valid())
        throw std::runtime_error("Pipeline::create_compute: invalid shader module");

    const vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = shader.handle(),
        .pName = entry,
    };
    const vk::ComputePipelineCreateInfo create_info{
        .stage = stage,
        .layout = layout.handle(),
    };

    Pipeline pipeline;
    pipeline.device_ = device;
    pipeline.bind_point_ = vk::PipelineBindPoint::eCompute;
    const vk::ResultValue<vk::Pipeline> result = device.createComputePipeline({}, create_info);
    check(result.result, "create compute pipeline");
    pipeline.pipeline_ = result.value;
    return pipeline;
}

void Pipeline::destroy() {
    if (pipeline_)
        device_.destroyPipeline(pipeline_);

    device_ = nullptr;
    pipeline_ = nullptr;
    bind_point_ = vk::PipelineBindPoint::eGraphics;
}

}

