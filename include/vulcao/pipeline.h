#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulcao/specialization.h"

namespace vulcao {

class PipelineCache;
class PipelineLayout;
class ShaderModule;

/// @brief Description of a graphics pipeline.
struct GraphicsPipelineInfo {
    vk::ShaderModule vertex_shader;
    vk::ShaderModule fragment_shader;
    const char* vertex_entry = "main";
    const char* fragment_entry = "main";
    const SpecializationInfo* vertex_specialization = nullptr;
    const SpecializationInfo* fragment_specialization = nullptr;

    vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
    vk::PolygonMode polygon_mode = vk::PolygonMode::eFill;
    vk::CullModeFlags cull_mode = {};
    vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;

    bool depth_test = false;
    bool depth_write = true;
    vk::CompareOp depth_compare = vk::CompareOp::eLess;
    bool depth_bias_enable = false;
    bool depth_bounds_test = false;
    float min_depth_bounds = 0.0f;
    float max_depth_bounds = 1.0f;
    bool stencil_test = false;
    vk::StencilOpState front_stencil{};
    vk::StencilOpState back_stencil{};

    bool blend = false;
    vk::LogicOp logic_op = vk::LogicOp::eCopy;
    bool logic_op_enable = false;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;

    std::vector<vk::DynamicState> dynamic_states{vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};
    std::vector<vk::VertexInputBindingDescription> vertex_bindings;
    std::vector<vk::VertexInputAttributeDescription> vertex_attributes;
    std::vector<vk::Format> color_formats;
    vk::Format depth_format = vk::Format::eUndefined;
};

/// @brief RAII wrapper around a Vulkan pipeline.
class Pipeline {
public:
    /// @brief Creates an empty pipeline.
    Pipeline() = default;

    /// @brief Destroys the pipeline.
    ~Pipeline();

    /// @brief Not copyable.
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /// @brief Moves the pipeline, leaving the source empty.
    Pipeline(Pipeline&& other) noexcept;

    /// @brief Move assignment. Destroys the current pipeline first.
    Pipeline& operator=(Pipeline&& other) noexcept;

    /// @brief Creates a graphics pipeline for dynamic rendering.
    /// @param device Device that creates the pipeline.
    /// @param layout Pipeline layout.
    /// @param info Pipeline description.
    /// @return The created pipeline.
    static Pipeline create_graphics(vk::Device device,
                                    const PipelineLayout& layout,
                                    const GraphicsPipelineInfo& info);

    /// @brief Creates a graphics pipeline using a pipeline cache.
    /// @param device Device that creates the pipeline.
    /// @param cache Pipeline cache used to accelerate creation.
    /// @param layout Pipeline layout.
    /// @param info Pipeline description.
    /// @return The created pipeline.
    static Pipeline create_graphics(vk::Device device,
                                    const PipelineCache& cache,
                                    const PipelineLayout& layout,
                                    const GraphicsPipelineInfo& info);

    /// @brief Creates a compute pipeline.
    /// @param device Device that creates the pipeline.
    /// @param layout Pipeline layout.
    /// @param shader Compute shader module.
    /// @param entry Entry point name.
    /// @param specialization Optional specialization constants.
    /// @return The created pipeline.
    static Pipeline create_compute(vk::Device device,
                                   const PipelineLayout& layout,
                                   const ShaderModule& shader,
                                   const char* entry = "main",
                                   const SpecializationInfo* specialization = nullptr);

    /// @brief Creates a compute pipeline using a pipeline cache.
    /// @param device Device that creates the pipeline.
    /// @param cache Pipeline cache used to accelerate creation.
    /// @param layout Pipeline layout.
    /// @param shader Compute shader module.
    /// @param entry Entry point name.
    /// @param specialization Optional specialization constants.
    /// @return The created pipeline.
    static Pipeline create_compute(vk::Device device,
                                   const PipelineCache& cache,
                                   const PipelineLayout& layout,
                                   const ShaderModule& shader,
                                   const char* entry = "main",
                                   const SpecializationInfo* specialization = nullptr);

    /// @brief Returns true if the pipeline holds a valid handle.
    bool valid() const { return static_cast<bool>(pipeline_); }

    /// @brief Returns true if the pipeline holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan pipeline handle.
    vk::Pipeline handle() const { return pipeline_; }

    /// @brief Returns the bind point of the pipeline.
    vk::PipelineBindPoint bind_point() const { return bind_point_; }

    /// @brief Destroys the pipeline and resets the wrapper.
    void destroy();

private:
    static Pipeline create_graphics_impl(vk::Device device,
                                         vk::PipelineCache cache,
                                         const PipelineLayout& layout,
                                         const GraphicsPipelineInfo& info);

    static Pipeline create_compute_impl(vk::Device device,
                                        vk::PipelineCache cache,
                                        const PipelineLayout& layout,
                                        const ShaderModule& shader,
                                        const char* entry,
                                        const SpecializationInfo* specialization);

    vk::Device device_;
    vk::Pipeline pipeline_;
    vk::PipelineBindPoint bind_point_ = vk::PipelineBindPoint::eGraphics;
};

}
