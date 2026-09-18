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
    vk::ShaderModule vertex_shader;   ///< Vertex shader module.
    vk::ShaderModule fragment_shader; ///< Fragment shader module.
    const char* vertex_entry = "main";   ///< Vertex entry point name.
    const char* fragment_entry = "main"; ///< Fragment entry point name.
    const SpecializationInfo* vertex_specialization = nullptr;   ///< Vertex specialization constants.
    const SpecializationInfo* fragment_specialization = nullptr; ///< Fragment specialization constants.

    vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList; ///< Input assembly topology.
    vk::PolygonMode polygon_mode = vk::PolygonMode::eFill;                 ///< Polygon rasterization mode.
    vk::CullModeFlags cull_mode = {};                                      ///< Face culling mode.
    vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;           ///< Front face winding order.

    bool depth_test = false;                          ///< Enable depth testing.
    bool depth_write = true;                          ///< Enable depth writes (when depth testing is on).
    vk::CompareOp depth_compare = vk::CompareOp::eLess; ///< Depth compare operation.
    bool depth_bias_enable = false;                   ///< Enable depth bias.
    bool depth_bounds_test = false;                   ///< Enable the depth bounds test.
    float min_depth_bounds = 0.0f;                    ///< Lower depth bounds.
    float max_depth_bounds = 1.0f;                    ///< Upper depth bounds.
    bool stencil_test = false;                        ///< Enable stencil testing.
    vk::StencilOpState front_stencil{};               ///< Front face stencil state.
    vk::StencilOpState back_stencil{};                ///< Back face stencil state.

    bool blend = false;                        ///< Enable alpha blending on the color attachments.
    vk::LogicOp logic_op = vk::LogicOp::eCopy; ///< Logic operation, used when logic_op_enable is true.
    bool logic_op_enable = false;              ///< Enable the logic operation.
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1; ///< Rasterization sample count.

    /// @brief Dynamic states, enabled on the pipeline and set while recording.
    std::vector<vk::DynamicState> dynamic_states{vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};
    std::vector<vk::VertexInputBindingDescription> vertex_bindings;      ///< Vertex input bindings.
    std::vector<vk::VertexInputAttributeDescription> vertex_attributes;  ///< Vertex input attributes.
    std::vector<vk::Format> color_formats;                               ///< Color attachment formats.
    vk::Format depth_format = vk::Format::eUndefined;                    ///< Depth attachment format.
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
    /// @throws std::runtime_error if the layout is invalid, a shader is missing, or creation fails.
    static Pipeline create_graphics(vk::Device device,
                                    const PipelineLayout& layout,
                                    const GraphicsPipelineInfo& info);

    /// @brief Creates a graphics pipeline using a pipeline cache.
    /// @param device Device that creates the pipeline.
    /// @param cache Pipeline cache used to accelerate creation.
    /// @param layout Pipeline layout.
    /// @param info Pipeline description.
    /// @return The created pipeline.
    /// @throws std::runtime_error if the layout is invalid, a shader is missing, or creation fails.
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
    /// @throws std::runtime_error if the layout or shader is invalid, or creation fails.
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
    /// @throws std::runtime_error if the layout or shader is invalid, or creation fails.
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
