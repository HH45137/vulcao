#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>

#include "common/window.h"
#include "vulcao/buffer.h"
#include "vulcao/command_buffer.h"
#include "vulcao/context.h"
#include "vulcao/frame_manager.h"
#include "vulcao/log.h"
#include "vulcao/pipeline.h"
#include "vulcao/pipeline_layout.h"
#include "vulcao/shader_module.h"
#include "vulcao/vertex_layout.h"

#ifndef VULCAO_SHADER_DIR
#define VULCAO_SHADER_DIR "shaders"
#endif

namespace {

struct Vertex {
    float position[3];
    float color[3];
};

const std::array<Vertex, 3> triangle_vertices{{
    {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
}};

const std::array<uint16_t, 3> triangle_indices{{0, 1, 2}};

std::filesystem::path shader_path(const char* name) {
    return std::filesystem::path(VULCAO_SHADER_DIR) / name;
}

}

int main() {
    vulcao::set_log_level(vulcao::LogLevel::info);
    vulcao::set_log_callback([](const vulcao::LogMessage& message) {
        std::cout << '[' << vulcao::to_string(message.level) << ": "
                  << vulcao::to_string(message.category) << "] " << message.message << std::endl;
    });

    try {
        sample::Window window{800, 600, "01_hello_triangle"};

        vulcao::Context context{{.app_name = "01_hello_triangle"}};
        context.initialize(window.create_surface(context.instance()), window.framebuffer_extent());

        const vulcao::ShaderModule vertex = vulcao::ShaderModule::create_from_file(
            context.device(), vk::ShaderStageFlagBits::eVertex, shader_path("triangle.vert.spv"));
        const vulcao::ShaderModule fragment = vulcao::ShaderModule::create_from_file(
            context.device(), vk::ShaderStageFlagBits::eFragment, shader_path("triangle.frag.spv"));

        vulcao::Buffer vertex_buffer = vulcao::Buffer::create(
            context.allocator(), triangle_vertices.size() * sizeof(Vertex),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        context.upload(vertex_buffer, triangle_vertices);

        vulcao::Buffer index_buffer = vulcao::Buffer::create(
            context.allocator(), triangle_indices.size() * sizeof(uint16_t),
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        context.upload(index_buffer, triangle_indices);

        const vulcao::VertexLayout vertex_layout = vulcao::make_vertex_layout<Vertex>(
            vertex.reflection(),
            {offsetof(Vertex, position), offsetof(Vertex, color)});

        const vulcao::PipelineLayout layout = vulcao::PipelineLayout::create_from_reflection(
            context.device(), std::span(&vertex.reflection(), 1));

        const vulcao::Pipeline pipeline = vulcao::Pipeline::create_graphics(
            context.device(), layout,
            vulcao::GraphicsPipelineInfo{
                .vertex_shader = vertex.handle(),
                .fragment_shader = fragment.handle(),
                .vertex_entry = "vertMain",
                .fragment_entry = "fragMain",
                .dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor},
                .vertex_bindings = vertex_layout.bindings,
                .vertex_attributes = vertex_layout.attributes,
                .color_formats = {context.swapchain_format()},
            });

        vulcao::FrameManager frames{context};

        const vk::ImageSubresourceRange range{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        auto render = [&]() {
            vulcao::Frame frame = frames.begin_frame();
            vulcao::CommandBuffer& cmd = *frame.command_buffer;

            const vk::Image image = context.swapchain_images()[frame.image_index];
            const vk::ImageView view = context.swapchain_image_views()[frame.image_index];
            const vk::Extent2D extent = context.swapchain_extent();

            vk::ClearValue clear{};
            clear.color.float32[0] = 0.05f;
            clear.color.float32[1] = 0.10f;
            clear.color.float32[2] = 0.20f;
            clear.color.float32[3] = 1.0f;

            const vk::RenderingAttachmentInfo color_attachment{
                .imageView = view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = clear,
            };
            const vk::RenderingInfo rendering_info{
                .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = extent},
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &color_attachment,
            };

            cmd.transition(image, vk::ImageLayout::eUndefined,
                           vk::ImageLayout::eColorAttachmentOptimal, range,
                           vulcao::FrameManager::acquire_wait_stage);
            cmd.begin_rendering(rendering_info);
            cmd.bind_pipeline(vk::PipelineBindPoint::eGraphics, pipeline.handle());
            cmd.set_viewport(extent);
            cmd.set_scissor(extent);
            cmd.bind_vertex_buffer(0, vertex_buffer);
            cmd.bind_index_buffer(index_buffer, 0, vk::IndexType::eUint16);
            cmd.draw_indexed(static_cast<uint32_t>(triangle_indices.size()));
            cmd.end_rendering();
            cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal,
                           vk::ImageLayout::ePresentSrcKHR, range);

            frames.end_frame(frame);
            return frames.present(frame);
        };

        auto recreate_swapchain = [&]() {
            const vk::Extent2D extent = window.framebuffer_extent();
            if (extent.width == 0 || extent.height == 0)
                return false;
            frames.recreate_swapchain(extent);
            return true;
        };

        while (!window.should_close()) {
            window.poll_events();

            if (window.consume_resized()) {
                if (!recreate_swapchain())
                    continue;
            }

            try {
                if (!render())
                    recreate_swapchain();
            } catch (const vk::OutOfDateKHRError&) {
                recreate_swapchain();
            }
        }

        context.wait_idle();
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
