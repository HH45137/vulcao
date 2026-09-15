#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "vulcao/buffer.h"
#include "vulcao/command_buffer.h"
#include "vulcao/context.h"
#include "vulcao/descriptor_set.h"
#include "vulcao/image.h"
#include "vulcao/pipeline.h"
#include "vulcao/pipeline_layout.h"
#include "vulcao/sampler.h"
#include "vulcao/semaphore.h"
#include "vulcao/shader_module.h"
#include "vulcao/vertex_layout.h"

#ifndef VULCAO_SHADER_DIR
#define VULCAO_SHADER_DIR "shaders"
#endif

namespace {

std::filesystem::path shader_path(const char* name) {
    return std::filesystem::path(VULCAO_SHADER_DIR) / name;
}

struct TexturedVertex {
    float position[2];
    float uv[2];
};

std::vector<uint8_t> make_checkerboard(uint32_t size, uint32_t cell) {
    std::vector<uint8_t> pixels(static_cast<size_t>(size) * size * 4);
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const bool light = ((x / cell) + (y / cell)) % 2 == 0;
            const uint8_t value = light ? 235 : 30;
            const size_t index = (static_cast<size_t>(y) * size + x) * 4;
            pixels[index + 0] = value;
            pixels[index + 1] = value;
            pixels[index + 2] = value;
            pixels[index + 3] = 255;
        }
    }
    return pixels;
}

vk::SurfaceKHR create_surface(vk::Instance instance, GLFWwindow* window) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");
    return vk::SurfaceKHR{surface};
}

vk::Extent2D framebuffer_extent(GLFWwindow* window) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    return vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

void run_compute_test(vulcao::Context& ctx) {
    constexpr uint32_t element_count = 64;
    const vk::DeviceSize data_bytes = element_count * sizeof(uint32_t);

    std::vector<uint32_t> initial(element_count);
    for (uint32_t i = 0; i < element_count; ++i)
        initial[i] = i;

    vulcao::Buffer buffer = vulcao::Buffer::create(
        ctx.allocator(), data_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst);
    ctx.upload(buffer, initial);

    vulcao::ShaderModule shader = vulcao::ShaderModule::create_from_file(
        ctx.device(), vk::ShaderStageFlagBits::eCompute, shader_path("reduce.comp.spv"));

    const vulcao::ShaderReflection& reflection = shader.reflection();
    std::cout << "compute reflection: " << reflection.sets.size() << " set(s), "
              << reflection.bindings_for_set(0).size() << " binding(s)" << std::endl;

    vulcao::PipelineLayout layout =
        vulcao::PipelineLayout::create_from_reflection(ctx.device(), std::span(&reflection, 1));
    vulcao::Pipeline pipeline =
        vulcao::Pipeline::create_compute(ctx.device(), layout, shader, "compMain");

    const vk::DescriptorPoolSize pool_size{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
    };
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(ctx.device(), pool_size, 1);
    vulcao::DescriptorSet set = pool.allocate(layout.set_layouts()[0]);
    set.write_storage_buffer(0, buffer);

    vulcao::Buffer readback = vulcao::Buffer::create(
        ctx.allocator(), data_bytes, vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    vulcao::CommandBuffer cmd = vulcao::CommandBuffer::allocate(ctx.device(), ctx.command_pool());
    const vk::DescriptorSet raw_set = set.handle();
    cmd.begin();
    cmd.bind_pipeline(vk::PipelineBindPoint::eCompute, pipeline.handle());
    cmd.bind_descriptor_sets(vk::PipelineBindPoint::eCompute, layout.handle(), raw_set);
    cmd.dispatch(element_count / 64);
    cmd.buffer_barrier(buffer.handle(),
                       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                       vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    cmd.copy_buffer(buffer.handle(), readback.handle(), data_bytes);
    cmd.end();
    ctx.submit_and_wait(cmd.handle());

    readback.invalidate();
    const auto* result = static_cast<const uint32_t*>(readback.map());
    bool ok = true;
    for (uint32_t i = 0; i < element_count; ++i)
        ok = ok && result[i] == initial[i] * 2 + 1;
    readback.unmap();

    std::cout << "compute dispatch: " << (ok ? "ok" : "MISMATCH") << std::endl;
}

void record_frame(vulcao::CommandBuffer& cmd,
                  const vulcao::Pipeline& pipeline,
                  const vulcao::PipelineLayout& pipeline_layout,
                  const vulcao::DescriptorSet& descriptor_set,
                  const vulcao::Buffer& vertex_buffer,
                  vk::Image image,
                  vk::ImageView view,
                  vk::Extent2D extent) {
    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    vk::ClearValue clear{};
    clear.color.float32[0] = 0.05f;
    clear.color.float32[1] = 0.10f;
    clear.color.float32[2] = 0.20f;
    clear.color.float32[3] = 1.0f;

    vk::RenderingAttachmentInfo color_attachment{
        .imageView = view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear,
    };

    vk::RenderingInfo rendering_info{
        .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
    };

    cmd.reset();
    cmd.begin();
    cmd.transition(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                   range);
    cmd.begin_rendering(rendering_info);
    cmd.bind_pipeline(vk::PipelineBindPoint::eGraphics, pipeline.handle());
    const vk::DescriptorSet raw_set = descriptor_set.handle();
    cmd.bind_descriptor_sets(vk::PipelineBindPoint::eGraphics, pipeline_layout.handle(), raw_set);
    cmd.set_viewport(extent);
    cmd.set_scissor(extent);
    cmd.bind_vertex_buffer(0, vertex_buffer);
    cmd.draw(6);
    cmd.end_rendering();
    cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                   range);
    cmd.end();
}

void render_frame(vulcao::Context& ctx,
                  const vulcao::Pipeline& pipeline,
                  const vulcao::PipelineLayout& pipeline_layout,
                  const vulcao::DescriptorSet& descriptor_set,
                  const vulcao::Buffer& vertex_buffer,
                  const vulcao::Semaphore& image_available,
                  const vulcao::Semaphore& render_finished) {
    vk::Device device = ctx.device();
    vk::SwapchainKHR swapchain = ctx.swapchain();
    vulcao::CommandBuffer& cmd = ctx.immediate_command_buffer();

    uint32_t image_index =
        device.acquireNextImageKHR(swapchain, UINT64_MAX, image_available.handle()).value;

    record_frame(cmd, pipeline, pipeline_layout, descriptor_set, vertex_buffer,
                 ctx.swapchain_images()[image_index],
                 ctx.swapchain_image_views()[image_index], ctx.swapchain_extent());

    const vk::Semaphore wait_semaphore = image_available.handle();
    const vk::Semaphore signal_semaphore = render_finished.handle();
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    const vk::CommandBuffer raw_cmd = cmd.handle();
    ctx.submit(vk::SubmitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &raw_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    });

    const vk::Result present_result = ctx.present_queue().presentKHR(vk::PresentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signal_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    });
    (void)present_result;
    device.waitIdle();
}

}

int main() {
    if (!glfwInit()) {
        std::cerr << "glfwInit failed" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "vulcao-game", nullptr, nullptr);
    if (!window) {
        std::cerr << "glfwCreateWindow failed" << std::endl;
        glfwTerminate();
        return 1;
    }

    try {
        vulcao::Context ctx{{.app_name = "vulcao-game"}};
        ctx.initialize(create_surface(ctx.instance(), window), framebuffer_extent(window),
                       vulcao::SwapchainInfo{
                           .extra_usage = vk::ImageUsageFlagBits::eTransferSrc});

        run_compute_test(ctx);

        constexpr uint32_t texture_size = 256;
        constexpr uint32_t texture_cell = 8;
        const uint32_t mip_levels = static_cast<uint32_t>(std::floor(std::log2(texture_size))) + 1;
        const std::vector<uint8_t> checkerboard = make_checkerboard(texture_size, texture_cell);
        vulcao::Image texture = vulcao::Image::create_2d(
            ctx.allocator(), vk::Extent2D{texture_size, texture_size}, vk::Format::eR8G8B8A8Srgb,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
                vk::ImageUsageFlagBits::eTransferDst,
            mip_levels);
        ctx.upload(texture, checkerboard, vk::ImageLayout::eShaderReadOnlyOptimal, true);
        vulcao::Sampler sampler = vulcao::Sampler::linear(ctx.device(), true);
        std::cout << "texture: " << texture_size << "x" << texture_size << ", " << mip_levels
                  << " mip levels" << std::endl;

        vulcao::ShaderModule vertex_shader = vulcao::ShaderModule::create_from_file(
            ctx.device(), vk::ShaderStageFlagBits::eVertex, shader_path("texture.vert.spv"));
        vulcao::ShaderModule fragment_shader = vulcao::ShaderModule::create_from_file(
            ctx.device(), vk::ShaderStageFlagBits::eFragment, shader_path("texture.frag.spv"));

        const vulcao::VertexLayout vertex_layout = vulcao::make_vertex_layout<TexturedVertex>(
            vertex_shader.reflection(),
            {offsetof(TexturedVertex, position), offsetof(TexturedVertex, uv)});
        std::cout << "vertex attributes from reflection: " << vertex_layout.attributes.size()
                  << std::endl;
        std::cout << "fragment bindings in set 0: "
                  << fragment_shader.reflection().bindings_for_set(0).size() << std::endl;

        const vulcao::GraphicsPipelineInfo pipeline_info{
            .vertex_shader = vertex_shader.handle(),
            .fragment_shader = fragment_shader.handle(),
            .vertex_entry = "vertMain",
            .fragment_entry = "fragMain",
            .vertex_bindings = vertex_layout.bindings,
            .vertex_attributes = vertex_layout.attributes,
            .color_formats = {ctx.swapchain_format()},
        };

        const std::array<vulcao::ShaderReflection, 2> stage_reflections{
            vertex_shader.reflection(), fragment_shader.reflection()};
        vulcao::PipelineLayout pipeline_layout =
            vulcao::PipelineLayout::create_from_reflection(ctx.device(), stage_reflections);
        vulcao::Pipeline pipeline =
            vulcao::Pipeline::create_graphics(ctx.device(), pipeline_layout, pipeline_info);

        const vk::DescriptorPoolSize descriptor_pool_size{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
        };
        vulcao::DescriptorPool descriptor_pool =
            vulcao::DescriptorPool::create(ctx.device(), descriptor_pool_size, 1);
        vulcao::DescriptorSet descriptor_set = descriptor_pool.allocate(pipeline_layout.set_layouts()[0]);
        descriptor_set.write_image(0, texture, sampler);

        const std::array<TexturedVertex, 6> vertices{{
            {{-0.8f, -0.8f}, {0.0f, 1.0f}},
            {{0.8f, -0.8f}, {1.0f, 1.0f}},
            {{0.8f, 0.8f}, {1.0f, 0.0f}},
            {{-0.8f, -0.8f}, {0.0f, 1.0f}},
            {{0.8f, 0.8f}, {1.0f, 0.0f}},
            {{-0.8f, 0.8f}, {0.0f, 0.0f}},
        }};
        vulcao::Buffer vertex_buffer = vulcao::Buffer::create(
            ctx.allocator(), sizeof(TexturedVertex) * vertices.size(),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        ctx.upload(vertex_buffer, vertices);

        vulcao::Semaphore image_available = vulcao::Semaphore::create(ctx.device());
        vulcao::Semaphore render_finished = vulcao::Semaphore::create(ctx.device());

        auto redraw = [&]() {
            try {
                render_frame(ctx, pipeline, pipeline_layout, descriptor_set, vertex_buffer, image_available, render_finished);
            } catch (const vk::OutOfDateKHRError&) {
                vk::Extent2D extent = framebuffer_extent(window);
                if (extent.width == 0 || extent.height == 0)
                    return;
                ctx.recreate_swapchain(extent);
                render_frame(ctx, pipeline, pipeline_layout, descriptor_set, vertex_buffer, image_available, render_finished);
            }
        };

        redraw();
        std::cout << "vulkan initialized" << std::endl;

        bool framebuffer_resized = false;
        glfwSetWindowUserPointer(window, &framebuffer_resized);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int) {
            *static_cast<bool*>(glfwGetWindowUserPointer(w)) = true;
        });

        while (!glfwWindowShouldClose(window)) {
            glfwWaitEvents();

            if (framebuffer_resized) {
                framebuffer_resized = false;
                vk::Extent2D extent = framebuffer_extent(window);
                if (extent.width == 0 || extent.height == 0)
                    continue;
                ctx.recreate_swapchain(extent);
                redraw();
            }
        }

        ctx.wait_idle();
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
