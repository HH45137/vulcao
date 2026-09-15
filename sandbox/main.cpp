#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.hpp>

#include "vulcao/buffer.h"
#include "vulcao/command_buffer.h"
#include "vulcao/context.h"
#include "vulcao/descriptor_set.h"
#include "vulcao/image.h"
#include "vulcao/pipeline.h"
#include "vulcao/pipeline_cache.h"
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

struct CubeVertex {
    float position[3];
    float uv[2];
};

struct Transform {
    glm::vec4 row0;
    glm::vec4 row1;
    glm::vec4 row2;
    glm::vec4 row3;
};

Transform make_transform(const glm::mat4& mvp) {
    return Transform{glm::row(mvp, 0), glm::row(mvp, 1), glm::row(mvp, 2), glm::row(mvp, 3)};
}

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

    vulcao::Semaphore timeline = vulcao::Semaphore::create_timeline(ctx.device());
    ctx.submit(ctx.graphics_queue(), cmd.handle(), timeline, 1);
    timeline.wait(1);
    std::cout << "timeline semaphore value: " << timeline.value() << std::endl;

    timeline.signal(2);
    timeline.wait(2);

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
                  const vulcao::Buffer& index_buffer,
                  vulcao::Image& depth,
                  const Transform& transform,
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

    vk::ClearValue depth_clear{};
    depth_clear.depthStencil.depth = 1.0f;
    depth_clear.depthStencil.stencil = 0;

    vk::RenderingAttachmentInfo depth_attachment{
        .imageView = depth.view(),
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = depth_clear,
    };

    vk::RenderingInfo rendering_info{
        .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    };

    cmd.reset();
    cmd.begin();
    cmd.begin_debug_label("frame", {0.2f, 0.5f, 0.9f, 1.0f});
    cmd.transition(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                   range);
    cmd.transition(depth, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    cmd.begin_rendering(rendering_info);
    cmd.bind_pipeline(vk::PipelineBindPoint::eGraphics, pipeline.handle());
    const vk::DescriptorSet raw_set = descriptor_set.handle();
    cmd.bind_descriptor_sets(vk::PipelineBindPoint::eGraphics, pipeline_layout.handle(), raw_set);
    cmd.set_viewport(extent);
    cmd.set_scissor(extent);
    cmd.set_depth_compare_op(vk::CompareOp::eLess);
    cmd.push_constants(pipeline_layout.handle(), vk::ShaderStageFlagBits::eVertex, 0, transform);
    cmd.bind_vertex_buffer(0, vertex_buffer);
    cmd.bind_index_buffer(index_buffer, 0, vk::IndexType::eUint16);
    cmd.draw_indexed(36);
    cmd.end_rendering();
    cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                   range);
    cmd.end_debug_label();
    cmd.end();
}

void render_frame(vulcao::Context& ctx,
                  const vulcao::Pipeline& pipeline,
                  const vulcao::PipelineLayout& pipeline_layout,
                  const vulcao::DescriptorSet& descriptor_set,
                  const vulcao::Buffer& vertex_buffer,
                  const vulcao::Buffer& index_buffer,
                  vulcao::Image& depth,
                  float time_seconds,
                  const vulcao::Semaphore& image_available,
                  const vulcao::Semaphore& render_finished) {
    vk::Device device = ctx.device();
    vk::SwapchainKHR swapchain = ctx.swapchain();
    vulcao::CommandBuffer& cmd = ctx.immediate_command_buffer();
    const vk::Extent2D extent = ctx.swapchain_extent();

    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(60.0f), aspect, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;
    const glm::mat4 view =
        glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 model =
        glm::rotate(glm::mat4(1.0f), time_seconds, glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), time_seconds * 0.6f, glm::vec3(1.0f, 0.0f, 0.0f));
    const Transform transform = make_transform(projection * view * model);

    uint32_t image_index =
        device.acquireNextImageKHR(swapchain, UINT64_MAX, image_available.handle()).value;

    record_frame(cmd, pipeline, pipeline_layout, descriptor_set, vertex_buffer, index_buffer, depth,
                 transform, ctx.swapchain_images()[image_index],
                 ctx.swapchain_image_views()[image_index], extent);

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
        vulcao::Context ctx{{.app_name = "vulcao-game",
                             .device_features = {.timeline_semaphore = true},
                             .separate_compute_queue = true,
                             .separate_transfer_queue = true,
                             .log = [](vulcao::LogLevel, std::string_view message) {
                                 std::cout << message << std::endl;
                             }}};
        ctx.initialize(create_surface(ctx.instance(), window), framebuffer_extent(window),
                       vulcao::SwapchainInfo{
                           .extra_usage = vk::ImageUsageFlagBits::eTransferSrc});

        std::cout << "dedicated compute queue: "
                  << (ctx.has_compute_queue() ? std::to_string(ctx.compute_queue_family_index())
                                              : "none")
                  << std::endl;
        std::cout << "dedicated transfer queue: "
                  << (ctx.has_transfer_queue() ? std::to_string(ctx.transfer_queue_family_index())
                                               : "none")
                  << std::endl;

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
        ctx.set_debug_name(vk::ObjectType::eImage,
                           reinterpret_cast<uint64_t>(static_cast<VkImage>(texture.handle())),
                           "checkerboard");
        std::cout << "texture: " << texture_size << "x" << texture_size << ", " << mip_levels
                  << " mip levels" << std::endl;

        vulcao::ShaderModule vertex_shader = vulcao::ShaderModule::create_from_file(
            ctx.device(), vk::ShaderStageFlagBits::eVertex, shader_path("cube.vert.spv"));
        vulcao::ShaderModule fragment_shader = vulcao::ShaderModule::create_from_file(
            ctx.device(), vk::ShaderStageFlagBits::eFragment, shader_path("cube.frag.spv"));

        const vulcao::VertexLayout vertex_layout = vulcao::make_vertex_layout<CubeVertex>(
            vertex_shader.reflection(),
            {offsetof(CubeVertex, position), offsetof(CubeVertex, uv)});
        std::cout << "vertex attributes from reflection: " << vertex_layout.attributes.size()
                  << std::endl;

        vulcao::SpecializationInfo fragment_specialization;
        fragment_specialization.map_constant(0, 0.85f);

        const vulcao::GraphicsPipelineInfo pipeline_info{
            .vertex_shader = vertex_shader.handle(),
            .fragment_shader = fragment_shader.handle(),
            .vertex_entry = "vertMain",
            .fragment_entry = "fragMain",
            .fragment_specialization = &fragment_specialization,
            .depth_test = true,
            .dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor,
                               vk::DynamicState::eDepthCompareOp},
            .vertex_bindings = vertex_layout.bindings,
            .vertex_attributes = vertex_layout.attributes,
            .color_formats = {ctx.swapchain_format()},
            .depth_format = vk::Format::eD32Sfloat,
        };

        const std::array<vulcao::ShaderReflection, 2> stage_reflections{
            vertex_shader.reflection(), fragment_shader.reflection()};
        vulcao::DescriptorSetLayoutCache layout_cache{ctx.device()};
        vulcao::PipelineLayout pipeline_layout =
            vulcao::PipelineLayout::create_from_reflection(ctx.device(), layout_cache,
                                                           stage_reflections);
        vulcao::PipelineCache pipeline_cache = vulcao::PipelineCache::create(ctx.device());
        vulcao::Pipeline pipeline = vulcao::Pipeline::create_graphics(
            ctx.device(), pipeline_cache, pipeline_layout, pipeline_info);
        std::cout << "pipeline cache size: " << pipeline_cache.data().size() << " bytes"
                  << std::endl;

        const vk::DescriptorPoolSize descriptor_pool_size{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
        };
        vulcao::DescriptorPool descriptor_pool =
            vulcao::DescriptorPool::create(ctx.device(), descriptor_pool_size, 1);
        vulcao::DescriptorSet descriptor_set =
            descriptor_pool.allocate(pipeline_layout.set_layout(0));
        vulcao::DescriptorSetWriter{descriptor_set}.write_image(0, texture, sampler).flush();

        const std::array<CubeVertex, 24> vertices{{
            {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f}}, {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}},
            {{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},   {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}},
            {{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}}, {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},
            {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f}},
            {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f}}, {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},
            {{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},   {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f}},
            {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f}},
            {{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f}}, {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f}},
            {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f}}, {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f}},
            {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f}},   {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f}},
            {{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f}},
            {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f}}, {{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f}},
        }};
        const std::array<uint16_t, 36> indices{{
            0,  1,  2,  0,  2,  3,  4,  5,  6,  4,  6,  7,  8,  9,  10, 8,  10, 11,
            12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
        }};

        vulcao::Buffer vertex_buffer = vulcao::Buffer::create(
            ctx.allocator(), sizeof(CubeVertex) * vertices.size(),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        ctx.upload(vertex_buffer, vertices);

        vulcao::Buffer index_buffer = vulcao::Buffer::create(
            ctx.allocator(), sizeof(uint16_t) * indices.size(),
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        ctx.upload(index_buffer, indices);

        vulcao::Image depth = vulcao::Image::create_depth(ctx.allocator(), ctx.swapchain_extent());

        vulcao::Semaphore image_available = vulcao::Semaphore::create(ctx.device());
        vulcao::Semaphore render_finished = vulcao::Semaphore::create(ctx.device());

        auto recreate_swapchain = [&]() {
            vk::Extent2D extent = framebuffer_extent(window);
            if (extent.width == 0 || extent.height == 0)
                return false;
            ctx.recreate_swapchain(extent);
            depth = vulcao::Image::create_depth(ctx.allocator(), ctx.swapchain_extent());
            return true;
        };

        auto redraw = [&](float time_seconds) {
            try {
                render_frame(ctx, pipeline, pipeline_layout, descriptor_set, vertex_buffer,
                             index_buffer, depth, time_seconds, image_available, render_finished);
            } catch (const vk::OutOfDateKHRError&) {
                if (!recreate_swapchain())
                    return;
                render_frame(ctx, pipeline, pipeline_layout, descriptor_set, vertex_buffer,
                             index_buffer, depth, time_seconds, image_available, render_finished);
            }
        };

        const double start_time = glfwGetTime();
        redraw(0.0f);
        std::cout << "vulkan initialized" << std::endl;

        bool framebuffer_resized = false;
        glfwSetWindowUserPointer(window, &framebuffer_resized);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int) {
            *static_cast<bool*>(glfwGetWindowUserPointer(w)) = true;
        });

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            if (framebuffer_resized) {
                framebuffer_resized = false;
                if (!recreate_swapchain())
                    continue;
                redraw(static_cast<float>(glfwGetTime() - start_time));
                continue;
            }

            redraw(static_cast<float>(glfwGetTime() - start_time));
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
