#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "vulcao/buffer.h"
#include "vulcao/context.h"
#include "vulcao/image.h"

namespace {

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

void record_clear(vulcao::CommandBuffer& cmd,
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
    cmd.handle().beginRendering(rendering_info);
    cmd.handle().endRendering();
    cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                   range);
    cmd.end();
}

void render_clear_frame(vulcao::Context& ctx,
                        vk::Semaphore image_available,
                        vk::Semaphore render_finished) {
    vk::Device device = ctx.device();
    vk::SwapchainKHR swapchain = ctx.swapchain();
    vulcao::CommandBuffer& cmd = ctx.immediate_command_buffer();

    uint32_t image_index =
        device.acquireNextImageKHR(swapchain, UINT64_MAX, image_available).value;

    record_clear(cmd, ctx.swapchain_images()[image_index],
                 ctx.swapchain_image_views()[image_index], ctx.swapchain_extent());

    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    const vk::CommandBuffer raw_cmd = cmd.handle();
    ctx.graphics_queue().submit(vk::SubmitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &image_available,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &raw_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &render_finished,
    });

    vk::Result present_result = ctx.present_queue().presentKHR(vk::PresentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_finished,
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
        vulcao::Context ctx{"vulcao-game"};
        ctx.initialize(create_surface(ctx.instance(), window), framebuffer_extent(window));

        const std::vector<uint32_t> vertex_data{0, 1, 2, 3, 4, 5};
        const vk::DeviceSize vertex_bytes = vertex_data.size() * sizeof(uint32_t);

        vulcao::Buffer vertex_buffer = vulcao::Buffer::create(
            ctx.allocator(), vertex_bytes,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst);
        ctx.upload(vertex_buffer, vertex_data);

        vulcao::Buffer readback = vulcao::Buffer::create(
            ctx.allocator(), vertex_bytes, vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        ctx.immediate([&](vulcao::CommandBuffer& cmd) {
            cmd.copy_buffer(vertex_buffer.handle(), readback.handle(), vertex_bytes);
        });

        readback.invalidate();
        const auto* read_data = static_cast<const uint32_t*>(readback.map());
        const bool upload_ok = std::equal(vertex_data.begin(), vertex_data.end(), read_data);
        readback.unmap();
        std::cout << "vertex buffer upload: " << (upload_ok ? "ok" : "MISMATCH") << std::endl;

        const std::array<uint8_t, 4> pixel{255, 128, 0, 255};
        const vk::ImageCreateInfo image_info{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = vk::Extent3D{1, 1, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        vulcao::Image image = vulcao::Image::create(ctx.allocator(), image_info);
        ctx.upload(image, pixel);
        std::cout << "image upload: ok, extent=" << image.extent().width << "x" << image.extent().height
                  << std::endl;

        vk::Device device = ctx.device();
        vk::Semaphore image_available = device.createSemaphore({});
        vk::Semaphore render_finished = device.createSemaphore({});

        auto redraw = [&]() {
            try {
                render_clear_frame(ctx, image_available, render_finished);
            } catch (const vk::OutOfDateKHRError&) {
                vk::Extent2D extent = framebuffer_extent(window);
                if (extent.width == 0 || extent.height == 0)
                    return;
                ctx.recreate_swapchain(extent);
                render_clear_frame(ctx, image_available, render_finished);
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

        device.waitIdle();
        device.destroySemaphore(image_available);
        device.destroySemaphore(render_finished);
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
