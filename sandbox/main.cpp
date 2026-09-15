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
#include "vulcao/fence.h"
#include "vulcao/image.h"
#include "vulcao/query_pool.h"
#include "vulcao/semaphore.h"

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
    cmd.begin_rendering(rendering_info);
    cmd.end_rendering();
    cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                   range);
    cmd.end();
}

void render_clear_frame(vulcao::Context& ctx,
                        const vulcao::Semaphore& image_available,
                        const vulcao::Semaphore& render_finished) {
    vk::Device device = ctx.device();
    vk::SwapchainKHR swapchain = ctx.swapchain();
    vulcao::CommandBuffer& cmd = ctx.immediate_command_buffer();

    uint32_t image_index =
        device.acquireNextImageKHR(swapchain, UINT64_MAX, image_available.handle()).value;

    record_clear(cmd, ctx.swapchain_images()[image_index],
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

    vk::Result present_result = ctx.present_queue().presentKHR(vk::PresentInfoKHR{
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

        vulcao::CommandBuffer update_cmd =
            vulcao::CommandBuffer::allocate(ctx.device(), ctx.command_pool());
        update_cmd.begin();
        update_cmd.update_buffer(readback.handle(), 0, vertex_data);
        update_cmd.end();
        ctx.submit_and_wait(update_cmd.handle());

        readback.invalidate();
        bool data_ok = std::equal(vertex_data.begin(), vertex_data.end(),
                                  static_cast<const uint32_t*>(readback.map()));
        readback.unmap();
        std::cout << "update_buffer: " << (data_ok ? "ok" : "MISMATCH") << std::endl;

        vulcao::CommandBuffer secondary = vulcao::CommandBuffer::allocate(
            ctx.device(), ctx.command_pool(), vk::CommandBufferLevel::eSecondary);
        secondary.begin();
        secondary.copy_buffer(vertex_buffer.handle(), readback.handle(), vertex_bytes);
        secondary.end();

        vulcao::QueryPool timestamps =
            vulcao::QueryPool::create(ctx.device(), vk::QueryType::eTimestamp, 2);
        vulcao::Buffer timing = vulcao::Buffer::create(
            ctx.allocator(), 2 * sizeof(uint64_t), vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        vulcao::CommandBuffer copy_cmd =
            vulcao::CommandBuffer::allocate(ctx.device(), ctx.command_pool());
        copy_cmd.begin();
        copy_cmd.fill_buffer(readback.handle(), 0, vertex_bytes, 0);
        copy_cmd.reset_query_pool(timestamps.handle(), 0, 2);
        copy_cmd.write_timestamp(timestamps.handle(), vk::PipelineStageFlagBits2::eTopOfPipe, 0);
        copy_cmd.buffer_barrier(vertex_buffer.handle(),
                                vk::PipelineStageFlagBits2::eTransfer,
                                vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eTransfer,
                                vk::AccessFlagBits2::eTransferRead);
        copy_cmd.execute_commands(secondary.handle());
        copy_cmd.write_timestamp(timestamps.handle(), vk::PipelineStageFlagBits2::eBottomOfPipe, 1);
        copy_cmd.copy_query_pool_results(
            timestamps.handle(), 0, 2, timing.handle(), 0, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        copy_cmd.end();

        vulcao::Fence copy_done = vulcao::Fence::create(ctx.device());
        ctx.submit(copy_cmd.handle(), copy_done.handle());
        std::cout << "copy submitted, doing CPU work while the GPU copies..." << std::endl;
        copy_done.wait();

        readback.invalidate();
        data_ok = std::equal(vertex_data.begin(), vertex_data.end(),
                             static_cast<const uint32_t*>(readback.map()));
        readback.unmap();
        std::cout << "fill + secondary + execute_commands: " << (data_ok ? "ok" : "MISMATCH")
                  << std::endl;

        timing.invalidate();
        const auto* stamp = static_cast<const uint64_t*>(timing.map());
        const float period = ctx.physical_device().getProperties().limits.timestampPeriod;
        std::cout << "copy time: " << static_cast<double>(stamp[1] - stamp[0]) * period << " ns"
                  << std::endl;
        timing.unmap();

        const std::array<uint8_t, 4> pixel{255, 128, 0, 255};
        const vk::ImageCreateInfo image_info{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = vk::Extent3D{1, 1, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
                     vk::ImageUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        vulcao::Image image = vulcao::Image::create(ctx.allocator(), image_info);
        ctx.upload(image, pixel);

        const uint32_t clear_value = 0x11223344u;
        vk::ClearColorValue clear_color{};
        clear_color.float32[0] = 68.0f / 255.0f;
        clear_color.float32[1] = 51.0f / 255.0f;
        clear_color.float32[2] = 34.0f / 255.0f;
        clear_color.float32[3] = 17.0f / 255.0f;

        vulcao::Buffer pixel_readback = vulcao::Buffer::create(
            ctx.allocator(), sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        vulcao::CommandBuffer clear_cmd =
            vulcao::CommandBuffer::allocate(ctx.device(), ctx.command_pool());
        clear_cmd.begin();
        clear_cmd.transition(image, vk::ImageLayout::eTransferDstOptimal);
        clear_cmd.clear_color_image(image, clear_color);
        clear_cmd.transition(image, vk::ImageLayout::eTransferSrcOptimal);
        clear_cmd.copy_image_to_buffer(pixel_readback.handle(), image);
        clear_cmd.end();
        ctx.submit_and_wait(clear_cmd.handle());

        pixel_readback.invalidate();
        const uint32_t read_value = *static_cast<const uint32_t*>(pixel_readback.map());
        pixel_readback.unmap();
        const bool clear_ok = read_value == clear_value;
        std::cout << "clear_color_image + copy_image_to_buffer: " << (clear_ok ? "ok" : "MISMATCH")
                  << " (read 0x" << std::hex << read_value << ", expected 0x" << clear_value << std::dec
                  << ")" << std::endl;

        vk::Device device = ctx.device();
        vulcao::Semaphore image_available = vulcao::Semaphore::create(device);
        vulcao::Semaphore render_finished = vulcao::Semaphore::create(device);

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
