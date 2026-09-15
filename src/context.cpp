#include "vulcao/context.h"
#include "vulcao/buffer.h"
#include "vulcao/check.h"
#include "vulcao/image.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vulcao {
namespace {

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        default: return "other";
    }
}

}

Context::Context(const ContextInfo& info) : info_(info) {
    create_instance(info_);
}

void Context::log(LogLevel level, std::string_view message) const {
    if (info_.log)
        info_.log(level, message);
}

Context::~Context() {
    if (device_) {
        device_.waitIdle();

        if (command_pool_) {
            immediate_command_buffer_.destroy();
            submit_fence_.destroy();
            device_.destroyCommandPool(command_pool_);
        }

        destroy_swapchain_resources();

        allocator_.destroy();

        if (vkb_swapchain_.swapchain)
            vkb::destroy_swapchain(vkb_swapchain_);
        vkb::destroy_device(vkb_device_);
    }

    if (surface_)
        instance_.destroySurfaceKHR(surface_);
    if (vkb_instance_.instance)
        vkb::destroy_instance(vkb_instance_);
}

void Context::initialize(vk::SurfaceKHR surface, vk::Extent2D extent, SwapchainInfo swapchain_info) {
    if (initialized())
        throw std::runtime_error("Context::initialize called twice");

    surface_ = surface;
    swapchain_info_ = std::move(swapchain_info);
    pick_physical_device();
    create_device();
    create_allocator();
    create_swapchain({}, extent);
    create_command_pool();
}

void Context::wait_idle() {
    if (device_)
        device_.waitIdle();
}

void Context::create_instance(const ContextInfo& info) {
    api_version_ = info.api_version;

    vkb::InstanceBuilder builder;
    builder.set_app_name(info.app_name.c_str())
        .set_app_version(info.app_version)
        .require_api_version(VK_VERSION_MAJOR(info.api_version), VK_VERSION_MINOR(info.api_version),
                             VK_VERSION_PATCH(info.api_version));

    if (info.validation)
        builder.request_validation_layers().use_default_debug_messenger();

    for (const char* extension : info.extensions)
        builder.enable_extension(extension);
    for (const char* layer : info.layers)
        builder.enable_layer(layer);

    vkb_instance_ = check(builder.build(), "create instance");
    instance_ = vk::Instance{vkb_instance_.instance};
}

std::vector<PhysicalDeviceInfo> Context::enumerate_physical_devices() const {
    std::vector<PhysicalDeviceInfo> result;
    for (const vk::PhysicalDevice& physical_device : instance_.enumeratePhysicalDevices()) {
        const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
        result.push_back(PhysicalDeviceInfo{
            .name = properties.deviceName,
            .type = properties.deviceType,
            .vendor_id = properties.vendorID,
            .driver_version = properties.driverVersion,
            .api_version = properties.apiVersion,
        });
    }
    return result;
}

void Context::pick_physical_device() {
    vkb::PhysicalDeviceSelector selector{vkb_instance_};
    selector.set_surface(surface_).set_minimum_version(1, 3);

    const DeviceFeatures& wanted = info_.device_features;
    selector.set_required_features(vk::PhysicalDeviceFeatures{
        .depthClamp = wanted.depth_clamp ? VK_TRUE : VK_FALSE,
        .fillModeNonSolid = wanted.fill_mode_non_solid ? VK_TRUE : VK_FALSE,
        .wideLines = wanted.wide_lines ? VK_TRUE : VK_FALSE,
        .samplerAnisotropy = wanted.sampler_anisotropy ? VK_TRUE : VK_FALSE,
        .drawIndirectFirstInstance = wanted.draw_indirect_first_instance ? VK_TRUE : VK_FALSE,
        .shaderInt64 = wanted.shader_int64 ? VK_TRUE : VK_FALSE,
    });

    selector.set_required_features_13(vk::PhysicalDeviceVulkan13Features{
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    });

    if (wanted.timeline_semaphore)
        selector.add_required_extension_features(
            vk::PhysicalDeviceTimelineSemaphoreFeatures{.timelineSemaphore = VK_TRUE});

    if (wanted.descriptor_indexing)
        selector.add_required_extension_features(vk::PhysicalDeviceDescriptorIndexingFeatures{
            .shaderInputAttachmentArrayDynamicIndexing = VK_TRUE,
            .shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
            .shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
            .shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
            .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
            .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
            .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
            .shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE,
            .shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
            .shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE,
            .descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE,
            .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
            .descriptorBindingPartiallyBound = VK_TRUE,
            .descriptorBindingVariableDescriptorCount = VK_TRUE,
            .runtimeDescriptorArray = VK_TRUE,
        });

    for (const char* extension : info_.device_extensions)
        selector.add_required_extension(extension);

    if (info_.separate_compute_queue)
        selector.require_separate_compute_queue();
    if (info_.separate_transfer_queue)
        selector.require_separate_transfer_queue();

    if (info_.customize_selector)
        info_.customize_selector(selector);

    vkb_physical_device_ = check(selector.select(), "select physical device");
    physical_device_ = vk::PhysicalDevice{vkb_physical_device_.physical_device};

    const vk::PhysicalDeviceProperties properties = physical_device_.getProperties();
    log(LogLevel::info, "GPU: " + std::string(properties.deviceName) + " (" +
                            device_type_name(static_cast<VkPhysicalDeviceType>(properties.deviceType)) +
                            ")");
}

void Context::create_device() {
    vkb_device_ = check(vkb::DeviceBuilder{vkb_physical_device_}.build(), "create device");
    device_ = vk::Device{vkb_device_.device};

    graphics_queue_family_index_ =
        check(vkb_device_.get_queue_index(vkb::QueueType::graphics), "get graphics queue index");
    graphics_queue_ = vk::Queue{check(vkb_device_.get_queue(vkb::QueueType::graphics), "get graphics queue")};
    present_queue_ = vk::Queue{check(vkb_device_.get_queue(vkb::QueueType::present), "get present queue")};

    if (info_.separate_compute_queue) {
        const vkb::Result<uint32_t> index = vkb_device_.get_queue_index(vkb::QueueType::compute);
        const vkb::Result<VkQueue> queue = vkb_device_.get_queue(vkb::QueueType::compute);
        if (index && queue) {
            compute_queue_family_index_ = index.value();
            compute_queue_ = vk::Queue{queue.value()};
            has_compute_queue_ = true;
        } else {
            log(LogLevel::warning, "dedicated compute queue was requested but is not available");
        }
    }

    if (info_.separate_transfer_queue) {
        const vkb::Result<uint32_t> index = vkb_device_.get_queue_index(vkb::QueueType::transfer);
        const vkb::Result<VkQueue> queue = vkb_device_.get_queue(vkb::QueueType::transfer);
        if (index && queue) {
            transfer_queue_family_index_ = index.value();
            transfer_queue_ = vk::Queue{queue.value()};
            has_transfer_queue_ = true;
        } else {
            log(LogLevel::warning, "dedicated transfer queue was requested but is not available");
        }
    }

    log(LogLevel::info, "queues: graphics=" + std::to_string(graphics_queue_family_index_) +
                            ", present=" +
                            std::to_string(
                                check(vkb_device_.get_queue_index(vkb::QueueType::present),
                                      "get present queue index")) +
                            ", compute=" +
                            (has_compute_queue_ ? std::to_string(compute_queue_family_index_) : "none") +
                            ", transfer=" +
                            (has_transfer_queue_ ? std::to_string(transfer_queue_family_index_) : "none"));
}

void Context::create_allocator() {
    allocator_.create(instance_, physical_device_, device_, api_version_);
}

void Context::create_swapchain(vk::SwapchainKHR oldSwapchain, vk::Extent2D extent) {
    vkb::SwapchainBuilder builder{vkb_device_, surface_};
    builder.set_desired_extent(extent.width, extent.height);

    if (!swapchain_info_.formats.empty()) {
        builder.set_desired_format(swapchain_info_.formats.front());
        for (size_t i = 1; i < swapchain_info_.formats.size(); ++i)
            builder.add_fallback_format(swapchain_info_.formats[i]);
    }

    if (!swapchain_info_.present_modes.empty()) {
        builder.set_desired_present_mode(
            static_cast<VkPresentModeKHR>(swapchain_info_.present_modes.front()));
        for (size_t i = 1; i < swapchain_info_.present_modes.size(); ++i)
            builder.add_fallback_present_mode(
                static_cast<VkPresentModeKHR>(swapchain_info_.present_modes[i]));
    }

    if (swapchain_info_.extra_usage != vk::ImageUsageFlags{})
        builder.add_image_usage_flags(static_cast<VkImageUsageFlags>(swapchain_info_.extra_usage));

    if (swapchain_info_.min_image_count > 0)
        builder.set_desired_min_image_count(swapchain_info_.min_image_count);

    if (oldSwapchain)
        builder.set_old_swapchain(oldSwapchain);

    vkb_swapchain_ = check(builder.build(), "create swapchain");

    swapchain_ = vk::SwapchainKHR{vkb_swapchain_.swapchain};
    swapchain_format_ = static_cast<vk::Format>(vkb_swapchain_.image_format);
    swapchain_extent_ = vk::Extent2D{vkb_swapchain_.extent.width, vkb_swapchain_.extent.height};

    auto images = check(vkb_swapchain_.get_images(), "get swapchain images");
    swapchain_images_.assign(images.begin(), images.end());

    auto views = check(vkb_swapchain_.get_image_views(), "create swapchain image views");
    swapchain_image_views_.assign(views.begin(), views.end());

    log(LogLevel::info, "swapchain: " + std::to_string(swapchain_extent_.width) + "x" +
                            std::to_string(swapchain_extent_.height) + ", " +
                            std::to_string(swapchain_images_.size()) + " images, " +
                            vk::to_string(swapchain_format_));
}

void Context::create_command_pool() {
    command_pool_ = device_.createCommandPool(vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = graphics_queue_family_index_,
    });

    immediate_command_buffer_ = CommandBuffer::allocate(device_, command_pool_);
    submit_fence_ = Fence::create(device_);
}

void Context::destroy_swapchain_resources() {
    for (auto view : swapchain_image_views_)
        device_.destroyImageView(view);
    swapchain_image_views_.clear();
    swapchain_images_.clear();
}

void Context::recreate_swapchain(vk::Extent2D extent) {
    if (!initialized())
        throw std::runtime_error("Context::recreate_swapchain before initialize");

    device_.waitIdle();

    destroy_swapchain_resources();

    vkb::Swapchain old_swapchain = vkb_swapchain_;
    create_swapchain(swapchain_, extent);
    if (old_swapchain.swapchain)
        vkb::destroy_swapchain(old_swapchain);
}

void Context::submit_and_wait(vk::CommandBuffer cmd) {
    submit_fence_.reset();
    submit(cmd, submit_fence_.handle());
    submit_fence_.wait();
}

void Context::submit(const vk::SubmitInfo& info, vk::Fence fence) {
    submit(graphics_queue_, info, fence);
}

void Context::submit(vk::Queue queue, const vk::SubmitInfo& info, vk::Fence fence) {
    queue.submit(info, fence);
}

void Context::submit(vk::Queue queue,
                     vk::CommandBuffer cmd,
                     const Semaphore& timeline_semaphore,
                     uint64_t signal_value,
                     vk::Fence fence) {
    const vk::TimelineSemaphoreSubmitInfo timeline_info{
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_value,
    };
    const vk::Semaphore semaphore = timeline_semaphore.handle();

    queue.submit(vk::SubmitInfo{
                     .pNext = &timeline_info,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &cmd,
                     .signalSemaphoreCount = 1,
                     .pSignalSemaphores = &semaphore,
                 },
                 fence);
}

void Context::submit(vk::CommandBuffer cmd, vk::Fence fence) {
    submit(vk::SubmitInfo{
               .commandBufferCount = 1,
               .pCommandBuffers = &cmd,
           },
           fence);
}

Fence Context::submit(vk::CommandBuffer cmd) {
    Fence fence = Fence::create(device_);
    submit(cmd, fence.handle());
    return fence;
}

void Context::upload(Buffer& dst, const void* data, vk::DeviceSize size) {
    if (size == 0)
        return;
    if (!dst.valid())
        throw std::runtime_error("Context::upload: invalid destination buffer");
    if (!(dst.usage() & vk::BufferUsageFlagBits::eTransferDst))
        throw std::runtime_error("Context::upload: destination buffer requires TransferDst usage");
    if (size > dst.size())
        throw std::runtime_error("Context::upload: data size exceeds destination buffer size");

    Buffer staging = Buffer::create(allocator_, size, vk::BufferUsageFlagBits::eTransferSrc,
                                    VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    staging.write_bytes(data, size);

    immediate([&](CommandBuffer& cmd) {
        cmd.copy_buffer(staging.handle(), dst.handle(), size);
    });
}

void Context::upload(Image& dst, const void* data, vk::DeviceSize size, vk::ImageLayout final_layout,
                     bool generate_mips) {
    if (size == 0)
        return;
    if (!dst.valid())
        throw std::runtime_error("Context::upload: invalid destination image");
    if (!(dst.usage() & vk::ImageUsageFlagBits::eTransferDst))
        throw std::runtime_error("Context::upload: destination image requires TransferDst usage");

    Buffer staging = Buffer::create(allocator_, size, vk::BufferUsageFlagBits::eTransferSrc,
                                    VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    staging.write_bytes(data, size);

    immediate([&](CommandBuffer& cmd) {
        cmd.transition(dst, vk::ImageLayout::eTransferDstOptimal);
        cmd.copy_buffer_to_image(staging.handle(), dst);
        if (generate_mips && dst.mip_levels() > 1)
            cmd.generate_mipmaps(dst, final_layout);
        else
            cmd.transition(dst, final_layout);
    });
}

}
