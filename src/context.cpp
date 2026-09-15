#include "vulcao/context.h"
#include "vulcao/buffer.h"
#include "vulcao/check.h"
#include "vulcao/image.h"

#include <iostream>
#include <stdexcept>
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

Context::Context(const ContextInfo& info) {
    create_instance(info);
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

void Context::inquery_physical_devices_info() {
    std::cout << "------------------- devices info -------------------" << std::endl;

    auto physical_devices = instance_.enumeratePhysicalDevices();
    if (physical_devices.empty())
        std::cerr << "not find physical device!!!!!!!\n";

    for (const auto& physical_device : physical_devices) {
        auto properties = physical_device.getProperties();

        std::string vendor_name;
        switch (properties.vendorID) {
            case 0x10DE: vendor_name = "NVIDIA"; break;
            case 0x1002: vendor_name = "AMD"; break;
            case 0x8086: vendor_name = "Intel"; break;
            case 0x13B5: vendor_name = "ARM"; break;
            default: vendor_name = "Unknown"; break;
        }
        std::cout << "find gpu vendor: " << vendor_name << ". \n";

        std::cout << "device name: " << properties.deviceName << ". \n";

        std::string device_type;
        switch (properties.deviceType) {
            case vk::PhysicalDeviceType::eCpu: device_type = "CPU"; break;
            case vk::PhysicalDeviceType::eOther: device_type = "Other"; break;
            case vk::PhysicalDeviceType::eVirtualGpu: device_type = "Virtual GPU"; break;
            case vk::PhysicalDeviceType::eDiscreteGpu: device_type = "Discrete GPU"; break;
            case vk::PhysicalDeviceType::eIntegratedGpu: device_type = "Integrated GPU"; break;
            default: device_type = "Unknown"; break;
        }
        std::cout << "device type: " << device_type << ". \n";

        std::cout << "driver version: " << properties.driverVersion << ". \n";
    }
}

void Context::pick_physical_device() {
    inquery_physical_devices_info();

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;

    vkb_physical_device_ =
        check(vkb::PhysicalDeviceSelector{vkb_instance_}
                      .set_surface(surface_)
                      .set_minimum_version(1, 3)
                      .set_required_features_13(features13)
                      .select(),
                  "select physical device");
    physical_device_ = vk::PhysicalDevice{vkb_physical_device_.physical_device};

    const auto& props = vkb_physical_device_.properties;
    std::cout << "GPU: " << props.deviceName << " (" << device_type_name(props.deviceType) << ")"
              << std::endl;
}

void Context::create_device() {
    std::vector<vk::QueueFamilyProperties> queue_family_properties =
        physical_device_.getQueueFamilyProperties();
    std::cout << "number of queue families: " << queue_family_properties.size() << std::endl;
    for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
        std::cout << "Queue family " << i << ": " << queue_family_properties[i].queueCount
                  << " queues, flags: " << vk::to_string(queue_family_properties[i].queueFlags)
                  << std::endl;
    }

    vkb_device_ = check(vkb::DeviceBuilder{vkb_physical_device_}.build(), "create device");
    device_ = vk::Device{vkb_device_.device};

    graphics_queue_family_index_ =
        check(vkb_device_.get_queue_index(vkb::QueueType::graphics), "get graphics queue index");
    graphics_queue_ = vk::Queue{check(vkb_device_.get_queue(vkb::QueueType::graphics), "get graphics queue")};
    present_queue_ = vk::Queue{check(vkb_device_.get_queue(vkb::QueueType::present), "get present queue")};

    std::cout << "queue families: graphics=" << graphics_queue_family_index_
              << ", present=" << check(vkb_device_.get_queue_index(vkb::QueueType::present), "get present queue index")
              << std::endl;
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

    std::cout << "swapchain: " << swapchain_extent_.width << "x" << swapchain_extent_.height
              << ", " << swapchain_images_.size() << " images, " << vk::to_string(swapchain_format_)
              << std::endl;
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
    graphics_queue_.submit(info, fence);
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
