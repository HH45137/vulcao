#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>

namespace vulcao {

class Context {
public:
    explicit Context(const std::string& appName = "vulcao",
                     uint32_t appVersion = VK_MAKE_VERSION(1, 0, 0));
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    void initialize(vk::SurfaceKHR surface, vk::Extent2D extent);

    void recreate_swapchain(vk::Extent2D extent);

    bool initialized() const { return static_cast<bool>(device_); }

    void inquery_physical_devices_info();

    vk::Instance instance() const { return instance_; }
    vk::SurfaceKHR surface() const { return surface_; }
    vk::PhysicalDevice physical_device() const { return physical_device_; }
    vk::Device device() const { return device_; }
    vk::Queue graphics_queue() const { return graphics_queue_; }
    vk::Queue present_queue() const { return present_queue_; }
    uint32_t graphics_queue_family_index() const { return graphics_queue_family_index_; }
    vk::SwapchainKHR swapchain() const { return swapchain_; }
    vk::Format swapchain_format() const { return swapchain_format_; }
    vk::Extent2D swapchain_extent() const { return swapchain_extent_; }
    const std::vector<vk::Image>& swapchain_images() const { return swapchain_images_; }
    const std::vector<vk::ImageView>& swapchain_image_views() const { return swapchain_image_views_; }
    vk::CommandPool command_pool() const { return command_pool_; }
    vk::CommandBuffer immediate_command_buffer() const { return immediate_command_buffer_; }

private:
    void create_instance(const std::string& appName, uint32_t appVersion);
    void pick_physical_device();
    void create_device();
    void create_swapchain(vk::SwapchainKHR oldSwapchain, vk::Extent2D extent);
    void create_command_pool();
    void destroy_swapchain_resources();

    vkb::Instance vkb_instance_;
    vkb::PhysicalDevice vkb_physical_device_;
    vkb::Device vkb_device_;
    vkb::Swapchain vkb_swapchain_;

    vk::Instance instance_;
    vk::SurfaceKHR surface_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::Queue graphics_queue_;
    vk::Queue present_queue_;
    uint32_t graphics_queue_family_index_ = 0;

    vk::SwapchainKHR swapchain_;
    vk::Format swapchain_format_ = vk::Format::eUndefined;
    vk::Extent2D swapchain_extent_{};
    std::vector<vk::Image> swapchain_images_;
    std::vector<vk::ImageView> swapchain_image_views_;

    vk::CommandPool command_pool_;
    vk::CommandBuffer immediate_command_buffer_;
};

}
