#pragma once

#include <cstdint>
#include <functional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>

#include "vulcao/allocator.h"
#include "vulcao/command_buffer.h"
#include "vulcao/command_pool.h"
#include "vulcao/fence.h"
#include "vulcao/log.h"
#include "vulcao/semaphore.h"

namespace vulcao {

class Buffer;
class Image;

/// @brief Device features that can be requested when creating the device.
struct DeviceFeatures {
    bool sampler_anisotropy = false;
    bool timeline_semaphore = false;
    bool descriptor_indexing = false;
    bool shader_int64 = false;
    bool fill_mode_non_solid = false;
    bool wide_lines = false;
    bool depth_clamp = false;
    bool draw_indirect_first_instance = false;
};

/// @brief Creation parameters of a Context.
struct ContextInfo {
    std::string app_name = "vulcao";
    uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);
    uint32_t api_version = VK_API_VERSION_1_3;
#ifdef NDEBUG
    bool validation = false;
#else
    bool validation = true;
#endif
    /// @brief Severities requested from the debug messenger when validation is enabled.
    vk::DebugUtilsMessageSeverityFlagsEXT validation_severity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    /// @brief Create the instance without presentation extensions and skip the swapchain.
    bool headless = false;
    std::vector<const char*> extensions;
    std::vector<const char*> layers;
    std::vector<const char*> device_extensions;
    DeviceFeatures device_features;
    bool separate_compute_queue = false;
    bool separate_transfer_queue = false;
    std::function<void(vkb::PhysicalDeviceSelector&)> customize_selector;
};

/// @brief Information about a physical device.
struct PhysicalDeviceInfo {
    std::string name;
    vk::PhysicalDeviceType type = vk::PhysicalDeviceType::eOther;
    uint32_t vendor_id = 0;
    uint32_t driver_version = 0;
    uint32_t api_version = 0;
};

/// @brief Creation parameters of the swapchain.
struct SwapchainInfo {
    std::vector<vk::SurfaceFormatKHR> formats{
        vk::SurfaceFormatKHR{.format = vk::Format::eB8G8R8A8Srgb,
                             .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear}};
    std::vector<vk::PresentModeKHR> present_modes{
        vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifo};
    vk::ImageUsageFlags extra_usage;
    uint32_t min_image_count = 0;
};

/// @brief Owns the Vulkan instance, device, swapchain, command pool and VMA allocator.
class Context {
public:
    /// @brief Creates the instance.
    /// @param info Context creation parameters.
    explicit Context(const ContextInfo& info = {});

    /// @brief Destroys all owned Vulkan objects.
    ~Context();

    /// @brief Not copyable.
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /// @brief Not movable.
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    /// @brief Picks a device, creates the allocator, swapchain and command pool.
    /// @param surface Presentation surface.
    /// @param extent Initial swapchain extent.
    /// @param swapchain_info Swapchain creation parameters.
    void initialize(vk::SurfaceKHR surface, vk::Extent2D extent, SwapchainInfo swapchain_info = {});

    /// @brief Picks a device and creates the allocator, command pool and queues without a swapchain.
    ///
    /// Requires ContextInfo::headless. No surface is created and present_queue() stays null.
    void initialize();

    /// @brief Recreates the swapchain with a new extent.
    /// @param extent New swapchain extent.
    void recreate_swapchain(vk::Extent2D extent);

    /// @brief Waits for the device to become idle.
    void wait_idle();

    /// @brief Returns true if the context was created without a surface or swapchain.
    bool headless() const { return info_.headless; }

    /// @brief Returns true if VK_EXT_debug_utils is enabled.
    bool debug_utils_enabled() const { return debug_utils_enabled_; }

    /// @brief Sets a debug name on a Vulkan object. No-op without debug utils.
    /// @param type Object type.
    /// @param handle Raw object handle.
    /// @param name Name to assign.
    void set_debug_name(vk::ObjectType type, uint64_t handle, const char* name) const;

    /// @brief Submits a command buffer on the graphics queue and waits for it.
    /// @param cmd Command buffer to submit.
    void submit_and_wait(vk::CommandBuffer cmd);

    /// @brief Submits commands on the graphics queue without waiting.
    /// @param info Submission parameters.
    /// @param fence Optional fence signaled when the submission completes.
    void submit(const vk::SubmitInfo& info, vk::Fence fence = {});

    /// @brief Submits a command buffer on the graphics queue without waiting.
    /// @param cmd Command buffer to submit.
    /// @param fence Fence signaled when the submission completes.
    void submit(vk::CommandBuffer cmd, vk::Fence fence);

    /// @brief Submits a command buffer on the graphics queue without waiting.
    /// @param cmd Command buffer to submit.
    /// @return A new fence signaled when the submission completes.
    Fence submit(vk::CommandBuffer cmd);

    /// @brief Submits commands on an arbitrary queue without waiting.
    /// @param queue Queue to submit to.
    /// @param info Submission parameters.
    /// @param fence Optional fence signaled when the submission completes.
    void submit(vk::Queue queue, const vk::SubmitInfo& info, vk::Fence fence = {});

    /// @brief Submits a command buffer that signals a timeline semaphore on completion.
    /// @param queue Queue to submit to.
    /// @param cmd Command buffer to submit.
    /// @param timeline_semaphore Timeline semaphore signaled by the submission.
    /// @param signal_value Value the semaphore is signaled with.
    /// @param fence Optional fence signaled when the submission completes.
    void submit(vk::Queue queue,
                vk::CommandBuffer cmd,
                const Semaphore& timeline_semaphore,
                uint64_t signal_value,
                vk::Fence fence = {});

    /// @brief Records one-time commands with the internal command buffer, submits and waits.
    /// @param fn Callable that records commands, invoked with a CommandBuffer reference.
    template <typename Fn>
    void immediate(Fn&& fn) {
        immediate_command_buffer_.reset();
        immediate_command_buffer_.begin();
        fn(immediate_command_buffer_);
        immediate_command_buffer_.end();
        submit_and_wait(immediate_command_buffer_.handle());
    }

    /// @brief Uploads raw bytes to a device local buffer through a staging buffer.
    /// @param dst Destination buffer, must have TransferDst usage.
    /// @param data Source pointer.
    /// @param size Number of bytes to upload.
    void upload(Buffer& dst, const void* data, vk::DeviceSize size);

    /// @brief Uploads a contiguous range to a device local buffer through a staging buffer.
    /// @param dst Destination buffer, must have TransferDst usage.
    /// @param data Source range.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    void upload(Buffer& dst, const Container& data) {
        using T = std::ranges::range_value_t<Container>;
        upload(dst,
               std::ranges::data(data),
               static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T));
    }

    /// @brief Uploads raw bytes to an image and transitions it to a final layout.
    /// @param dst Destination image, must have TransferDst usage.
    /// @param data Source pointer.
    /// @param size Number of bytes to upload.
    /// @param final_layout Layout the image is transitioned to after the upload.
    /// @param generate_mips True to generate the mip chain after uploading level 0.
    void upload(Image& dst,
                const void* data,
                vk::DeviceSize size,
                vk::ImageLayout final_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
                bool generate_mips = false);

    /// @brief Uploads a contiguous range to an image and transitions it to a final layout.
    /// @param dst Destination image, must have TransferDst usage.
    /// @param data Source range.
    /// @param final_layout Layout the image is transitioned to after the upload.
    /// @param generate_mips True to generate the mip chain after uploading level 0.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    void upload(Image& dst,
                const Container& data,
                vk::ImageLayout final_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
                bool generate_mips = false) {
        using T = std::ranges::range_value_t<Container>;
        upload(dst,
               std::ranges::data(data),
               static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T),
               final_layout,
               generate_mips);
    }

    /// @brief Returns true if the context has been initialized.
    bool initialized() const { return static_cast<bool>(device_); }

    /// @brief Returns information about all physical devices.
    std::vector<PhysicalDeviceInfo> enumerate_physical_devices() const;

    /// @brief Returns the allocator owned by this context.
    Allocator& allocator() { return allocator_; }

    /// @brief Returns the allocator owned by this context.
    const Allocator& allocator() const { return allocator_; }

    /// @brief Returns the Vulkan instance.
    vk::Instance instance() const { return instance_; }

    /// @brief Returns the presentation surface.
    vk::SurfaceKHR surface() const { return surface_; }

    /// @brief Returns the selected physical device.
    vk::PhysicalDevice physical_device() const { return physical_device_; }

    /// @brief Returns the logical device.
    vk::Device device() const { return device_; }

    /// @brief Returns the graphics queue.
    vk::Queue graphics_queue() const { return graphics_queue_; }

    /// @brief Returns the present queue.
    vk::Queue present_queue() const { return present_queue_; }

    /// @brief Returns the dedicated compute queue, or null if none was requested.
    vk::Queue compute_queue() const { return compute_queue_; }

    /// @brief Returns the dedicated transfer queue, or null if none was requested.
    vk::Queue transfer_queue() const { return transfer_queue_; }

    /// @brief Returns the queue family index used for graphics.
    uint32_t graphics_queue_family_index() const { return graphics_queue_family_index_; }

    /// @brief Returns the queue family index used for compute.
    uint32_t compute_queue_family_index() const { return compute_queue_family_index_; }

    /// @brief Returns the queue family index used for transfer.
    uint32_t transfer_queue_family_index() const { return transfer_queue_family_index_; }

    /// @brief Returns true if a dedicated compute queue is available.
    bool has_compute_queue() const { return has_compute_queue_; }

    /// @brief Returns true if a dedicated transfer queue is available.
    bool has_transfer_queue() const { return has_transfer_queue_; }

    /// @brief Returns the swapchain.
    vk::SwapchainKHR swapchain() const { return swapchain_; }

    /// @brief Returns the swapchain image format.
    vk::Format swapchain_format() const { return swapchain_format_; }

    /// @brief Returns the swapchain extent.
    vk::Extent2D swapchain_extent() const { return swapchain_extent_; }

    /// @brief Returns the swapchain images.
    const std::vector<vk::Image>& swapchain_images() const { return swapchain_images_; }

    /// @brief Returns the swapchain image views.
    const std::vector<vk::ImageView>& swapchain_image_views() const { return swapchain_image_views_; }

    /// @brief Returns the command pool used for one-time commands.
    vk::CommandPool command_pool() const { return command_pool_.handle(); }

    /// @brief Returns the command buffer used by immediate().
    CommandBuffer& immediate_command_buffer() { return immediate_command_buffer_; }

    /// @brief Returns the command buffer used by immediate().
    const CommandBuffer& immediate_command_buffer() const { return immediate_command_buffer_; }

private:
    /// @brief Creates the instance and the debug messenger.
    void create_instance(const ContextInfo& info);

    /// @brief Sends a message to the log callback if one is set.
    void log(LogLevel level, std::string_view message) const;

    /// @brief Selects a suitable physical device.
    void pick_physical_device();

    /// @brief Creates the logical device and retrieves the queues.
    void create_device();

    /// @brief Creates the VMA allocator.
    void create_allocator();

    /// @brief Creates the swapchain and its image views.
    void create_swapchain(vk::SwapchainKHR oldSwapchain, vk::Extent2D extent);

    /// @brief Creates the command pool and the immediate command buffer.
    void create_command_pool();

    /// @brief Destroys swapchain images and image views.
    void destroy_swapchain_resources();

    vkb::Instance vkb_instance_;
    vkb::PhysicalDevice vkb_physical_device_;
    vkb::Device vkb_device_;
    vkb::Swapchain vkb_swapchain_;

    vk::Instance instance_;
    vk::SurfaceKHR surface_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    uint32_t api_version_ = VK_API_VERSION_1_3;
    vk::Queue graphics_queue_;
    vk::Queue present_queue_;
    vk::Queue compute_queue_;
    vk::Queue transfer_queue_;
    uint32_t graphics_queue_family_index_ = 0;
    uint32_t compute_queue_family_index_ = 0;
    uint32_t transfer_queue_family_index_ = 0;
    bool has_compute_queue_ = false;
    bool has_transfer_queue_ = false;
    bool debug_utils_enabled_ = false;
    PFN_vkSetDebugUtilsObjectNameEXT set_debug_name_ext_ = nullptr;
    Allocator allocator_;
    ContextInfo info_;

    vk::SwapchainKHR swapchain_;
    vk::Format swapchain_format_ = vk::Format::eUndefined;
    vk::Extent2D swapchain_extent_{};
    std::vector<vk::Image> swapchain_images_;
    std::vector<vk::ImageView> swapchain_image_views_;
    SwapchainInfo swapchain_info_;

    CommandPool command_pool_;
    CommandBuffer immediate_command_buffer_;
    Fence submit_fence_;
};

}
