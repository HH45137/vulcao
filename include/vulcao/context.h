#pragma once

#include <cstdint>
#include <functional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>

#include "vulcao/allocator.h"
#include "vulcao/buffer.h"
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
    bool sampler_anisotropy = false;         ///< Enable anisotropic filtering.
    bool timeline_semaphore = false;         ///< Enable timeline semaphores.
    bool descriptor_indexing = false;        ///< Enable the descriptor indexing feature set.
    bool shader_int64 = false;               ///< Enable 64 bit integers in shaders.
    bool fill_mode_non_solid = false;        ///< Enable point and wireframe polygon modes.
    bool wide_lines = false;                 ///< Enable line widths other than 1.
    bool depth_clamp = false;                ///< Enable depth clamping.
    bool draw_indirect_first_instance = false; ///< Enable the firstInstance parameter of indirect draws.
    bool shader_draw_parameters = false;       ///< Enable the DrawParameters capability for gl_VertexIndex.
};

/// @brief Creation parameters of a Context.
struct ContextInfo {
    std::string app_name = "vulcao";                     ///< Application name reported to the driver.
    uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);     ///< Application version reported to the driver.
    uint32_t api_version = VK_API_VERSION_1_3;           ///< Vulkan version to require.
    /// @brief Enable the validation layers and the debug messenger.
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
    std::vector<const char*> extensions;         ///< Additional instance extensions to enable.
    std::vector<const char*> layers;             ///< Additional instance layers to enable.
    std::vector<const char*> device_extensions;  ///< Device extensions to require.
    DeviceFeatures device_features;              ///< Device features to require.
    bool separate_compute_queue = false;         ///< Require a compute queue family distinct from graphics.
    bool separate_transfer_queue = false;        ///< Require a transfer queue family distinct from graphics.
    std::function<void(vkb::PhysicalDeviceSelector&)> customize_selector; ///< Hook to customize device selection.
};

/// @brief Information about a physical device.
struct PhysicalDeviceInfo {
    std::string name;                                    ///< Device name.
    vk::PhysicalDeviceType type = vk::PhysicalDeviceType::eOther; ///< Device type.
    uint32_t vendor_id = 0;                              ///< Vendor id.
    uint32_t driver_version = 0;                         ///< Driver version.
    uint32_t api_version = 0;                            ///< Highest supported Vulkan version.
};

/// @brief Creation parameters of the swapchain.
struct SwapchainInfo {
    /// @brief Surface formats to try, in priority order.
    std::vector<vk::SurfaceFormatKHR> formats{
        vk::SurfaceFormatKHR{.format = vk::Format::eB8G8R8A8Srgb,
                             .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear}};
    /// @brief Present modes to try, in priority order.
    std::vector<vk::PresentModeKHR> present_modes{
        vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifo};
    /// @brief Extra usage flags to request for the swapchain images.
    vk::ImageUsageFlags extra_usage;
    /// @brief Minimum image count, or 0 to let the implementation decide.
    uint32_t min_image_count = 0;
};

/// @brief Owns the Vulkan instance, device, swapchain, command pool and VMA allocator.
/// @note immediate(), upload() and download() share an internal command buffer and
///       staging buffer and are therefore not thread safe.
class Context {
public:
    /// @brief Creates the instance.
    /// @param info Context creation parameters.
    /// @throws std::runtime_error if the instance cannot be created.
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
    /// @throws std::runtime_error if already initialized or created headless.
    void initialize(vk::SurfaceKHR surface, vk::Extent2D extent, SwapchainInfo swapchain_info = {});

    /// @brief Picks a device and creates the allocator, command pool and queues without a swapchain.
    ///
    /// Requires ContextInfo::headless. No surface is created and present_queue() stays null.
    /// @throws std::runtime_error if already initialized or ContextInfo::headless is false.
    void initialize();

    /// @brief Recreates the swapchain with a new extent.
    /// @param extent New swapchain extent.
    /// @throws std::runtime_error if the context is not initialized, is headless, or the
    ///         swapchain cannot be created.
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
    ///
    /// Not reentrant: the context owns a single command buffer, so calling
    /// immediate() from inside @p fn throws instead of resetting the buffer that
    /// is currently being recorded. The flag is cleared even if @p fn throws.
    /// @param fn Callable that records commands, invoked with a CommandBuffer reference.
    /// @throws std::runtime_error if called reentrantly or if the submission fails.
    template <typename Fn>
    void immediate(Fn&& fn) {
        if (immediate_active_)
            throw std::runtime_error("Context::immediate is not reentrant");

        const ImmediateGuard guard{*this};
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
    /// @throws std::runtime_error if dst is invalid, lacks TransferDst usage or is too small.
    void upload(Buffer& dst, const void* data, vk::DeviceSize size);

    /// @brief Uploads a contiguous range to a device local buffer through a staging buffer.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param dst Destination buffer, must have TransferDst usage.
    /// @param data Source range.
    /// @throws std::runtime_error if dst is invalid, lacks TransferDst usage or is too small.
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
    /// @param size Number of bytes to upload. Must cover mip 0 of the image.
    /// @param final_layout Layout the image is transitioned to after the upload.
    /// @param generate_mips True to generate the mip chain after uploading level 0.
    /// @throws std::runtime_error if dst is invalid, lacks TransferDst usage, or size is
    ///         smaller than image_byte_size(dst.extent(), dst.format()).
    void upload(Image& dst,
                const void* data,
                vk::DeviceSize size,
                vk::ImageLayout final_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
                bool generate_mips = false);

    /// @brief Uploads a contiguous range to an image and transitions it to a final layout.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param dst Destination image, must have TransferDst usage.
    /// @param data Source range.
    /// @param final_layout Layout the image is transitioned to after the upload.
    /// @param generate_mips True to generate the mip chain after uploading level 0.
    /// @throws std::runtime_error if dst is invalid or lacks TransferDst usage.
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

    /// @brief Reads raw bytes from a buffer into host memory through a staging buffer.
    /// @param src Source buffer, must have TransferSrc usage.
    /// @param data Destination pointer.
    /// @param size Number of bytes to read.
    /// @throws std::runtime_error if src is invalid, lacks TransferSrc usage or is too small.
    void download(const Buffer& src, void* data, vk::DeviceSize size);

    /// @brief Reads from a buffer into a contiguous range.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param src Source buffer, must have TransferSrc usage.
    /// @param data Destination range.
    /// @throws std::runtime_error if src is invalid, lacks TransferSrc usage or is too small.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    void download(const Buffer& src, Container& data) {
        using T = std::ranges::range_value_t<Container>;
        static_assert(std::is_trivially_copyable_v<T>, "buffer data must be trivially copyable");
        download(src,
                 std::ranges::data(data),
                 static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T));
    }

    /// @brief Reads mip 0, layer 0 of an image into host memory and restores its layout.
    /// @param src Source image, must have TransferSrc usage.
    /// @param data Destination pointer.
    /// @param size Number of bytes to read. Must cover mip 0 of the image.
    /// @throws std::runtime_error if src is invalid, lacks TransferSrc usage, or size is
    ///         smaller than image_byte_size(src.extent(), src.format()).
    void download(Image& src, void* data, vk::DeviceSize size);

    /// @brief Reads mip 0, layer 0 of an image into a contiguous range.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param src Source image, must have TransferSrc usage.
    /// @param data Destination range.
    /// @throws std::runtime_error if src is invalid or lacks TransferSrc usage.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    void download(Image& src, Container& data) {
        using T = std::ranges::range_value_t<Container>;
        static_assert(std::is_trivially_copyable_v<T>, "image data must be trivially copyable");
        download(src,
                 std::ranges::data(data),
                 static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T));
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
    /// @brief Marks the context as recording and clears the flag on destruction.
    ///
    /// Keeps the reentrancy flag correct when the recording callable throws.
    struct ImmediateGuard {
        explicit ImmediateGuard(Context& context) : context_(context) {
            context_.immediate_active_ = true;
        }

        ~ImmediateGuard() { context_.immediate_active_ = false; }

        ImmediateGuard(const ImmediateGuard&) = delete;
        ImmediateGuard& operator=(const ImmediateGuard&) = delete;

        Context& context_;
    };

    /// @brief Creates the instance and the debug messenger.
    void create_instance(const ContextInfo& info);

    /// @brief Creates the debug messenger for the current instance.
    /// @param severity Severities the messenger forwards.
    /// @throws std::runtime_error if the entry point is missing or creation fails.
    void create_debug_messenger(vk::DebugUtilsMessageSeverityFlagsEXT severity);

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

    /// @brief Returns a host visible staging buffer that holds at least size bytes.
    Buffer& staging(vk::DeviceSize size);

    vkb::Instance vkb_instance_;
    vkb::PhysicalDevice vkb_physical_device_;
    vkb::Device vkb_device_;
    vkb::Swapchain vkb_swapchain_;

    vk::Instance instance_;
    vk::SurfaceKHR surface_;
    vk::DebugUtilsMessengerEXT debug_messenger_;
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
    Buffer staging_;
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
    bool immediate_active_ = false;
};

}
