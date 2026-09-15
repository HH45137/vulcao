#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

class Image;

/// @brief RAII wrapper around a Vulkan command buffer with recording helpers.
class CommandBuffer {
public:
    /// @brief Creates an empty command buffer.
    CommandBuffer() = default;

    /// @brief Frees the command buffer from its pool.
    ~CommandBuffer();

    /// @brief Not copyable.
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    /// @brief Moves the command buffer, leaving the source empty.
    CommandBuffer(CommandBuffer&& other) noexcept;

    /// @brief Move assignment. Destroys the current command buffer first.
    CommandBuffer& operator=(CommandBuffer&& other) noexcept;

    /// @brief Allocates one command buffer from a pool.
    /// @param device Device that owns the pool.
    /// @param pool Pool to allocate from.
    /// @param level Command buffer level.
    /// @return The allocated command buffer.
    static CommandBuffer allocate(vk::Device device,
                                  vk::CommandPool pool,
                                  vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);

    /// @brief Returns true if the command buffer holds a valid handle.
    bool valid() const { return static_cast<bool>(cmd_); }

    /// @brief Returns true if the command buffer holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan command buffer handle.
    vk::CommandBuffer handle() const { return cmd_; }

    /// @brief Frees the command buffer and resets the wrapper.
    void destroy();

    /// @brief Resets the command buffer to the initial recording state.
    /// @param flags Reset flags.
    /// @return This command buffer.
    CommandBuffer& reset(vk::CommandBufferResetFlags flags = {});

    /// @brief Starts recording.
    /// @param flags Usage flags for the recording.
    /// @return This command buffer.
    CommandBuffer& begin(vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    /// @brief Stops recording.
    /// @return This command buffer.
    CommandBuffer& end();

    /// @brief Inserts a global memory barrier.
    /// @param src_stage Source pipeline stages.
    /// @param src_access Source access flags.
    /// @param dst_stage Destination pipeline stages.
    /// @param dst_access Destination access flags.
    /// @return This command buffer.
    CommandBuffer& barrier(vk::PipelineStageFlags2 src_stage,
                           vk::AccessFlags2 src_access,
                           vk::PipelineStageFlags2 dst_stage,
                           vk::AccessFlags2 dst_access);

    /// @brief Transitions an image layout using raw handles.
    /// @param image Image to transition.
    /// @param old_layout Current layout.
    /// @param new_layout Target layout.
    /// @param range Subresource range to transition.
    /// @param src_stage Source stages. Derived from old_layout when zero.
    /// @param src_access Source access. Derived from old_layout when zero.
    /// @param dst_stage Destination stages. Derived from new_layout when zero.
    /// @param dst_access Destination access. Derived from new_layout when zero.
    /// @return This command buffer.
    CommandBuffer& transition(vk::Image image,
                              vk::ImageLayout old_layout,
                              vk::ImageLayout new_layout,
                              const vk::ImageSubresourceRange& range,
                              vk::PipelineStageFlags2 src_stage = {},
                              vk::AccessFlags2 src_access = {},
                              vk::PipelineStageFlags2 dst_stage = {},
                              vk::AccessFlags2 dst_access = {});

    /// @brief Transitions an image layout and updates its tracked layout.
    /// @param image Image to transition.
    /// @param new_layout Target layout.
    /// @param src_stage Source stages. Derived from the tracked layout when zero.
    /// @param src_access Source access. Derived from the tracked layout when zero.
    /// @param dst_stage Destination stages. Derived from new_layout when zero.
    /// @param dst_access Destination access. Derived from new_layout when zero.
    /// @return This command buffer.
    CommandBuffer& transition(Image& image,
                              vk::ImageLayout new_layout,
                              vk::PipelineStageFlags2 src_stage = {},
                              vk::AccessFlags2 src_access = {},
                              vk::PipelineStageFlags2 dst_stage = {},
                              vk::AccessFlags2 dst_access = {});

    /// @brief Copies data between buffers.
    /// @param src Source buffer.
    /// @param dst Destination buffer.
    /// @param size Number of bytes to copy.
    /// @param src_offset Byte offset in the source.
    /// @param dst_offset Byte offset in the destination.
    /// @return This command buffer.
    CommandBuffer& copy_buffer(vk::Buffer src,
                               vk::Buffer dst,
                               vk::DeviceSize size,
                               vk::DeviceSize src_offset = 0,
                               vk::DeviceSize dst_offset = 0);

    /// @brief Copies a buffer into an image using raw handles.
    /// @param src Source buffer.
    /// @param dst Destination image, must be in TransferDst layout.
    /// @param extent Size of the copied region.
    /// @param layers Image subresource layers to copy into.
    /// @param offset Offset in the destination image.
    /// @return This command buffer.
    CommandBuffer& copy_buffer_to_image(vk::Buffer src,
                                        vk::Image dst,
                                        vk::Extent3D extent,
                                        const vk::ImageSubresourceLayers& layers,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Copies a buffer into mip 0, layer 0 of an image.
    /// @param src Source buffer.
    /// @param dst Destination image, must be in TransferDst layout.
    /// @param offset Offset in the destination image.
    /// @return This command buffer.
    CommandBuffer& copy_buffer_to_image(vk::Buffer src,
                                        const Image& dst,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Copies an image into a buffer using raw handles.
    /// @param dst Destination buffer.
    /// @param src Source image, must be in TransferSrc layout.
    /// @param extent Size of the copied region.
    /// @param layers Image subresource layers to copy from.
    /// @param offset Offset in the source image.
    /// @return This command buffer.
    CommandBuffer& copy_image_to_buffer(vk::Buffer dst,
                                        vk::Image src,
                                        vk::Extent3D extent,
                                        const vk::ImageSubresourceLayers& layers,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Copies mip 0, layer 0 of an image into a buffer.
    /// @param dst Destination buffer.
    /// @param src Source image, must be in TransferSrc layout.
    /// @param offset Offset in the source image.
    /// @return This command buffer.
    CommandBuffer& copy_image_to_buffer(vk::Buffer dst,
                                        const Image& src,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

private:
    vk::Device device_;
    vk::CommandPool pool_;
    vk::CommandBuffer cmd_;
};

}
