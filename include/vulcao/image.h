#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "vulcao/allocator.h"

namespace vulcao {

/// @brief Returns the image aspect mask that matches a format.
/// @param format Image format.
/// @return Color, depth or depth-stencil aspect flags.
vk::ImageAspectFlags image_aspect_for_format(vk::Format format);

/// @brief Returns the number of bytes a tightly packed image of that extent occupies.
///
/// Follows the texel block layout, so compressed formats round the extent up to
/// whole blocks. Describes one mip level of one array layer, which is what
/// Context::upload and Context::download move.
/// @param extent Image extent. Depth is ignored for non-3D images, which use 1.
/// @param format Image format.
/// @return Size in bytes.
vk::DeviceSize image_byte_size(vk::Extent3D extent, vk::Format format);

/// @brief RAII wrapper around a VMA-allocated image, its view and tracked layout.
class Image {
public:
    /// @brief Creates an empty image.
    Image() = default;

    /// @brief Destroys the view, the image and its memory.
    ~Image();

    /// @brief Not copyable.
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    /// @brief Moves the image, leaving the source empty.
    Image(Image&& other) noexcept;

    /// @brief Move assignment. Destroys the current image first.
    Image& operator=(Image&& other) noexcept;

    /// @brief Creates an image, allocates its memory and creates its view.
    ///
    /// Fields of @p view_info left at their unset sentinel are derived from
    /// @p image_info: viewType when it is e1D, format when it is undefined, and
    /// the whole subresourceRange when its levelCount is zero. Set a field
    /// explicitly to keep it as it is.
    /// @param allocator Allocator used for the memory.
    /// @param image_info Image creation parameters.
    /// @param view_info View creation parameters, possibly only partially filled.
    /// @return The created image.
    /// @throws std::runtime_error if the allocator is invalid or the image cannot be created.
    static Image create(Allocator& allocator,
                        const vk::ImageCreateInfo& image_info,
                        vk::ImageViewCreateInfo view_info = {});

    /// @brief Creates a 2D image with a view.
    /// @param allocator Allocator used for the memory.
    /// @param extent Image width and height.
    /// @param format Image format.
    /// @param usage Image usage flags.
    /// @param mip_levels Number of mip levels.
    /// @param samples Sample count.
    /// @return The created image.
    /// @throws std::runtime_error if the allocator is invalid or the image cannot be created.
    static Image create_2d(Allocator& allocator,
                           vk::Extent2D extent,
                           vk::Format format,
                           vk::ImageUsageFlags usage,
                           uint32_t mip_levels = 1,
                           vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);

    /// @brief Creates a depth image with a view.
    /// @param allocator Allocator used for the memory.
    /// @param extent Image width and height.
    /// @param format Depth format.
    /// @return The created image.
    /// @throws std::runtime_error if the allocator is invalid or the image cannot be created.
    static Image create_depth(Allocator& allocator,
                              vk::Extent2D extent,
                              vk::Format format = vk::Format::eD32Sfloat);

    /// @brief Returns true if the image holds a valid handle.
    bool valid() const { return image_ != VK_NULL_HANDLE; }

    /// @brief Returns true if the image holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan image handle.
    vk::Image handle() const { return vk::Image{image_}; }

    /// @brief Returns the image view owned by this image.
    vk::ImageView view() const { return vk::ImageView{view_}; }

    /// @brief Returns the image extent.
    vk::Extent3D extent() const { return extent_; }

    /// @brief Returns the number of mip levels covered by the view.
    uint32_t mip_levels() const { return range_.levelCount; }

    /// @brief Returns the image format.
    vk::Format format() const { return format_; }

    /// @brief Returns the usage flags the image was created with.
    vk::ImageUsageFlags usage() const { return usage_; }

    /// @brief Returns the tracked current layout of the image.
    vk::ImageLayout layout() const { return layout_; }

    /// @brief Returns the subresource range covered by the image view.
    vk::ImageSubresourceRange subresource_range() const { return range_; }

    /// @brief Returns the VMA allocation of the image.
    VmaAllocation allocation() const { return allocation_; }

private:
    friend class CommandBuffer;

    /// @brief Updates the tracked layout. Called by CommandBuffer.
    void set_layout(vk::ImageLayout layout) { layout_ = layout; }

    /// @brief Destroys the view, image and memory, then resets the wrapper.
    void destroy();

    vk::Device device_;
    VmaAllocator allocator_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VkImageView view_ = VK_NULL_HANDLE;
    vk::Extent3D extent_{};
    vk::Format format_ = vk::Format::eUndefined;
    vk::ImageUsageFlags usage_;
    vk::ImageLayout layout_ = vk::ImageLayout::eUndefined;
    vk::ImageSubresourceRange range_{};
};

}
