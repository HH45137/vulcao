#include "vulcao/image.h"

#include "vulcao/check.h"

#include <array>
#include <stdexcept>
#include <utility>

#include <vulkan/vulkan_format_traits.hpp>

namespace vulcao {
namespace {

vk::ImageViewType view_type_for(const vk::ImageCreateInfo& image_info) {
    switch (image_info.imageType) {
        case vk::ImageType::e1D:
            return image_info.arrayLayers > 1 ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
        case vk::ImageType::e3D:
            return vk::ImageViewType::e3D;
        default:
            if (image_info.flags & vk::ImageCreateFlagBits::eCubeCompatible)
                return image_info.arrayLayers > 6 ? vk::ImageViewType::eCubeArray
                                                  : vk::ImageViewType::eCube;
            return image_info.arrayLayers > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
    }
}

}

vk::ImageAspectFlags image_aspect_for_format(vk::Format format) {
    switch (format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
        case vk::Format::eD32Sfloat:
            return vk::ImageAspectFlagBits::eDepth;
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        default:
            return vk::ImageAspectFlagBits::eColor;
    }
}

vk::DeviceSize image_byte_size(vk::Extent3D extent, vk::Format format) {
    const std::array<uint8_t, 3> block_extent = vk::blockExtent(format);
    const vk::DeviceSize block_size = vk::blockSize(format);

    const vk::DeviceSize blocks_x = (extent.width + block_extent[0] - 1) / block_extent[0];
    const vk::DeviceSize blocks_y = (extent.height + block_extent[1] - 1) / block_extent[1];
    const vk::DeviceSize blocks_z = (extent.depth + block_extent[2] - 1) / block_extent[2];
    return blocks_x * blocks_y * blocks_z * block_size;
}

Image::~Image() {
    destroy();
}

Image::Image(Image&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      allocator_(std::exchange(other.allocator_, nullptr)),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, nullptr)),
      view_(std::exchange(other.view_, VK_NULL_HANDLE)),
      extent_(std::exchange(other.extent_, vk::Extent3D{})),
      format_(std::exchange(other.format_, vk::Format::eUndefined)),
      usage_(std::exchange(other.usage_, vk::ImageUsageFlags{})),
      layout_(std::exchange(other.layout_, vk::ImageLayout::eUndefined)),
      range_(std::exchange(other.range_, vk::ImageSubresourceRange{})) {}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        allocator_ = std::exchange(other.allocator_, nullptr);
        image_ = std::exchange(other.image_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, nullptr);
        view_ = std::exchange(other.view_, VK_NULL_HANDLE);
        extent_ = std::exchange(other.extent_, vk::Extent3D{});
        format_ = std::exchange(other.format_, vk::Format::eUndefined);
        usage_ = std::exchange(other.usage_, vk::ImageUsageFlags{});
        layout_ = std::exchange(other.layout_, vk::ImageLayout::eUndefined);
        range_ = std::exchange(other.range_, vk::ImageSubresourceRange{});
    }
    return *this;
}

Image Image::create(Allocator& allocator,
                    const vk::ImageCreateInfo& image_info,
                    vk::ImageViewCreateInfo view_info) {
    if (!allocator.valid())
        throw std::runtime_error("Image::create: invalid allocator");

    Image image;
    image.allocator_ = allocator.handle();
    image.device_ = allocator.device();

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    const VkImageCreateInfo raw_image_info = image_info;
    check(static_cast<vk::Result>(vmaCreateImage(image.allocator_, &raw_image_info, &allocation_info,
                                                 &image.image_, &image.allocation_, nullptr)),
          "create image");

    if (view_info.viewType == vk::ImageViewType::e1D)
        view_info.viewType = view_type_for(image_info);

    if (view_info.format == vk::Format::eUndefined)
        view_info.format = image_info.format;

    if (view_info.subresourceRange.levelCount == 0) {
        view_info.subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = image_aspect_for_format(view_info.format),
            .baseMipLevel = 0,
            .levelCount = image_info.mipLevels,
            .baseArrayLayer = 0,
            .layerCount = image_info.arrayLayers,
        };
    }

    view_info.image = image.handle();
    image.view_ = image.device_.createImageView(view_info);

    image.extent_ = image_info.extent;
    image.format_ = image_info.format;
    image.usage_ = image_info.usage;
    image.layout_ = image_info.initialLayout;
    image.range_ = view_info.subresourceRange;
    return image;
}

Image Image::create_2d(Allocator& allocator,
                       vk::Extent2D extent,
                       vk::Format format,
                       vk::ImageUsageFlags usage,
                       uint32_t mip_levels,
                       vk::SampleCountFlagBits samples) {
    return create(allocator, vk::ImageCreateInfo{
                                 .imageType = vk::ImageType::e2D,
                                 .format = format,
                                 .extent = vk::Extent3D{extent.width, extent.height, 1},
                                 .mipLevels = mip_levels,
                                 .arrayLayers = 1,
                                 .samples = samples,
                                 .tiling = vk::ImageTiling::eOptimal,
                                 .usage = usage,
                                 .sharingMode = vk::SharingMode::eExclusive,
                                 .initialLayout = vk::ImageLayout::eUndefined,
                             });
}

Image Image::create_depth(Allocator& allocator, vk::Extent2D extent, vk::Format format) {
    return create_2d(allocator, extent, format, vk::ImageUsageFlagBits::eDepthStencilAttachment);
}

void Image::destroy() {
    if (view_ != VK_NULL_HANDLE)
        device_.destroyImageView(vk::ImageView{view_});
    if (image_ != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_, image_, allocation_);

    device_ = nullptr;
    allocator_ = nullptr;
    image_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    view_ = VK_NULL_HANDLE;
    extent_ = vk::Extent3D{};
    format_ = vk::Format::eUndefined;
    usage_ = {};
    layout_ = vk::ImageLayout::eUndefined;
    range_ = vk::ImageSubresourceRange{};
}

}
