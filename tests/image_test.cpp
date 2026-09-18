#include <cstdint>

#include <doctest/doctest.h>

#include <vulcao/context.h>
#include <vulcao/image.h>
#include <vulcao/log.h>

#include "common.h"

TEST_CASE("image_aspect_for_format maps depth and stencil formats") {
    using vk::Format;
    using vk::ImageAspectFlagBits;

    CHECK(vulcao::image_aspect_for_format(Format::eD16Unorm) == ImageAspectFlagBits::eDepth);
    CHECK(vulcao::image_aspect_for_format(Format::eX8D24UnormPack32) == ImageAspectFlagBits::eDepth);
    CHECK(vulcao::image_aspect_for_format(Format::eD32Sfloat) == ImageAspectFlagBits::eDepth);

    CHECK(vulcao::image_aspect_for_format(Format::eD16UnormS8Uint) ==
          (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil));
    CHECK(vulcao::image_aspect_for_format(Format::eD24UnormS8Uint) ==
          (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil));
    CHECK(vulcao::image_aspect_for_format(Format::eD32SfloatS8Uint) ==
          (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil));

    CHECK(vulcao::image_aspect_for_format(Format::eR8G8B8A8Unorm) == ImageAspectFlagBits::eColor);
}

TEST_CASE("image_byte_size follows the texel block layout") {
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 1}, vk::Format::eR8G8B8A8Unorm) == 8u * 4u * 4u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 1}, vk::Format::eR8Unorm) == 8u * 4u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 1}, vk::Format::eR32G32B32A32Sfloat) ==
          8u * 4u * 16u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 3}, vk::Format::eR8G8B8A8Unorm) ==
          8u * 4u * 3u * 4u);

    // BC1 packs a 4x4 texel block into 8 bytes, so partial blocks still count whole.
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 8, 1}, vk::Format::eBc1RgbaUnormBlock) ==
          2u * 2u * 8u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{5, 5, 1}, vk::Format::eBc1RgbaUnormBlock) ==
          2u * 2u * 8u);
}

TEST_CASE("Image::create derives the fields of a partial view") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D{16, 16, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    // Only the format is set: viewType and the subresource range must be derived,
    // otherwise the view would be created with a zero level count.
    const vulcao::Image derived =
        vulcao::Image::create(context.allocator(), image_info,
                              vk::ImageViewCreateInfo{.format = vk::Format::eR8G8B8A8Unorm});

    CHECK(derived.valid());
    CHECK(derived.view() != VK_NULL_HANDLE);
    CHECK(derived.subresource_range().levelCount == 1);
    CHECK(derived.subresource_range().layerCount == 1);
    CHECK(derived.subresource_range().aspectMask == vk::ImageAspectFlagBits::eColor);

    // An explicit subresource range must survive instead of being replaced.
    vk::ImageCreateInfo mipmapped = image_info;
    mipmapped.mipLevels = 4;

    const vulcao::Image explicit_range = vulcao::Image::create(
        context.allocator(), mipmapped,
        vk::ImageViewCreateInfo{
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });

    CHECK(explicit_range.valid());
    CHECK(explicit_range.mip_levels() == 1);
}
