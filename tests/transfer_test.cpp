#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/image.h>
#include <vulcao/log.h>

#include "common.h"

TEST_CASE("download round trips a buffer and a padded image") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::vector<uint32_t> data{10, 20, 30, 40, 50};
    vulcao::Buffer buffer = vulcao::Buffer::create(
        context.allocator(), data.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    context.upload(buffer, data);

    std::vector<uint32_t> readback(data.size());
    context.download(buffer, readback);
    CHECK(readback == data);

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 3;
    constexpr uint32_t padding = 2;
    const uint32_t padded = width + padding;

    std::vector<uint8_t> source(static_cast<size_t>(padded) * height * 4);
    std::vector<uint8_t> expected(static_cast<size_t>(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t src = (static_cast<size_t>(y) * padded + x) * 4;
            source[src + 0] = static_cast<uint8_t>(x * 20);
            source[src + 1] = static_cast<uint8_t>(y * 30);
            source[src + 2] = 90;
            source[src + 3] = 255;

            const size_t dst = (static_cast<size_t>(y) * width + x) * 4;
            expected[dst + 0] = source[src + 0];
            expected[dst + 1] = source[src + 1];
            expected[dst + 2] = source[src + 2];
            expected[dst + 3] = source[src + 3];
        }
    }

    vulcao::Buffer staging = vulcao::Buffer::create(
        context.allocator(), source.size(), vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    staging.write(source);

    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst);

    const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.transition(image, vk::ImageLayout::eTransferDstOptimal);
        cmd.copy_buffer_to_image(staging.handle(), image.handle(),
                                 vk::BufferImageCopy{
                                     .bufferOffset = 0,
                                     .bufferRowLength = padded,
                                     .bufferImageHeight = 0,
                                     .imageSubresource = layers,
                                     .imageOffset = vk::Offset3D{0, 0, 0},
                                     .imageExtent = vk::Extent3D{width, height, 1},
                                 });
        cmd.transition(image, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    context.download(image, pixels);
    CHECK(pixels == expected);
    CHECK(image.layout() == vk::ImageLayout::eShaderReadOnlyOptimal);

    CHECK(capture.errors.empty());
}

TEST_CASE("upload and download reject an image payload that is too small") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    constexpr uint32_t width = 8;
    constexpr uint32_t height = 8;
    constexpr size_t texel = 4;

    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst);

    const size_t required = static_cast<size_t>(width) * height * texel;
    const std::vector<uint8_t> exact(required, 0x7f);
    const std::vector<uint8_t> too_small(required - texel, 0x7f);

    CHECK_NOTHROW(context.upload(image, exact));
    CHECK_THROWS_AS(context.upload(image, too_small), std::runtime_error);

    std::vector<uint8_t> readback(required);
    CHECK_NOTHROW(context.download(image, readback));
    CHECK(readback == exact);

    std::vector<uint8_t> short_readback(required - texel);
    CHECK_THROWS_AS(context.download(image, short_readback), std::runtime_error);

    // A payload larger than the image stays accepted, matching the buffer path.
    const std::vector<uint8_t> oversized(required + texel, 0x7f);
    CHECK_NOTHROW(context.upload(image, oversized));
}
