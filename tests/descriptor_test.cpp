#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/context.h>
#include <vulcao/descriptor_set.h>
#include <vulcao/image.h>
#include <vulcao/log.h>
#include <vulcao/sampler.h>

#include "common.h"

namespace {

using vk::DescriptorSetLayoutBinding;
using vk::DescriptorType;
using vk::ShaderStageFlagBits;

constexpr uint32_t uniform_binding = 0;
constexpr uint32_t storage_binding = 1;
constexpr uint32_t image_binding = 2;
constexpr uint32_t storage_image_binding = 3;

const std::array<DescriptorSetLayoutBinding, 4> all_bindings{{
    DescriptorSetLayoutBinding{.binding = uniform_binding,
                               .descriptorType = DescriptorType::eUniformBuffer,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eVertex},
    DescriptorSetLayoutBinding{.binding = storage_binding,
                               .descriptorType = DescriptorType::eStorageBuffer,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eCompute},
    DescriptorSetLayoutBinding{.binding = image_binding,
                               .descriptorType = DescriptorType::eCombinedImageSampler,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eFragment},
    DescriptorSetLayoutBinding{.binding = storage_image_binding,
                               .descriptorType = DescriptorType::eStorageImage,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eCompute},
}};

const std::array<vk::DescriptorPoolSize, 4> all_sizes{{
    vk::DescriptorPoolSize{DescriptorType::eUniformBuffer, 8},
    vk::DescriptorPoolSize{DescriptorType::eStorageBuffer, 8},
    vk::DescriptorPoolSize{DescriptorType::eCombinedImageSampler, 8},
    vk::DescriptorPoolSize{DescriptorType::eStorageImage, 8},
}};

}

TEST_CASE("descriptor pool allocates up to max_sets and resets") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), all_bindings);
    CHECK(layout.valid());

    constexpr uint32_t max_sets = 4;
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(context.device(), all_sizes, max_sets);
    CHECK(pool.valid());

    for (uint32_t i = 0; i < max_sets; ++i) {
        const vulcao::DescriptorSet set = pool.allocate(layout);
        CHECK(set.valid());
        CHECK(set.handle() != VK_NULL_HANDLE);
    }

    // The pool is exhausted, so the next allocation has to fail rather than
    // hand out an invalid handle.
    CHECK_THROWS_AS(pool.allocate(layout), std::runtime_error);

    pool.reset();
    CHECK(pool.allocate(layout).valid());
}

TEST_CASE("descriptor pool rejects an empty layout") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(context.device(), all_sizes, 1);
    CHECK_THROWS_AS(pool.allocate(vulcao::DescriptorSetLayout{}), std::runtime_error);
}

TEST_CASE("descriptor set writes match the declared binding types") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), all_bindings);
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(context.device(), all_sizes, 2);

    vulcao::Buffer uniform = vulcao::Buffer::create(context.allocator(), 64,
                                                   vk::BufferUsageFlagBits::eUniformBuffer);
    vulcao::Buffer storage = vulcao::Buffer::create(context.allocator(), 256,
                                                   vk::BufferUsageFlagBits::eStorageBuffer);
    const vulcao::Sampler sampler = vulcao::Sampler::linear(context.device());
    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{4, 4}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);

    // The write helpers are const: they update the Vulkan set, not this handle.
    const vulcao::DescriptorSet set = pool.allocate(layout);

    CHECK_NOTHROW(set.write_uniform_buffer(uniform_binding, uniform));
    CHECK_NOTHROW(set.write_storage_buffer(storage_binding, storage));
    CHECK_NOTHROW(set.write_image(image_binding, image, sampler));
    CHECK_NOTHROW(set.write_storage_image(storage_image_binding, image));

    CHECK_NOTHROW(set.write_buffer(uniform_binding, uniform, DescriptorType::eUniformBuffer, 0, 64));

    // The batching writer must produce the same result through one update call.
    vulcao::DescriptorSetWriter writer{set};
    writer.write_uniform_buffer(uniform_binding, uniform)
        .write_storage_buffer(storage_binding, storage)
        .write_image(image_binding, image, sampler)
        .write_storage_image(storage_image_binding, image);
    CHECK_NOTHROW(writer.flush());

    // Flushing twice is a no-op, and clear() drops pending writes.
    CHECK_NOTHROW(writer.flush());
    writer.write_uniform_buffer(uniform_binding, uniform);
    writer.clear();
    CHECK_NOTHROW(writer.flush());

    CHECK(capture.errors.empty());
}

TEST_CASE("descriptor set layout cache reuses layouts by binding set") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::DescriptorSetLayoutCache cache{context.device()};

    const vk::DescriptorSetLayout first = cache.get(all_bindings);
    CHECK(first != VK_NULL_HANDLE);
    CHECK(cache.get(all_bindings) == first);

    // The cache key normalises binding order, so a permuted list hits the same entry.
    const std::array<DescriptorSetLayoutBinding, 4> permuted{{all_bindings[3], all_bindings[1],
                                                              all_bindings[0], all_bindings[2]}};
    CHECK(cache.get(permuted) == first);

    const std::array<DescriptorSetLayoutBinding, 1> single{{all_bindings[0]}};
    CHECK(cache.get(single) != first);

    // Clearing drops the cached layouts; the cache stays usable afterwards.
    cache.clear();
    CHECK(cache.get(all_bindings) != VK_NULL_HANDLE);
}
