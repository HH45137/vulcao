#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/log.h>
#include <vulcao/pipeline_cache.h>
#include <vulcao/query_pool.h>
#include <vulcao/sampler.h>

#include "common.h"

TEST_CASE("sampler factories produce distinct usable samplers") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::Sampler repeat = vulcao::Sampler::linear(context.device(), true);
    const vulcao::Sampler clamp = vulcao::Sampler::linear(context.device(), false);
    const vulcao::Sampler nearest = vulcao::Sampler::nearest(context.device());
    const vulcao::Sampler custom = vulcao::Sampler::create(
        context.device(), vk::SamplerCreateInfo{
                              .magFilter = vk::Filter::eNearest,
                              .minFilter = vk::Filter::eNearest,
                              .mipmapMode = vk::SamplerMipmapMode::eNearest,
                              .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                              .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                              .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                              .maxLod = VK_LOD_CLAMP_NONE,
                          });

    CHECK(repeat.valid());
    CHECK(clamp.valid());
    CHECK(nearest.valid());
    CHECK(custom.valid());
    CHECK(repeat.handle() != VK_NULL_HANDLE);

    // Each factory must return a fresh sampler, not a shared one.
    CHECK(repeat.handle() != clamp.handle());
    CHECK(repeat.handle() != nearest.handle());
    CHECK(nearest.handle() != custom.handle());

    vulcao::Sampler empty;
    CHECK_FALSE(empty.valid());
    CHECK_NOTHROW(empty.destroy());
}

TEST_CASE("query pool reports its type and count") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::QueryPool timestamps =
        vulcao::QueryPool::create(context.device(), vk::QueryType::eTimestamp, 8);
    CHECK(timestamps.valid());
    CHECK(timestamps.handle() != VK_NULL_HANDLE);
    CHECK(timestamps.query_count() == 8);
    CHECK(timestamps.type() == vk::QueryType::eTimestamp);

    const vulcao::QueryPool occlusion =
        vulcao::QueryPool::create(context.device(), vk::QueryType::eOcclusion, 4);
    CHECK(occlusion.valid());
    CHECK(occlusion.query_count() == 4);
    CHECK(occlusion.type() == vk::QueryType::eOcclusion);
}

TEST_CASE("timestamps record through a command buffer and read back") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vk::QueueFamilyProperties queue_properties =
        context.physical_device().getQueueFamilyProperties()[context.graphics_queue_family_index()];
    if (queue_properties.timestampValidBits == 0) {
        MESSAGE("skipping: the graphics queue family does not support timestamps");
        return;
    }

    constexpr uint32_t query_count = 2;
    vulcao::QueryPool pool =
        vulcao::QueryPool::create(context.device(), vk::QueryType::eTimestamp, query_count);

    vulcao::Buffer results = vulcao::Buffer::create(
        context.allocator(), query_count * sizeof(uint64_t),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.reset_query_pool(pool.handle(), 0, query_count);
        cmd.write_timestamp(pool.handle(), vk::PipelineStageFlagBits2::eAllCommands, 0);
        cmd.write_timestamp(pool.handle(), vk::PipelineStageFlagBits2::eAllCommands, 1);
        cmd.copy_query_pool_results(pool.handle(), 0, query_count, results.handle(), 0,
                                    sizeof(uint64_t),
                                    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    });

    std::vector<uint64_t> timestamps(query_count, 0);
    context.download(results, timestamps);
    CHECK(timestamps[1] >= timestamps[0]);

    CHECK(capture.errors.empty());
}

TEST_CASE("pipeline cache serializes and reloads") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::PipelineCache cache = vulcao::PipelineCache::create(context.device());
    CHECK(cache.valid());
    CHECK(cache.handle() != VK_NULL_HANDLE);

    const std::vector<uint8_t> blob = cache.data();
    CHECK_NOTHROW(vulcao::PipelineCache::create(context.device(), blob.data(), blob.size()));

    vulcao::PipelineCache empty;
    CHECK_FALSE(empty.valid());
    CHECK_NOTHROW(empty.destroy());
}
