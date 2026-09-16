#include <cstddef>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/log.h>

namespace {

/// Restores the global log level even if the test aborts early.
struct LogLevelGuard {
    vulcao::LogLevel previous = vulcao::log_level();
    ~LogLevelGuard() { vulcao::set_log_level(previous); }
};

}

TEST_CASE("headless context runs commands without a surface or swapchain") {
    // Request validation so the device path is checked when the layers are
    // installed; quiet the informational messages while doing so.
    const LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    try {
        vulcao::Context context{info};
        context.initialize();

        REQUIRE(context.initialized());
        CHECK(context.headless());
        CHECK(!context.swapchain());
        CHECK(!context.present_queue());
        CHECK(context.graphics_queue());
        CHECK_THROWS(context.recreate_swapchain(vk::Extent2D{32, 32}));

        const std::vector<uint32_t> data{1, 2, 3, 4, 5, 6, 7, 8};
        const vk::DeviceSize bytes = data.size() * sizeof(uint32_t);

        vulcao::Buffer buffer = vulcao::Buffer::create(
            context.allocator(), bytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst);
        context.upload(buffer, data);

        vulcao::Buffer readback =
            vulcao::Buffer::create(context.allocator(), bytes, vk::BufferUsageFlagBits::eTransferDst,
                                   VMA_MEMORY_USAGE_AUTO,
                                   VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        context.immediate([&](vulcao::CommandBuffer& cmd) {
            cmd.copy_buffer(buffer.handle(), readback.handle(), bytes);
        });

        readback.invalidate();
        const auto* result = static_cast<const uint32_t*>(readback.map());
        REQUIRE(result != nullptr);
        for (size_t i = 0; i < data.size(); ++i)
            CHECK(result[i] == data[i]);
        readback.unmap();
    } catch (const std::exception& e) {
        MESSAGE("skipping headless test (no usable Vulkan device): " << e.what());
    }
}
