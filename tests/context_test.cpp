#include <stdexcept>

#include <doctest/doctest.h>

#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/log.h>

#include "common.h"

TEST_CASE("immediate rejects reentrant recording and recovers afterwards") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    // Nesting would reset the command buffer that is currently being recorded.
    CHECK_THROWS_AS(
        context.immediate([&](vulcao::CommandBuffer&) {
            context.immediate([](vulcao::CommandBuffer&) {});
        }),
        std::runtime_error);

    // The guard must be released when the callable throws, otherwise the context
    // would stay locked for the rest of its lifetime.
    CHECK_THROWS_AS(
        context.immediate([](vulcao::CommandBuffer&) { throw std::runtime_error("boom"); }),
        std::runtime_error);

    CHECK_NOTHROW(context.immediate([](vulcao::CommandBuffer&) {}));
}

TEST_CASE("contexts with different validation settings coexist in one process") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    // vk-bootstrap caches its instance function pointers process wide on the first
    // instance it builds and never invalidates them, so a context created without
    // VK_EXT_debug_utils used to break the debug messenger of every later context
    // that asked for validation.
    {
        vulcao::ContextInfo quiet;
        quiet.headless = true;
        quiet.validation = false;

        vulcao::Context first{quiet};
        first.initialize();

        CHECK(first.initialized());
        CHECK_FALSE(first.debug_utils_enabled());
    }

    {
        vulcao::ContextInfo loud;
        loud.headless = true;
        loud.validation = true;

        vulcao::Context second{loud};
        second.initialize();

        CHECK(second.initialized());
        CHECK(second.debug_utils_enabled());
    }
}
