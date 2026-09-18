#pragma once

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/context.h>
#include <vulcao/log.h>

namespace vulcao::test {

/// @brief Restores the global log level even if the test aborts early.
struct LogLevelGuard {
    LogLevel previous = log_level();
    ~LogLevelGuard() { set_log_level(previous); }
};

/// @brief Collects the error level messages logged while it is alive.
///
/// Lets a test assert that a code path stays free of library and validation
/// errors, instead of only checking that it did not throw.
struct ErrorCapture {
    std::vector<std::string> errors;
    LogCallback previous = log_callback();

    ErrorCapture() {
        set_log_callback([this](const LogMessage& message) {
            if (message.level == LogLevel::error)
                errors.emplace_back(message.message);
        });
    }

    ~ErrorCapture() { set_log_callback(previous); }
};

/// @brief Returns true when the headless device path can run on this machine.
///
/// Probes by building a throwaway headless context and immediately tearing it
/// down. This is the only place in the suite allowed to swallow an exception: it
/// answers "is a device present", not "does the library work". The tests
/// themselves run without a catch, so a failure inside the library fails the
/// test instead of being reported as a skip.
inline bool has_device() {
    const LogLevel previous = log_level();
    set_log_level(LogLevel::error);

    bool available = false;
    try {
        ContextInfo info;
        info.headless = true;
        info.validation = false;
        Context context{info};
        context.initialize();
        available = context.initialized();
    } catch (const std::exception&) {
        available = false;
    }

    set_log_level(previous);
    return available;
}

/// @brief Returns true when a missing device is tolerated instead of failing.
///
/// Set the environment variable VULCAO_ALLOW_NO_DEVICE=1 to run the device tests
/// on a machine without a usable Vulkan device. Unset, a missing device fails
/// the test, so that a green run always means the device path really executed.
inline bool missing_device_allowed() {
#ifdef _MSC_VER
    char* buffer = nullptr;
    size_t size = 0;
    if (_dupenv_s(&buffer, &size, "VULCAO_ALLOW_NO_DEVICE") != 0 || buffer == nullptr)
        return false;

    const std::string_view value{buffer};
    const bool allowed = !value.empty() && value != "0";
    std::free(buffer);
    return allowed;
#else
    const char* value = std::getenv("VULCAO_ALLOW_NO_DEVICE");
    return value != nullptr && std::string_view(value) != "0";
#endif
}

/// @brief Abandons the current test case unless a usable device is present.
///
/// Use at the top of a test case that needs Vulkan:
/// @code
/// TEST_CASE("something") {
///     VULCAO_REQUIRE_DEVICE();
///     vulcao::Context context{info};
///     // no try/catch: failures must fail the test
/// }
/// @endcode
#define VULCAO_REQUIRE_DEVICE()                                                              \
    do {                                                                                     \
        if (vulcao::test::has_device())                                                      \
            break;                                                                           \
        if (vulcao::test::missing_device_allowed()) {                                        \
            MESSAGE("skipping: no usable Vulkan device (VULCAO_ALLOW_NO_DEVICE is set)");     \
            return;                                                                          \
        }                                                                                    \
        FAIL("no usable Vulkan device; set VULCAO_ALLOW_NO_DEVICE=1 to tolerate this");       \
        return;                                                                              \
    } while (false)

}

#ifdef VULCAO_SHADER_DIR

/// @brief Marks the test cases that need the compiled Slang shaders.
///
/// VULCAO_SHADER_DIR is only defined when slangc was found at configure time, so
/// the shader dependent cases are compiled out on machines without the Vulkan SDK
/// command line tools. The build prints a status message when that happens.
#define VULCAO_HAVE_TEST_SHADERS 1

namespace vulcao::test {

/// @brief Returns the directory holding the compiled test shaders.
inline std::filesystem::path shader_dir() { return std::filesystem::path{VULCAO_SHADER_DIR}; }

}

#endif
