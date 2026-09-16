#include "vulcao/log.h"

#include <iostream>
#include <mutex>
#include <utility>

namespace vulcao {
namespace {

std::mutex g_mutex;
LogCallback g_callback = default_log_callback;

}

std::string_view to_string(LogLevel level) {
    switch (level) {
        case LogLevel::trace: return "trace";
        case LogLevel::debug: return "debug";
        case LogLevel::info: return "info";
        case LogLevel::warning: return "warning";
        case LogLevel::error: return "error";
    }
    return "unknown";
}

std::string_view to_string(LogCategory category) {
    switch (category) {
        case LogCategory::general: return "general";
        case LogCategory::validation: return "validation";
        case LogCategory::performance: return "performance";
    }
    return "unknown";
}

void default_log_callback(const LogMessage& message) {
    if (message.category == LogCategory::general && message.level < LogLevel::warning)
        return;

    std::cerr << '[' << to_string(message.level) << ": " << to_string(message.category) << ']';
    if (!message.message_id.empty())
        std::cerr << ' ' << message.message_id << " -";
    std::cerr << ' ' << message.message << '\n';
}

void set_log_callback(LogCallback callback) {
    std::lock_guard lock(g_mutex);
    g_callback = callback ? std::move(callback) : LogCallback{default_log_callback};
}

LogCallback log_callback() {
    std::lock_guard lock(g_mutex);
    return g_callback;
}

void log(LogLevel level,
         LogCategory category,
         std::string_view message,
         std::string_view message_id) {
    const LogCallback callback = log_callback();
    if (callback)
        callback(LogMessage{level, category, message, message_id});
}

}
