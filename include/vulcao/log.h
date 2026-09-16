#pragma once

#include <functional>
#include <string_view>

namespace vulcao {

/// @brief Severity of a log message.
enum class LogLevel {
    trace,
    debug,
    info,
    warning,
    error,
};

/// @brief Source of a log message.
enum class LogCategory {
    general,     ///< Messages emitted by the library itself.
    validation,  ///< Messages from the validation layers.
    performance, ///< Performance warnings from the validation layers.
};

/// @brief A single log message sent to the global log callback.
///
/// The string views point to storage owned by the caller and are only valid for
/// the duration of the callback. Copy them if they must outlive the call.
struct LogMessage {
    LogLevel level = LogLevel::info;
    LogCategory category = LogCategory::general;
    std::string_view message;
    std::string_view message_id;
};

/// @brief Callback invoked for every log message.
using LogCallback = std::function<void(const LogMessage&)>;

/// @brief Returns the name of a log level.
/// @param level Log level.
/// @return Name of the level.
std::string_view to_string(LogLevel level);

/// @brief Returns the name of a log category.
/// @param category Log category.
/// @return Name of the category.
std::string_view to_string(LogCategory category);

/// @brief Sets the global log callback.
///
/// An empty callback restores default_log_callback. Replacing the callback is
/// safe while messages are being logged.
/// @param callback Callback to install, or an empty function for the default.
void set_log_callback(LogCallback callback);

/// @brief Returns a copy of the global log callback.
/// @return The current callback.
LogCallback log_callback();

/// @brief Sets the minimum level delivered to the callback.
///
/// Messages below the level are dropped before the callback is invoked. This
/// also avoids the cost of formatting messages that would be filtered out.
/// Defaults to LogLevel::trace, which delivers everything.
/// @param level Minimum level to deliver.
void set_log_level(LogLevel level);

/// @brief Returns the minimum level delivered to the callback.
/// @return The current minimum level.
LogLevel log_level();

/// @brief Default callback: writes warning/error and all validation/performance
///        messages to stderr. Drops general trace/debug/info messages.
/// @param message Message to write.
void default_log_callback(const LogMessage& message);

/// @brief Emits a message through the global log callback.
/// @param level Severity of the message.
/// @param category Source of the message.
/// @param message Message text.
/// @param message_id Optional message identifier, such as a VUID.
void log(LogLevel level,
         LogCategory category,
         std::string_view message,
         std::string_view message_id = {});

}
