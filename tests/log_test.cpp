#include <string>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/log.h>

namespace {

struct Recorded {
    vulcao::LogLevel level;
    vulcao::LogCategory category;
    std::string message;
    std::string message_id;
};

}

TEST_CASE("log callback receives structured messages") {
    std::vector<Recorded> records;
    vulcao::set_log_callback([&](const vulcao::LogMessage& message) {
        records.push_back(Recorded{message.level, message.category, std::string(message.message),
                                   std::string(message.message_id)});
    });

    vulcao::log(vulcao::LogLevel::warning, vulcao::LogCategory::general, "hello");
    vulcao::log(vulcao::LogLevel::error, vulcao::LogCategory::validation, "bad", "VUID-1234");

    vulcao::set_log_callback({});

    CHECK(records.size() == 2);
    if (records.size() == 2) {
        CHECK(records[0].level == vulcao::LogLevel::warning);
        CHECK(records[0].category == vulcao::LogCategory::general);
        CHECK(records[0].message == "hello");
        CHECK(records[0].message_id.empty());
        CHECK(records[1].level == vulcao::LogLevel::error);
        CHECK(records[1].category == vulcao::LogCategory::validation);
        CHECK(records[1].message == "bad");
        CHECK(records[1].message_id == "VUID-1234");
    }

    CHECK(static_cast<bool>(vulcao::log_callback()));
}

TEST_CASE("to_string names levels and categories") {
    CHECK(vulcao::to_string(vulcao::LogLevel::trace) == "trace");
    CHECK(vulcao::to_string(vulcao::LogLevel::debug) == "debug");
    CHECK(vulcao::to_string(vulcao::LogLevel::info) == "info");
    CHECK(vulcao::to_string(vulcao::LogLevel::warning) == "warning");
    CHECK(vulcao::to_string(vulcao::LogLevel::error) == "error");
    CHECK(vulcao::to_string(vulcao::LogCategory::general) == "general");
    CHECK(vulcao::to_string(vulcao::LogCategory::validation) == "validation");
    CHECK(vulcao::to_string(vulcao::LogCategory::performance) == "performance");
}
