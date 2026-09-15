#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief Owns specialization constant values for one pipeline stage.
class SpecializationInfo {
public:
    /// @brief Maps a constant id to a value.
    /// @tparam T Trivially copyable value type.
    /// @param constant_id Constant id from the shader.
    /// @param value Value to substitute.
    /// @return This object.
    template <typename T>
    SpecializationInfo& map_constant(uint32_t constant_id, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "specialization constants must be trivially copyable");

        const uint32_t size = static_cast<uint32_t>(sizeof(T));
        const uint32_t alignment = size > 4 ? 8u : 4u;
        const uint32_t offset =
            static_cast<uint32_t>((data_.size() + alignment - 1) & ~(alignment - 1));

        data_.resize(offset + size);
        std::memcpy(data_.data() + offset, &value, size);
        entries_.push_back(vk::SpecializationMapEntry{
            .constantID = constant_id,
            .offset = offset,
            .size = size,
        });
        return *this;
    }

    /// @brief Returns the Vulkan specialization info referencing the owned data.
    vk::SpecializationInfo get() const {
        return vk::SpecializationInfo{
            .mapEntryCount = static_cast<uint32_t>(entries_.size()),
            .pMapEntries = entries_.data(),
            .dataSize = data_.size(),
            .pData = data_.data(),
        };
    }

    /// @brief Returns true if no constant was mapped.
    bool empty() const { return entries_.empty(); }

private:
    std::vector<vk::SpecializationMapEntry> entries_;
    std::vector<std::byte> data_;
};

}
