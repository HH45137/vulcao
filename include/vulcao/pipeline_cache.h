#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a Vulkan pipeline cache.
class PipelineCache {
public:
    /// @brief Creates an empty pipeline cache.
    PipelineCache() = default;

    /// @brief Destroys the pipeline cache.
    ~PipelineCache();

    /// @brief Not copyable.
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    /// @brief Moves the pipeline cache, leaving the source empty.
    PipelineCache(PipelineCache&& other) noexcept;

    /// @brief Move assignment. Destroys the current cache first.
    PipelineCache& operator=(PipelineCache&& other) noexcept;

    /// @brief Creates a pipeline cache, optionally from previously saved data.
    /// @param device Device that creates the cache.
    /// @param data Optional serialized cache data.
    /// @param size Size of the initial data.
    /// @param flags Pipeline cache creation flags.
    /// @return The created pipeline cache.
    static PipelineCache create(vk::Device device,
                                const void* data = nullptr,
                                size_t size = 0,
                                vk::PipelineCacheCreateFlags flags = {});

    /// @brief Returns true if the cache holds a valid handle.
    bool valid() const { return static_cast<bool>(cache_); }

    /// @brief Returns true if the cache holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan pipeline cache handle.
    vk::PipelineCache handle() const { return cache_; }

    /// @brief Returns the serialized cache data, suitable for PipelineCache::create.
    std::vector<uint8_t> data() const;

    /// @brief Destroys the cache and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::PipelineCache cache_;
};

}
