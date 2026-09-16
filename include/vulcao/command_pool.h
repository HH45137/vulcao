#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

#include "vulcao/command_buffer.h"

namespace vulcao {

/// @brief RAII wrapper around a Vulkan command pool.
class CommandPool {
public:
    /// @brief Creates an empty command pool.
    CommandPool() = default;

    /// @brief Destroys the command pool.
    ~CommandPool();

    /// @brief Not copyable.
    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;

    /// @brief Moves the command pool, leaving the source empty.
    CommandPool(CommandPool&& other) noexcept;

    /// @brief Move assignment. Destroys the current command pool first.
    CommandPool& operator=(CommandPool&& other) noexcept;

    /// @brief Creates a command pool.
    /// @param device Device that creates the pool.
    /// @param queue_family_index Queue family the pool allocates for.
    /// @param flags Pool creation flags.
    /// @return The created command pool.
    static CommandPool create(vk::Device device,
                              uint32_t queue_family_index,
                              vk::CommandPoolCreateFlags flags = {});

    /// @brief Returns true if the pool holds a valid handle.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns true if the pool holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan command pool handle.
    vk::CommandPool handle() const { return pool_; }

    /// @brief Allocates one command buffer from the pool.
    /// @param level Command buffer level.
    /// @param debug_utils True to enable debug labels on the command buffer.
    /// @return The allocated command buffer.
    CommandBuffer allocate(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
                           bool debug_utils = false) const;

    /// @brief Resets the pool and frees all command buffers allocated from it.
    /// @param flags Reset flags.
    void reset(vk::CommandPoolResetFlags flags = {}) const;

    /// @brief Destroys the pool and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::CommandPool pool_;
};

}
