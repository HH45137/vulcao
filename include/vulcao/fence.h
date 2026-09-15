#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a Vulkan fence.
class Fence {
public:
    /// @brief Creates an empty fence.
    Fence() = default;

    /// @brief Destroys the fence.
    ~Fence();

    /// @brief Not copyable.
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;

    /// @brief Moves the fence, leaving the source empty.
    Fence(Fence&& other) noexcept;

    /// @brief Move assignment. Destroys the current fence first.
    Fence& operator=(Fence&& other) noexcept;

    /// @brief Creates a fence.
    /// @param device Device that creates the fence.
    /// @param flags Fence creation flags.
    /// @return The created fence.
    static Fence create(vk::Device device, vk::FenceCreateFlags flags = {});

    /// @brief Returns true if the fence holds a valid handle.
    bool valid() const { return static_cast<bool>(fence_); }

    /// @brief Returns true if the fence holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan fence handle.
    vk::Fence handle() const { return fence_; }

    /// @brief Waits until the fence is signaled.
    /// @param timeout Timeout in nanoseconds, or UINT64_MAX to wait forever.
    void wait(uint64_t timeout = UINT64_MAX) const;

    /// @brief Resets the fence to the unsignaled state.
    void reset() const;

    /// @brief Destroys the fence and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::Fence fence_;
};

}
