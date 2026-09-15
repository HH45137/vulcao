#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a binary or timeline Vulkan semaphore.
class Semaphore {
public:
    /// @brief Creates an empty semaphore.
    Semaphore() = default;

    /// @brief Destroys the semaphore.
    ~Semaphore();

    /// @brief Not copyable.
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;

    /// @brief Moves the semaphore, leaving the source empty.
    Semaphore(Semaphore&& other) noexcept;

    /// @brief Move assignment. Destroys the current semaphore first.
    Semaphore& operator=(Semaphore&& other) noexcept;

    /// @brief Creates a binary semaphore.
    /// @param device Device that creates the semaphore.
    /// @param flags Semaphore creation flags.
    /// @return The created semaphore.
    static Semaphore create(vk::Device device, vk::SemaphoreCreateFlags flags = {});

    /// @brief Creates a timeline semaphore.
    /// @param device Device that creates the semaphore.
    /// @param initial_value Initial counter value.
    /// @return The created semaphore.
    static Semaphore create_timeline(vk::Device device, uint64_t initial_value = 0);

    /// @brief Returns true if the semaphore holds a valid handle.
    bool valid() const { return static_cast<bool>(semaphore_); }

    /// @brief Returns true if the semaphore holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan semaphore handle.
    vk::Semaphore handle() const { return semaphore_; }

    /// @brief Returns the type the semaphore was created with.
    vk::SemaphoreType type() const { return type_; }

    /// @brief Returns the current counter value of a timeline semaphore.
    uint64_t value() const;

    /// @brief Signals a timeline semaphore from the host.
    /// @param value Value to signal.
    void signal(uint64_t value) const;

    /// @brief Waits for a timeline semaphore to reach a value from the host.
    /// @param value Value to wait for.
    /// @param timeout Timeout in nanoseconds, or UINT64_MAX to wait forever.
    void wait(uint64_t value, uint64_t timeout = UINT64_MAX) const;

    /// @brief Destroys the semaphore and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::Semaphore semaphore_;
    vk::SemaphoreType type_ = vk::SemaphoreType::eBinary;
};

}
