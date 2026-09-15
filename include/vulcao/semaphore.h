#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a binary Vulkan semaphore.
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

    /// @brief Creates a semaphore.
    /// @param device Device that creates the semaphore.
    /// @param flags Semaphore creation flags.
    /// @return The created semaphore.
    static Semaphore create(vk::Device device, vk::SemaphoreCreateFlags flags = {});

    /// @brief Returns true if the semaphore holds a valid handle.
    bool valid() const { return static_cast<bool>(semaphore_); }

    /// @brief Returns true if the semaphore holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan semaphore handle.
    vk::Semaphore handle() const { return semaphore_; }

    /// @brief Destroys the semaphore and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::Semaphore semaphore_;
};

}
