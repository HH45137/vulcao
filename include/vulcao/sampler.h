#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a Vulkan sampler.
class Sampler {
public:
    /// @brief Creates an empty sampler.
    Sampler() = default;

    /// @brief Destroys the sampler.
    ~Sampler();

    /// @brief Not copyable.
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    /// @brief Moves the sampler, leaving the source empty.
    Sampler(Sampler&& other) noexcept;

    /// @brief Move assignment. Destroys the current sampler first.
    Sampler& operator=(Sampler&& other) noexcept;

    /// @brief Creates a sampler.
    /// @param device Device that creates the sampler.
    /// @param info Sampler creation parameters.
    /// @return The created sampler.
    static Sampler create(vk::Device device, const vk::SamplerCreateInfo& info);

    /// @brief Creates a linear filtering sampler.
    /// @param device Device that creates the sampler.
    /// @param repeat True for repeat addressing, false for clamp to edge.
    /// @return The created sampler.
    static Sampler linear(vk::Device device, bool repeat = true);

    /// @brief Creates a nearest filtering sampler.
    /// @param device Device that creates the sampler.
    /// @return The created sampler.
    static Sampler nearest(vk::Device device);

    /// @brief Returns true if the sampler holds a valid handle.
    bool valid() const { return static_cast<bool>(sampler_); }

    /// @brief Returns true if the sampler holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan sampler handle.
    vk::Sampler handle() const { return sampler_; }

    /// @brief Destroys the sampler and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::Sampler sampler_;
};

}
