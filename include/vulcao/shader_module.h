#pragma once

#include <filesystem>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vulcao/reflection.h"

namespace vulcao {

/// @brief RAII wrapper around a Vulkan shader module with its reflection data.
class ShaderModule {
public:
    /// @brief Creates an empty shader module.
    ShaderModule() = default;

    /// @brief Destroys the shader module.
    ~ShaderModule();

    /// @brief Not copyable.
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    /// @brief Moves the shader module, leaving the source empty.
    ShaderModule(ShaderModule&& other) noexcept;

    /// @brief Move assignment. Destroys the current shader module first.
    ShaderModule& operator=(ShaderModule&& other) noexcept;

    /// @brief Creates a shader module from SPIR-V code and reflects it.
    /// @param device Device that creates the shader module.
    /// @param stage Shader stage of the entry point.
    /// @param spirv SPIR-V code.
    /// @return The created shader module.
    /// @throws std::runtime_error if the SPIR-V is empty, cannot be reflected, or module creation fails.
    static ShaderModule create(vk::Device device,
                               vk::ShaderStageFlagBits stage,
                               std::span<const uint32_t> spirv);

    /// @brief Creates a shader module by loading a SPIR-V file and reflects it.
    /// @param device Device that creates the shader module.
    /// @param stage Shader stage of the entry point.
    /// @param path Path to the SPIR-V file.
    /// @return The created shader module.
    /// @throws std::runtime_error if the file cannot be read, its size is invalid, or
    ///         reflection or module creation fails.
    static ShaderModule create_from_file(vk::Device device,
                                         vk::ShaderStageFlagBits stage,
                                         const std::filesystem::path& path);

    /// @brief Returns true if the shader module holds a valid handle.
    bool valid() const { return static_cast<bool>(module_); }

    /// @brief Returns true if the shader module holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan shader module handle.
    vk::ShaderModule handle() const { return module_; }

    /// @brief Returns the shader stage the module was created with.
    vk::ShaderStageFlagBits stage() const { return stage_; }

    /// @brief Returns the reflection data of the shader.
    const ShaderReflection& reflection() const { return reflection_; }

    /// @brief Destroys the shader module and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::ShaderModule module_;
    vk::ShaderStageFlagBits stage_ = vk::ShaderStageFlagBits::eVertex;
    ShaderReflection reflection_;
};

}
