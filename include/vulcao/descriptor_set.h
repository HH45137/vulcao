#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

#include "vulcao/reflection.h"

namespace vulcao {

class Buffer;
class DescriptorSet;
class Image;
class Sampler;

/// @brief RAII wrapper around a Vulkan descriptor set layout.
class DescriptorSetLayout {
public:
    /// @brief Creates an empty descriptor set layout.
    DescriptorSetLayout() = default;

    /// @brief Destroys the descriptor set layout.
    ~DescriptorSetLayout();

    /// @brief Not copyable.
    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

    /// @brief Moves the descriptor set layout, leaving the source empty.
    DescriptorSetLayout(DescriptorSetLayout&& other) noexcept;

    /// @brief Move assignment. Destroys the current layout first.
    DescriptorSetLayout& operator=(DescriptorSetLayout&& other) noexcept;

    /// @brief Creates a descriptor set layout from bindings.
    /// @param device Device that creates the layout.
    /// @param bindings Bindings of the layout.
    /// @return The created descriptor set layout.
    static DescriptorSetLayout create(vk::Device device,
                                      vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings);

    /// @brief Creates a descriptor set layout from reflected shader bindings.
    /// @param device Device that creates the layout.
    /// @param reflection Shader reflection data.
    /// @param set Descriptor set index to create the layout for.
    /// @return The created descriptor set layout.
    static DescriptorSetLayout create(vk::Device device,
                                      const ShaderReflection& reflection,
                                      uint32_t set);

    /// @brief Returns true if the layout holds a valid handle.
    bool valid() const { return static_cast<bool>(layout_); }

    /// @brief Returns true if the layout holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan descriptor set layout handle.
    vk::DescriptorSetLayout handle() const { return layout_; }

    /// @brief Destroys the layout and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::DescriptorSetLayout layout_;
};

/// @brief RAII wrapper around a Vulkan descriptor pool.
class DescriptorPool {
public:
    /// @brief Creates an empty descriptor pool.
    DescriptorPool() = default;

    /// @brief Destroys the descriptor pool and all sets allocated from it.
    ~DescriptorPool();

    /// @brief Not copyable.
    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    /// @brief Moves the descriptor pool, leaving the source empty.
    DescriptorPool(DescriptorPool&& other) noexcept;

    /// @brief Move assignment. Destroys the current pool first.
    DescriptorPool& operator=(DescriptorPool&& other) noexcept;

    /// @brief Creates a descriptor pool.
    /// @param device Device that creates the pool.
    /// @param sizes Number of descriptors of each type.
    /// @param max_sets Maximum number of sets that can be allocated.
    /// @param flags Pool creation flags.
    /// @return The created descriptor pool.
    static DescriptorPool create(vk::Device device,
                                 vk::ArrayProxy<const vk::DescriptorPoolSize> sizes,
                                 uint32_t max_sets,
                                 vk::DescriptorPoolCreateFlags flags = {});

    /// @brief Returns true if the pool holds a valid handle.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns true if the pool holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan descriptor pool handle.
    vk::DescriptorPool handle() const { return pool_; }

    /// @brief Allocates one descriptor set. The pool must outlive the returned set.
    /// @param layout Layout of the set.
    /// @return The allocated descriptor set.
    DescriptorSet allocate(const DescriptorSetLayout& layout);

    /// @brief Resets the pool and frees all sets allocated from it.
    /// @param flags Reset flags.
    void reset(vk::DescriptorPoolResetFlags flags = {});

    /// @brief Destroys the pool and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::DescriptorPool pool_;
};

/// @brief Non-owning handle to a Vulkan descriptor set with writing helpers.
class DescriptorSet {
public:
    /// @brief Creates an empty descriptor set handle.
    DescriptorSet() = default;

    /// @brief Returns true if the handle is valid.
    bool valid() const { return static_cast<bool>(set_); }

    /// @brief Returns true if the handle is valid.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan descriptor set handle.
    vk::DescriptorSet handle() const { return set_; }

    /// @brief Writes a buffer descriptor with an explicit descriptor type.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param type Descriptor type.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This descriptor set.
    DescriptorSet& write_buffer(uint32_t binding,
                                const Buffer& buffer,
                                vk::DescriptorType type,
                                vk::DeviceSize offset = 0,
                                vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Writes a uniform buffer descriptor.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This descriptor set.
    DescriptorSet& write_uniform_buffer(uint32_t binding,
                                        const Buffer& buffer,
                                        vk::DeviceSize offset = 0,
                                        vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Writes a storage buffer descriptor.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This descriptor set.
    DescriptorSet& write_storage_buffer(uint32_t binding,
                                        const Buffer& buffer,
                                        vk::DeviceSize offset = 0,
                                        vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Writes a combined image sampler descriptor.
    /// @param binding Binding index.
    /// @param image Image to bind.
    /// @param sampler Sampler to bind.
    /// @param layout Layout the image is sampled in.
    /// @return This descriptor set.
    DescriptorSet& write_image(uint32_t binding,
                               const Image& image,
                               const Sampler& sampler,
                               vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal);

    /// @brief Writes a storage image descriptor.
    /// @param binding Binding index.
    /// @param image Image to bind.
    /// @param layout Layout the image is accessed in.
    /// @return This descriptor set.
    DescriptorSet& write_storage_image(uint32_t binding,
                                       const Image& image,
                                       vk::ImageLayout layout = vk::ImageLayout::eGeneral);

private:
    friend class DescriptorPool;

    vk::Device device_;
    vk::DescriptorSet set_;
};

}
