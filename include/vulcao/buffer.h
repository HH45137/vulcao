#pragma once

#include <cstddef>
#include <ranges>
#include <type_traits>

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "vulcao/allocator.h"

namespace vulcao {

/// @brief RAII wrapper around a VMA-allocated buffer.
class Buffer {
public:
    /// @brief Creates an empty buffer.
    Buffer() = default;

    /// @brief Frees the buffer and its memory.
    ~Buffer();

    /// @brief Not copyable.
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /// @brief Moves the buffer, leaving the source empty.
    Buffer(Buffer&& other) noexcept;

    /// @brief Move assignment. Destroys the current buffer first.
    Buffer& operator=(Buffer&& other) noexcept;

    /// @brief Creates a buffer and allocates its memory.
    /// @param allocator Allocator used for the memory.
    /// @param size Buffer size in bytes.
    /// @param usage Buffer usage flags.
    /// @param memory_usage VMA memory usage hint.
    /// @param flags VMA allocation flags.
    /// @return The created buffer.
    /// @throws std::runtime_error if the allocator is invalid or the buffer cannot be created.
    static Buffer create(Allocator& allocator,
                         vk::DeviceSize size,
                         vk::BufferUsageFlags usage,
                         VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                         VmaAllocationCreateFlags flags = 0);

    /// @brief Returns true if the buffer holds a valid handle.
    bool valid() const { return buffer_ != VK_NULL_HANDLE; }

    /// @brief Returns true if the buffer holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan buffer handle.
    vk::Buffer handle() const { return vk::Buffer{buffer_}; }

    /// @brief Returns the buffer size in bytes.
    vk::DeviceSize size() const { return size_; }

    /// @brief Returns the usage flags the buffer was created with.
    vk::BufferUsageFlags usage() const { return usage_; }

    /// @brief Returns the VMA allocation of the buffer.
    VmaAllocation allocation() const { return allocation_; }

    /// @brief Returns information about the VMA allocation.
    const VmaAllocationInfo& allocation_info() const { return info_; }

    /// @brief Returns true if the memory is host visible.
    bool host_visible() const { return host_visible_; }

    /// @brief Maps the memory and returns a pointer to it.
    /// @return Pointer to the mapped memory.
    /// @throws std::runtime_error if the buffer is not host visible or mapping fails.
    void* map();

    /// @brief Unmaps the memory if it was mapped by map().
    void unmap();

    /// @brief Flushes a range of the allocation to make writes visible to the device.
    /// @param offset Byte offset of the range.
    /// @param size Size of the range, or VK_WHOLE_SIZE.
    /// @throws std::runtime_error if the flush fails.
    void flush(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);

    /// @brief Invalidates a range of the allocation to make device writes visible to the host.
    /// @param offset Byte offset of the range.
    /// @param size Size of the range, or VK_WHOLE_SIZE.
    /// @throws std::runtime_error if the invalidate fails.
    void invalidate(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);

    /// @brief Copies raw bytes into a host visible buffer.
    /// @param data Source pointer.
    /// @param size Number of bytes to copy.
    /// @param offset Byte offset in the buffer.
    /// @throws std::runtime_error if the range is out of bounds, the buffer is not host
    ///         visible, or the copy fails.
    void write_bytes(const void* data, vk::DeviceSize size, vk::DeviceSize offset = 0);

    /// @brief Copies a contiguous range of trivially copyable values into a host visible buffer.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param data Source range.
    /// @param offset Byte offset in the buffer.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    void write(const Container& data, vk::DeviceSize offset = 0) {
        using T = std::ranges::range_value_t<Container>;
        static_assert(std::is_trivially_copyable_v<T>, "buffer data must be trivially copyable");
        write_bytes(std::ranges::data(data),
                    static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T),
                    offset);
    }

private:
    /// @brief Frees the buffer and resets the wrapper.
    void destroy();

    VmaAllocator allocator_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VmaAllocationInfo info_{};
    vk::DeviceSize size_ = 0;
    vk::BufferUsageFlags usage_;
    void* mapped_data_ = nullptr;
    bool mapped_ = false;
    bool host_visible_ = false;
};

}
