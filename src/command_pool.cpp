#include "vulcao/command_pool.h"

#include <utility>

namespace vulcao {

CommandPool::~CommandPool() {
    destroy();
}

CommandPool::CommandPool(CommandPool&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pool_(std::exchange(other.pool_, vk::CommandPool{})) {}

CommandPool& CommandPool::operator=(CommandPool&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::CommandPool{});
    }
    return *this;
}

CommandPool CommandPool::create(vk::Device device,
                                uint32_t queue_family_index,
                                vk::CommandPoolCreateFlags flags) {
    CommandPool pool;
    pool.device_ = device;
    pool.pool_ = device.createCommandPool(vk::CommandPoolCreateInfo{
        .flags = flags,
        .queueFamilyIndex = queue_family_index,
    });
    return pool;
}

CommandBuffer CommandPool::allocate(vk::CommandBufferLevel level, bool debug_utils) const {
    return CommandBuffer::allocate(device_, pool_, level, debug_utils);
}

void CommandPool::reset(vk::CommandPoolResetFlags flags) const {
    device_.resetCommandPool(pool_, flags);
}

void CommandPool::destroy() {
    if (pool_)
        device_.destroyCommandPool(pool_);

    device_ = nullptr;
    pool_ = nullptr;
}

}
