#include "vulcao/fence.h"

#include "vulcao/check.h"

#include <utility>

namespace vulcao {

Fence::~Fence() {
    destroy();
}

Fence::Fence(Fence&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      fence_(std::exchange(other.fence_, vk::Fence{})) {}

Fence& Fence::operator=(Fence&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        fence_ = std::exchange(other.fence_, vk::Fence{});
    }
    return *this;
}

Fence Fence::create(vk::Device device, vk::FenceCreateFlags flags) {
    Fence fence;
    fence.device_ = device;
    fence.fence_ = device.createFence(vk::FenceCreateInfo{.flags = flags});
    return fence;
}

void Fence::wait(uint64_t timeout) const {
    check(device_.waitForFences(fence_, VK_TRUE, timeout), "wait for fence");
}

void Fence::reset() const {
    device_.resetFences(fence_);
}

void Fence::destroy() {
    if (fence_)
        device_.destroyFence(fence_);

    device_ = nullptr;
    fence_ = nullptr;
}

}
