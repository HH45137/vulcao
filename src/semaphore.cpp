#include "vulcao/semaphore.h"

#include <utility>

namespace vulcao {

Semaphore::~Semaphore() {
    destroy();
}

Semaphore::Semaphore(Semaphore&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      semaphore_(std::exchange(other.semaphore_, vk::Semaphore{})) {}

Semaphore& Semaphore::operator=(Semaphore&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        semaphore_ = std::exchange(other.semaphore_, vk::Semaphore{});
    }
    return *this;
}

Semaphore Semaphore::create(vk::Device device, vk::SemaphoreCreateFlags flags) {
    Semaphore semaphore;
    semaphore.device_ = device;
    semaphore.semaphore_ = device.createSemaphore(vk::SemaphoreCreateInfo{.flags = flags});
    return semaphore;
}

void Semaphore::destroy() {
    if (semaphore_)
        device_.destroySemaphore(semaphore_);

    device_ = nullptr;
    semaphore_ = nullptr;
}

}
