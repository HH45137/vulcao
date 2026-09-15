#include "vulcao/pipeline_cache.h"

#include <utility>

namespace vulcao {

PipelineCache::~PipelineCache() {
    destroy();
}

PipelineCache::PipelineCache(PipelineCache&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      cache_(std::exchange(other.cache_, vk::PipelineCache{})) {}

PipelineCache& PipelineCache::operator=(PipelineCache&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        cache_ = std::exchange(other.cache_, vk::PipelineCache{});
    }
    return *this;
}

PipelineCache PipelineCache::create(vk::Device device,
                                    const void* data,
                                    size_t size,
                                    vk::PipelineCacheCreateFlags flags) {
    PipelineCache cache;
    cache.device_ = device;
    cache.cache_ = device.createPipelineCache(vk::PipelineCacheCreateInfo{
        .flags = flags,
        .initialDataSize = size,
        .pInitialData = data,
    });
    return cache;
}

std::vector<uint8_t> PipelineCache::data() const {
    return device_.getPipelineCacheData(cache_);
}

void PipelineCache::destroy() {
    if (cache_)
        device_.destroyPipelineCache(cache_);

    device_ = nullptr;
    cache_ = nullptr;
}

}
