#include "vulcao/sampler.h"

#include <utility>

namespace vulcao {

Sampler::~Sampler() {
    destroy();
}

Sampler::Sampler(Sampler&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      sampler_(std::exchange(other.sampler_, vk::Sampler{})) {}

Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        sampler_ = std::exchange(other.sampler_, vk::Sampler{});
    }
    return *this;
}

Sampler Sampler::create(vk::Device device, const vk::SamplerCreateInfo& info) {
    Sampler sampler;
    sampler.device_ = device;
    sampler.sampler_ = device.createSampler(info);
    return sampler;
}

Sampler Sampler::linear(vk::Device device, bool repeat) {
    const vk::SamplerAddressMode address_mode =
        repeat ? vk::SamplerAddressMode::eRepeat : vk::SamplerAddressMode::eClampToEdge;

    return create(device, vk::SamplerCreateInfo{
                              .magFilter = vk::Filter::eLinear,
                              .minFilter = vk::Filter::eLinear,
                              .mipmapMode = vk::SamplerMipmapMode::eLinear,
                              .addressModeU = address_mode,
                              .addressModeV = address_mode,
                              .addressModeW = address_mode,
                              .maxLod = VK_LOD_CLAMP_NONE,
                          });
}

Sampler Sampler::nearest(vk::Device device) {
    return create(device, vk::SamplerCreateInfo{
                              .magFilter = vk::Filter::eNearest,
                              .minFilter = vk::Filter::eNearest,
                              .mipmapMode = vk::SamplerMipmapMode::eNearest,
                              .addressModeU = vk::SamplerAddressMode::eRepeat,
                              .addressModeV = vk::SamplerAddressMode::eRepeat,
                              .addressModeW = vk::SamplerAddressMode::eRepeat,
                              .maxLod = VK_LOD_CLAMP_NONE,
                          });
}

void Sampler::destroy() {
    if (sampler_)
        device_.destroySampler(sampler_);

    device_ = nullptr;
    sampler_ = nullptr;
}

}
