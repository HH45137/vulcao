#include "vulcao/descriptor_set.h"

#include "vulcao/buffer.h"
#include "vulcao/image.h"
#include "vulcao/sampler.h"

#include <stdexcept>
#include <utility>

namespace vulcao {

DescriptorSetLayout::~DescriptorSetLayout() {
    destroy();
}

DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      layout_(std::exchange(other.layout_, vk::DescriptorSetLayout{})) {}

DescriptorSetLayout& DescriptorSetLayout::operator=(DescriptorSetLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        layout_ = std::exchange(other.layout_, vk::DescriptorSetLayout{});
    }
    return *this;
}

DescriptorSetLayout DescriptorSetLayout::create(
    vk::Device device, vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings) {
    DescriptorSetLayout layout;
    layout.device_ = device;
    layout.layout_ = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    });
    return layout;
}

DescriptorSetLayout DescriptorSetLayout::create(vk::Device device,
                                                const ShaderReflection& reflection,
                                                uint32_t set) {
    return create(device, reflection.bindings_for_set(set));
}

void DescriptorSetLayout::destroy() {
    if (layout_)
        device_.destroyDescriptorSetLayout(layout_);

    device_ = nullptr;
    layout_ = nullptr;
}

DescriptorPool::~DescriptorPool() {
    destroy();
}

DescriptorPool::DescriptorPool(DescriptorPool&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pool_(std::exchange(other.pool_, vk::DescriptorPool{})) {}

DescriptorPool& DescriptorPool::operator=(DescriptorPool&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::DescriptorPool{});
    }
    return *this;
}

DescriptorPool DescriptorPool::create(vk::Device device,
                                      vk::ArrayProxy<const vk::DescriptorPoolSize> sizes,
                                      uint32_t max_sets,
                                      vk::DescriptorPoolCreateFlags flags) {
    DescriptorPool pool;
    pool.device_ = device;
    pool.pool_ = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .flags = flags,
        .maxSets = max_sets,
        .poolSizeCount = static_cast<uint32_t>(sizes.size()),
        .pPoolSizes = sizes.data(),
    });
    return pool;
}

DescriptorSet DescriptorPool::allocate(const DescriptorSetLayout& layout) {
    if (!layout.valid())
        throw std::runtime_error("DescriptorPool::allocate: invalid layout");

    const vk::DescriptorSetLayout raw_layout = layout.handle();
    const vk::DescriptorSet set = device_
                                      .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
                                          .descriptorPool = pool_,
                                          .descriptorSetCount = 1,
                                          .pSetLayouts = &raw_layout,
                                      })
                                      .front();

    DescriptorSet result;
    result.device_ = device_;
    result.set_ = set;
    return result;
}

void DescriptorPool::reset(vk::DescriptorPoolResetFlags flags) {
    device_.resetDescriptorPool(pool_, flags);
}

void DescriptorPool::destroy() {
    if (pool_)
        device_.destroyDescriptorPool(pool_);

    device_ = nullptr;
    pool_ = nullptr;
}

DescriptorSet& DescriptorSet::write_buffer(uint32_t binding,
                                           const Buffer& buffer,
                                           vk::DescriptorType type,
                                           vk::DeviceSize offset,
                                           vk::DeviceSize range) {
    const vk::DescriptorBufferInfo info{
        .buffer = buffer.handle(),
        .offset = offset,
        .range = range,
    };

    device_.updateDescriptorSets(vk::WriteDescriptorSet{
                                     .dstSet = set_,
                                     .dstBinding = binding,
                                     .descriptorCount = 1,
                                     .descriptorType = type,
                                     .pBufferInfo = &info,
                                 },
                                 {});
    return *this;
}

DescriptorSet& DescriptorSet::write_uniform_buffer(uint32_t binding,
                                                   const Buffer& buffer,
                                                   vk::DeviceSize offset,
                                                   vk::DeviceSize range) {
    return write_buffer(binding, buffer, vk::DescriptorType::eUniformBuffer, offset, range);
}

DescriptorSet& DescriptorSet::write_storage_buffer(uint32_t binding,
                                                   const Buffer& buffer,
                                                   vk::DeviceSize offset,
                                                   vk::DeviceSize range) {
    return write_buffer(binding, buffer, vk::DescriptorType::eStorageBuffer, offset, range);
}

DescriptorSet& DescriptorSet::write_image(uint32_t binding,
                                          const Image& image,
                                          const Sampler& sampler,
                                          vk::ImageLayout layout) {
    const vk::DescriptorImageInfo info{
        .sampler = sampler.handle(),
        .imageView = image.view(),
        .imageLayout = layout,
    };

    device_.updateDescriptorSets(vk::WriteDescriptorSet{
                                     .dstSet = set_,
                                     .dstBinding = binding,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                     .pImageInfo = &info,
                                 },
                                 {});
    return *this;
}

DescriptorSet& DescriptorSet::write_storage_image(uint32_t binding,
                                                  const Image& image,
                                                  vk::ImageLayout layout) {
    const vk::DescriptorImageInfo info{
        .sampler = nullptr,
        .imageView = image.view(),
        .imageLayout = layout,
    };

    device_.updateDescriptorSets(vk::WriteDescriptorSet{
                                     .dstSet = set_,
                                     .dstBinding = binding,
                                     .descriptorCount = 1,
                                     .descriptorType = vk::DescriptorType::eStorageImage,
                                     .pImageInfo = &info,
                                 },
                                 {});
    return *this;
}

}
