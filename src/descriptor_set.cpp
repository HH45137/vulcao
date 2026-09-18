#include "vulcao/descriptor_set.h"

#include "vulcao/buffer.h"
#include "vulcao/image.h"
#include "vulcao/sampler.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
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
    vk::Device device, vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings,
    vk::DescriptorSetLayoutCreateFlags flags,
    vk::ArrayProxy<const vk::DescriptorBindingFlags> binding_flags) {
    if (!binding_flags.empty() && binding_flags.size() != bindings.size())
        throw std::runtime_error(
            "DescriptorSetLayout::create: binding_flags must be empty or match the binding count");
    for (const vk::DescriptorSetLayoutBinding& binding : bindings)
        if (binding.descriptorCount == 0)
            throw std::runtime_error(
                "DescriptorSetLayout::create: descriptorCount 0 is not valid; a reflected runtime "
                "array needs a concrete upper bound, see set_binding_count");

    const vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{
        .bindingCount = static_cast<uint32_t>(binding_flags.size()),
        .pBindingFlags = binding_flags.data(),
    };

    DescriptorSetLayout layout;
    layout.device_ = device;
    layout.layout_ = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        .pNext = binding_flags.empty() ? nullptr : &binding_flags_info,
        .flags = flags,
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

DescriptorSet DescriptorPool::allocate(const DescriptorSetLayout& layout,
                                       uint32_t variable_descriptor_count) {
    if (!layout.valid())
        throw std::runtime_error("DescriptorPool::allocate: invalid layout");
    return allocate(layout.handle(), variable_descriptor_count);
}

DescriptorSet DescriptorPool::allocate(vk::DescriptorSetLayout layout,
                                       uint32_t variable_descriptor_count) {
    if (!layout)
        throw std::runtime_error("DescriptorPool::allocate: invalid layout");

    const vk::DescriptorSetVariableDescriptorCountAllocateInfo variable_info{
        .descriptorSetCount = 1,
        .pDescriptorCounts = &variable_descriptor_count,
    };

    const vk::DescriptorSet set = device_
                                      .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
                                          .pNext = variable_descriptor_count > 0 ? &variable_info
                                                                                 : nullptr,
                                          .descriptorPool = pool_,
                                          .descriptorSetCount = 1,
                                          .pSetLayouts = &layout,
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

const DescriptorSet& DescriptorSet::write_buffer(uint32_t binding,
                                           const Buffer& buffer,
                                           vk::DescriptorType type,
                                           vk::DeviceSize offset,
                                           vk::DeviceSize range) const {
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

const DescriptorSet& DescriptorSet::write_uniform_buffer(uint32_t binding,
                                                   const Buffer& buffer,
                                                   vk::DeviceSize offset,
                                                   vk::DeviceSize range) const {
    return write_buffer(binding, buffer, vk::DescriptorType::eUniformBuffer, offset, range);
}

const DescriptorSet& DescriptorSet::write_storage_buffer(uint32_t binding,
                                                   const Buffer& buffer,
                                                   vk::DeviceSize offset,
                                                   vk::DeviceSize range) const {
    return write_buffer(binding, buffer, vk::DescriptorType::eStorageBuffer, offset, range);
}

const DescriptorSet& DescriptorSet::write_image(uint32_t binding,
                                          const Image& image,
                                          const Sampler& sampler,
                                          vk::ImageLayout layout) const {
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

const DescriptorSet& DescriptorSet::write_storage_image(uint32_t binding,
                                                  const Image& image,
                                                  vk::ImageLayout layout) const {
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

DescriptorSetWriter::DescriptorSetWriter(const DescriptorSet& set)
    : device_(set.device_), set_(set.set_) {}

DescriptorSetWriter& DescriptorSetWriter::write_buffer(uint32_t binding,
                                                       const Buffer& buffer,
                                                       vk::DescriptorType type,
                                                       uint32_t array_element,
                                                       vk::DeviceSize offset,
                                                       vk::DeviceSize range) {
    records_.push_back(Record{
        .binding = binding,
        .array_element = array_element,
        .type = type,
        .is_image = false,
        .buffer_info = vk::DescriptorBufferInfo{
            .buffer = buffer.handle(),
            .offset = offset,
            .range = range,
        },
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::write_uniform_buffer(uint32_t binding,
                                                               const Buffer& buffer,
                                                               uint32_t array_element,
                                                               vk::DeviceSize offset,
                                                               vk::DeviceSize range) {
    return write_buffer(binding, buffer, vk::DescriptorType::eUniformBuffer, array_element, offset,
                        range);
}

DescriptorSetWriter& DescriptorSetWriter::write_storage_buffer(uint32_t binding,
                                                               const Buffer& buffer,
                                                               uint32_t array_element,
                                                               vk::DeviceSize offset,
                                                               vk::DeviceSize range) {
    return write_buffer(binding, buffer, vk::DescriptorType::eStorageBuffer, array_element, offset,
                        range);
}

DescriptorSetWriter& DescriptorSetWriter::write_image(uint32_t binding,
                                                      const Image& image,
                                                      const Sampler& sampler,
                                                      vk::ImageLayout layout,
                                                      uint32_t array_element) {
    records_.push_back(Record{
        .binding = binding,
        .array_element = array_element,
        .type = vk::DescriptorType::eCombinedImageSampler,
        .is_image = true,
        .image_info = vk::DescriptorImageInfo{
            .sampler = sampler.handle(),
            .imageView = image.view(),
            .imageLayout = layout,
        },
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::write_storage_image(uint32_t binding,
                                                              const Image& image,
                                                              vk::ImageLayout layout,
                                                              uint32_t array_element) {
    records_.push_back(Record{
        .binding = binding,
        .array_element = array_element,
        .type = vk::DescriptorType::eStorageImage,
        .is_image = true,
        .image_info = vk::DescriptorImageInfo{
            .sampler = nullptr,
            .imageView = image.view(),
            .imageLayout = layout,
        },
    });
    return *this;
}

void DescriptorSetWriter::flush() {
    if (records_.empty())
        return;

    std::vector<vk::DescriptorBufferInfo> buffer_infos(records_.size());
    std::vector<vk::DescriptorImageInfo> image_infos(records_.size());
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(records_.size());

    for (size_t i = 0; i < records_.size(); ++i) {
        const Record& record = records_[i];
        vk::WriteDescriptorSet write{
            .dstSet = set_,
            .dstBinding = record.binding,
            .dstArrayElement = record.array_element,
            .descriptorCount = 1,
            .descriptorType = record.type,
        };

        if (record.is_image) {
            image_infos[i] = record.image_info;
            write.pImageInfo = &image_infos[i];
        } else {
            buffer_infos[i] = record.buffer_info;
            write.pBufferInfo = &buffer_infos[i];
        }

        writes.push_back(write);
    }

    device_.updateDescriptorSets(writes, {});
    records_.clear();
}

void DescriptorSetWriter::clear() {
    records_.clear();
}

DescriptorSetLayoutCache::DescriptorSetLayoutCache(vk::Device device) : device_(device) {}

bool DescriptorSetLayoutCache::Key::Entry::operator<(const Entry& other) const {
    const auto lhs = std::tie(binding, type, count, stages, flags);
    const auto rhs = std::tie(other.binding, other.type, other.count, other.stages, other.flags);
    if (lhs != rhs)
        return lhs < rhs;
    return immutable_samplers < other.immutable_samplers;
}

bool DescriptorSetLayoutCache::Key::operator<(const Key& other) const {
    if (create_flags != other.create_flags)
        return create_flags < other.create_flags;
    return entries < other.entries;
}

vk::DescriptorSetLayout DescriptorSetLayoutCache::get(
    vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings,
    vk::DescriptorSetLayoutCreateFlags flags,
    vk::ArrayProxy<const vk::DescriptorBindingFlags> binding_flags) {
    if (!binding_flags.empty() && binding_flags.size() != bindings.size())
        throw std::runtime_error(
            "DescriptorSetLayoutCache::get: binding_flags must be empty or match the binding count");

    Key key;
    key.create_flags = flags;
    key.entries.reserve(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        const vk::DescriptorSetLayoutBinding& binding = bindings.data()[i];
        Key::Entry entry;
        entry.binding = binding.binding;
        entry.type = binding.descriptorType;
        entry.count = binding.descriptorCount;
        entry.stages = binding.stageFlags;
        entry.flags = binding_flags.empty() ? vk::DescriptorBindingFlags{} : binding_flags.data()[i];
        // Immutable samplers are baked into the layout, so they must be part of
        // the key or two layouts with different samplers would collide.
        if (binding.pImmutableSamplers != nullptr)
            entry.immutable_samplers.assign(binding.pImmutableSamplers,
                                            binding.pImmutableSamplers + binding.descriptorCount);
        key.entries.push_back(std::move(entry));
    }
    std::sort(key.entries.begin(), key.entries.end());

    const auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second.handle();

    DescriptorSetLayout layout = DescriptorSetLayout::create(device_, bindings, flags, binding_flags);
    const vk::DescriptorSetLayout handle = layout.handle();
    cache_.emplace(std::move(key), std::move(layout));
    return handle;
}

void DescriptorSetLayoutCache::clear() {
    cache_.clear();
}

}
