#include "vulcao/command_buffer.h"

#include "vulcao/image.h"

#include <utility>

namespace vulcao {
namespace {

vk::PipelineStageFlags2 stage_for_layout(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
            return vk::PipelineStageFlagBits2::eTopOfPipe;
        case vk::ImageLayout::eTransferSrcOptimal:
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::PipelineStageFlagBits2::eTransfer;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                   vk::PipelineStageFlagBits2::eLateFragmentTests;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
        case vk::ImageLayout::ePresentSrcKHR:
            return vk::PipelineStageFlagBits2::eBottomOfPipe;
        default:
            return vk::PipelineStageFlagBits2::eAllCommands;
    }
}

vk::AccessFlags2 access_for_layout(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
        case vk::ImageLayout::ePresentSrcKHR:
            return {};
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits2::eTransferRead;
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits2::eTransferWrite;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::AccessFlagBits2::eColorAttachmentWrite;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::AccessFlagBits2::eShaderRead;
        default:
            return vk::AccessFlagBits2::eMemoryRead;
    }
}

vk::ImageSubresourceLayers single_layer(const vk::ImageSubresourceRange& range) {
    return vk::ImageSubresourceLayers{
        .aspectMask = range.aspectMask,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

}

CommandBuffer::~CommandBuffer() {
    destroy();
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pool_(std::exchange(other.pool_, vk::CommandPool{})),
      cmd_(std::exchange(other.cmd_, vk::CommandBuffer{})) {}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::CommandPool{});
        cmd_ = std::exchange(other.cmd_, vk::CommandBuffer{});
    }
    return *this;
}

CommandBuffer CommandBuffer::allocate(vk::Device device,
                                      vk::CommandPool pool,
                                      vk::CommandBufferLevel level) {
    CommandBuffer command_buffer;
    command_buffer.device_ = device;
    command_buffer.pool_ = pool;
    command_buffer.cmd_ = device
                              .allocateCommandBuffers(vk::CommandBufferAllocateInfo{
                                  .commandPool = pool,
                                  .level = level,
                                  .commandBufferCount = 1,
                              })
                              .front();
    return command_buffer;
}

void CommandBuffer::destroy() {
    if (cmd_)
        device_.freeCommandBuffers(pool_, cmd_);

    device_ = nullptr;
    pool_ = nullptr;
    cmd_ = nullptr;
}

CommandBuffer& CommandBuffer::reset(vk::CommandBufferResetFlags flags) {
    cmd_.reset(flags);
    return *this;
}

CommandBuffer& CommandBuffer::begin(vk::CommandBufferUsageFlags flags) {
    cmd_.begin(vk::CommandBufferBeginInfo{.flags = flags});
    return *this;
}

CommandBuffer& CommandBuffer::end() {
    cmd_.end();
    return *this;
}

CommandBuffer& CommandBuffer::barrier(vk::PipelineStageFlags2 src_stage,
                                      vk::AccessFlags2 src_access,
                                      vk::PipelineStageFlags2 dst_stage,
                                      vk::AccessFlags2 dst_access) {
    const vk::MemoryBarrier2 memory_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &memory_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::transition(vk::Image image,
                                         vk::ImageLayout old_layout,
                                         vk::ImageLayout new_layout,
                                         const vk::ImageSubresourceRange& range,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::AccessFlags2 src_access,
                                         vk::PipelineStageFlags2 dst_stage,
                                         vk::AccessFlags2 dst_access) {
    if (!src_stage)
        src_stage = stage_for_layout(old_layout);
    if (!src_access)
        src_access = access_for_layout(old_layout);
    if (!dst_stage)
        dst_stage = stage_for_layout(new_layout);
    if (!dst_access)
        dst_access = access_for_layout(new_layout);

    const vk::ImageMemoryBarrier2 image_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = range,
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &image_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::transition(Image& image,
                                         vk::ImageLayout new_layout,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::AccessFlags2 src_access,
                                         vk::PipelineStageFlags2 dst_stage,
                                         vk::AccessFlags2 dst_access) {
    const vk::ImageLayout old_layout = image.layout();
    transition(image.handle(), old_layout, new_layout, image.subresource_range(), src_stage, src_access,
               dst_stage, dst_access);
    image.set_layout(new_layout);
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer(vk::Buffer src,
                                          vk::Buffer dst,
                                          vk::DeviceSize size,
                                          vk::DeviceSize src_offset,
                                          vk::DeviceSize dst_offset) {
    const vk::BufferCopy region{
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = size,
    };
    cmd_.copyBuffer(src, dst, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer_to_image(vk::Buffer src,
                                                   vk::Image dst,
                                                   vk::Extent3D extent,
                                                   const vk::ImageSubresourceLayers& layers,
                                                   vk::Offset3D offset) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = layers,
        .imageOffset = offset,
        .imageExtent = extent,
    };
    cmd_.copyBufferToImage(src, dst, vk::ImageLayout::eTransferDstOptimal, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer_to_image(vk::Buffer src,
                                                   const Image& dst,
                                                   vk::Offset3D offset) {
    return copy_buffer_to_image(src, dst.handle(), dst.extent(),
                                single_layer(dst.subresource_range()), offset);
}

CommandBuffer& CommandBuffer::copy_image_to_buffer(vk::Buffer dst,
                                                   vk::Image src,
                                                   vk::Extent3D extent,
                                                   const vk::ImageSubresourceLayers& layers,
                                                   vk::Offset3D offset) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = layers,
        .imageOffset = offset,
        .imageExtent = extent,
    };
    cmd_.copyImageToBuffer(src, vk::ImageLayout::eTransferSrcOptimal, dst, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_image_to_buffer(vk::Buffer dst,
                                                   const Image& src,
                                                   vk::Offset3D offset) {
    return copy_image_to_buffer(dst, src.handle(), src.extent(),
                                single_layer(src.subresource_range()), offset);
}

}
