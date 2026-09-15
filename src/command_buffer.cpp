#include "vulcao/command_buffer.h"

#include "vulcao/buffer.h"
#include "vulcao/image.h"

#include <algorithm>
#include <stdexcept>
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
      cmd_(std::exchange(other.cmd_, vk::CommandBuffer{})),
      level_(std::exchange(other.level_, vk::CommandBufferLevel::ePrimary)) {}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::CommandPool{});
        cmd_ = std::exchange(other.cmd_, vk::CommandBuffer{});
        level_ = std::exchange(other.level_, vk::CommandBufferLevel::ePrimary);
    }
    return *this;
}

CommandBuffer CommandBuffer::allocate(vk::Device device,
                                      vk::CommandPool pool,
                                      vk::CommandBufferLevel level) {
    CommandBuffer command_buffer;
    command_buffer.device_ = device;
    command_buffer.pool_ = pool;
    command_buffer.level_ = level;
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
    level_ = vk::CommandBufferLevel::ePrimary;
}

CommandBuffer& CommandBuffer::reset(vk::CommandBufferResetFlags flags) {
    cmd_.reset(flags);
    return *this;
}

CommandBuffer& CommandBuffer::begin(vk::CommandBufferUsageFlags flags) {
    if (level_ == vk::CommandBufferLevel::eSecondary) {
        const vk::CommandBufferInheritanceInfo inheritance{};
        cmd_.begin(vk::CommandBufferBeginInfo{
            .flags = flags,
            .pInheritanceInfo = &inheritance,
        });
        return *this;
    }

    cmd_.begin(vk::CommandBufferBeginInfo{.flags = flags});
    return *this;
}

CommandBuffer& CommandBuffer::begin(const vk::CommandBufferInheritanceInfo& inheritance,
                                    vk::CommandBufferUsageFlags flags) {
    cmd_.begin(vk::CommandBufferBeginInfo{
        .flags = flags,
        .pInheritanceInfo = &inheritance,
    });
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

CommandBuffer& CommandBuffer::buffer_barrier(vk::Buffer buffer,
                                             vk::PipelineStageFlags2 src_stage,
                                             vk::AccessFlags2 src_access,
                                             vk::PipelineStageFlags2 dst_stage,
                                             vk::AccessFlags2 dst_access,
                                             vk::DeviceSize offset,
                                             vk::DeviceSize size) {
    const vk::BufferMemoryBarrier2 buffer_memory_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .buffer = buffer,
        .offset = offset,
        .size = size,
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &buffer_memory_barrier,
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

CommandBuffer& CommandBuffer::generate_mipmaps(Image& image, vk::ImageLayout final_layout) {
    const uint32_t level_count = image.mip_levels();
    if (level_count <= 1) {
        transition(image, final_layout);
        return *this;
    }

    const vk::ImageAspectFlags aspect = image.subresource_range().aspectMask;
    const vk::Extent3D extent = image.extent();
    const vk::Image handle = image.handle();

    for (uint32_t level = 1; level < level_count; ++level) {
        const vk::ImageSubresourceRange src_range{aspect, level - 1, 1, 0, 1};
        const vk::ImageSubresourceRange dst_range{aspect, level, 1, 0, 1};

        const vk::ImageLayout src_old =
            level == 1 ? vk::ImageLayout::eTransferDstOptimal : final_layout;
        transition(handle, src_old, vk::ImageLayout::eTransferSrcOptimal, src_range);

        const uint32_t src_width = std::max(1u, extent.width >> (level - 1));
        const uint32_t src_height = std::max(1u, extent.height >> (level - 1));
        const uint32_t dst_width = std::max(1u, extent.width >> level);
        const uint32_t dst_height = std::max(1u, extent.height >> level);

        vk::ImageBlit region{};
        region.srcSubresource = vk::ImageSubresourceLayers{aspect, level - 1, 0, 1};
        region.srcOffsets[0] = vk::Offset3D{0, 0, 0};
        region.srcOffsets[1] = vk::Offset3D{static_cast<int32_t>(src_width),
                                            static_cast<int32_t>(src_height), 1};
        region.dstSubresource = vk::ImageSubresourceLayers{aspect, level, 0, 1};
        region.dstOffsets[0] = vk::Offset3D{0, 0, 0};
        region.dstOffsets[1] = vk::Offset3D{static_cast<int32_t>(dst_width),
                                            static_cast<int32_t>(dst_height), 1};

        blit_image(handle, vk::ImageLayout::eTransferSrcOptimal, handle,
                   vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);

        transition(handle, vk::ImageLayout::eTransferDstOptimal, final_layout, dst_range);
    }

    const vk::ImageSubresourceRange source_levels{aspect, 0, level_count - 1, 0, 1};
    transition(handle, vk::ImageLayout::eTransferSrcOptimal, final_layout, source_levels);

    image.set_layout(final_layout);
    return *this;
}

CommandBuffer& CommandBuffer::copy_image(vk::Image src,
                                         vk::ImageLayout src_layout,
                                         vk::Image dst,
                                         vk::ImageLayout dst_layout,
                                         const vk::ImageCopy& region) {
    cmd_.copyImage(src, src_layout, dst, dst_layout, region);
    return *this;
}

CommandBuffer& CommandBuffer::blit_image(vk::Image src,
                                         vk::ImageLayout src_layout,
                                         vk::Image dst,
                                         vk::ImageLayout dst_layout,
                                         const vk::ImageBlit& region,
                                         vk::Filter filter) {
    cmd_.blitImage(src, src_layout, dst, dst_layout, region, filter);
    return *this;
}

CommandBuffer& CommandBuffer::clear_color_image(vk::Image image,
                                                vk::ImageLayout layout,
                                                const vk::ClearColorValue& color,
                                                const vk::ImageSubresourceRange& range) {
    cmd_.clearColorImage(image, layout, color, range);
    return *this;
}

CommandBuffer& CommandBuffer::clear_color_image(const Image& image, const vk::ClearColorValue& color) {
    return clear_color_image(image.handle(), image.layout(), color, image.subresource_range());
}

CommandBuffer& CommandBuffer::clear_depth_stencil_image(vk::Image image,
                                                        vk::ImageLayout layout,
                                                        const vk::ClearDepthStencilValue& depth_stencil,
                                                        const vk::ImageSubresourceRange& range) {
    cmd_.clearDepthStencilImage(image, layout, depth_stencil, range);
    return *this;
}

CommandBuffer& CommandBuffer::clear_depth_stencil_image(const Image& image,
                                                        const vk::ClearDepthStencilValue& depth_stencil) {
    return clear_depth_stencil_image(image.handle(), image.layout(), depth_stencil,
                                     image.subresource_range());
}

CommandBuffer& CommandBuffer::execute_commands(vk::CommandBuffer cmd) {
    cmd_.executeCommands(cmd);
    return *this;
}

CommandBuffer& CommandBuffer::fill_buffer(vk::Buffer dst,
                                          vk::DeviceSize offset,
                                          vk::DeviceSize size,
                                          uint32_t data) {
    if (size == 0 || offset % 4 != 0 || size % 4 != 0)
        throw std::runtime_error("CommandBuffer::fill_buffer: size must be non-zero and offset and size must be multiples of 4");

    cmd_.fillBuffer(dst, offset, size, data);
    return *this;
}

CommandBuffer& CommandBuffer::update_buffer(vk::Buffer dst,
                                            vk::DeviceSize offset,
                                            const void* data,
                                            vk::DeviceSize size) {
    if (size == 0 || size % 4 != 0 || size > 65536)
        throw std::runtime_error("CommandBuffer::update_buffer: size must be a non-zero multiple of 4 and at most 65536 bytes");

    cmd_.updateBuffer(dst, offset, size, data);
    return *this;
}

CommandBuffer& CommandBuffer::bind_pipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) {
    cmd_.bindPipeline(bind_point, pipeline);
    return *this;
}

CommandBuffer& CommandBuffer::bind_vertex_buffer(uint32_t binding,
                                                 const Buffer& buffer,
                                                 vk::DeviceSize offset) {
    const vk::Buffer raw_buffer = buffer.handle();
    cmd_.bindVertexBuffers(binding, raw_buffer, offset);
    return *this;
}

CommandBuffer& CommandBuffer::bind_vertex_buffers(uint32_t first_binding,
                                                  vk::ArrayProxy<const vk::Buffer> buffers,
                                                  vk::ArrayProxy<const vk::DeviceSize> offsets) {
    cmd_.bindVertexBuffers(first_binding, buffers, offsets);
    return *this;
}

CommandBuffer& CommandBuffer::bind_index_buffer(const Buffer& buffer,
                                                vk::DeviceSize offset,
                                                vk::IndexType index_type) {
    cmd_.bindIndexBuffer(buffer.handle(), offset, index_type);
    return *this;
}

CommandBuffer& CommandBuffer::set_viewport(const vk::Viewport& viewport) {
    cmd_.setViewport(0, viewport);
    return *this;
}

CommandBuffer& CommandBuffer::set_viewport(vk::Extent2D extent) {
    return set_viewport(vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    });
}

CommandBuffer& CommandBuffer::set_scissor(const vk::Rect2D& scissor) {
    cmd_.setScissor(0, scissor);
    return *this;
}

CommandBuffer& CommandBuffer::set_scissor(vk::Extent2D extent) {
    return set_scissor(vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = extent});
}

CommandBuffer& CommandBuffer::set_cull_mode(vk::CullModeFlags cull_mode) {
    cmd_.setCullMode(cull_mode);
    return *this;
}

CommandBuffer& CommandBuffer::set_front_face(vk::FrontFace front_face) {
    cmd_.setFrontFace(front_face);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_bias(float constant_factor, float clamp, float slope_factor) {
    cmd_.setDepthBias(constant_factor, clamp, slope_factor);
    return *this;
}

CommandBuffer& CommandBuffer::set_blend_constants(const std::array<float, 4>& constants) {
    cmd_.setBlendConstants(constants.data());
    return *this;
}

CommandBuffer& CommandBuffer::set_stencil_reference(vk::StencilFaceFlags face_mask,
                                                    uint32_t reference) {
    cmd_.setStencilReference(face_mask, reference);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_bounds(float min_depth_bounds, float max_depth_bounds) {
    cmd_.setDepthBounds(min_depth_bounds, max_depth_bounds);
    return *this;
}

CommandBuffer& CommandBuffer::set_line_width(float line_width) {
    cmd_.setLineWidth(line_width);
    return *this;
}

CommandBuffer& CommandBuffer::set_primitive_topology(vk::PrimitiveTopology topology) {
    cmd_.setPrimitiveTopology(topology);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_test_enable(bool enable) {
    cmd_.setDepthTestEnable(enable ? VK_TRUE : VK_FALSE);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_write_enable(bool enable) {
    cmd_.setDepthWriteEnable(enable ? VK_TRUE : VK_FALSE);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_compare_op(vk::CompareOp compare_op) {
    cmd_.setDepthCompareOp(compare_op);
    return *this;
}

CommandBuffer& CommandBuffer::push_constants(vk::PipelineLayout layout,
                                             vk::ShaderStageFlags stages,
                                             uint32_t offset,
                                             const void* data,
                                             uint32_t size) {
    cmd_.pushConstants(layout, stages, offset, size, data);
    return *this;
}

CommandBuffer& CommandBuffer::draw(uint32_t vertex_count,
                                   uint32_t instance_count,
                                   uint32_t first_vertex,
                                   uint32_t first_instance) {
    cmd_.draw(vertex_count, instance_count, first_vertex, first_instance);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indexed(uint32_t index_count,
                                           uint32_t instance_count,
                                           uint32_t first_index,
                                           int32_t vertex_offset,
                                           uint32_t first_instance) {
    cmd_.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indirect(vk::Buffer buffer,
                                            vk::DeviceSize offset,
                                            uint32_t draw_count,
                                            uint32_t stride) {
    cmd_.drawIndirect(buffer, offset, draw_count, stride);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indexed_indirect(vk::Buffer buffer,
                                                    vk::DeviceSize offset,
                                                    uint32_t draw_count,
                                                    uint32_t stride) {
    cmd_.drawIndexedIndirect(buffer, offset, draw_count, stride);
    return *this;
}

CommandBuffer& CommandBuffer::begin_rendering(const vk::RenderingInfo& info) {
    cmd_.beginRendering(info);
    return *this;
}

CommandBuffer& CommandBuffer::end_rendering() {
    cmd_.endRendering();
    return *this;
}

CommandBuffer& CommandBuffer::bind_descriptor_sets(vk::PipelineBindPoint bind_point,
                                                   vk::PipelineLayout layout,
                                                   vk::ArrayProxy<const vk::DescriptorSet> descriptor_sets,
                                                   vk::ArrayProxy<const uint32_t> dynamic_offsets) {
    cmd_.bindDescriptorSets(bind_point, layout, 0, descriptor_sets, dynamic_offsets);
    return *this;
}

CommandBuffer& CommandBuffer::dispatch(uint32_t group_count_x,
                                       uint32_t group_count_y,
                                       uint32_t group_count_z) {
    cmd_.dispatch(group_count_x, group_count_y, group_count_z);
    return *this;
}

CommandBuffer& CommandBuffer::dispatch_indirect(vk::Buffer buffer, vk::DeviceSize offset) {
    cmd_.dispatchIndirect(buffer, offset);
    return *this;
}

CommandBuffer& CommandBuffer::reset_query_pool(vk::QueryPool pool,
                                               uint32_t first_query,
                                               uint32_t query_count) {
    cmd_.resetQueryPool(pool, first_query, query_count);
    return *this;
}

CommandBuffer& CommandBuffer::write_timestamp(vk::QueryPool pool,
                                              vk::PipelineStageFlags2 stage,
                                              uint32_t query) {
    cmd_.writeTimestamp2(stage, pool, query);
    return *this;
}

CommandBuffer& CommandBuffer::begin_query(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) {
    cmd_.beginQuery(pool, query, flags);
    return *this;
}

CommandBuffer& CommandBuffer::end_query(vk::QueryPool pool, uint32_t query) {
    cmd_.endQuery(pool, query);
    return *this;
}

CommandBuffer& CommandBuffer::copy_query_pool_results(vk::QueryPool pool,
                                                      uint32_t first_query,
                                                      uint32_t query_count,
                                                      vk::Buffer dst,
                                                      vk::DeviceSize dst_offset,
                                                      vk::DeviceSize stride,
                                                      vk::QueryResultFlags flags) {
    cmd_.copyQueryPoolResults(pool, first_query, query_count, dst, dst_offset, stride, flags);
    return *this;
}

}
