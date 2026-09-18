#include "vulcao/rendering.h"

namespace vulcao {

vk::RenderingAttachmentInfo color_attachment(vk::ImageView view,
                                             vk::ImageLayout layout,
                                             const vk::ClearColorValue& clear) {
    vk::ClearValue clear_value;
    clear_value.color = clear;
    return vk::RenderingAttachmentInfo{
        .imageView = view,
        .imageLayout = layout,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear_value,
    };
}

vk::RenderingAttachmentInfo color_attachment(vk::ImageView view, vk::ImageLayout layout) {
    return vk::RenderingAttachmentInfo{
        .imageView = view,
        .imageLayout = layout,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

vk::RenderingAttachmentInfo depth_attachment(vk::ImageView view,
                                             vk::ImageLayout layout,
                                             float clear_depth,
                                             uint32_t clear_stencil) {
    vk::ClearValue clear_value;
    clear_value.depthStencil = vk::ClearDepthStencilValue{clear_depth, clear_stencil};
    return vk::RenderingAttachmentInfo{
        .imageView = view,
        .imageLayout = layout,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear_value,
    };
}

vk::RenderingAttachmentInfo depth_attachment(vk::ImageView view, vk::ImageLayout layout) {
    return vk::RenderingAttachmentInfo{
        .imageView = view,
        .imageLayout = layout,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
}

}
