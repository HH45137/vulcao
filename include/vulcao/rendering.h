#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief Builders for the dynamic rendering structures, removing the per-frame
///        boilerplate of filling vk::RenderingAttachmentInfo by hand.
///
/// The builders cover the two dominant attachment patterns: clearing on load
/// and loading the previous contents. Anything more involved (resolves, store
/// op none, layered rendering) should keep using the raw structs.

/// @brief Returns a color attachment that is cleared on load and stored.
/// @param view Attachment image view.
/// @param layout Layout the attachment is in during rendering.
/// @param clear Clear color.
vk::RenderingAttachmentInfo color_attachment(vk::ImageView view,
                                             vk::ImageLayout layout,
                                             const vk::ClearColorValue& clear);

/// @brief Returns a color attachment that loads and stores its contents.
/// @param view Attachment image view.
/// @param layout Layout the attachment is in during rendering.
vk::RenderingAttachmentInfo color_attachment(vk::ImageView view, vk::ImageLayout layout);

/// @brief Returns a depth attachment that is cleared on load and stored.
///
/// The clear depth is explicit so that the two-argument call always means
/// "load the previous contents".
/// @param view Attachment image view.
/// @param layout Layout the attachment is in during rendering.
/// @param clear_depth Clear depth value, typically 1.0f or 0.0f.
/// @param clear_stencil Clear stencil value.
vk::RenderingAttachmentInfo depth_attachment(vk::ImageView view,
                                             vk::ImageLayout layout,
                                             float clear_depth,
                                             uint32_t clear_stencil = 0);

/// @brief Returns a depth attachment that loads and stores its contents.
/// @param view Attachment image view.
/// @param layout Layout the attachment is in during rendering.
vk::RenderingAttachmentInfo depth_attachment(vk::ImageView view, vk::ImageLayout layout);

}
