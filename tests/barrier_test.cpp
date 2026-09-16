#include <doctest/doctest.h>

#include <vulcao/barrier.h>

namespace {

constexpr vk::PipelineStageFlags2 stages(vk::PipelineStageFlagBits2 bits) {
    return vk::PipelineStageFlags2{bits};
}

constexpr vk::AccessFlags2 access(vk::AccessFlagBits2 bits) {
    return vk::AccessFlags2{bits};
}

}

TEST_CASE("stage_for_layout maps common layouts") {
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eUndefined) ==
          stages(vk::PipelineStageFlagBits2::eTopOfPipe));
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eTransferSrcOptimal) ==
          stages(vk::PipelineStageFlagBits2::eTransfer));
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eTransferDstOptimal) ==
          stages(vk::PipelineStageFlagBits2::eTransfer));
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eColorAttachmentOptimal) ==
          stages(vk::PipelineStageFlagBits2::eColorAttachmentOutput));
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eDepthStencilAttachmentOptimal) ==
          (stages(vk::PipelineStageFlagBits2::eEarlyFragmentTests) |
           stages(vk::PipelineStageFlagBits2::eLateFragmentTests)));
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eShaderReadOnlyOptimal) ==
          (stages(vk::PipelineStageFlagBits2::eFragmentShader) |
           stages(vk::PipelineStageFlagBits2::eComputeShader)));
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::ePresentSrcKHR) ==
          stages(vk::PipelineStageFlagBits2::eBottomOfPipe));
}

TEST_CASE("stage_for_layout falls back to all commands") {
    CHECK(vulcao::stage_for_layout(vk::ImageLayout::eGeneral) ==
          stages(vk::PipelineStageFlagBits2::eAllCommands));
}

TEST_CASE("access_for_layout maps common layouts") {
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eUndefined) == vk::AccessFlags2{});
    CHECK(vulcao::access_for_layout(vk::ImageLayout::ePresentSrcKHR) == vk::AccessFlags2{});
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eTransferSrcOptimal) ==
          access(vk::AccessFlagBits2::eTransferRead));
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eTransferDstOptimal) ==
          access(vk::AccessFlagBits2::eTransferWrite));
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eColorAttachmentOptimal) ==
          access(vk::AccessFlagBits2::eColorAttachmentWrite));
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eDepthStencilAttachmentOptimal) ==
          access(vk::AccessFlagBits2::eDepthStencilAttachmentWrite));
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eShaderReadOnlyOptimal) ==
          access(vk::AccessFlagBits2::eShaderRead));
}

TEST_CASE("access_for_layout falls back to memory reads") {
    CHECK(vulcao::access_for_layout(vk::ImageLayout::eGeneral) ==
          access(vk::AccessFlagBits2::eMemoryRead));
}
