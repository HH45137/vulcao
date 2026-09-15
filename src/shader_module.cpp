#include "vulcao/shader_module.h"

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vulcao {
namespace {

std::vector<uint32_t> read_spirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("ShaderModule: cannot open " + path.string());

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0)
        throw std::runtime_error("ShaderModule: invalid SPIR-V size in " + path.string());

    std::vector<uint32_t> spirv(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(spirv.data()), size))
        throw std::runtime_error("ShaderModule: cannot read " + path.string());

    return spirv;
}

}

ShaderModule::~ShaderModule() {
    destroy();
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      module_(std::exchange(other.module_, vk::ShaderModule{})),
      stage_(std::exchange(other.stage_, vk::ShaderStageFlagBits::eVertex)),
      reflection_(std::move(other.reflection_)) {}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        module_ = std::exchange(other.module_, vk::ShaderModule{});
        stage_ = std::exchange(other.stage_, vk::ShaderStageFlagBits::eVertex);
        reflection_ = std::move(other.reflection_);
    }
    return *this;
}

ShaderModule ShaderModule::create(vk::Device device,
                                  vk::ShaderStageFlagBits stage,
                                  std::span<const uint32_t> spirv) {
    if (spirv.empty())
        throw std::runtime_error("ShaderModule::create: empty SPIR-V");

    ShaderModule shader;
    shader.device_ = device;
    shader.stage_ = stage;
    shader.module_ = device.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = spirv.size() * sizeof(uint32_t),
        .pCode = spirv.data(),
    });
    shader.reflection_ = reflect_spirv(spirv);
    return shader;
}

ShaderModule ShaderModule::create_from_file(vk::Device device,
                                            vk::ShaderStageFlagBits stage,
                                            const std::filesystem::path& path) {
    const std::vector<uint32_t> spirv = read_spirv(path);
    return create(device, stage, spirv);
}

void ShaderModule::destroy() {
    if (module_)
        device_.destroyShaderModule(module_);

    device_ = nullptr;
    module_ = nullptr;
    stage_ = vk::ShaderStageFlagBits::eVertex;
    reflection_ = {};
}

}
