#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <vector>
#include <fstream>

#include "vulcao/buffer.h"
#include "vulcao/command_buffer.h"
#include "vulcao/context.h"
#include "vulcao/descriptor_set.h"
#include "vulcao/log.h"
#include "vulcao/pipeline.h"
#include "vulcao/pipeline_layout.h"
#include "vulcao/shader_module.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#ifndef VULCAO_SHADER_DIR
#define VULCAO_SHADER_DIR "shaders"
#endif

namespace MLP_INFERENCE {

namespace DEVICE {

struct PushParams {
    glm::uint32 mat_dim;
};

}

namespace HOST {
constexpr uint32_t MAT_DIM = 2;
constexpr uint32_t MAT_ITEM_COUNT = MAT_DIM * MAT_DIM;
constexpr uint32_t WORKGROUP_DIM = 8;
constexpr uint32_t WORKGROUP_SIZE = WORKGROUP_DIM * WORKGROUP_DIM * 1;
constexpr uint32_t DISPATCHED_NUM = (MAT_DIM + (WORKGROUP_DIM - 1)) / WORKGROUP_DIM;


enum act_func_e : glm::int32 {
    relu = 1,
    sine,
    sigmoid,
};

struct linear_layer_s {
    glm::int32 in_dim{};
    glm::int32 out_dim{};
    act_func_e act_func{relu};
    std::vector<float> weight_data{};
    std::vector<float> bias_data{};
};

struct coordinate_mlp_s {
    glm::ivec2 img_size{};

    std::vector<float> frequencies_data{};

    linear_layer_s in_layer;
    std::vector<linear_layer_s> hidden_layers{};
    linear_layer_s out_layer;
};


static act_func_e get_act_func_type(std::string str) {
    if (str == "relu") {
        return relu;
    } else if (str == "sigmoid") {
        return sigmoid;
    }
}

static bool load_model_from_json(coordinate_mlp_s& ref_model, const std::string& json_path) {
    using json = nlohmann::json;

    std::ifstream json_file_handle(json_path.c_str());
    json json_data = json::parse(json_file_handle);

    ref_model.img_size.x = json_data["width"];
    ref_model.img_size.y = json_data["height"];

    std::string act_func_str = json_data["activation"];
    ref_model.in_layer.act_func = get_act_func_type(act_func_str);

    // TODO

    return true;
}

}

}

int main(int argc, char* argv[]) {
    using namespace MLP_INFERENCE;
    using namespace MLP_INFERENCE::HOST;

    try {
        /* ----------------- Prepare ----------------- */

        coordinate_mlp_s mlp_model{};
        {
            std::string model_weight_path = "neural_texture.json";
            if (argc >= 2) {
                model_weight_path = std::string(argv[1]);
            }
            bool result = load_model_from_json(mlp_model, model_weight_path);
        }

        // A compute pass needs no surface, so the context is created headless and
        // this sample runs on a machine without a display.
        vulcao::Context context{{.app_name = "07_mlp", .headless = true}};
        context.initialize();

        /* ----------------- Pipeline ----------------- */

        const vulcao::ShaderModule shader = vulcao::ShaderModule::create_from_file(
            context.device(), vk::ShaderStageFlagBits::eCompute,
            std::filesystem::path(VULCAO_SHADER_DIR) / "mlp_inference.comp.spv");

        const vulcao::DescriptorSetLayout set_layout =
            vulcao::DescriptorSetLayout::create(context.device(), shader.reflection(), 0);
        const vulcao::PipelineLayout pipeline_layout =
            vulcao::PipelineLayout::create_from_reflection(
                context.device(), std::span(&shader.reflection(), 1));
        const vulcao::Pipeline pipeline =
            vulcao::Pipeline::create_compute(context.device(),
                                             pipeline_layout,
                                             shader,
                                             "mlp_inference");

        // The pool is sized from the reflected bindings: set 0 carries three
        // storage buffers (a, b, c), so a hand-counted single-storage-buffer
        // pool would run out of pool memory on allocation.
        vulcao::DescriptorPool pool = vulcao::DescriptorPool::create_for_bindings(
            context.device(), shader.reflection().bindings_for_set(0), 1);

        /* ----------------- Matrix ----------------- */

        std::vector<float> mat_value_a(MAT_ITEM_COUNT);
        std::vector<float> mat_value_b(MAT_ITEM_COUNT);
        std::vector<float> mat_value_c(MAT_ITEM_COUNT);
        for (uint32_t i = 0; i < MAT_ITEM_COUNT; ++i) {
            mat_value_a[i] = 0.0f;
            mat_value_b[i] = 0.0f;
            mat_value_c[i] = 0.0f;
        }

        constexpr vk::DeviceSize mat_bytes = MAT_ITEM_COUNT * sizeof(float);

        vulcao::Buffer mat_buffer_a = vulcao::Buffer::create_with_data(
            context, mat_value_a,
            vk::BufferUsageFlagBits::eStorageBuffer
            );
        vulcao::Buffer mat_buffer_b = vulcao::Buffer::create_with_data(
            context, mat_value_b,
            vk::BufferUsageFlagBits::eStorageBuffer
            );
        vulcao::Buffer mat_buffer_c = vulcao::Buffer::create(
            context.allocator(), mat_bytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO, 0
            );

        const vulcao::DescriptorSet set = pool.allocate(set_layout);
        {
            set.write_storage_buffer(0, mat_buffer_a);
            set.write_storage_buffer(1, mat_buffer_b);
            set.write_storage_buffer(2, mat_buffer_c);
        }

        /* ----------------- Run and readback ----------------- */

        vulcao::Buffer readback = vulcao::Buffer::create(
            context.allocator(), mat_bytes, vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        context.immediate([&](vulcao::CommandBuffer& cmd) {
            // Sync for setup matrix
            {
                cmd.buffer_barrier(mat_buffer_a.handle(),
                                   vk::PipelineStageFlagBits2::eTransfer,
                                   vk::AccessFlagBits2::eTransferWrite,
                                   vk::PipelineStageFlagBits2::eComputeShader,
                                   vk::AccessFlagBits2::eShaderStorageRead
                    );
                cmd.buffer_barrier(mat_buffer_b.handle(),
                                   vk::PipelineStageFlagBits2::eTransfer,
                                   vk::AccessFlagBits2::eTransferWrite,
                                   vk::PipelineStageFlagBits2::eComputeShader,
                                   vk::AccessFlagBits2::eShaderStorageRead
                    );
                cmd.buffer_barrier(mat_buffer_c.handle(),
                                   vk::PipelineStageFlagBits2::eTransfer,
                                   vk::AccessFlagBits2::eTransferWrite,
                                   vk::PipelineStageFlagBits2::eComputeShader,
                                   vk::AccessFlagBits2::eShaderStorageRead
                    );
            }

            // Push constants
            {
                cmd.push_constants(pipeline_layout.handle(),
                                   vk::ShaderStageFlagBits::eCompute,
                                   0,
                                   DEVICE::PushParams{MAT_DIM});
            }

            cmd.bind_pipeline(pipeline);
            cmd.bind_descriptor_sets(vk::PipelineBindPoint::eCompute, pipeline_layout.handle(),
                                     set.handle());
            cmd.dispatch(DISPATCHED_NUM, DISPATCHED_NUM);

            // Sync for readback matrix
            {
                cmd.buffer_barrier(mat_buffer_c.handle(),
                                   vk::PipelineStageFlagBits2::eComputeShader,
                                   vk::AccessFlagBits2::eShaderStorageWrite,
                                   vk::PipelineStageFlagBits2::eTransfer,
                                   vk::AccessFlagBits2::eTransferRead);

                cmd.copy_buffer(mat_buffer_c.handle(), readback.handle(), mat_bytes);
            }
        });

        readback.invalidate();
        const auto* mat_result_c = static_cast<const float*>(readback.map());

        /* ----------------- Verifier ----------------- */

        std::cout << "[";
        for (int i = 0; i < MAT_ITEM_COUNT; i++) {
            if (i % MAT_DIM == 0 && i != 0) {
                std::cout << std::endl;
            }
            std::cout << mat_result_c[i];
            if (i != MAT_ITEM_COUNT - 1) {
                std::cout << "\t";
            }
        }
        std::cout << "]" << std::endl;

        std::cout <<
            "dispatched " << DISPATCHED_NUM <<
            " workgroup of " << WORKGROUP_SIZE <<
            " threads over " << MAT_ITEM_COUNT <<
            " elements." << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}