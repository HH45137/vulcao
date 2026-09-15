#include <array>

#include <doctest/doctest.h>

#include <vulcao/reflection.h>

namespace {

vulcao::ShaderReflection make_stage(vk::ShaderStageFlagBits stage,
                                    uint32_t binding,
                                    uint32_t offset,
                                    uint32_t size) {
    vulcao::ShaderReflection reflection;
    reflection.stage = stage;
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 0,
        .bindings = {vk::DescriptorSetLayoutBinding{
            .binding = binding,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = stage,
        }},
    });
    reflection.push_constants.push_back(vk::PushConstantRange{
        .stageFlags = stage,
        .offset = offset,
        .size = size,
    });
    return reflection;
}

}

TEST_CASE("bindings_for_set returns bindings of a known set") {
    const vulcao::ShaderReflection reflection =
        make_stage(vk::ShaderStageFlagBits::eVertex, 3, 0, 16);

    CHECK(reflection.bindings_for_set(0).size() == 1);
    CHECK(reflection.bindings_for_set(0).front().binding == 3);
    CHECK(reflection.bindings_for_set(1).empty());
}

TEST_CASE("merge_reflections merges binding and push constant stage flags") {
    const std::array<vulcao::ShaderReflection, 2> stages{
        make_stage(vk::ShaderStageFlagBits::eVertex, 0, 0, 64),
        make_stage(vk::ShaderStageFlagBits::eFragment, 0, 0, 64),
    };

    const vulcao::PipelineReflection merged = vulcao::merge_reflections(stages);

    REQUIRE(merged.sets.size() == 1);
    REQUIRE(merged.sets.front().bindings.size() == 1);
    CHECK(merged.sets.front().bindings.front().binding == 0);
    CHECK(merged.sets.front().bindings.front().stageFlags ==
          (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));

    REQUIRE(merged.push_constants.size() == 1);
    CHECK(merged.push_constants.front().stageFlags ==
          (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
}

TEST_CASE("merge_reflections keeps distinct bindings and ranges") {
    vulcao::ShaderReflection vertex = make_stage(vk::ShaderStageFlagBits::eVertex, 0, 0, 64);
    vulcao::ShaderReflection fragment = make_stage(vk::ShaderStageFlagBits::eFragment, 1, 64, 16);

    const std::array<vulcao::ShaderReflection, 2> stages{vertex, fragment};
    const vulcao::PipelineReflection merged = vulcao::merge_reflections(stages);

    REQUIRE(merged.sets.size() == 1);
    REQUIRE(merged.sets.front().bindings.size() == 2);
    CHECK(merged.sets.front().bindings[0].binding == 0);
    CHECK(merged.sets.front().bindings[1].binding == 1);
    CHECK(merged.push_constants.size() == 2);
}

TEST_CASE("merge_reflections sorts sets and bindings") {
    vulcao::ShaderReflection reflection;
    reflection.stage = vk::ShaderStageFlagBits::eFragment;
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 1,
        .bindings = {vk::DescriptorSetLayoutBinding{.binding = 2}},
    });
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 0,
        .bindings = {vk::DescriptorSetLayoutBinding{.binding = 5},
                     vk::DescriptorSetLayoutBinding{.binding = 1}},
    });

    const std::array<vulcao::ShaderReflection, 1> stages{reflection};
    const vulcao::PipelineReflection merged = vulcao::merge_reflections(stages);

    REQUIRE(merged.sets.size() == 2);
    CHECK(merged.sets[0].set == 0);
    CHECK(merged.sets[1].set == 1);
    REQUIRE(merged.sets[0].bindings.size() == 2);
    CHECK(merged.sets[0].bindings[0].binding == 1);
    CHECK(merged.sets[0].bindings[1].binding == 5);
}
