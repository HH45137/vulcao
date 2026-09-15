#include <cstddef>

#include <doctest/doctest.h>

#include <vulcao/vertex_layout.h>

namespace {

struct Vertex {
    float position[3];
    float uv[2];
};

}

TEST_CASE("make_vertex_layout applies offsets in location order") {
    vulcao::ShaderReflection reflection;
    reflection.stage = vk::ShaderStageFlagBits::eVertex;
    reflection.vertex_attributes = {
        vk::VertexInputAttributeDescription{
            .location = 1, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = 0},
        vk::VertexInputAttributeDescription{
            .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = 0},
    };

    const vulcao::VertexLayout layout = vulcao::make_vertex_layout<Vertex>(
        reflection, {offsetof(Vertex, position), offsetof(Vertex, uv)});

    REQUIRE(layout.bindings.size() == 1);
    CHECK(layout.bindings.front().stride == sizeof(Vertex));
    CHECK(layout.bindings.front().inputRate == vk::VertexInputRate::eVertex);

    REQUIRE(layout.attributes.size() == 2);
    CHECK(layout.attributes[0].location == 0);
    CHECK(layout.attributes[0].offset == offsetof(Vertex, position));
    CHECK(layout.attributes[0].format == vk::Format::eR32G32B32Sfloat);
    CHECK(layout.attributes[1].location == 1);
    CHECK(layout.attributes[1].offset == offsetof(Vertex, uv));
}

TEST_CASE("make_vertex_layout throws when counts differ") {
    vulcao::ShaderReflection reflection;
    reflection.vertex_attributes = {
        vk::VertexInputAttributeDescription{.location = 0},
    };

    CHECK_THROWS(vulcao::make_vertex_layout<Vertex>(reflection, {0, 4}));
}
