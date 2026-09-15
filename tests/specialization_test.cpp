#include <cstddef>
#include <cstring>

#include <doctest/doctest.h>

#include <vulcao/specialization.h>

TEST_CASE("specialization info maps constants") {
    vulcao::SpecializationInfo info;
    info.map_constant(0, 5);
    info.map_constant(1, 2.5f);

    const vk::SpecializationInfo spec = info.get();
    REQUIRE(spec.mapEntryCount == 2);
    CHECK(spec.pMapEntries[0].constantID == 0);
    CHECK(spec.pMapEntries[0].offset == 0);
    CHECK(spec.pMapEntries[0].size == 4);
    CHECK(spec.pMapEntries[1].constantID == 1);
    CHECK(spec.dataSize == 8);

    int first = 0;
    std::memcpy(&first, spec.pData, sizeof(first));
    CHECK(first == 5);
}

TEST_CASE("specialization info aligns 64 bit constants") {
    vulcao::SpecializationInfo info;
    info.map_constant(0, uint32_t{1});
    info.map_constant(1, uint64_t{2});

    const vk::SpecializationInfo spec = info.get();
    REQUIRE(spec.mapEntryCount == 2);
    CHECK(spec.pMapEntries[1].offset == 8);
    CHECK(spec.pMapEntries[1].size == 8);
    CHECK(spec.dataSize == 16);
}

TEST_CASE("specialization info is empty by default") {
    const vulcao::SpecializationInfo info;
    CHECK(info.empty());
    CHECK(info.get().mapEntryCount == 0);
}
