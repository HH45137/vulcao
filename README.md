# vulcao

A thin, RAII wrapper around Vulkan. It maps close to the Vulkan API, owns objects
with move-only copy semantics, throws on failure and derives descriptor and vertex
layouts from SPIR-V reflection.

## Features

- `Context`: instance, physical device, logical device, swapchain, queues and
  command pool, with configurable device features/extensions and dedicated
  compute/transfer queues. Created headless (`ContextInfo::headless`) it skips
  the surface and swapchain for compute-only or offscreen work.
- Resources: VMA-backed `Allocator`, `Buffer`, `Image` (2D/depth factories,
  mipmap generation), `Sampler` and `CommandPool`.
- Synchronization: `Fence`, binary and timeline `Semaphore`, `QueryPool` and a
  `FrameManager` that owns frames in flight, swapchain acquire and present.
- Recording: `CommandBuffer` with layout transitions, copies, mipmaps, draws,
  dispatches, queries, dynamic state, debug labels and queue family ownership
  transfers (release/acquire).
- Pipelines: `ShaderModule` with SPIR-V reflection, `DescriptorSetLayout/Pool/Set`
  with a batching writer and layout cache, `PipelineLayout`, `Pipeline` with
  graphics/compute factories, specialization constants and `PipelineCache`.
- Reflection: descriptor sets, push constants and vertex attributes extracted from
  SPIR-V with SPIRV-Reflect, plus `make_vertex_layout`.

## Requirements

- CMake 3.24+
- A C++20 compiler
- The Vulkan SDK (also provides `slangc`, used by the sandbox)

All other dependencies are git submodules.

## Building

```sh
git clone --recursive https://github.com/<you>/vulcao
cmake -S . -B build
cmake --build build
```

Options:

- `VULCAO_BUILD_SANDBOX` (default `ON`): build the sandbox example.
- `VULCAO_BUILD_TESTS` (default `ON` when top level): build and register tests.
- `VULCAO_INSTALL` (default `ON` when top level): generate install rules.

Run the tests with `ctest --test-dir build`.

## Using from another project

```sh
cmake --install build --prefix <prefix>
```

```cmake
find_package(vulcao REQUIRED)
target_link_libraries(app PRIVATE vulcao::vulcao)
```

The installed package is self-contained: it bundles vk-bootstrap, SPIRV-Reflect and
the VMA headers.

## Example

```cpp
#include <vulcao/context.h>
#include <vulcao/shader_module.h>

vulcao::Context ctx{{.app_name = "my-app"}};
ctx.initialize(surface, extent);

auto shader = vulcao::ShaderModule::create_from_file(
    ctx.device(), vk::ShaderStageFlagBits::eCompute, "reduce.comp.spv");

// reflection-driven layouts
auto layout = vulcao::PipelineLayout::create_from_reflection(
    ctx.device(), std::span(&shader.reflection(), 1));
auto pipeline = vulcao::Pipeline::create_compute(ctx.device(), layout, shader, "compMain");
```

## License

TBD.
