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
- Transfer: staging-backed `upload`/`download` for buffers and images, with the
  full `vk::BufferImageCopy` region exposed for row-pitched data.
- Synchronization: `Fence`, binary and timeline `Semaphore`, `QueryPool` and a
  `FrameManager` that owns frames in flight, swapchain acquire and present.
- Recording: `CommandBuffer` with layout transitions, copies, mipmaps, draws,
  dispatches, queries, dynamic state, debug labels and queue family ownership
  transfers (release/acquire).
- Pipelines: `ShaderModule` with SPIR-V reflection, `DescriptorSetLayout/Pool/Set`
  with a batching writer and layout cache, `PipelineLayout`, `Pipeline` with
  graphics/compute factories, specialization constants and `PipelineCache`.
- Descriptor indexing: layouts accept creation flags and per-binding flags
  (`PARTIALLY_BOUND`, `UPDATE_AFTER_BIND`, `VARIABLE_DESCRIPTOR_COUNT`), pools
  allocate variable descriptor counts, and reflected runtime arrays report
  descriptor count 0 which `set_binding_count` turns into a concrete bound. The
  building blocks for bindless, without prescribing a bindless design.
- Reflection: descriptor sets, push constants and vertex attributes extracted from
  SPIR-V with SPIRV-Reflect, plus `make_vertex_layout`.
- Logging: a global `set_log_callback` receives structured `LogMessage`s from the
  library and from the validation layers; `set_log_level` filters them and the
  default prints warnings, errors and validation output to stderr.

## Requirements

- CMake 3.24+
- A C++20 compiler
- The Vulkan SDK (also provides `slangc`, used by the samples)

All other dependencies are git submodules.

## Building

```sh
git clone --recursive https://github.com/<you>/vulcao
cmake -S . -B build
cmake --build build
```

Options:

- `VULCAO_BUILD_SAMPLES` (default `ON`): build the samples.
- `VULCAO_BUILD_TESTS` (default `ON` when top level): build and register tests.
- `VULCAO_INSTALL` (default `ON` when top level): generate install rules.

Run the tests with `ctest --test-dir build`.

## Samples

- `01_hello_triangle`: a CPU-side vertex and index buffer drawn through a full
  graphics pipeline.
- `02_uniforms`: the triangle spun by a uniform buffer and descriptor set.
- `03_compute`: a storage buffer transformed by a compute shader, with the result
  verified on the CPU. Needs no display, and exits non-zero on a wrong result or
  on a validation error.
- `04_offscreen`: a triangle rendered into an image, then read back and checked
  pixel by pixel. Also needs no display.

```sh
./build/samples/01_hello_triangle/hello_triangle
./build/samples/02_uniforms/uniforms
./build/samples/03_compute/compute
./build/samples/04_offscreen/offscreen
```

## Documentation

The API reference is generated with Doxygen from `docs/`:

```sh
cmake -S . -B build -DVULCAO_BUILD_DOCS=ON
cmake --build build --target vulcao_docs
```

The HTML output is written to `docs/generated/html` and is published to GitHub
Pages and Codeberg Pages by CI.

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

MIT, see [LICENSE](LICENSE).
