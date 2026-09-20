add_executable(mlp main.cpp)

# No samples_common: a compute pass needs no window, so this sample stays headless.
target_link_libraries(mlp PRIVATE vulcao)

vulcao_add_slang_shaders(mlp
    "mlp_inference.slang|compMain|compute|mlp_inference.comp.spv"
)
