add_executable(mlp main.cpp)

find_package(nlohmann_json REQUIRED)

target_link_libraries(mlp PRIVATE vulcao nlohmann_json::nlohmann_json)

vulcao_add_slang_shaders(mlp
    "mlp_inference.slang|mlp_inference|compute|mlp_inference.comp.spv"
)
