// Shader used by the reflection, layout and pipeline tests.
//
// Every entry point pulls in a different slice of the module so the tests can
// check that reflection reports exactly what a stage uses:
//   vertMain    -> set 0 binding 0 (uniform buffer), two vertex inputs
//   fragMain    -> push constants only
//   computeMain -> set 0 binding 0 (uniform buffer), set 0 binding 1 (storage
//                  buffer) and push constants

struct Uniforms
{
    float4x4 mvp;
};

struct PushData
{
    float4 tint;
    uint count;
};

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> uniforms;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> values;
[[vk::push_constant]] PushData push_data;

struct VSOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color;
};

[shader("vertex")]
VSOut vertMain([[vk::location(0)]] float3 position : POSITION,
               [[vk::location(1)]] float3 color : COLOR)
{
    VSOut output;
    output.position = mul(uniforms.mvp, float4(position, 1.0));
    output.color = color;
    return output;
}

// Reads SV_VertexID, which makes Slang declare the SPIR-V DrawParameters
// capability: it lowers SV_VertexID to gl_VertexIndex plus BaseVertex, and
// BaseVertex is what needs the capability. Building a module from this entry
// point therefore requires DeviceFeatures::shader_draw_parameters.
[shader("vertex")]
VSOut vertexIndexMain(uint id : SV_VertexID)
{
    VSOut output;
    output.position = float4(float(id), 0.0, 0.0, 1.0);
    output.color = float3(1.0, 0.0, 0.0);
    return output;
}

[shader("fragment")]
float4 fragMain([[vk::location(0)]] float3 color) : SV_Target
{
    return float4(color, 1.0) + push_data.tint;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    values[id.x] = values[id.x] + push_data.tint.x + uniforms.mvp[0][0];
}
