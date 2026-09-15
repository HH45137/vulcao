struct PushConstants
{
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;
};
[[vk::push_constant]] PushConstants transform;

struct VSOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

[shader("vertex")]
VSOut vertMain([[vk::location(0)]] float3 position : POSITION,
               [[vk::location(1)]] float2 uv : TEXCOORD)
{
    const float4 p = float4(position, 1.0);
    VSOut output;
    output.position = float4(dot(transform.row0, p), dot(transform.row1, p),
                             dot(transform.row2, p), dot(transform.row3, p));
    output.uv = uv;
    return output;
}

[[vk::binding(0, 0)]] Sampler2D tex;

[SpecializationConstant]
const float tint = 1.0;

[shader("fragment")]
float4 fragMain([[vk::location(0)]] float2 uv) : SV_Target
{
    return tex.Sample(uv) * tint;
}
