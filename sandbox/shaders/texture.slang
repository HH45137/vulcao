struct VSOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

[shader("vertex")]
VSOut vertMain([[vk::location(0)]] float2 position : POSITION,
               [[vk::location(1)]] float2 uv : TEXCOORD)
{
    VSOut output;
    output.position = float4(position, 0.0, 1.0);
    output.uv = uv;
    return output;
}

[[vk::binding(0, 0)]] Sampler2D tex;

[shader("fragment")]
float4 fragMain([[vk::location(0)]] float2 uv) : SV_Target
{
    return tex.Sample(uv);
}
