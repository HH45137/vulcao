[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> data;

[shader("compute")]
[numthreads(64, 1, 1)]
void compMain(uint3 tid : SV_DispatchThreadID)
{
    data[tid.x] = data[tid.x] * 2u + 1u;
}
