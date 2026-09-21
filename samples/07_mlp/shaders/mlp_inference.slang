// MLP neural network inference by the 07_mlp sample.

struct PushParams
{
    uint mat_dim;
}

[[vk::push_constant]] PushParams params;

[[vk::binding(0, 0)]] StructuredBuffer<float> a;
[[vk::binding(1, 0)]] StructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c;

[shader("compute")]
[numthreads(8, 8, 1)]
void compMain(uint3 tid : SV_DispatchThreadID)
{
    const uint DIM = params.mat_dim;

    if (tid.y >= DIM || tid.x >= DIM)
    {
        return;
    }

    float sum = 0.0;
    for (uint idx_dim = 0; idx_dim < DIM; ++idx_dim)
    {
        const float a_item = a[tid.y * DIM + idx_dim];
        const float b_item = b[idx_dim * DIM + tid.x];
        sum += a_item * b_item;
    }
    c[tid.y * DIM + tid.x] = sum;
}
