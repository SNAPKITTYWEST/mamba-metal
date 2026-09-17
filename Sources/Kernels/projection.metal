// projection.metal – simple linear layers used by Mamba
// y = x @ W + b   (row-major)
#include <metal_stdlib>
using namespace metal;

struct ProjParams {
    uint M;   // rows of x (B*L)
    uint K;   // cols of x / rows of W
    uint N;   // cols of W / output dim
    uint has_bias;
};

kernel void linear_f32(
    device const float *x     [[buffer(0)]],  // [M, K]
    device const float *W     [[buffer(1)]],  // [K, N]
    device const float *bias  [[buffer(2)]],  // [N] or null
    device       float *y     [[buffer(3)]],  // [M, N]
    constant ProjParams &p    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.y;
    uint n = gid.x;
    if (m >= p.M || n >= p.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k)
        acc += x[m * p.K + k] * W[k * p.N + n];
    if (p.has_bias) acc += bias[n];
    y[m * p.N + n] = acc;
}

kernel void linear_f16(
    device const half *x     [[buffer(0)]],
    device const half *W     [[buffer(1)]],
    device const half *bias  [[buffer(2)]],
    device       half *y     [[buffer(3)]],
    constant ProjParams &p   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.y;
    uint n = gid.x;
    if (m >= p.M || n >= p.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k)
        acc += float(x[m * p.K + k]) * float(W[k * p.N + n]);
    if (p.has_bias) acc += float(bias[n]);
    y[m * p.N + n] = half(acc);
}
