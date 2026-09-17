#include <metal_stdlib>
using namespace metal;

// SiLU(x) = x * sigmoid(x)
kernel void silu_f32(
    device const float *x [[buffer(0)]],
    device       float *y [[buffer(1)]],
    constant uint &n     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    y[gid] = v / (1.0f + exp(-v));
}

kernel void silu_f16(
    device const half *x [[buffer(0)]],
    device       half *y [[buffer(1)]],
    constant uint &n    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = float(x[gid]);
    y[gid] = half(v / (1.0f + exp(-v)));
}
