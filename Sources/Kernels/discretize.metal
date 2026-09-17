#include <metal_stdlib>
using namespace metal;

// Softplus used for Δ
inline float softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

// Produce Δ, B, C from the projected input (fused projection already done)
// This kernel is mainly a helper; the fused scan usually absorbs discretization.
kernel void discretize_delta_f32(
    device const float *delta_raw [[buffer(0)]],
    device       float *delta     [[buffer(1)]],
    constant uint &n             [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    delta[gid] = softplus(delta_raw[gid]);
}
