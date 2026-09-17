// gated_mlp.metal – SwiGLU-style gated MLP used around Mamba blocks
#include <metal_stdlib>
using namespace metal;

struct MLPParams {
    uint B;
    uint L;
    uint D;          // input dim
    uint H;          // hidden dim (usually 2*D or expand*D)
};

// y = silu(x @ Wgate) * (x @ Wup)   then later @ Wdown
// Here we assume the two projections are already done; this kernel does the gate.
kernel void gated_silu_mul_f32(
    device const float *gate [[buffer(0)]],  // [B*L, H]
    device const float *up   [[buffer(1)]],  // [B*L, H]
    device       float *out  [[buffer(2)]],
    constant uint &n         [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float u = up[gid];
    // SiLU(g) * u
    float silu = g / (1.0f + exp(-g));
    out[gid] = silu * u;
}

kernel void gated_silu_mul_f16(
    device const half *gate [[buffer(0)]],
    device const half *up   [[buffer(1)]],
    device       half *out  [[buffer(2)]],
    constant uint &n        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float silu = g / (1.0f + exp(-g));
    out[gid] = half(silu * u);
}
