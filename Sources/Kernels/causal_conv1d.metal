// causal_conv1d.metal
// Causal 1-D convolution used by Mamba (depthwise, typically width 4).
// Supports full-sequence and incremental (streaming) modes.

#include <metal_stdlib>
using namespace metal;

struct ConvParams {
    uint B;
    uint L;
    uint D;          // channels (depthwise)
    uint K;          // kernel width (usually 4)
    uint has_bias;
};

// Full-sequence causal depthwise conv
// weight: [D, K]
// bias  : [D] (optional)
// x     : [B, L, D]
// y     : [B, L, D]
kernel void causal_conv1d_f32(
    device const float *x      [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias   [[buffer(2)]],
    device       float *y      [[buffer(3)]],
    constant ConvParams &p     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;
    uint t = gid.y;
    uint d = gid.x;
    if (b >= p.B || t >= p.L || d >= p.D) return;

    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        int src = int(t) - int(p.K) + 1 + int(k);
        if (src >= 0) {
            float xv = x[(b * p.L + uint(src)) * p.D + d];
            float wv = weight[d * p.K + k];
            acc += xv * wv;
        }
    }
    if (p.has_bias) acc += bias[d];
    y[(b * p.L + t) * p.D + d] = acc;
}

// Incremental / streaming update
// Maintains a circular buffer of the last (K-1) inputs per channel.
// conv_state: [B, D, K-1]
kernel void causal_conv1d_step_f32(
    device const float *x_t        [[buffer(0)]],  // [B, D] current token
    device const float *weight     [[buffer(1)]],  // [D, K]
    device const float *bias       [[buffer(2)]],
    device       float *y_t        [[buffer(3)]],  // [B, D]
    device       float *conv_state [[buffer(4)]],  // [B, D, K-1]
    constant ConvParams &p         [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.y;
    uint d = gid.x;
    if (b >= p.B || d >= p.D) return;

    float acc = 0.0f;
    // oldest .. newest in state, then current x
    for (uint k = 0; k < p.K - 1; ++k) {
        float xv = conv_state[(b * p.D + d) * (p.K - 1) + k];
        float wv = weight[d * p.K + k];
        acc += xv * wv;
    }
    float xt = x_t[b * p.D + d];
    acc += xt * weight[d * p.K + (p.K - 1)];
    if (p.has_bias) acc += bias[d];
    y_t[b * p.D + d] = acc;

    // shift state left and append current token
    for (uint k = 0; k < p.K - 2; ++k) {
        conv_state[(b * p.D + d) * (p.K - 1) + k] =
            conv_state[(b * p.D + d) * (p.K - 1) + k + 1];
    }
    if (p.K > 1)
        conv_state[(b * p.D + d) * (p.K - 1) + (p.K - 2)] = xt;
}

// FP16 versions follow the same pattern (omitted for brevity – identical structure)
kernel void causal_conv1d_f16(
    device const half *x, device const half *weight, device const half *bias,
    device half *y, constant ConvParams &p,
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z, t = gid.y, d = gid.x;
    if (b >= p.B || t >= p.L || d >= p.D) return;
    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        int src = int(t) - int(p.K) + 1 + int(k);
        if (src >= 0)
            acc += float(x[(b * p.L + uint(src)) * p.D + d]) * float(weight[d * p.K + k]);
    }
    if (p.has_bias) acc += float(bias[d]);
    y[(b * p.L + t) * p.D + d] = half(acc);
}
