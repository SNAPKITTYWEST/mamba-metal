// rmsnorm.metal
#include <metal_stdlib>
using namespace metal;

struct RMSNormParams {
    uint N;          // number of elements per row
    uint rows;       // batch * seq or just batch
    float eps;
};

kernel void rmsnorm_f32(
    device const float *x     [[buffer(0)]],
    device const float *weight[[buffer(1)]],  // [N]
    device       float *y     [[buffer(2)]],
    constant RMSNormParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]])
{
    uint row = gid.y;
    if (row >= p.rows) return;

    // Threadgroup reduction for sum of squares
    threadgroup float shared[256];
    float local_sum = 0.0f;
    for (uint i = lid; i < p.N; i += lsize) {
        float v = x[row * p.N + i];
        local_sum += v * v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared[0] / float(p.N) + p.eps);

    for (uint i = lid; i < p.N; i += lsize) {
        float v = x[row * p.N + i] * inv_rms;
        if (weight) v *= weight[i];
        y[row * p.N + i] = v;
    }
}

kernel void rmsnorm_f16(
    device const half *x      [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device       half *y      [[buffer(2)]],
    constant RMSNormParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]])
{
    uint row = gid.y;
    if (row >= p.rows) return;

    threadgroup float shared[256];
    float local_sum = 0.0f;
    for (uint i = lid; i < p.N; i += lsize) {
        float v = float(x[row * p.N + i]);
        local_sum += v * v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared[0] / float(p.N) + p.eps);

    for (uint i = lid; i < p.N; i += lsize) {
        float v = float(x[row * p.N + i]) * inv_rms;
        if (weight) v *= float(weight[i]);
        y[row * p.N + i] = half(v);
    }
}
