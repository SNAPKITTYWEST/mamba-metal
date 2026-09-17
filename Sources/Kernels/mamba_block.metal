// mamba_block.metal
// Fused path that keeps intermediate values in registers / threadgroup
// as much as practical. Full projection is still a separate matmul;
// this kernel fuses: after in-proj → SiLU → (optional conv already done)
// → Δ/B/C split → selective scan → multiply by gate → write.
//
// For maximum fusion the host can also launch a single large kernel
// that does the whole block when d_inner is modest.

#include <metal_stdlib>
using namespace metal;

// Re-use the Affine + compose from selective_scan (duplicated for independence)
struct Affine {
    float a;
    float b;
};

inline Affine compose(Affine t2, Affine t1) {
    return {t2.a * t1.a, t2.a * t1.b + t2.b};
}

constant Affine AFFINE_ID = {1.0f, 0.0f};

inline float softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

struct FusedScanParams {
    uint B, L, D, N, tile_L;
};

// Fused selective scan + residual D + optional gate multiply
// xz : [B, L, 2*D]   (x and z gate concatenated after in-proj + silu on z)
// The first D channels are the SSM input, second D are the gate.
kernel void fused_selective_scan_gate_f32(
    device const float *xz        [[buffer(0)]],  // [B,L,2D]
    device const float *delta     [[buffer(1)]],  // [B,L,D]
    device const float *Bmat      [[buffer(2)]],  // [B,L,N]
    device const float *Cmat      [[buffer(3)]],  // [B,L,N]
    device const float *A         [[buffer(4)]],  // [D,N]
    device const float *Dskip     [[buffer(5)]],  // [D]
    device       float *y         [[buffer(6)]],  // [B,L,D]
    device       float *state     [[buffer(7)]],  // [B,D,N] optional
    constant FusedScanParams &p   [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]])
{
    const uint b = gid.y, d = gid.z, tile = gid.x;
    if (b >= p.B || d >= p.D) return;
    const uint t0 = tile * p.tile_L;
    if (t0 >= p.L) return;
    const uint local_t = lid.x;
    const uint t = t0 + local_t;
    const bool valid = (t < p.L);

    thread float A_n[16];
    for (uint n = 0; n < p.N; ++n) A_n[n] = A[d * p.N + n];

    thread Affine T[16];
    float xt = 0.0f, zt = 0.0f;

    if (valid) {
        const uint base = ((b * p.L) + t) * (2 * p.D);
        xt = xz[base + d];
        zt = xz[base + p.D + d];               // gate
        float dt = softplus(delta[((b * p.L) + t) * p.D + d]);

        for (uint n = 0; n < p.N; ++n) {
            float a = A_n[n];
            float da = dt * a;
            float e = exp(da);
            T[n].a = e;
            float Bn = Bmat[((b * p.L) + t) * p.N + n];
            T[n].b = (fabs(a) > 1e-6f) ? ((e - 1.0f) / a) * Bn * xt : dt * Bn * xt;
        }
    } else {
        for (uint n = 0; n < p.N; ++n) T[n] = AFFINE_ID;
    }

    // SIMD + threadgroup scan (identical to selective_scan.metal)
    threadgroup Affine tg_carry[32];
    for (uint n = 0; n < p.N; ++n) {
        Affine val = T[n];
        for (uint off = 1; off < simd_size; off <<= 1) {
            Affine other = simd_shuffle_up(val, off);
            if (simd_lane >= off) val = compose(val, other);
        }
        T[n] = val;
        if (simd_lane == simd_size - 1) tg_carry[simd_gid] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lane < (lsize.x / simd_size)) {
        Affine acc = AFFINE_ID;
        uint nsg = lsize.x / simd_size;
        for (uint s = 0; s < nsg; ++s) {
            Affine tmp = tg_carry[s];
            tg_carry[s] = compose(tmp, acc);
            acc = tg_carry[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint n = 0; n < p.N; ++n)
        if (simd_gid > 0) T[n] = compose(T[n], tg_carry[simd_gid - 1]);

    if (valid) {
        float yt = 0.0f;
        for (uint n = 0; n < p.N; ++n) {
            float Cn = Cmat[((b * p.L) + t) * p.N + n];
            yt += Cn * T[n].b;
        }
        yt += Dskip[d] * xt;
        // gate
        float gate = zt / (1.0f + exp(-zt));   // SiLU already applied on host or here
        yt *= gate;
        y[((b * p.L) + t) * p.D + d] = yt;
    }

    if (state && valid && t == p.L - 1)
        for (uint n = 0; n < p.N; ++n)
            state[(b * p.D + d) * p.N + n] = T[n].b;
}
