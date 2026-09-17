// selective_scan.metal
// Native Metal implementation of the Mamba selective state-space scan
// using an associative affine transformation formulation.
//
// Each timestep t is represented as the affine map:
//   h' = a_t * h + b_t
// stored as the pair T_t = (a_t, b_t).
//
// Composition:
//   (a2, b2) ⊗ (a1, b1) = (a2*a1, a2*b1 + b2)
// is associative, enabling a parallel prefix-scan.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Data layout conventions (all device buffers are contiguous)
//
// For a batch of sequences:
//   x     : [B, L, D]          input
//   delta : [B, L, D]          Δ (softplus projected)
//   B     : [B, L, N]          input-dependent B
//   C     : [B, L, N]          input-dependent C
//   A     : [D, N]             log-space or already negative A (constant across time)
//   D     : [D]                skip connection
//   y     : [B, L, D]          output
//   state : [B, D, N]          final SSM state (optional write-back)
//
// N = d_state (typically 16), D = d_inner (expand * d_model)
// ---------------------------------------------------------------------------

struct ScanParams {
    uint B;          // batch
    uint L;          // sequence length
    uint D;          // inner dimension
    uint N;          // state dimension
    uint tile_L;     // sequence tile size processed by one threadgroup
};

// ---------------------------------------------------------------------------
// Associative operator on FP32 for numerical stability of the scan
// ---------------------------------------------------------------------------
struct Affine {
    float a;   // multiplicative factor (Ā)
    float b;   // additive factor (B̄ * x)
};

inline Affine compose(Affine t2, Affine t1) {
    // t2 ⊗ t1
    Affine out;
    out.a = t2.a * t1.a;
    out.b = t2.a * t1.b + t2.b;
    return out;
}

// ---------------------------------------------------------------------------
// Identity element of the monoid
// ---------------------------------------------------------------------------
constant Affine AFFINE_IDENTITY = {1.0f, 0.0f};

// ---------------------------------------------------------------------------
// Discretization helpers (zero-order hold)
// Ā = exp(Δ * A)
// B̄ = (exp(Δ * A) - 1) / A * B   ≈ Δ * B when using the simplified form
// We keep the full form for correctness; a fast approximation can be swapped in.
// ---------------------------------------------------------------------------
inline float softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// ---------------------------------------------------------------------------
// Hierarchical parallel selective scan
//
// Launch geometry (recommended):
//   grid  = (B, D, ceil(L / tile_L))
//   tg    = (tile_L, 1, 1)   or a power-of-two that fits in threadgroup memory
//
// Each thread owns one timestep inside its tile.
// SIMD groups perform a local inclusive scan.
// Threadgroup memory holds the per-SIMD carry for the next level.
// A final device-wide carry propagation stitches tiles together.
// ---------------------------------------------------------------------------

kernel void selective_scan_f32(
    device const float  *x          [[buffer(0)]],   // [B, L, D]
    device const float  *delta      [[buffer(1)]],   // [B, L, D]
    device const float  *Bmat       [[buffer(2)]],   // [B, L, N]
    device const float  *Cmat       [[buffer(3)]],   // [B, L, N]
    device const float  *A          [[buffer(4)]],   // [D, N]  (already negative / log)
    device const float  *Dskip      [[buffer(5)]],   // [D]
    device       float  *y          [[buffer(6)]],   // [B, L, D]
    device       float  *final_state[[buffer(7)]],   // [B, D, N]  (optional, may be null)
    constant     ScanParams &params [[buffer(8)]],
    uint3        gid                [[thread_position_in_grid]],
    uint3        lid                [[thread_position_in_threadgroup]],
    uint3        lsize              [[threads_per_threadgroup]],
    uint         simd_lane          [[thread_index_in_simdgroup]],
    uint         simd_size          [[threads_per_simdgroup]],
    uint         simd_group_id      [[simdgroup_index_in_threadgroup]])
{
    const uint b = gid.y;          // batch
    const uint d = gid.z;          // channel
    const uint tile = gid.x;       // sequence tile
    const uint L = params.L;
    const uint D = params.D;
    const uint N = params.N;
    const uint tile_L = params.tile_L;

    if (b >= params.B || d >= D) return;

    const uint t0 = tile * tile_L;                 // global start of this tile
    if (t0 >= L) return;

    const uint local_t = lid.x;                    // 0 .. tile_L-1
    const uint t = t0 + local_t;                   // global timestep
    const bool valid = (t < L);

    // -----------------------------------------------------------------------
    // Load A for this channel (constant across time)
    // A is stored as the continuous-time coefficient (usually negative)
    // -----------------------------------------------------------------------
    thread float A_n[16];                          // N <= 16 for register pressure
    for (uint n = 0; n < N; ++n) {
        A_n[n] = A[d * N + n];
    }

    // -----------------------------------------------------------------------
    // Per-timestep affine coefficients for each state dimension
    // We keep them in registers / thread-local arrays
    // -----------------------------------------------------------------------
    thread Affine T[16];                           // one affine per state dim

    if (valid) {
        const uint idx = ((b * L) + t) * D + d;    // index into x / delta
        float xt = x[idx];
        float dt = softplus(delta[idx]);           // Δ_t > 0

        for (uint n = 0; n < N; ++n) {
            float a_cont = A_n[n];                 // continuous A
            // Zero-order hold discretization
            float da = dt * a_cont;
            float exp_da = exp(da);
            // Ā
            T[n].a = exp_da;
            // B̄ * x  (simplified ZOH: (exp(da)-1)/a * B * x)
            // When a is very small we fall back to dt * B * x
            float Bn = Bmat[((b * L) + t) * N + n];
            float Bbar_x;
            if (fabs(a_cont) > 1e-6f) {
                Bbar_x = ((exp_da - 1.0f) / a_cont) * Bn * xt;
            } else {
                Bbar_x = dt * Bn * xt;
            }
            T[n].b = Bbar_x;
        }
    } else {
        for (uint n = 0; n < N; ++n) T[n] = AFFINE_IDENTITY;
    }

    // -----------------------------------------------------------------------
    // Level 1: SIMD-group inclusive scan (associative)
    // Metal provides simd_shuffle / simd_prefix operations on recent GPUs;
    // we implement a generic Hillis-Steele style scan that works everywhere.
    // -----------------------------------------------------------------------
    threadgroup Affine tg_carry[32];               // one carry per SIMD group (max 32)

    for (uint n = 0; n < N; ++n) {
        Affine val = T[n];

        // Hillis-Steele inclusive scan inside the SIMD group
        for (uint offset = 1; offset < simd_size; offset <<= 1) {
            Affine other = simd_shuffle_up(val, offset);
            if (simd_lane >= offset) {
                val = compose(val, other);
            }
        }
        T[n] = val;

        // Last lane of each SIMD group writes the carry for the next level
        if (simd_lane == simd_size - 1) {
            tg_carry[simd_group_id] = val;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Level 2: scan the SIMD-group carries inside the threadgroup
    // Only the first few lanes participate
    // -----------------------------------------------------------------------
    if (simd_group_id == 0 && simd_lane < (lsize.x / simd_size)) {
        // Simple serial scan of the carries (number of SIMD groups is small)
        Affine acc = AFFINE_IDENTITY;
        uint num_sg = lsize.x / simd_size;
        for (uint s = 0; s < num_sg; ++s) {
            Affine tmp = tg_carry[s];
            tg_carry[s] = compose(tmp, acc);       // inclusive
            acc = tg_carry[s];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Apply the preceding SIMD-group carry to every element
    // -----------------------------------------------------------------------
    for (uint n = 0; n < N; ++n) {
        if (simd_group_id > 0) {
            Affine prefix = tg_carry[simd_group_id - 1];
            T[n] = compose(T[n], prefix);
        }
    }

    // -----------------------------------------------------------------------
    // At this point T[n] holds the inclusive scan result for the local tile
    // relative to the beginning of the tile (identity start).
    // A later kernel / host-side pass can propagate inter-tile carries
    // for very long sequences. For moderate L the tile can cover the whole
    // sequence (tile_L >= L).
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Produce output y_t = C · h_t + D · x_t
    // h_t is exactly the 'b' component of the inclusive scan when the
    // scan starts from the zero state (or from a previous tile carry).
    // -----------------------------------------------------------------------
    if (valid) {
        float yt = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float Cn = Cmat[((b * L) + t) * N + n];
            yt += Cn * T[n].b;                     // C · h
        }
        const uint idx = ((b * L) + t) * D + d;
        yt += Dskip[d] * x[idx];                   // D residual
        y[idx] = yt;
    }

    // Optional: write final state of the last timestep of the whole sequence
    if (final_state && valid && t == L - 1) {
        for (uint n = 0; n < N; ++n) {
            final_state[(b * D + d) * N + n] = T[n].b;
        }
    }
}

// ---------------------------------------------------------------------------
// FP16 variant (state still accumulated in FP32 for stability)
// ---------------------------------------------------------------------------
kernel void selective_scan_f16(
    device const half   *x          [[buffer(0)]],
    device const half   *delta      [[buffer(1)]],
    device const half   *Bmat       [[buffer(2)]],
    device const half   *Cmat       [[buffer(3)]],
    device const half   *A          [[buffer(4)]],
    device const half   *Dskip      [[buffer(5)]],
    device       half   *y          [[buffer(6)]],
    device       float  *final_state[[buffer(7)]],   // keep state in FP32
    constant     ScanParams &params [[buffer(8)]],
    uint3        gid                [[thread_position_in_grid]],
    uint3        lid                [[thread_position_in_threadgroup]],
    uint3        lsize              [[threads_per_threadgroup]],
    uint         simd_lane          [[thread_index_in_simdgroup]],
    uint         simd_size          [[threads_per_simdgroup]],
    uint         simd_group_id      [[simdgroup_index_in_threadgroup]])
{
    // Identical structure; loads/stores are half, arithmetic promoted to float
    const uint b = gid.y;
    const uint d = gid.z;
    const uint tile = gid.x;
    const uint L = params.L;
    const uint D = params.D;
    const uint N = params.N;
    const uint tile_L = params.tile_L;

    if (b >= params.B || d >= D) return;

    const uint t0 = tile * tile_L;
    if (t0 >= L) return;

    const uint local_t = lid.x;
    const uint t = t0 + local_t;
    const bool valid = (t < L);

    thread float A_n[16];
    for (uint n = 0; n < N; ++n)
        A_n[n] = float(A[d * N + n]);

    thread Affine T[16];

    if (valid) {
        const uint idx = ((b * L) + t) * D + d;
        float xt = float(x[idx]);
        float dt = softplus(float(delta[idx]));

        for (uint n = 0; n < N; ++n) {
            float a_cont = A_n[n];
            float da = dt * a_cont;
            float exp_da = exp(da);
            T[n].a = exp_da;
            float Bn = float(Bmat[((b * L) + t) * N + n]);
            float Bbar_x = (fabs(a_cont) > 1e-6f)
                ? ((exp_da - 1.0f) / a_cont) * Bn * xt
                : dt * Bn * xt;
            T[n].b = Bbar_x;
        }
    } else {
        for (uint n = 0; n < N; ++n) T[n] = AFFINE_IDENTITY;
    }

    threadgroup Affine tg_carry[32];

    for (uint n = 0; n < N; ++n) {
        Affine val = T[n];
        for (uint offset = 1; offset < simd_size; offset <<= 1) {
            Affine other = simd_shuffle_up(val, offset);
            if (simd_lane >= offset)
                val = compose(val, other);
        }
        T[n] = val;
        if (simd_lane == simd_size - 1)
            tg_carry[simd_group_id] = val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0 && simd_lane < (lsize.x / simd_size)) {
        Affine acc = AFFINE_IDENTITY;
        uint num_sg = lsize.x / simd_size;
        for (uint s = 0; s < num_sg; ++s) {
            Affine tmp = tg_carry[s];
            tg_carry[s] = compose(tmp, acc);
            acc = tg_carry[s];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint n = 0; n < N; ++n) {
        if (simd_group_id > 0) {
            Affine prefix = tg_carry[simd_group_id - 1];
            T[n] = compose(T[n], prefix);
        }
    }

    if (valid) {
        float yt = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float Cn = float(Cmat[((b * L) + t) * N + n]);
            yt += Cn * T[n].b;
        }
        const uint idx = ((b * L) + t) * D + d;
        yt += float(Dskip[d]) * float(x[idx]);
        y[idx] = half(yt);
    }

    if (final_state && valid && t == L - 1) {
        for (uint n = 0; n < N; ++n)
            final_state[(b * D + d) * N + n] = T[n].b;
    }
}
