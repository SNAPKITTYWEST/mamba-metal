// state_update.metal
// Single-token SSM state update for incremental inference.
// h_t = Ā h_{t-1} + B̄ x_t
// y_t = C h_t + D x_t

#include <metal_stdlib>
using namespace metal;

struct StepParams {
    uint B;
    uint D;
    uint N;
};

kernel void ssm_step_f32(
    device const float *x_t        [[buffer(0)]],  // [B, D]
    device const float *delta_t    [[buffer(1)]],  // [B, D]
    device const float *B_t        [[buffer(2)]],  // [B, N]
    device const float *C_t        [[buffer(3)]],  // [B, N]
    device const float *A          [[buffer(4)]],  // [D, N]
    device const float *Dskip      [[buffer(5)]],  // [D]
    device       float *y_t        [[buffer(6)]],  // [B, D]
    device       float *ssm_state  [[buffer(7)]],  // [B, D, N]  in/out
    constant StepParams &p         [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.y;
    uint d = gid.x;
    if (b >= p.B || d >= p.D) return;

    float xt = x_t[b * p.D + d];
    float dt = delta_t[b * p.D + d];
    if (dt < 0) dt = log(1.0f + exp(dt));   // softplus safety

    float yt = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        float a = A[d * p.N + n];
        float da = dt * a;
        float exp_da = exp(da);
        float Bn = B_t[b * p.N + n];
        float Bbar;
        if (fabs(a) > 1e-6f)
            Bbar = ((exp_da - 1.0f) / a) * Bn;
        else
            Bbar = dt * Bn;

        // state update
        float h = ssm_state[(b * p.D + d) * p.N + n];
        h = exp_da * h + Bbar * xt;
        ssm_state[(b * p.D + d) * p.N + n] = h;

        float Cn = C_t[b * p.N + n];
        yt += Cn * h;
    }
    yt += Dskip[d] * xt;
    y_t[b * p.D + d] = yt;
}
