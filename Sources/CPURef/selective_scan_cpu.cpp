#include "selective_scan_cpu.hpp"
#include <cstring>

namespace mamba {
namespace cpu {

void selective_scan_cpu(
    const float *x, const float *delta,
    const float *B, const float *C,
    const float *A, const float *Dskip,
    float *y, float *state_out,
    int L, int D, int N)
{
    // Sequential reference (correctness oracle)
    std::vector<float> h(D * N, 0.f);

    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            float xt = x[t * D + d];
            float dt = softplus(delta[t * D + d]);
            float yt = 0.f;

            for (int n = 0; n < N; ++n) {
                float a = A[d * N + n];
                float da = dt * a;
                float exp_da = std::exp(da);
                float Bn = B[t * N + n];
                float Bbar = (std::fabs(a) > 1e-6f)
                    ? ((exp_da - 1.f) / a) * Bn
                    : dt * Bn;

                float &ht = h[d * N + n];
                ht = exp_da * ht + Bbar * xt;

                float Cn = C[t * N + n];
                yt += Cn * ht;
            }
            yt += Dskip[d] * xt;
            y[t * D + d] = yt;
        }
    }

    if (state_out)
        std::memcpy(state_out, h.data(), sizeof(float) * D * N);
}

} // namespace cpu
} // namespace mamba
