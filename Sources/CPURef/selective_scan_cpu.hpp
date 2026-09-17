#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace mamba {
namespace cpu {

// Pure CPU reference of the selective scan for numerical verification.
// Uses the same associative formulation.

struct Affine {
    float a = 1.f;
    float b = 0.f;
};

inline Affine compose(const Affine &t2, const Affine &t1) {
    return {t2.a * t1.a, t2.a * t1.b + t2.b};
}

inline float softplus(float x) {
    return x > 20.f ? x : std::log1p(std::exp(x));
}

// x, delta: [L, D]
// B, C: [L, N]
// A: [D, N]
// Dskip: [D]
// y: [L, D]
// state_out: [D, N] (optional)
void selective_scan_cpu(
    const float *x, const float *delta,
    const float *B, const float *C,
    const float *A, const float *Dskip,
    float *y, float *state_out,
    int L, int D, int N);

} // namespace cpu
} // namespace mamba
