// Minimal verification harness (compile with the CPU reference)
#include "../Sources/CPURef/selective_scan_cpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace mamba::cpu;

int main() {
    const int L = 128, D = 64, N = 16;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> x(L*D), delta(L*D), B(L*N), C(L*N), A(D*N), Dskip(D);
    std::vector<float> y(L*D), state(D*N);

    for (auto &v : x) v = dist(rng);
    for (auto &v : delta) v = dist(rng) * 0.1f;
    for (auto &v : B) v = dist(rng);
    for (auto &v : C) v = dist(rng);
    for (auto &v : A) v = -std::abs(dist(rng)); // negative
    for (auto &v : Dskip) v = dist(rng) * 0.1f;

    selective_scan_cpu(x.data(), delta.data(), B.data(), C.data(),
                       A.data(), Dskip.data(), y.data(), state.data(),
                       L, D, N);

    // Sanity: output should be finite
    int nans = 0;
    for (float v : y) if (!std::isfinite(v)) ++nans;
    printf("CPU selective scan: L=%d D=%d N=%d  NaNs=%d  last_y=%.6f\n",
           L, D, N, nans, y.back());
    return nans ? 1 : 0;
}
