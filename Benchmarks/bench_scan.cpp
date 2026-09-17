// bench_scan.cpp – micro-benchmark for selective scan
#include "../Sources/CPURef/selective_scan_cpu.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using namespace mamba::cpu;
using Clock = std::chrono::high_resolution_clock;

int main() {
    const int D = 512, N = 16;
    std::vector<int> lengths = {1024, 4096, 16384, 65536};

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int L : lengths) {
        std::vector<float> x(L*D), delta(L*D), B(L*N), C(L*N), A(D*N), Dskip(D), y(L*D);

        for (auto &v : x) v = dist(rng);
        for (auto &v : delta) v = dist(rng) * 0.05f;
        for (auto &v : B) v = dist(rng);
        for (auto &v : C) v = dist(rng);
        for (auto &v : A) v = -std::abs(dist(rng));
        for (auto &v : Dskip) v = dist(rng) * 0.1f;

        // warm-up
        selective_scan_cpu(x.data(), delta.data(), B.data(), C.data(),
                           A.data(), Dskip.data(), y.data(), nullptr, L, D, N);

        auto t0 = Clock::now();
        const int iters = 5;
        for (int i = 0; i < iters; ++i)
            selective_scan_cpu(x.data(), delta.data(), B.data(), C.data(),
                               A.data(), Dskip.data(), y.data(), nullptr, L, D, N);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double tokens = double(L);
        double tok_per_s = tokens / (ms / 1000.0);

        printf("CPU scan  L=%6d  D=%d  N=%d  %.2f ms  %.1f ktok/s\n",
               L, D, N, ms, tok_per_s / 1000.0);
    }
    return 0;
}
