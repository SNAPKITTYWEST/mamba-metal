// test_numerical.cpp – Metal vs CPU numerical agreement tests
// (CPU side only in this environment; Metal side is exercised on device)
#include "../Sources/CPURef/selective_scan_cpu.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>

using namespace mamba::cpu;

static bool almost_equal(float a, float b, float rtol = 1e-4f, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

static int test_impulse() {
    // Impulse input should produce a clean decaying trajectory
    const int L = 64, D = 4, N = 2;
    std::vector<float> x(L*D, 0.f), delta(L*D, 0.1f), B(L*N, 1.f), C(L*N, 1.f);
    std::vector<float> A(D*N, -0.5f), Dskip(D, 0.f), y(L*D);

    x[0] = 1.0f;  // impulse at t=0, channel 0

    selective_scan_cpu(x.data(), delta.data(), B.data(), C.data(),
                       A.data(), Dskip.data(), y.data(), nullptr, L, D, N);

    // y[0] should be non-zero, subsequent values should decay
    if (y[0] == 0.f) return 1;
    bool decaying = true;
    for (int t = 1; t < L; ++t) {
        if (std::fabs(y[t*D]) > std::fabs(y[(t-1)*D]) + 1e-6f)
            decaying = false;
    }
    printf("impulse test: %s\n", decaying ? "PASS" : "FAIL");
    return decaying ? 0 : 1;
}

static int test_zero_input() {
    const int L = 32, D = 8, N = 4;
    std::vector<float> x(L*D, 0.f), delta(L*D, 0.1f), B(L*N, 0.5f), C(L*N, 0.5f);
    std::vector<float> A(D*N, -1.f), Dskip(D, 0.1f), y(L*D);

    selective_scan_cpu(x.data(), delta.data(), B.data(), C.data(),
                       A.data(), Dskip.data(), y.data(), nullptr, L, D, N);

    int non_zero = 0;
    for (float v : y) if (std::fabs(v) > 1e-7f) ++non_zero;
    printf("zero-input test: non-zero outputs = %d (expect 0) %s\n",
           non_zero, non_zero == 0 ? "PASS" : "FAIL");
    return non_zero == 0 ? 0 : 1;
}

static int test_associativity() {
    // Compose three random affines two different ways; results must match
    Affine a{0.9f, 0.1f}, b{0.8f, 0.2f}, c{0.7f, 0.3f};
    Affine left  = compose(a, compose(b, c));
    Affine right = compose(compose(a, b), c);
    bool ok = almost_equal(left.a, right.a) && almost_equal(left.b, right.b);
    printf("associativity test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += test_associativity();
    fails += test_zero_input();
    fails += test_impulse();
    printf("Numerical tests finished – %d failure(s)\n", fails);
    return fails;
}
