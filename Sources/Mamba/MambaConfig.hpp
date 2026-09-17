#pragma once
#include <cstdint>

namespace mamba {

enum class DType { F32, F16 };

struct MambaConfig {
    int d_model   = 512;   // model dimension
    int d_state   = 16;    // SSM state size N
    int d_conv    = 4;     // conv kernel width
    int expand    = 2;     // expansion factor
    int n_layer   = 1;     // for multi-layer later
    DType dtype   = DType::F16;

    // derived
    int d_inner() const { return expand * d_model; }
};

} // namespace mamba
