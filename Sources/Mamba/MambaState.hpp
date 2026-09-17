#pragma once
#include "MambaConfig.hpp"
#include "../Runtime/MetalBuffer.hpp"
#include <memory>

namespace mamba {

struct MambaState {
    // Persistent state for incremental inference
    // conv_state : [B, d_inner, d_conv-1]
    // ssm_state  : [B, d_inner, d_state]
    MetalBuffer conv_state;
    MetalBuffer ssm_state;
    int batch = 1;

    void reset();
};

} // namespace mamba
