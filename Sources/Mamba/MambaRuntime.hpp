#pragma once
#include "MambaConfig.hpp"
#include "MambaState.hpp"
#include "../Runtime/MetalDevice.hpp"
#include "../Runtime/MetalBuffer.hpp"
#include "../Runtime/MetalPipeline.hpp"
#include <vector>
#include <string>
#include <memory>

namespace mamba {

struct MambaWeights {
    // All weights stored as device buffers
    MetalBuffer in_proj;      // [d_model, 2*d_inner]
    MetalBuffer conv1d;       // [d_inner, d_conv]
    MetalBuffer conv_bias;    // [d_inner]
    MetalBuffer x_proj;       // [d_inner, dt_rank + 2*d_state]
    MetalBuffer dt_proj;      // [dt_rank, d_inner]
    MetalBuffer A;            // [d_inner, d_state]  (log-space or negative)
    MetalBuffer D;            // [d_inner]
    MetalBuffer out_proj;     // [d_inner, d_model]
    // optional RMSNorm weights
    MetalBuffer norm_weight;
};

class MambaRuntime {
public:
    explicit MambaRuntime(const MambaConfig &cfg);
    ~MambaRuntime();

    bool init();
    bool load_weights(const void *data, size_t bytes);  // simple contiguous blob for demo

    // Full-sequence forward
    // input  : [B, L, d_model]
    // output : [B, L, d_model]
    bool forward(const float *input, float *output, int batch, int seq_len);

    // Incremental single-token step
    bool step(MambaState &state, const float *token, float *out, int batch = 1);

    MambaState create_state(int batch = 1);
    void reset_state(MambaState &state);

    const MambaConfig &config() const { return cfg_; }

private:
    MambaConfig cfg_;
    std::unique_ptr<MetalDevice> device_;
    MambaWeights weights_;

    // Pipelines
    MetalPipeline pipe_scan_f32_;
    MetalPipeline pipe_scan_f16_;
    MetalPipeline pipe_conv_;
    MetalPipeline pipe_step_;
    MetalPipeline pipe_rmsnorm_;
    MetalPipeline pipe_linear_;
    MetalPipeline pipe_silu_;
    MetalPipeline pipe_fused_gate_;

    bool create_pipelines();
};

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------
extern "C" {

typedef struct MambaModel MambaModel;
typedef struct MambaStateHandle MambaStateHandle;

MambaModel *mamba_create(const MambaConfig *cfg);
void        mamba_destroy(MambaModel *m);

int  mamba_load_weights(MambaModel *m, const void *data, size_t size);
int  mamba_forward(MambaModel *m, const float *input, float *output, int batch, int seq_len);

MambaStateHandle *mamba_create_state(MambaModel *m, int batch);
void              mamba_reset_state(MambaStateHandle *s);
int               mamba_step(MambaModel *m, MambaStateHandle *s, const float *token, float *out);
void              mamba_destroy_state(MambaStateHandle *s);

} // extern "C"

} // namespace mamba
