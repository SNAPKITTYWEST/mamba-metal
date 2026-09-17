#include "MambaRuntime.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>

namespace mamba {

// Helper to read a whole .metal file into a string (for embedding / loading)
static std::string read_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

MambaRuntime::MambaRuntime(const MambaConfig &cfg) : cfg_(cfg) {
    device_ = std::make_unique<MetalDevice>();
}

MambaRuntime::~MambaRuntime() = default;

bool MambaRuntime::init() {
    if (!device_ || !device_->device()) return false;

    // In a real build the metallib is precompiled; here we load sources
    // that were placed next to the binary or embedded.
    // For the demo we assume the kernels are compiled into a single library
    // by the Xcode/Metal build system.  The pipelines are created by name.

    return create_pipelines();
}

bool MambaRuntime::create_pipelines() {
    // Pipelines are created from the default library that Xcode embeds.
    // When running outside Xcode the host must call loadLibraryFromSource
    // with the concatenated kernel sources.
    auto *lib = device_->library();
    if (!lib) {
        // Fallback: try to compile from a known source tree (development only)
        std::cerr << "mamba-metal: no Metal library loaded – pipelines will be null\n";
        return false;
    }

    auto *dev = device_->device();
    pipe_scan_f32_   = MetalPipeline(dev, lib, "selective_scan_f32");
    pipe_scan_f16_   = MetalPipeline(dev, lib, "selective_scan_f16");
    pipe_conv_       = MetalPipeline(dev, lib, "causal_conv1d_f32");
    pipe_step_       = MetalPipeline(dev, lib, "ssm_step_f32");
    pipe_rmsnorm_    = MetalPipeline(dev, lib, "rmsnorm_f32");
    pipe_linear_     = MetalPipeline(dev, lib, "linear_f32");
    pipe_silu_       = MetalPipeline(dev, lib, "silu_f32");
    pipe_fused_gate_ = MetalPipeline(dev, lib, "fused_selective_scan_gate_f32");

    return pipe_scan_f32_.valid() || pipe_step_.valid();
}

bool MambaRuntime::load_weights(const void *data, size_t bytes) {
    // Demo loader: expects a simple header + raw float buffers.
    // Production code would parse a safetensors / custom format.
    if (!data || bytes < 64) return false;
    // Allocate device buffers according to cfg_ and copy data.
    // Omitted full parsing for length; the structure is ready.
    std::cerr << "mamba-metal: load_weights stub – replace with real weight loader\n";
    return true;
}

MambaState MambaRuntime::create_state(int batch) {
    MambaState s;
    s.batch = batch;
    size_t conv_bytes = size_t(batch) * cfg_.d_inner() * (cfg_.d_conv - 1) * sizeof(float);
    size_t ssm_bytes  = size_t(batch) * cfg_.d_inner() * cfg_.d_state * sizeof(float);
    s.conv_state = MetalBuffer(device_->device(), conv_bytes);
    s.ssm_state  = MetalBuffer(device_->device(), ssm_bytes);
    s.reset();
    return s;
}

void MambaRuntime::reset_state(MambaState &state) {
    if (state.conv_state.contents())
        std::memset(state.conv_state.contents(), 0, state.conv_state.length());
    if (state.ssm_state.contents())
        std::memset(state.ssm_state.contents(), 0, state.ssm_state.length());
}

void MambaState::reset() {
    if (conv_state.contents()) std::memset(conv_state.contents(), 0, conv_state.length());
    if (ssm_state.contents())  std::memset(ssm_state.contents(), 0, ssm_state.length());
}

bool MambaRuntime::forward(const float *input, float *output, int batch, int seq_len) {
    // High-level orchestration:
    // 1. RMSNorm
    // 2. in_proj → split x / z
    // 3. causal_conv on x
    // 4. x_proj → delta, B, C
    // 5. selective_scan (or fused)
    // 6. gate with z
    // 7. out_proj
    // 8. residual
    //
    // Each step is a Metal dispatch.  For production the fused kernel
    // reduces the number of launches.
    std::cerr << "mamba-metal: forward() – full graph dispatch not yet wired "
                 "(pipelines are ready)\n";
    // Copy input → output as a placeholder so the API is callable
    size_t elems = size_t(batch) * seq_len * cfg_.d_model;
    std::memcpy(output, input, elems * sizeof(float));
    return true;
}

bool MambaRuntime::step(MambaState &state, const float *token, float *out, int batch) {
    // Single-token path using ssm_step + conv_step kernels
    std::cerr << "mamba-metal: step() – incremental path ready for wiring\n";
    size_t elems = size_t(batch) * cfg_.d_model;
    std::memcpy(out, token, elems * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// C API implementation
// ---------------------------------------------------------------------------
struct MambaModel {
    std::unique_ptr<MambaRuntime> rt;
};

struct MambaStateHandle {
    MambaState state;
};

extern "C" {

MambaModel *mamba_create(const MambaConfig *cfg) {
    if (!cfg) return nullptr;
    auto *m = new MambaModel;
    m->rt = std::make_unique<MambaRuntime>(*cfg);
    if (!m->rt->init()) {
        delete m;
        return nullptr;
    }
    return m;
}

void mamba_destroy(MambaModel *m) {
    delete m;
}

int mamba_load_weights(MambaModel *m, const void *data, size_t size) {
    if (!m || !m->rt) return -1;
    return m->rt->load_weights(data, size) ? 0 : -1;
}

int mamba_forward(MambaModel *m, const float *input, float *output, int batch, int seq_len) {
    if (!m || !m->rt) return -1;
    return m->rt->forward(input, output, batch, seq_len) ? 0 : -1;
}

MambaStateHandle *mamba_create_state(MambaModel *m, int batch) {
    if (!m || !m->rt) return nullptr;
    auto *h = new MambaStateHandle;
    h->state = m->rt->create_state(batch);
    return h;
}

void mamba_reset_state(MambaStateHandle *s) {
    if (s) s->state.reset();
}

int mamba_step(MambaModel *m, MambaStateHandle *s, const float *token, float *out) {
    if (!m || !m->rt || !s) return -1;
    return m->rt->step(s->state, token, out) ? 0 : -1;
}

void mamba_destroy_state(MambaStateHandle *s) {
    delete s;
}

} // extern "C"

} // namespace mamba
