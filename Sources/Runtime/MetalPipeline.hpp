#pragma once
#include <Metal/Metal.hpp>
#include <string>

namespace mamba {

class MetalPipeline {
public:
    MetalPipeline() = default;
    MetalPipeline(MTL::Device *dev, MTL::Library *lib, const std::string &func_name);
    ~MetalPipeline();

    MetalPipeline(MetalPipeline &&o) noexcept;
    MetalPipeline &operator=(MetalPipeline &&o) noexcept;
    MetalPipeline(const MetalPipeline &) = delete;
    MetalPipeline &operator=(const MetalPipeline &) = delete;

    MTL::ComputePipelineState *pso() const { return pso_; }
    bool valid() const { return pso_ != nullptr; }

    NS::UInteger threadExecutionWidth() const {
        return pso_ ? pso_->threadExecutionWidth() : 0;
    }
    NS::UInteger maxTotalThreadsPerThreadgroup() const {
        return pso_ ? pso_->maxTotalThreadsPerThreadgroup() : 0;
    }

private:
    MTL::ComputePipelineState *pso_ = nullptr;
};

} // namespace mamba
