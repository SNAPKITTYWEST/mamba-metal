#include "MetalPipeline.hpp"
#include <iostream>

namespace mamba {

MetalPipeline::MetalPipeline(MTL::Device *dev, MTL::Library *lib, const std::string &name) {
    if (!dev || !lib) return;
    auto ns_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    auto fn = lib->newFunction(ns_name);
    if (!fn) {
        std::cerr << "mamba-metal: function not found: " << name << "\n";
        return;
    }
    NS::Error *err = nullptr;
    pso_ = dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso_) {
        std::cerr << "mamba-metal: pipeline creation failed for " << name << "\n";
    }
}

MetalPipeline::~MetalPipeline() {
    if (pso_) pso_->release();
}

MetalPipeline::MetalPipeline(MetalPipeline &&o) noexcept : pso_(o.pso_) {
    o.pso_ = nullptr;
}

MetalPipeline &MetalPipeline::operator=(MetalPipeline &&o) noexcept {
    if (this != &o) {
        if (pso_) pso_->release();
        pso_ = o.pso_;
        o.pso_ = nullptr;
    }
    return *this;
}

} // namespace mamba
