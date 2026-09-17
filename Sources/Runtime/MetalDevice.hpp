#pragma once
#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <memory>

namespace mamba {

class MetalDevice {
public:
    MetalDevice();
    ~MetalDevice();

    MTL::Device *device() const { return device_; }
    MTL::CommandQueue *queue() const { return queue_; }
    MTL::Library *library() const { return library_; }

    bool loadLibraryFromSource(const std::string &source);
    bool loadLibraryFromFile(const std::string &metallib_path);

    // Query hardware characteristics
    NS::UInteger threadExecutionWidth(MTL::ComputePipelineState *pso) const;
    NS::UInteger maxTotalThreadsPerThreadgroup(MTL::ComputePipelineState *pso) const;

private:
    MTL::Device *device_ = nullptr;
    MTL::CommandQueue *queue_ = nullptr;
    MTL::Library *library_ = nullptr;
};

} // namespace mamba
