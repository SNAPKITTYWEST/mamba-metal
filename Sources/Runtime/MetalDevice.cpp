#include "MetalDevice.hpp"
#include <iostream>

namespace mamba {

MetalDevice::MetalDevice() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "mamba-metal: no Metal device available\n";
        return;
    }
    queue_ = device_->newCommandQueue();
    if (!queue_) {
        std::cerr << "mamba-metal: failed to create command queue\n";
    }
}

MetalDevice::~MetalDevice() {
    if (library_) library_->release();
    if (queue_) queue_->release();
    if (device_) device_->release();
}

bool MetalDevice::loadLibraryFromSource(const std::string &source) {
    if (!device_) return false;
    NS::Error *err = nullptr;
    auto ns_src = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    library_ = device_->newLibrary(ns_src, nullptr, &err);
    if (!library_) {
        std::cerr << "mamba-metal: shader compile error: "
                  << (err ? err->localizedDescription()->utf8String() : "unknown")
                  << "\n";
        return false;
    }
    return true;
}

bool MetalDevice::loadLibraryFromFile(const std::string &path) {
    if (!device_) return false;
    NS::Error *err = nullptr;
    auto ns_path = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    library_ = device_->newLibrary(ns_path, &err);
    if (!library_) {
        std::cerr << "mamba-metal: failed to load metallib\n";
        return false;
    }
    return true;
}

NS::UInteger MetalDevice::threadExecutionWidth(MTL::ComputePipelineState *pso) const {
    return pso ? pso->threadExecutionWidth() : 0;
}

NS::UInteger MetalDevice::maxTotalThreadsPerThreadgroup(MTL::ComputePipelineState *pso) const {
    return pso ? pso->maxTotalThreadsPerThreadgroup() : 0;
}

} // namespace mamba
