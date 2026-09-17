#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>

namespace mamba {

class MetalBuffer {
public:
    MetalBuffer() = default;
    MetalBuffer(MTL::Device *dev, size_t bytes, MTL::ResourceOptions opts = MTL::ResourceStorageModeShared);
    ~MetalBuffer();

    // move only
    MetalBuffer(MetalBuffer &&o) noexcept;
    MetalBuffer &operator=(MetalBuffer &&o) noexcept;
    MetalBuffer(const MetalBuffer &) = delete;
    MetalBuffer &operator=(const MetalBuffer &) = delete;

    MTL::Buffer *get() const { return buf_; }
    void *contents() const { return buf_ ? buf_->contents() : nullptr; }
    size_t length() const { return buf_ ? buf_->length() : 0; }

    template <typename T>
    T *data() { return static_cast<T *>(contents()); }

private:
    MTL::Buffer *buf_ = nullptr;
};

} // namespace mamba
