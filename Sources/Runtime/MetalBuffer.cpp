#include "MetalBuffer.hpp"

namespace mamba {

MetalBuffer::MetalBuffer(MTL::Device *dev, size_t bytes, MTL::ResourceOptions opts) {
    if (dev && bytes)
        buf_ = dev->newBuffer(bytes, opts);
}

MetalBuffer::~MetalBuffer() {
    if (buf_) buf_->release();
}

MetalBuffer::MetalBuffer(MetalBuffer &&o) noexcept : buf_(o.buf_) {
    o.buf_ = nullptr;
}

MetalBuffer &MetalBuffer::operator=(MetalBuffer &&o) noexcept {
    if (this != &o) {
        if (buf_) buf_->release();
        buf_ = o.buf_;
        o.buf_ = nullptr;
    }
    return *this;
}

} // namespace mamba
