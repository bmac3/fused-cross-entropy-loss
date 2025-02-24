#pragma once

#include <bit>
#include <pybind11/pybind11.h>
#include "kernel_helpers.h"

template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
    return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
    return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <template <typename> typename Kernel, typename Descriptor>
void kernel_wrapper(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    Descriptor descriptor = *UnpackDescriptor<Descriptor>(opaque, opaque_len);
    Kernel<decltype(descriptor.kernel_traits())> kernel;
    typename Kernel::Params params(buffers, descriptor);
    kernel(params, stream);
}

template <template <typename> typename Kernel, typename Descriptor>
pybind11::capsule EncapsulateKernel() {
    return EncapsulateFunction(kernel_wrapper<Kernel>);
}
