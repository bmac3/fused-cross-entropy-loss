#pragma once

namespace fused_ce {

template <template<typename...> typename Adapter, typename Descriptor, typename... AdapterArgs>
void dispatch_adapter(cudaStream_t stream, void **buffers, Descriptor descriptor) {
    Adapter<AdapterArgs...> adapter;
    adapter(stream, buffers, descriptor);
}

} // fused_ce