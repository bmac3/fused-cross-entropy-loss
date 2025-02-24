#pragma once

#include <cute/tensor.hpp>
#include <thrust/device_vector.h>

#include "descriptor.h"
#include "kernel/fwd_kernels.h"


namespace fused_ce {

using namespace cute;


template <
    typename Element, 
    typename ElementReduce, 
    typename ElementAccumulate,
    typename Layout = LayoutRight
>
class FusedCELossFwdAdapter {

public:
    // methods
    void operator()(cudaStream_t stream, void** buffers, Descriptor descriptor) {
        auto embed_size = descriptor.embed_size;
        auto vocab_size = descriptor.vocab_size;
        auto batch_size = descriptor.batch_size;
        auto sequence_len = descriptor.sequence_len;
        auto ptr_X = static_cast<Element const*>(buffers[0]);
        Tensor X = make_tensor(
            make_gmem_ptr(ptr_X),
            make_shape(batch_size * sequence_len, embed_size),
            Layout{}
        );
        auto ptr_V = static_cast<Element const*>(buffers[1]);
        Tensor V = make_tensor(
            make_gmem_ptr(ptr_V),
            make_shape(vocab_size, embed_size),
            Layout{}
        );
        auto ptr_Y = static_cast<int const*>(buffers[2]);
        Tensor Y = make_tensor(
            make_gmem_ptr(ptr_Y),
            make_shape(batch_size * sequence_len),
            Layout{}
        );
        auto ptr_O = static_cast<Element*>(buffers[3]);
        Tensor O = make_tensor(
            make_gmem_ptr(ptr_O),
            make_shape(batch_size * sequence_len),
            Layout{}
        );
        thrust::device_vector<ElementReduce> d_N(size(O));
        Tensor N = make_tensor(
            make_gmem_ptr(d_N.data().get()),
            layout(O)
        );
        thrust::device_vector<ElementAccumulate> d_M(size(O));
        Tensor M = make_tensor(
            make_gmem_ptr(d_M.data().get()),
            layout(O)
        );
        


    }

};

} // fused_ce

