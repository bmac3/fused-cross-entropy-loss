#pragma once

#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include "kernel_helpers.h"


namespace fused_ce {

using namespace cute;

namespace kernel {


template <
    typename ElementReduce_,
    typename TensorV_,
    typename TensorX_,
    typename TensorN_
>
class FusedCENormalizer {
public:
    // type definitions
    using TensorV = TensorV_;
    using TensorX = TensorX_;
    using TensorN = TensorN_;

    struct Params {
        // input arguments
        TensorV V;
        TensorX X;

        // output arguments
        TensorN N;

    };

    struct SharedStorage {

    };

public:
    // methods
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        return;
    }

};

} // kernel

template <typename KTraits>
class FusedCELossFwd {
public:
    // type definitions
    using Element = KTraits::Element;
    using ElementReduce = KTraits::ElementReduce;
    using ElementAccumulate = KTraits::ElementAccumulate;
    using Layout = Layout_;

    struct Params {
        // input arguments
        Element const* ptr_V;
        Element const* ptr_X;
        int     const* ptr_Y;

        // output arguments
        Element      * ptr_O;

        // opaque arguments
        Descriptor desc;

        Params(void** buffers, Descriptor descriptor)
            : ptr_X{static_cast<Element const*>(buffers[0])},
              ptr_V{static_cast<Element const*>(buffers[1])},
              ptr_Y{static_cast<int     const*>(buffers[2])},
              ptr_O{static_cast<Element      *>(buffers[3])},
              desc{descriptor} {}

        // Params(void** buffers, const char* opaque, std::size_t opaque_len)
        //     : ptr_X{static_cast<Element const*>(buffers[0])},
        //       ptr_V{static_cast<Element const*>(buffers[1])},
        //       ptr_Y{static_cast<int     const*>(buffers[2])},
        //       ptr_O{static_cast<Element      *>(buffers[3])},
        //       desc{*UnpackDescriptor<Descriptor>(opaque, opaque_len)} {}
    };

public:
    // methods
    cutlass::Status operator()(Params const &params, cudaStream_t stream) {
        // auto desc = params.desc;

        

        // Tensor V = make_tensor(
        //     make_gmem_ptr(params.ptr_V),
        //     make_shape(params.desc.vocab_size, params.desc.embed_size),
        //     Layout{}
        // );
        // Tensor X = make_tensor(
        //     make_gmem_ptr(params.ptr_X),
        //     make_shape(params.desc.batch_size * params.desc.sequence_len, params.desc.embed_size),
        //     Layout{}
        // );
        // Tensor Y = make_tensor(
        //     make_gmem_ptr(params.ptr_Y),
        //     make_shape(params.desc.batch_size * params.desc.sequence_len),
        //     Layout{}
        // );
        // Tensor O = make_tensor(
        //     make_gmem_ptr(params.ptr_O),
        //     make_shape(params.desc.batch_size * params.desc.sequence_len),
        //     Layout{}
        // );

        // thrust::device_vector<float> d_N(size(O));
        // Tensor N = make_tensor(
        //     make_gmem_ptr(d_N.data().get()),
        //     layout(O)
        // );

        // using Normalizer = kernel::FusedCENormalizer<ElementReduce, decltype(V), decltype(X), decltype(N)>;
        // typename Normalizer::Params normalizer_params{V, X, N};

        



        return cutlass::Status::kSuccess;
    }

};

} // namespace fused_ce
