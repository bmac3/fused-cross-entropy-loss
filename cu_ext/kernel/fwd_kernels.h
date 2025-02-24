#pragma once

#include <cute/container/array_aligned.hpp>
#include <cute/tensor.hpp>

#include "descriptor.h"


namespace fused_ce {

using namespace cute;


namespace kernel {

template <
    typename TensorV,
    typename TensorX,
    typename TensorN,
    typename TensorM,
    typename CtaTiler, 
    typename ReduceTiler,
    typename TiledMma,
    typename VSmemLayout,
    typename XSmemLayout,
    typename NSmemLayout,
    typename MSmemLayout,
    typename TiledCopyX,
    typename TiledCopyV
>
class FusedCENormalizer {
public:
    // type definitions
    using Element = typename TensorV::value_type;



    struct Params {
        // input arguments
        TensorV V;
        TensorX X;

        // output arguments
        TensorN N;
        TensorM M;
    };

    struct SharedStorage {
        array_aligned<ElementA, cosize_v<VSmemLayout>> smem_a;
    };

public:
    // methods
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        return;
    }

};

} // kernel

} // namespace fused_ce
