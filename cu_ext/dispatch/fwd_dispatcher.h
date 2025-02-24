#pragma once

#include "descriptor.h"
#include "dispatcher.h"
#include "element_types.h"
#include "device/fwd_adapter.h"
#include "kernel_helpers.h"


namespace fused_ce {

void fused_ce_loss_fwd_dispatcher(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    Descriptor descriptor = *UnpackDescriptor<Descriptor>(opaque, opaque_len);
    auto element_type = descriptor.element_type;
    auto reduction_type = descriptor.reduction_type;
    auto accumulation_type = descriptor.accumulation_type;
    if (
        (element_type      == ElementType::BF16) && 
        (reduction_type    == ElementType::F32 ) && 
        (accumulation_type == ElementType::F32 )
    ) {
        dispatch_adapter<
            FusedCELossFwdAdapter,
            Descriptor,
            CutlassType<ElementType::BF16>,
            CutlassType<ElementType::F32 >,
            CutlassType<ElementType::F32 >
        >(stream, buffers, descriptor);
    } else {
        throw std::runtime_error("Unsupported kernel traits");
    }
};

} // fused_ce
