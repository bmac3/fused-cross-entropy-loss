#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

namespace cu_ext {

struct FusedCELossDescriptor {
    int batch_size;
    int sequence_len;
    int vocab_size;
    int embed_size;
};

void fused_ce_loss_fwd_bf16(cudaStream_t stream, void** buffers, const char* opaque,
                            std::size_t opaque_len);

}
