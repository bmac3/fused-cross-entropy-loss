#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "kernel_helpers.h"
#include "kernels.h"

namespace cu_ext {

namespace {

constexpr int BLOCK_SIZE = 16;


template <typename T>
__global__ void cuFusedCrossEntropyLossFwd(const T *xs, const T *vocab, T *out, 
                                           int batch_size, int sequence_len, int vocab_size, int embed_size) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx >= sequence_len || tidy >= vocab_size)
        return;

    out[tidy * sequence_len + tidx] = 1.6;
}


void ThrowIfError(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}


template <typename T>
void apply_fused_ce_loss_fwd(cudaStream_t stream, void **buffers, const char *opaque, 
                             std::size_t opaque_len) {
    const FusedCELossDescriptor &d = *UnpackDescriptor<FusedCELossDescriptor>(opaque, opaque_len);
    const int batch_size = d.batch_size;
    const int vocab_size = d.vocab_size;
    const int sequence_len = d.sequence_len;
    const int embed_size = d.embed_size;

    const T *xs = static_cast<const T*>(buffers[0]);
    const T *vocab = static_cast<const T*>(buffers[1]);
    T *out = static_cast<T*>(buffers[2]);

    const dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    int grid_x = (sequence_len % BLOCK_SIZE == 0) ? sequence_len / BLOCK_SIZE : (sequence_len / BLOCK_SIZE) + 1;
    int grid_y = (vocab_size % BLOCK_SIZE == 0) ? vocab_size / BLOCK_SIZE : (vocab_size / BLOCK_SIZE) + 1;
    const dim3 grid_dim(grid_x, grid_y);

    cuFusedCrossEntropyLossFwd<<<grid_dim, block_dim>>>(vocab, xs, out, batch_size, sequence_len, vocab_size, embed_size);

    ThrowIfError(cudaGetLastError());
}

}

void fused_ce_loss_fwd_bf16(cudaStream_t stream, void **buffers, const char *opaque,
                       std::size_t opaque_len) {
    apply_fused_ce_loss_fwd<__nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

}
