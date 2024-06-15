#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cooperative_groups.h>
#include <stdexcept>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "device_utils.h"
#include "kernels.h"

namespace cu_ext {

using namespace cute;
using namespace cooperative_groups;

namespace {

template <class T>
T infinity() {
    return cutlass::platform::numeric_limits<T>::infinity();
}


template <class T,
          class TensorX, class XSmemLayout, class TiledCopyX, 
          class TensorV, class VSmemLayout, class TiledCopyV,
          class TensorN, class TensorM,
          class CSmemLayout, class TiledMma>
__device__ void computeNormalizer(TensorX const gX, XSmemLayout sX_layout, TiledCopyX copy_x,
                                  TensorV const gV, VSmemLayout sV_layout, TiledCopyV copy_v,
                                  TensorN sN, TensorM sM,
                                  CSmemLayout sC_layout, TiledMma mma) {
    // create shared memory buffers for X and V
    __shared__ T smemX[cosize_v<XSmemLayout>];
    __shared__ T smemv[cosize_v<VSmemLayout>];
    Tensor sX = make_tensor(make_smem_ptr(smemX), sX_layout);       // (BLK_B,BLK_E)
    Tensor sV = make_tensor(make_smem_ptr(smemV), sV_layout);       // (BLK_V,BLK_E)

    // partition X and V for copying
    ThrCopy thr_copy_x = copy_x.get_slice(threadIdx.x);
    Tensor tXgX = thr_copy_x.partition_S(gX);                       // (CPY,CPY_B,CPY_E,e)
    Tensor tXsX = thr_copy_x.partition_D(sX);                       // (CPY,CPY_B,CPY_E)

    ThrCopy thr_copy_v = copy_v.get_slice(threadIdx.x);
    Tensor tVgV = thr_copy_v.partition_S(gV);                       // (CPY,CPY_V,CPY_E,v,e)
    Tensor tVsV = thr_copy_v.partition_D(sV);                       // (CPY,CPY_V,CPY_E)

    // initialize normalizer and max values
    fill(sN, 0.);
    fill(sM, -infinity<T>());

    // partition X and V for mma
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsX = thr_mma.partition_A(sX);                          // (MMA,MMA_B,MMA_E)
    Tensor tCsV = thr_mma.partition_B(sV);                          // (MMA,MMA_V,MMA_E)
    auto sC_TVlayout = mma.get_layoutC_TV();

    // create accumulator in registers
    auto rC_shape = replace<2>(shape(tCsX), shape<1>(tCsV));
    Tensor tCrC = make_tensor<mma.FrgTypeC>(rC_shape);              // (MMA,MMA_B,MMA_V)

    auto V_BLOCK_MAX = size<3>(tVgV);
    auto E_TILE_MAX = size<3>(tXgX);
    auto B_REG_MAX = size<1>(tCrC);
    auto V_REG_MAX = size<2>(tCrC);

    Tensor reg_max = make_tensor<mma.FrgTypeC>(shape<0>(sX));

    // init cooperative groups for local reductions
    thread_block


    for (int v_block = 0; v_block < V_BLOCK_MAX; ++v_block) {
        // clear the accumulator
        clear(tCrC);
        for (int e_tile = 0; e_tile < E_TILE_MAX; ++e_tile) {
            // copy X and V from gmem to smem
            copy(copy_x, tXgX(_,_,_,e_tile),        tXsX);
            copy(copy_v, tVgV(_,_,_,v_block,e_tile),tVgV);
            __syncthreads();

            // compute gemm
            gemm(mma, tCsX, tCsV, tCrC);
            __syncthreads();
        }
        // compute local max

        // first find max across own registers
        fill(reg_max, -infinity<T>());

        for int (b_reg = 0; b_reg < B_REG_MAX;, ++b_reg) {
            for int (v_reg = 0; v_reg < V_REG_MAX; ++v_reg) {
                reg_max(b_reg) = max(reg_max(b_reg), tCrC(b_reg, v_reg));
            }
        }

        // find max across the tile
        // We want threads to communicate with every other thread in its tileC row



    }


}




template <class CtaTiler, class T,
          class TensorX, class XSmemLayout, class TiledCopyX, 
          class TensorV, class VSmemLayout, class TiledCopyV,
          class TensorY, class YSmemLayout,
          class TensorO, 
          class CSmemLayout, class NSmemLayout, class MSmemLayout,
          class TiledMma>
__global__ void cuFusedCrossEntropyLossFwd(CtaTiler cta_tiler,
                                           TensorX const X, XSmemLayout sX_layout, TiledCopyX copy_x,
                                           TensorV const V, VSmemLayout sV_layout, TiledCopyV copy_v,
                                           TensorY const Y, YSmemLayout sY_layout,
                                           TensorO       O, 
                                           CSmemLayout sC_layout, NSmemLayout sN_layout, MSmemLayout sM_layout,
                                           TiledMma mma) {
    // get appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, _, _);
    Tensor gX = local_tile(X, cta_tiler, cta_coord, Step<_1,X,_1>);     // (BLK_B,BLK_E,e)
    Tensor gV = flat_divide(V, cta_tiler);                              // (BLK_V,BLK_E,v,e)
    Tensor gY = local_tile(Y, cta_tiler, cta_coord, Step<_1,X, X>);     // (BLK_B)
    Tensor gO = local_tile(O, cta_tiler, cta_coord, Step<_1,X, X>);     // (BLK_B)

    // create shared memory buffers for normalizers and max scores
    __shared__ T smemN[cosize_v<NSmemLayout>]; 
    __shared__ T smemM[cosize_v<MSmemLayout>];
    Tensor sN = make_tensor(make_smem_ptr(smemN), sN_layout);
    Tensor sM = make_tensor(make_smem_ptr(smemM), sM_layout);

    computeNormalizer<T>(gX, sX_layout, copy_x,
                         gV, sV_layout, copy_v,
                         sN, sM,
                         sC_layout, mma)

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
    // we can treat batch size and seq len as same dimension since no tokens depend on any other token for
    // this op
    const int batch_size = d.batch_size * d.sequence_len;
    const int vocab_size = d.vocab_size;
    const int embed_size = d.embed_size;

    const T *X = static_cast<const T*>(buffers[0]);
    const T *V = static_cast<const T*>(buffers[1]);
    const int *Y = static_cast<const int*>(buffers[2]);
    T *O = static_cast<T*>(buffers[3]);

    // make full tensors
    Tensor tensor_X = make_tensor(make_gmem_ptr(X), (batch_size, embed_size), LayoutRight{});
    Tensor tensor_V = make_tensor(make_gmem_ptr(V), (vocab_size, embed_size), LayoutRight{});
    Tensor tensor_Y = make_tensor(make_gmem_ptr(Y), batch_size);
    Tensor tensor_O = make_tensor(make_gmem_ptr(O), batch_size);

    // define CTA tile sizes
    auto bB = Int<128>{};
    auto bV = Int<128>{};
    auto bE = Int<  8>{};
    auto cta_tiler = make_shape(bB, bV, bE);

    // define smem layouts
    auto sX = make_layout(make_shape(bB, bE), LayoutRight{});
    auto sV = make_layout(make_shape(bV, bE), LayoutRight{});
    auto sY = make_layout(bB);
    auto sC = make_layout(make_shape(bB, bV), LayoutRight{});
    auto sN = make_layout(bB);
    auto sM = make_layout(bB);


    TiledCopy copyX = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                      Layout<Shape<_32,_8>,Stride<_8,_1>>{},
                                      Layout<Shape<_1,_1>>{});

    TiledCopy copyV = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                      Layout<Shape<_32,_8>,Stride<_8,_1>>{},
                                      Layout<Shape<_1,_1>>{});

    TiledMMA mma = make_tiled_mma(UniversalFMA<T,T,T>{},
                                  Layout<Shape<_16,_16,_1>>{});

    dim3 dimBlock(size(mma));
    dim3 dimGrid(size(ceil_div(batch_size, bB)));

    cuFusedCrossEntropyLossFwd<<<dimGrid, dimBlock>>>
        (cta_tiler, 
         X, sX, copyX,
         V, sV, copyV,
         Y, sY,
         O, 
         sC, sN, sM,
         mma);

    ThrowIfError(cudaGetLastError());
}

}

void fused_ce_loss_fwd_bf16(cudaStream_t stream, void **buffers, const char *opaque,
                       std::size_t opaque_len) {
    apply_fused_ce_loss_fwd<__nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

}
