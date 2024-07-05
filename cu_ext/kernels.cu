#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cooperative_groups.h>
#include <stdexcept>
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "kernels.h"
#include "kernel_helpers.h"
#include "utils.h"

namespace cu_ext {

using namespace cute;

namespace {

template <class T>
CUTE_HOST_DEVICE 
T infinity() {
    return std::numeric_limits<T>::infinity();
}


// template <class T, class ReduceTiler,
//           class TensorX, class XSmemLayout, class TiledCopyX, 
//           class TensorV, class VSmemLayout, class TiledCopyV,
//           class TensorN, class TensorM,
//           class CSmemLayout, class TiledMma>
// CUTE_DEVICE 
// void computeNormalizer(ReduceTiler reduce_tiler,
//                        TensorX const gX, XSmemLayout sX_layout, TiledCopyX copy_x,
//                        TensorV const gV, VSmemLayout sV_layout, TiledCopyV copy_v,
//                        TensorN sN, TensorM sM,
//                        CSmemLayout sC_layout, TiledMma mma) {
//     // if (thread0()) {
//     //     print(" gX : "); print(gX); print("\n");
//     //     print(" gV : "); print(gV); print("\n");
//     // }

//     // create shared memory buffers for X and V
//     __shared__ T smemX[cosize_v<XSmemLayout>];
//     __shared__ T smemV[cosize_v<VSmemLayout>];
//     __shared__ T smemC[cosize_v<CSmemLayout>];
//     Tensor sX = make_tensor(make_smem_ptr(smemX), sX_layout);       // (BLK_B,BLK_E)
//     Tensor sV = make_tensor(make_smem_ptr(smemV), sV_layout);       // (BLK_V,BLK_E)
//     Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);       // (BLK_B,BLK_V)

//     // partition X and V for copying
//     ThrCopy thr_copy_x = copy_x.get_slice(threadIdx.x);
//     Tensor tXgX = thr_copy_x.partition_S(gX);                       // (CPY,CPY_B,CPY_E,e)
//     Tensor tXsX = thr_copy_x.partition_D(sX);                       // (CPY,CPY_B,CPY_E)

//     ThrCopy thr_copy_v = copy_v.get_slice(threadIdx.x);
//     Tensor tVgV = thr_copy_v.partition_S(gV);                       // (CPY,CPY_V,CPY_E,v,e)
//     Tensor tVsV = thr_copy_v.partition_D(sV);                       // (CPY,CPY_V,CPY_E)

//     // partition X and V for mma
//     ThrMMA thr_mma = mma.get_slice(threadIdx.x);
//     Tensor tCsX = thr_mma.partition_A(sX);                          // (MMA,MMA_B,MMA_E)
//     Tensor tCsV = thr_mma.partition_B(sV);                          // (MMA,MMA_V,MMA_E)
//     Tensor tCsC = thr_mma.partition_C(sC);                          // (MMA,MMA_B,MMA_V)

//     // create accumulator in registers
//     Tensor tCrC = thr_mma.make_fragment_C(tCsC);                    // (MMA,MMA_B,MMA_V)
//     Tensor tCrC2 = thr_mma.make_fragment_C(tCsC);

//     auto V_BLOCK_MAX = size<3>(tVgV);
//     auto E_TILE_MAX = size<3>(tXgX);
//     auto B_REG_MAX = size<1>(tCrC);
//     auto V_REG_MAX = size<2>(tCrC);

//     // register vars to local stats
//     T r_sum;
//     T r_max;

//     // partition C for reduction
//     auto reduce_coord = idx2crd(threadIdx.x, shape(reduce_tiler));
//     Tensor tRsC = local_partition(sC, reduce_tiler, threadIdx.x);
//     Tensor tRsM = local_partition(sM, get<0>(reduce_tiler), get<0>(reduce_coord));
//     Tensor tRsN = local_partition(sN, get<0>(reduce_tiler), get<0>(reduce_coord));

//     // if (thread0()) {
//     //     print(" tVgV : "); print(tVgV); print("\n");
//     //     print(" tXgX : "); print(tXgX); print("\n");
//     //     print(" sC : ");   print(sC);   print("\n");
//     //     print(" tCsC : "); print(tCsC); print("\n");
//     //     print(" tCrC : "); print(tCrC); print("\n");
//     //     print(" reduce_coord : "); print(reduce_coord); print("\n");
//     //     print(" tRsC : "); print(tRsC); print("\n");
//     //     print(" tRsN : "); print(tRsN); print("\n");
//     //     print(" tRsM : "); print(tRsM); print("\n");
//     //     print(" get<0>(reduce_tiler) : "); print(get<0>(reduce_tiler)); print("\n");
//     //     print(" select<0>(reduce_tiler) : "); print(select<0>(reduce_tiler)); print("\n");
//     //     print(" size(tRsC) "), print(size(tRsC)); print("\n");
//     //     print(" V_BLOCK_MAX : "); print(V_BLOCK_MAX); print("\n");
//     //     print(" E_TILE_MAX : "); print(E_TILE_MAX); print("\n");
//     // }

    
//     // TODO: deal with imperfect tiling
//     for (int v_block = 0; v_block < V_BLOCK_MAX; ++v_block) {
//         // clear the accumulator
//         clear(tCrC);
//         for (int e_tile = 0; e_tile < E_TILE_MAX; ++e_tile) {
//             // copy X and V from gmem to smem
//             copy(copy_x, tXgX(_,_,_,e_tile),         tXsX);
//             copy(copy_v, tVgV(_,_,_,v_block,e_tile), tVsV);
//             fill(tCrC2, static_cast<T>(0.f));
//             __syncthreads();

//             // compute gemm
//             gemm(mma, tCsX, tCsV, tCrC);
//             gemm(mma, tCsX, tCsV, tCrC2);
//             __syncthreads();
//             // if (thread0()) {
//             //     print(" tCrC(0) : "); print(tCrC(0)); print("\n");
//             //     print(" tCrC(0) + 1 : "); print(tCrC(0) + static_cast<T>(1.f)); print("\n");
//             //     print(" tXsX(2) : "); print(tXsX(2)); print("\n");
//             //     print(" tVsV(2) : "); print(tVsV(2)); print("\n");
//             //     print(" tCrC2(0) : "); print(tCrC2(0)); print("\n");
//             // }
//         }
//         // copy gemm tile to smem
//         copy(tCrC, tCsC);

//         // initialize normalizer and max values
//         fill(sM, -infinity<T>());
//         r_max = -infinity<T>();
//         fill(sN, static_cast<T>(0.f));
//         r_sum = static_cast<T>(0.f);
//         __syncthreads();

//         // compute local max
//         for (int i = 0; i < size(tRsC); ++i) {
//             r_max = fast_max(r_max, tRsC(i));
//         }
//         cutlass::atomic_maximum<T>{}(&tRsM(0), r_max);
//         __syncthreads();

//         // compute local sum
//         r_max = tRsM(0);
//         for (int i = 0; i < size(tRsC); ++i) {
//             r_sum += exp(tRsC(i) - r_max);
//         }
//         cutlass::atomic_add<T>{}(&tRsN(0), r_sum);
//         __syncthreads();
//     }
// }




template <class T, class CtaTiler, class ReduceTiler,
          class XGmemLayout, class XSmemLayout, class TiledCopyX, 
          class VGmemLayout, class VSmemLayout, class TiledCopyV,
          class YGmemLayout, class YSmemLayout,
          class OGmemLayout, 
          class CSmemLayout, class NSmemLayout, class MSmemLayout,
          class TiledMma>
__global__ void cuFusedCrossEntropyLossFwd(CtaTiler cta_tiler, ReduceTiler reduce_tiler,
                                           T const* X, XGmemLayout mX_layout, XSmemLayout sX_layout, TiledCopyX copy_x,
                                           T const* V, VGmemLayout mV_layout, VSmemLayout sV_layout, TiledCopyV copy_v,
                                           int const* Y, YGmemLayout mY_layout, YSmemLayout sY_layout,
                                           T      * O, OGmemLayout mO_layout,
                                           CSmemLayout sC_layout, NSmemLayout sN_layout, MSmemLayout sM_layout,
                                           TiledMma mma) {
    // get full tensors
    Tensor mX = make_tensor(make_gmem_ptr(X), mX_layout);   // (b,e)
    Tensor mV = make_tensor(make_gmem_ptr(V), mV_layout);   // (v,e)
    Tensor mY = make_tensor(make_gmem_ptr(Y), mY_layout);   // (b)
    Tensor mO = make_tensor(make_gmem_ptr(O), mO_layout);   // (b)

    // get appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, _, _);
    Tensor gX = local_tile(mX, cta_tiler, cta_coord, Step<_1,cute::X,_1>{});          // (BLK_B,BLK_E,e)
    Tensor gV = flat_divide(mV, select<1,2>(cta_tiler));                              // (BLK_V,BLK_E,v,e)
    Tensor gY = local_tile(mY, cta_tiler, cta_coord, Step<_1,cute::X,cute::X>{});     // (BLK_B)
    Tensor gO = local_tile(mO, cta_tiler, cta_coord, Step<_1,cute::X,cute::X>{});     // (BLK_B)

    // create shared memory buffers for normalizers, max scores, and label scores
    __shared__ T smemN[cosize_v<NSmemLayout>]; 
    __shared__ T smemM[cosize_v<MSmemLayout>];

    Tensor sN = make_tensor(make_smem_ptr(smemN), sN_layout);
    Tensor sM = make_tensor(make_smem_ptr(smemM), sM_layout);

    __shared__ T smemX[cosize_v<XSmemLayout>];
    __shared__ T smemV[cosize_v<VSmemLayout>];
    __shared__ T smemC[cosize_v<CSmemLayout>];
    Tensor sX = make_tensor(make_smem_ptr(smemX), sX_layout);       // (BLK_B,BLK_E)
    Tensor sV = make_tensor(make_smem_ptr(smemV), sV_layout);       // (BLK_V,BLK_E)
    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);       // (BLK_B,BLK_V)

    // partition X and V for copying
    ThrCopy thr_copy_x = copy_x.get_slice(threadIdx.x);
    Tensor tXgX = thr_copy_x.partition_S(gX);                       // (CPY,CPY_B,CPY_E,e)
    Tensor tXsX = thr_copy_x.partition_D(sX);                       // (CPY,CPY_B,CPY_E)

    ThrCopy thr_copy_v = copy_v.get_slice(threadIdx.x);
    Tensor tVgV = thr_copy_v.partition_S(gV);                       // (CPY,CPY_V,CPY_E,v,e)
    Tensor tVsV = thr_copy_v.partition_D(sV);                       // (CPY,CPY_V,CPY_E)

    // partition X and V for mma
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsX = thr_mma.partition_A(sX);                          // (MMA,MMA_B,MMA_E)
    Tensor tCsV = thr_mma.partition_B(sV);                          // (MMA,MMA_V,MMA_E)
    Tensor tCsC = thr_mma.partition_C(sC);                          // (MMA,MMA_B,MMA_V)

    // create accumulator in registers
    Tensor tCrC = thr_mma.make_fragment_C(tCsC);                    // (MMA,MMA_B,MMA_V)
    Tensor tCrC2 = thr_mma.make_fragment_C(tCsC);

    auto V_BLOCK_MAX = size<3>(tVgV);
    auto E_TILE_MAX = size<3>(tXgX);
    auto B_REG_MAX = size<1>(tCrC);
    auto V_REG_MAX = size<2>(tCrC);

    // register vars for local stats
    T r_sum;
    T r_max;

    // partition C for reduction
    auto reduce_coord = idx2crd(threadIdx.x, shape(reduce_tiler));
    Tensor tRsC = local_partition(sC, reduce_tiler, threadIdx.x);
    Tensor tRsM = local_partition(sM, get<0>(reduce_tiler), get<0>(reduce_coord));
    Tensor tRsN = local_partition(sN, get<0>(reduce_tiler), get<0>(reduce_coord));

    // if (thread0()) {
    //     print(" tVgV : "); print(tVgV); print("\n");
    //     print(" tXgX : "); print(tXgX); print("\n");
    //     print(" sC : ");   print(sC);   print("\n");
    //     print(" tCsC : "); print(tCsC); print("\n");
    //     print(" tCrC : "); print(tCrC); print("\n");
    //     print(" reduce_coord : "); print(reduce_coord); print("\n");
    //     print(" tRsC : "); print(tRsC); print("\n");
    //     print(" tRsN : "); print(tRsN); print("\n");
    //     print(" tRsM : "); print(tRsM); print("\n");
    //     print(" get<0>(reduce_tiler) : "); print(get<0>(reduce_tiler)); print("\n");
    //     print(" select<0>(reduce_tiler) : "); print(select<0>(reduce_tiler)); print("\n");
    //     print(" size(tRsC) "), print(size(tRsC)); print("\n");
    //     print(" V_BLOCK_MAX : "); print(V_BLOCK_MAX); print("\n");
    //     print(" E_TILE_MAX : "); print(E_TILE_MAX); print("\n");
    // }

    
    // TODO: deal with imperfect tiling
    for (int v_block = 0; v_block < V_BLOCK_MAX; ++v_block) {
        // clear the accumulator
        clear(tCrC);
        for (int e_tile = 0; e_tile < E_TILE_MAX; ++e_tile) {
            // copy X and V from gmem to smem
            copy(copy_x, tXgX(_,_,_,e_tile),         tXsX);
            copy(copy_v, tVgV(_,_,_,v_block,e_tile), tVsV);
            fill(tCrC2, static_cast<T>(0.f));
            __syncthreads();

            // compute gemm
            gemm(mma, tCsX, tCsV, tCrC);
            gemm(mma, tCsX, tCsV, tCrC2);
            __syncthreads();
            // if (thread0()) {
            //     print(" tCrC(0) : "); print(tCrC(0)); print("\n");
            //     print(" tCrC(0) + 1 : "); print(tCrC(0) + static_cast<T>(1.f)); print("\n");
            //     print(" tXsX(2) : "); print(tXsX(2)); print("\n");
            //     print(" tVsV(2) : "); print(tVsV(2)); print("\n");
            //     print(" tCrC2(0) : "); print(tCrC2(0)); print("\n");
            // }
        }
        // copy gemm tile to smem
        copy(tCrC, tCsC);

        // initialize normalizer and max values
        fill(sM, -infinity<T>());
        r_max = -infinity<T>();
        fill(sN, static_cast<T>(0.f));
        r_sum = static_cast<T>(0.f);
        __syncthreads();

        // compute local max
        for (int i = 0; i < size(tRsC); ++i) {
            r_max = fast_max(r_max, tRsC(i));
        }
        cutlass::atomic_maximum<T>{}(&tRsM(0), r_max);
        __syncthreads();

        // compute local sum
        r_max = tRsM(0);
        for (int i = 0; i < size(tRsC); ++i) {
            r_sum += exp(tRsC(i) - r_max);
        }
        cutlass::atomic_add<T>{}(&tRsN(0), r_sum);
        __syncthreads();
    }

    // compute label logits

    // partition threads over 





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

    // define gmem layouts
    auto mX = make_layout(make_shape(batch_size, embed_size), LayoutRight{});
    auto mV = make_layout(make_shape(vocab_size, embed_size), LayoutRight{});
    auto mY = make_layout(batch_size);
    auto mO = make_layout(batch_size);

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

    // define reduction tiler
    auto reduce_tiler = make_layout(make_shape(Int<128>{}, Int<2>{}));

    TiledCopy copyX = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                      Layout<Shape<_32,_8>,Stride<_8,_1>>{},
                                      Layout<Shape<_1,_1>>{});

    TiledCopy copyV = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                      Layout<Shape<_32,_8>,Stride<_8,_1>>{},
                                      Layout<Shape<_1,_1>>{});

    TiledMMA mma = make_tiled_mma(UniversalFMA<T,T,T,T>{},
                                  Layout<Shape<_16,_16,_1>>{});

    dim3 dimBlock(size(mma));
    dim3 dimGrid(size(ceil_div(batch_size, bB)));

    cuFusedCrossEntropyLossFwd<<<dimGrid, dimBlock>>>
        (cta_tiler, reduce_tiler,
         X, mX, sX, copyX,
         V, mV, sV, copyV,
         Y, mY, sY,
         O, mO,
         sC, sN, sM,
         mma);

    ThrowIfError(cudaGetLastError());
}

}

void fused_ce_loss_fwd_bf16(cudaStream_t stream, void **buffers, const char *opaque,
                       std::size_t opaque_len) {
    apply_fused_ce_loss_fwd<cutlass::bfloat16_t>(stream, buffers, opaque, opaque_len);
}

}
