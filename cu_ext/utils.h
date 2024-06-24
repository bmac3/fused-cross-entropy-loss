#include "cutlass/bfloat16.h"
#include "cutlass/functional.h"

namespace cutlass {


CUTLASS_DEVICE
cutlass::bfloat16_t atomic_binop_bfloat16(cutlass::bfloat16_t *ptr, cutlass::bfloat16_t value, cutlass::bfloat16_t(*bin_op)(cutlass::bfloat16_t, cutlass::bfloat16_t)) {
#if defined(__CUDA_ARCH__)
    unsigned short int* address_as_us = (unsigned short int*)ptr;
    unsigned short int old = *address_as_us, assumed, replace;

    do {
        assumed = old;
        replace = static_cast<unsigned short int>((bin_op(value, cutlass::bfloat16_t::bitcast(assumed))).raw());
        old = atomicCAS(address_as_us, assumed, replace);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return cutlass::bfloat16_t::bitcast(old);
#else
    CUTLASS_UNUSED(ptr);
    CUTLASS_UNUSED(value);
    CUTLASS_NOT_IMPLEMENTED();
    return cutlass::bfloat16_t(0);
#endif
}

template <>
struct atomic_maximum<cutlass::bfloat16_t> {
CUTLASS_DEVICE
float operator()(cutlass::bfloat16_t *ptr, cutlass::bfloat16_t value) const {
    return atomic_binop_bfloat16(ptr, value, [](cutlass::bfloat16_t a, cutlass::bfloat16_t b) { return (a < b ? b : a); });
}
};


template <>
struct atomic_add<cutlass::bfloat16_t> {
CUTLASS_DEVICE
float operator()(cutlass::bfloat16_t *ptr, cutlass::bfloat16_t value) const {
    return atomic_binop_bfloat16(ptr, value, [](cutlass::bfloat16_t a, cutlass::bfloat16_t b) { return a + b; });
}
};


CUTLASS_HOST_DEVICE
cutlass::bfloat16_t exp(cutlass::bfloat16_t const& h) {
#if defined(__CUDACC_RTC__)
  return cutlass::bfloat16_t(expf(float(h)));
#else
  return cutlass::bfloat16_t(std::exp(float(h)));
#endif
}


}
