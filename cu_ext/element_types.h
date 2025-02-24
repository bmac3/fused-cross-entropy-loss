#pragma once

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"


namespace fused_ce {

enum ElementType { BF16, F16, F32, F64 };

template <ElementType T>
struct ElementTypeTraits;

template <>
struct ElementTypeTraits<ElementType::BF16> {
    using Type = cutlass::bfloat16_t;
};

template <>
struct ElementTypeTraits<ElementType::F16> {
    using Type = cutlass::half_t;
};

template <>
struct ElementTypeTraits<ElementType::F32> {
    using Type = float;
};

template <>
struct ElementTypeTraits<ElementType::F64> {
    using Type = double;
};


template <ElementType T>
using CutlassType = typename ElementTypeTraits<T>::Type;


} // fused_ce
