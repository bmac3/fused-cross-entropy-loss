#include "cutlass/bfloat16.h"
#include "cutlass/half.h"


namespace fused_ce {


enum ElementType { BF16, F16, F32, F64 };


// template <ElementType Type>
// struct ElementTypeMapping_;

// template <>
// struct ElementTypeMapping_<BF16> {
//     using Element = cutlass::bfloat16_t;
// };

// template <>
// struct ElementTypeMapping_<F16> {
//     using Element = cutlass::half_t;
// };

// template <>
// struct ElementTypeMapping_<F32> {
//     using Element = float;
// };

// template <>
// struct ElementTypeMapping_<F64> {
//     using Element = double;
// };

// template <ElementType Type>
//     using ElementTypeMapping = typename ElementTypeMapping_<Type>::Element;


struct Descriptor {
    int batch_size;
    int sequence_len;
    int vocab_size;
    int embed_size;
    ElementType element_type;
    ElementType reduce_type;
    ElementType acc_type;
};


// template <Descriptor desc>
// struct KernelTraits {
//     using Element = ElementTypeMapping<desc.element_type>;
//     using ElementReduce = ElementTypeMapping<desc.reduce_type>;
//     using ElementAccumulator = ElementTypeMapping<desc.acc_type>;
// }


} // fused_ce