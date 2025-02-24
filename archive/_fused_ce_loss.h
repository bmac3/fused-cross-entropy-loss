#include "element_types.h"


namespace fused_ce {


template <ElementType Element_, ElementType ElementReduce_, ElementType ElementAccumulate_>
struct KernelTraits {
    using Element = ElementTypeMapping<Element_>;
    using ElementReduce = ElementTypeMapping<ElementReduce_>;
    using ElementAccumulate = ElementTypeMapping<ElementAccumulate_>;
};


struct Descriptor {
    int batch_size;
    int sequence_len;
    int vocab_size;
    int embed_size;
    ElementType element_type;
    ElementType reduction_type;
    ElementType accumulation_type;

    auto kernel_traits() const {
        return KernelTraits<element_type, reduction_type, accumulation_type>{};
    }
};


} // fused_ce