#pragma once

#include "element_types.h"

namespace fused_ce {

struct Descriptor {
    int batch_size;
    int sequence_len;
    int vocab_size;
    int embed_size;
    ElementType element_type;
    ElementType reduction_type;
    ElementType accumulation_type;
};

} // fused_ce
