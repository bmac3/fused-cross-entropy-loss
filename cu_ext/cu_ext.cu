#include <pybind11/pybind11.h>

#include "descriptor.h"
#include "element_types.h"
#include "dispatch/fwd_dispatcher.h"
#include "pybind_helpers.h"


namespace {

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["fused_ce_loss_fwd"] = EncapsulateFunction(fused_ce::fused_ce_loss_fwd_dispatcher);
    return dict;
}

PYBIND11_MODULE(cu_ext, m) {
    m.def("registrations", &Registrations);
    m.def("build_fused_ce_loss_descriptor",
          [](int batch_size, int sequence_len, int vocab_size, int embed_size, 
             fused_ce::ElementType element_type, fused_ce::ElementType reduction_type, fused_ce::ElementType accumulation_type) 
          { return PackDescriptor(fused_ce::Descriptor{
                batch_size, sequence_len, vocab_size, embed_size, element_type, reduction_type, accumulation_type
          }); }
    );

    pybind11::enum_<fused_ce::ElementType>(m, "ElementType")
        .value("BF16", fused_ce::ElementType::BF16)
        .value("F16", fused_ce::ElementType::F16)
        .value("F32", fused_ce::ElementType::F32)
        .value("F64", fused_ce::ElementType::F64);

}


} // namespace
