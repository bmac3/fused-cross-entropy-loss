#include <pybind11/pybind11.h>
#include "cutlass/bfloat16.h"

#include "fused_ce_loss.h"
#include "fused_ce_loss_fwd.h"
#include "pybind_helpers.h"


namespace {

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["fused_ce_loss_fwd"] = EncapsulateKernel<fused_ce::FusedCELossFwd>();
    return dict;
}

PYBIND11_MODULE(fused_ce, m) {
    m.def("registrations", &Registrations);
    m.def("build_fused_ce_loss_descriptor",
          [](int batch_size, int sequence_len, int vocab_size, int embed_size, 
             fused_ce::ElementType element_type, fused_ce::ElementType reduce_type, fused_ce::ElementType acc_type) 
          { return PackDescriptor(fused_ce::Descriptor{
                batch_size, sequence_len, vocab_size, embed_size, element_type, reduce_type, acc_type
          }); }
    );

    pybind11::enum_<fused_ce::ElementType>(m, "ElementType")
        .value("BF16", fused_ce::ElementType::BF16)
        .value("F16", fused_ce::ElementType::F16)
        .value("F32", fused_ce::ElementType::F32)
        .value("F64", fused_ce::ElementType::F64);
    }

} // namespace
