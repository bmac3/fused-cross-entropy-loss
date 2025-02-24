#include <pybind11/pybind11.h>
#include "cutlass/bfloat16.h"

#include "fused_ce_loss.h"
#include "fused_ce_loss_fwd.h"
#include "pybind_helpers.h"


namespace {

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["fused_ce_loss_fwd_bf16"] = EncapsulateKernel<fused_ce::FusedCELossFwd>();
    return dict;
}

PYBIND11_MODULE(fused_ce, m) {
    m.def("registrations", &Registrations);
    m.def("build_fused_ce_loss_descriptor",
          [](int batch_size, int sequence_len, int vocab_size, int embed_size) 
          { return PackDescriptor(fused_ce::Descriptor{
                batch_size, sequence_len, vocab_size, embed_size
          }); }
    );
}


} // namespace
