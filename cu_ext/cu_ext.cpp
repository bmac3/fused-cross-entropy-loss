#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace cu_ext;

namespace {
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["fused_ce_loss_fwd_bf16"] = EncapsulateFunction(fused_ce_loss_fwd_bf16);
    return dict;
}

PYBIND11_MODULE(cu_ext, m) {
    m.def("registrations", &Registrations);
    m.def("build_fused_ce_loss_descriptor",
          [](int batch_size, int sequence_len, int vocab_size, int embed_size) 
          { return PackDescriptor(FusedCELossDescriptor{batch_size, sequence_len, vocab_size, embed_size}); }
    );
}
}
