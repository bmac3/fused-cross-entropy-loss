from functools import partial

from jax import core, dtypes
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

from . import cu_ext


# create primative
_fused_ce_loss_fwd_p = core.Primitive('fused_ce_loss_fwd')
_fused_ce_loss_fwd_p.multiple_results = False
_fused_ce_loss_fwd_p.def_impl(partial(xla.apply_primitive, _fused_ce_loss_fwd_p))


def fused_ce_loss_fwd(xs, vocab, ys):
    return _fused_ce_loss_fwd_p.bind(xs, vocab, ys)


# register lowering
for _name, _value in cu_ext.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform='gpu')


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): cu_ext.ElementType.BF16,
        ir.F16Type.get(): cu_ext.ElementType.F16,
        ir.F32Type.get(): cu_ext.ElementType.F32,
        ir.F64Type.get(): cu_ext.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def check_shapes(xs_shape, ys_shape, vocab_shape):
    assert xs_shape[0] == ys_shape[0], 'batch size of labels does not match batch size of inputs'
    assert xs_shape[1] == ys_shape[1], 'sequence length of labels does not match sequence length of inputs'
    assert xs_shape[2] == vocab_shape[1], 'embedding size of vocab does not match embedding size of xs'


def make_row_major_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _fused_ce_loss_fwd_cuda_lowering(ctx, xs, vocab, ys):
    xs_type = ir.RankedTensorType(xs.type)
    xs_shape = xs_type.shape
    ys_type = ir.RankedTensorType(ys.type)
    ys_shape = ys_type.shape
    vocab_type = ir.RankedTensorType(vocab.type)
    vocab_shape = vocab_type.shape

    check_shapes(xs_shape, ys_shape, vocab_shape)

    batch_size, sequence_len = ys_shape
    vocab_size, embed_size = vocab_shape

    opaque = cu_ext.build_fused_ce_loss_descriptor(
        batch_size, 
        sequence_len, 
        vocab_size, 
        embed_size
        element_type_to_descriptor_type_mapping(xs_type),
        element_type_to_descriptor_type_mapping(ir.F32Type.get()),
        element_type_to_descriptor_type_mapping(ir.F32Type.get())
    )
    result_shape = ys_shape

    return custom_call(
        b'fused_ce_loss_fwd',
        result_types=[ir.RankedTensorType.get(result_shape, xs_type)],
        operands=[xs, vocab, ys],
        backend_config=opaque,
        operand_layouts=make_row_major_layouts(xs_shape, ys_shape, vocab_shape),
        result_layouts=make_row_major_layouts(result_shape)
    ).results
     

mlir.register_lowering(
    _fused_ce_loss_fwd_p, 
    _fused_ce_loss_fwd_cuda_lowering,
    platform='gpu'
)

# define abstract evaluation

def _fused_ce_loss_fwd_abstract_eval(xs, vocab, ys):
    check_shapes(xs.shape, ys.shape, vocab.shape)
    dtype = dtypes.canonicalize_dtype(xs.dtype)
    return core.ShapedArray(ys.shape, dtype)
    

_fused_ce_loss_fwd_p.def_abstract_eval(_fused_ce_loss_fwd_abstract_eval)
