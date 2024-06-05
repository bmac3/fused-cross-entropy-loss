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


def fused_ce_loss_fwd(xs, vocab):
    return _fused_ce_loss_fwd_p.bind(xs, vocab)


# register lowering
for _name, _value in cu_ext.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform='gpu')


def make_row_major_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _fused_ce_loss_fwd_cuda_lowering(ctx, xs, vocab):
    xs_type = ir.RankedTensorType(xs.type)
    xs_shape = xs_type.shape
    vocab_type = ir.RankedTensorType(vocab.type)
    vocab_shape = vocab_type.shape

    assert vocab_shape[-1] == xs_shape[-1], 'embed_size of vocab does not match embed_size of xs'

    vocab_size, embed_size = vocab_shape
    batch_size, sequence_len, _ = xs_shape

    opaque = cu_ext.build_fused_ce_loss_descriptor(batch_size, sequence_len, vocab_size, embed_size)
    result_shape = (batch_size, sequence_len)

    return custom_call(
        b'fused_ce_loss_fwd_bf16',
        result_types=[ir.RankedTensorType.get(result_shape, ir.BF16Type.get())],
        operands=[xs, vocab],
        backend_config=opaque,
        operand_layouts=make_row_major_layouts(xs_shape, vocab_shape),
        result_layouts=make_row_major_layouts(result_shape)
    ).results
     

mlir.register_lowering(
    _fused_ce_loss_fwd_p, 
    _fused_ce_loss_fwd_cuda_lowering,
    platform='gpu'
)

# define abstract evaluation

def _fused_ce_loss_fwd_abstract_eval(xs, vocab):
    assert xs.shape[-1] == vocab.shape[-1]
    batch_size, sequence_len, _ = xs.shape
    dtype = dtypes.canonicalize_dtype(xs.dtype)
    return core.ShapedArray((batch_size, sequence_len), dtype)
    

_fused_ce_loss_fwd_p.def_abstract_eval(_fused_ce_loss_fwd_abstract_eval)
