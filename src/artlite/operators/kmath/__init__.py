"""Math functions for kernel sources, generated per contract: ``km`` in every kernel template (``JINJA_ENV``).

A function's arguments state what its caller needs (lanes, the argument's form, the precision, what arguments below
the range give, inlined or one out-of-line copy per core) and it returns an ``Fn``: the definition to render before
the first use (``{{ e.src }}``) and the name to call (``{{ e }}``):

    {% set exp2 = km.exp2(64, below="zero") %}
    {{ exp2.src }}
    ...
        const aie::vector<float, 64> e = {{ exp2 }}(aie::sub(s, m));

A contract no implementation meets is an error. The definitions render under one stem (``km``: ``kernel_source``):
equal contracts render equal text, so a fused ELF holds one copy whichever operators ask for it, and a core that
runs several of them one body. They assume ``aie::rounding_mode::conv_even``.

An ``Fn`` also carries its contract (``func``, ``params``), its domain and its error bound there (stated in its
comment): tests/operators/test_kmath.py checks them on the NPU against the variant's numpy model
(tests/operators/kmath_sim.py), bit-exact (rope: 99%, the rest within 2 ULP, the model's MACs after cancellations;
the softmax passes through the attention operators' tests).

The package: ``fn`` (``Fn``, the renderer of the templates, the contract checks), ``exponent`` (``exp2``: three forms),
``reciprocal``, ``composite`` (exp, expneg, expm1, softplus, sigmoid: fp32 or a bf16 gate, silu, situ), ``acttable`` (the gather tables' host fit),
``gather`` (the tables' kernels: ``table_pass``, ``act``, ``silu_tanh``), ``rows`` (a head's row:
``rms_norm``, ``rope``), ``softmax`` (a softmax's P in two passes: ``p_pass1``, ``p_pass2``), ``rotate`` (a PARO
rotation's rounds in their tan form: ``rotate``). Each function's C++ is a template file
``<module>.<part>.cc.jinja`` next to its module.
"""

from .acttable import GELU, GELU_ERF, SIGMOID, SILU, TABLES, ActTable, bf16_values, silu64
from .composite import ATANH_C, EXPM1_C, SITU_TANH, exp, expm1, expneg, sigmoid, silu, situ, situ_coefficients, softplus
from .exponent import ACC_C, ACC_Q, CHORD2_C, CHORD2_SCALE, CUBIC, LN2, TAYLOR, exp2
from .fn import KM, LANES, Fn
from .gather import act, silu_tanh, table_pass
from .reciprocal import recip
from .rotate import rotate
from .rows import rms_norm, rope
from .softmax import p_pass1, p_pass2

__all__ = [
    "ACC_C", "ACC_Q", "ATANH_C", "CHORD2_C", "CHORD2_SCALE", "CUBIC", "EXPM1_C", "GELU", "GELU_ERF", "KM", "LANES", "LN2",
    "SIGMOID", "SILU", "SITU_TANH", "TABLES", "TAYLOR", "ActTable", "Fn", "act", "bf16_values", "exp", "exp2", "expm1",
    "expneg", "p_pass1", "p_pass2", "recip", "rms_norm", "rope", "rotate", "sigmoid", "silu", "silu64", "silu_tanh",
    "situ", "situ_coefficients", "softplus", "table_pass",
]
