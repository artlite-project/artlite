"""The bfp16ebs8 MMUL loop shared by the prefill GEMMs (GemmQOperator, GemmQdOperator).

``render_mac`` emits an ``always_inline`` helper that accumulates one k-tile into
an m x n C stripe of 8x8 row-major sub-blocks (sub-block (rb, cb) at
rb * 8 n + cb * 64): one pipelined loop over all (16-row band, 16-col block)
pairs, four 8x8 accumulators per block (2 A panels x 2 B panels per k-step).
A arrives as bfp16ebs8 panels in pop order [band][k-step][half], B as bfp16ebs8
panels in pop order [16-col block][k-step][half].

II 33 per 16x16 block, one above the resource bound of one MMUL per cycle (the
post-pipeliner misses 32 by latency on the C-load chain; no pragma helps). A
per-band loop with a C pair prefetched across iterations runs at II 42;
this one is near the bound because:
 - every accumulator loads its C sub-block with vlda.conv inside the
   iteration: nothing is carried across iterations in the x registers the
   operand pops need;
 - C loads and C stores walk separate pointers: one pointer used by the early
   loads and the late stores lives ~50 cycles and keeps the next iteration
   from redefining it;
 - pCld and pCst are distinct __restrict parameters (callers pass the same
   stripe): the inlined noalias scopes let the scheduler overlap a block's
   stores with the next blocks' loads. C blocks are disjoint and a block's
   loads reach its stores through the MMUL chain, so this is safe; unlike
   AIE_LOOP_IGNORE_MEM_DEPS it keeps spill/reload ordering (that hint
   miscompiled a k=32 variant whose loop spilled).
The A/B streams persist and are rewound by 2D seeks (the band's A panels are
re-read for every block, B once per band); separate stream base pointers
spilled. Each block pops 2 k / 8 panels per stream, a multiple of 8 (k = 32 or
64), so a stream is exactly drained at every seek.

``split`` (GemmQ's dequant pairs): B in two halves of n / 2 columns at two addresses (the
core's own half and its partner's, in two memories), the own half (blocks of
half h) first: one loop over [half][band][block of the half] with 3-D walks
(A restarts per half; B rewinds per band, then jumps to the other half, in
bytes; C walks the half's blocks, then the other half's). II 33, 563 cycles per
k-tile at m = n = 64 and 864 B of code (two loops of n / 2: II 32 each, but 569
cycles and 1584 B).
"""

from ..operators import templating
from ..operators.kernel_source import kernel_prefix

MAC = kernel_prefix("mac")  # the helper's name (``kernel_source``: one copy for all operators that render it alike)

_MAC_TPL = templating.load("artlite/gemm/mac.cc.jinja")


def render_mac(*, name: str = MAC, k: int, split: bool = False, f32: bool = False, nbr_arg: bool = False,
               zero: bool = False, pats: bool = False, c2_bank: str = "") -> str:
    """C++ of the MAC helper ``name<M, N>(pA, pB, pCld, pCst)`` for an M x N stripe (a template: every stripe shape of
    a translation unit is one definition) and a k-tile of k
    (``split``: ``name<M, N>(pA, pB, pB1, pCld, pCst, h)``, B in halves, the module docstring; ``f32``: an fp32 C stripe,
    II 47 per 16x16 block at k 64 instead of 33; ``nbr_arg``: the stripe's 16-row bands a runtime argument ``nbr``
    (M: their maximum), the same schedule; ``zero``: C := A B^T, pCld unread; ``pats``: the walks of the 16-row
    bands' A (``seek_byte`` after each block: rewind by the block's 2 k / 8 panels, at a band's end the move to the next
    band's first panel, wherever it is), of B (``seek_byte`` after each block: to the next block's panels) and of C
    (``add_3d_ptr``; one iterator per pointer) are the caller's ``patA``, ``patB``, ``patL``, ``patS``: blocks in any
    order, bands from several buffers); ``c2_bank``: the bank the second 8-row band's C loads are annotated with
    (the scheduler's memory model only; the fp32 C at M = 64 with "b": II 41 instead of 43, the same code otherwise)."""
    # (every block pops a multiple of 8 panels per stream)
    assert k % 32 == 0
    assert not (nbr_arg and split) and not (pats and split)
    return _MAC_TPL.render(name=name, k=k, split=split, f32=f32, nbr_arg=nbr_arg, zero=zero, pats=pats,
                           c2_bank=c2_bank)
