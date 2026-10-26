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

from ..operators.base import JINJA_ENV
from ..operators.kernel_source import kernel_prefix

MAC = kernel_prefix("mac")  # the helper's name (``kernel_source``: one copy for all operators that render it alike)

_MAC_TPL = JINJA_ENV.from_string(r"""
{% set r, s, t = 8, 8, 8 %}
{% set ks = k // s %}
{% set ct = "float" if f32 else "bfloat16" %}
{% macro kstep(bfill) %}
{#  B pops first on the fill k-steps (pops 0 and 8 trigger fifo_ld_fill), so the
    fill issues ahead of the A-stream ops: keeps RA from spilling on the C chain. #}
    {
      aie::block_vector<bfp16ebs8, {{ r * s }}> A0, A1;
      aie::block_vector<bfp16ebs8, {{ s * t }}> B0, B1;
{% if bfill %}
      B0 = b_stream.pop();
      B1 = b_stream.pop();
      A0 = a_stream.pop();
      A1 = a_stream.pop();
{% else %}
      A0 = a_stream.pop();
      A1 = a_stream.pop();
      B0 = b_stream.pop();
      B1 = b_stream.pop();
{% endif %}
      acc0.mac(A0, aie::op_transpose(B0));
      acc1.mac(A0, aie::op_transpose(B1));
      acc2.mac(A1, aie::op_transpose(B0));
      acc3.mac(A1, aie::op_transpose(B1));
    }
{% endmacro %}
// C (M x N, 8x8 row-major sub-blocks) += A @ B over one {{ k }}-k tile (gemm/mac.py{{ "; B: halves pB (h), pB1" if split else "" }})
template <int M, int N>
static inline __attribute__((always_inline)) void {{ name }}(
    const bfp16ebs8 *__restrict pA, const void *__restrict pB,
{% if split %}
    const void *__restrict pB1,
{% endif %}
    const {{ ct }} __aie_dm_resource_c *__restrict pCld,
    {{ ct }} __aie_dm_resource_c *__restrict pCst{{ ", int h" if split else "" }}{{ ", int nbr" if nbr_arg else "" }}{{ ", dims_3d_t &patA, dims_3d_t &patB, dims_3d_t &patL, dims_3d_t &patS" if pats else "" }}) {
  using AStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, {{ r * s }}, aie_dm_resource::a>;
  using BStream = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, {{ s * t }}, aie_dm_resource::b>;
  using MMUL = aie::mmul<{{ r }}, {{ s }}, {{ t }}, bfp16ebs8>;
  static_assert(M % {{ 2 * r }} == 0 && N % {{ 2 * t }} == 0{{ " && N % " ~ 4 * t ~ " == 0" if split else "" }});
  constexpr int NB = N / {{ 2 * t }}, NBR = M / {{ 2 * r }};  // (16-col blocks, 16-row bands)
  AStream a_stream(pA);
  BStream b_stream((const bfp16ebs8 *)pB);
{% if split %}
  constexpr int NBH = NB / 2;
  // [half][band][block of the half]: A rewinds per block, continues per band, restarts per half; B rewinds per band,
  // then jumps to the other half; C: the half's blocks (own: half h), then the other half's
  dims_3d_t patA(NBH - 1, {{ -2 * ks }}, NBR - 1, 0, {{ -2 * ks }} * NBR);
  dims_3d_t patB(NBH - 1, 0, NBR - 1, -NBH * {{ 2 * ks * 72 }},
                 (int)((const char *)pB1 - (const char *)pB) - NBH * {{ 2 * ks * 72 }});
  const int c3 = -(NBR - 1) * {{ 2 * r }} * N + {{ 2 * r * t }} - 2 * NBH * {{ 2 * r * t }} * h;
  dims_3d_t patL(NBH - 1, {{ 2 * r * t }}, NBR - 1, {{ 2 * r }} * N - (NBH - 1) * {{ 2 * r * t }}, c3);
  dims_3d_t patS(NBH - 1, {{ 2 * r * t }}, NBR - 1, {{ 2 * r }} * N - (NBH - 1) * {{ 2 * r * t }}, c3);
  pCld += NBH * {{ 2 * r * t }} * h;
  pCst += NBH * {{ 2 * r * t }} * h;
{% elif not pats %}
  dims_2d_t patA(NB - 1, {{ -2 * ks }}, 0);
{% endif %}
{% if not split and not pats %}
  dims_2d_t patB(NB - 1, 0, -NB * {{ 2 * ks }});
{% endif %}
{% if not split %}
{% if not pats %}
  int cL = 0, cS = 0;
{% endif %}
{% endif %}
{% if nbr_arg %}
  AIE_LOOP_MIN_ITERATION_COUNT(NB)
  for (int it = 0; it < nbr * NB; it++) {
{% else %}
  for (int it = 0; it < NBR * NB; it++) {
{% endif %}
{% if zero %}
    MMUL acc0(aie::zeros<accfloat, {{ r * t }}>()), acc1(aie::zeros<accfloat, {{ r * t }}>()),
        acc2(aie::zeros<accfloat, {{ r * t }}>()), acc3(aie::zeros<accfloat, {{ r * t }}>());
{% else %}
    MMUL acc0((aie::accum<accfloat, {{ r * t }}>(aie::load_v<{{ r * t }}>(pCld))));
    MMUL acc1((aie::accum<accfloat, {{ r * t }}>(aie::load_v<{{ r * t }}>(pCld + {{ r * t }}))));
{% if c2_bank %}
    MMUL acc2((aie::accum<accfloat, {{ r * t }}>(aie::load_v<{{ r * t }}>((const {{ ct }} __aie_dm_resource_{{ c2_bank }} *)(pCld + N * {{ r }})))));
    MMUL acc3((aie::accum<accfloat, {{ r * t }}>(aie::load_v<{{ r * t }}>((const {{ ct }} __aie_dm_resource_{{ c2_bank }} *)(pCld + {{ r * t }} + N * {{ r }})))));
{% else %}
    MMUL acc2((aie::accum<accfloat, {{ r * t }}>(aie::load_v<{{ r * t }}>(pCld + N * {{ r }}))));
    MMUL acc3((aie::accum<accfloat, {{ r * t }}>(aie::load_v<{{ r * t }}>(pCld + {{ r * t }} + N * {{ r }}))));
{% endif %}
{% endif %}
{% for step in range(ks) %}
    {{ kstep(step % 4 == 0) }}
{%- endfor %}
    aie::store_v(pCst, acc0.template to_vector<{{ ct }}>());
    aie::store_v(pCst + {{ r * t }}, acc1.template to_vector<{{ ct }}>());
    aie::store_v(pCst + N * {{ r }}, acc2.template to_vector<{{ ct }}>());
    aie::store_v(pCst + {{ r * t }} + N * {{ r }}, acc3.template to_vector<{{ ct }}>());
{% if pats %}
    a_stream.seek_byte(patA);
{% else %}
    a_stream.seek(patA);
{% endif %}
{% if split or pats %}
    b_stream.seek_byte(patB);
    pCld = add_3d_ptr(pCld, patL);
    pCst = add_3d_ptr(pCst, patS);
{% else %}
    b_stream.seek(patB);
    pCld = add_2d_ptr(pCld, {{ 2 * r }} * N - (NB - 1) * {{ 2 * r * t }}, NB - 1, cL, {{ 2 * r * t }});
    pCst = add_2d_ptr(pCst, {{ 2 * r }} * N - (NB - 1) * {{ 2 * r * t }}, NB - 1, cS, {{ 2 * r * t }});
{% endif %}
  }
}
""")


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
