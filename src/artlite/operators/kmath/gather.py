"""Activations through the gather LUT: ``table_pass`` (the table's quadratic of a reduced argument) and ``act`` (the
whole activation), over ``acttable``'s tables, and ``silu_tanh`` (templates: ``gather.*.cc.jinja``).

AIE2P's gather (``load_4x64_lo / hi`` = ``vldb.4x64``: four 64-bit table reads per instruction, ldb slot; probed on
the NPU): lane k of an instruction reads table ab (k % 4 < 2) or cd, its odd lanes 32 bytes further; offset bits 5:4
select the entry of a 64-byte group, bits 3:0 are ignored. A table is stored as groups of 4 entries, each group twice
(64 bytes per 4 segments), in two copies (ab, cd) in two 16 KB regions: 32 bytes per segment, 3.2 KB for SiLU (in one
region every gather waits a cycle for a bank: ~1.3x the pass's cycles). Entries: [c2 bf16 | c1 bf16 | c0 fp32]
(``shuffle(.., T16_16x4_lo)`` gives c2 | c1 per 16 lanes, ``T32_16x2_hi`` c0).

SiLU: silu(x) = relu(x) + g(t), g(t) = silu(-t), t = min(|x|, 24) (silu(x) - silu(-x) = x). g is a table of 97
segments of 1/4 over [0, 24], each a quadratic in the offset r from its centre: c0 + c1 r + c2 r^2 (c0 fp32, c1 and c2
bf16; minimax over the segment's bf16 points, in units of the result's bf16 ULP):
    v = bf16(t + 32)              t rounded to the 1/4 grid ([32, 56] is a binade of ULP 1/4)
    r = bf16((32 - v) + t)        exact ((32 - v) first: tiny t stay exact)
    y = bf16(c0 + c1 r + c2 bf16(r^2) + max(x, 0))
    table byte offset = 64 v - 2048 (``bfloat16_to_int(v, 6)``: ``vfloor``)
Max 0.59 bf16 ULP over [-24, inf), 99.84% correctly rounded (below -24: silu(-24), within 1e-9 of the exact value).
The vtanh / vexp2 form: 6.6 ULP; ``silu``: 1.17 ULP at ~16 cycles per element. ~2000 cycles per 4096 elements on the
NPU (the vtanh / vexp2 form: 2682 per 1024), in two loops: t into y (II 2), then the table pass in place (II 14 per 32
lanes; bound 9: 8 table reads, the shuffles; post-RA only: 21, its register allocation fails below). Fewer, longer
loops ran slower: the constant vectors (1, 24, the abs mask, 0, the lanes' table pointers) take 5 of the 12 x
registers.

Sigmoid (``full``: the function itself over [-16, 16], x + 48 in the binade [32, 64] of ULP 1/4: 129 segments, no
symmetry, x clamped to the range): within 0.6 bf16 ULP, ~1 cycle per element (an fp32 exp2 and reciprocal: ~5).
"""

from __future__ import annotations

import numpy as np

from .acttable import ActTable
from .exponent import exp2
from .fn import Fn, bound, check, make


def _walk(walk: tuple[int, int], ptr: str, counter: str) -> dict[str, str]:
    """A pointer's walk over x in groups of ``walk[0]`` vectors of 32 values, ``walk[1]`` values apart (a gate/up C
    tile's gate halves: (2, 128)): its counter's declaration and the step after each vector."""
    count, period = walk
    if count < 1 or period % 32 or period < 32 * count:
        raise ValueError(f"kmath: walk = {walk}: groups of whole vectors (period a multiple of 32, >= 32 x count)")
    if count == 1:
        return {"decl": "", "step": f"{ptr} += {period};"}
    return {"decl": f"    int {counter} = 0;",
            "step": f"{ptr} = add_2d_ptr({ptr}, {period - 32 * (count - 1)}, {count - 1}, {counter}, 32);"}


def _walk_tag(walk: tuple[int, int]) -> str:
    return "" if walk == (1, 32) else f"_w{walk[0]}x{walk[1]}"


def _lut16() -> Fn:
    return make("lut16", "gather.lut16.cc.jinja", variables={}, inline=True, lanes=16, arg="ptr", ret="void")


def table_pass(table: ActTable, *, table_arg: bool = False, walk: tuple[int, int] = (1, 32), n: int | None = None,
               f32: bool = False, copies: int = 2, small: bool = False) -> Fn:
    """The table pass of ``table`` (an ActTable): a noinline template ``<N>(x, t_, t2_, y[, tab_ab, tab_cd])``, y = the
    table's quadratic of t (t_, t2_: t = min(|x|, t_max) as bf16, or x clamped for a full table; may be y) plus relu(x)
    (``f32``: y fp32, not rounded to bf16; t_, t2_ then buffers of their own).
    The table's two copies in two 16 KB regions: static, in the linker's sections of banks 0 and 1 (one section per
    table: a core links only the tables it reads), or (``table_arg``)
    the core's buffers ``tab_ab``, ``tab_cd`` (``table_words`` each; two parameters: llvm-aie gives the gathers a memory
    operand only when each base of their address vector is a pointer of its own: from one pointer and an offset of it,
    each gather orders against every load and store, and the loop does not pipeline). ``walk``: x in groups of
    ``walk[0]`` vectors of 32 values, ``walk[1]`` apart (a GEMM's gate/up C tile [gate 64 | up 64]: (2, 128)). ``n``: a plain function of n values (a multiple of 32), not a template (a kernel of a
    program-memory-patched design must be emitted where it is defined). ``copies`` 1: one static copy (bank 0), for a
    core short of L1 (every gather then waits a cycle for its bank: ~1.3x the pass's cycles). ``small``: the loop
    pipelined post-RA only (a full table's always is), for a core short of program memory (II 21 instead of 14, 112 B
    less)."""
    check("copies", copies, (1, 2))
    if copies == 1 and table_arg:
        raise ValueError("kmath: copies = 1 is a static table's")
    if not isinstance(table, ActTable):
        raise TypeError(f"kmath: table = {table!r}, not an ActTable")
    shift = int(np.log2(16 / table.seg))  # 16 bytes of offset per segment
    words = [f"0x{w:08x}u" for w in table.table_words()]
    l16 = _lut16()
    if n is not None and (n <= 0 or n % 32):
        raise ValueError(f"kmath: a table pass of n = {n} values (a multiple of 32)")
    variables = {"table": table.name, "lut16": l16.name, "M": repr(table.m), "shift": shift, "t_max": table.t_max,
                 "full": table.full, "words": words, "bias": int(table.base * 2 ** shift),
                 "T": "float" if f32 else "bfloat16", "table_arg": table_arg, "walk": walk,
                 "wx": _walk(walk, "px", "cx"), "n": n, "copies": copies, "post": table.full or small}
    local = (f"{table.name}_tab" + ("_t" if table_arg else "") + _walk_tag(walk) + (f"_n{n}" if n else "")
             + ("_f" if f32 else "") + ("_c1" if copies == 1 else "") + ("_s" if small and not table.full else ""))
    params = {"table": table, "table_arg": table_arg, "walk": walk, "n": n, "f32": f32} | \
        ({"copies": copies} if copies != 2 else {}) | ({"small": True} if small and not table.full else {})
    return make(local, "gather.table_pass.cc.jinja", variables=variables, inline=False, deps=(l16,), lanes=32,
                arg="ptr", ret="void", func="table_pass", params=params)


# act's bounds (bf16 ULPs of the exact value) and where they hold (below: tests/operators/test_kmath.py)
_ACT_BOUND = {"silu": (0.59, (-24.0, np.inf)), "sigmoid": (0.54, (-16.0, 16.0)), "gelu": (0.59, (-4.0, np.inf)),
              "gelu_erf": (0.59, (-4.0, np.inf))}


_ACT_REL = {"silu": 7e-4}  # (the same, relative, of ``f32`` results: tests/operators/test_kmath.py)


def act(table: ActTable, *, table_arg: bool = False, walk: tuple[int, int] = (1, 32), n: int | None = None,
        scale: int = 1, clamp: float | None = None, inline: bool = False, f32: bool = False, copies: int = 2,
        small: bool = False) -> Fn:
    """``table``'s activation (an ActTable: SILU, GELU, GELU_ERF, SIGMOID) of N bf16 into y (x, y distinct): a noinline
    template ``<N>(x, y[, tab_ab, tab_cd])``, two loops (t = min(|x|, t_max) into y, then the table pass in place): ~0.5
    cycles per element (SiLU: ~2000 per 4096 on the NPU). ``f32``: ``<N>(x, t, y)``, y fp32 (the quadratic not rounded to
    bf16: the table's fit, ~0.09 bf16 ULP of the exact value, where a fp32 consumer takes ``silu``'s 16 cycles per element
    otherwise), t a bf16 scratch of N values. ``table_arg``, ``walk`` (x's), ``n`` (a plain function of n
    values), ``copies``, ``small``: ``table_pass``'s. ``scale`` 2, ``clamp``: the table fed t = min(2 |x|, clamp) (a table of another function
    on this one's grid: the vision overlay's GELU, ``vision.layers.gelu_table``; its words the caller's buffers).
    ``inline``: the clamp pass inlined into the caller (a call per few thousand values costs ~7 cycles: the PLE's
    0.25%)."""
    check("scale", scale, (1, 2))
    clamp = table.t_max * scale if clamp is None else clamp
    if table.full and (scale, clamp) != (1, table.t_max):
        raise ValueError(f"kmath: {table.name} (a full table) takes x itself, clamped to its range")
    tab = table_pass(table, table_arg=table_arg, walk=walk, n=n, f32=f32, copies=copies, small=small)
    ulp, dom = _ACT_BOUND[table.name] if (scale, clamp) == (1, table.t_max) else (None, (-np.inf, np.inf))
    t_text = (f"x clamped to +-{clamp:g}" if table.full else f"min({scale} |x|, {clamp:g})" if scale > 1
              else f"min(|x|, {clamp:g})")
    bound_text = f"{bound(ulp, None)} for x in [{dom[0]:g}, {dom[1]:g})" if ulp else "the caller's words"
    what = f"{table.name} through its table" if ulp else f"a table on {table.name}'s grid"
    variables = {"what": what, "table": table.name, "tab": tab.name, "clamp": repr(float(clamp)), "full": table.full,
                 "bound": bound_text, "t_text": t_text, "scale": scale, "n": n, "table_arg": table_arg, "walk": walk,
                 "wx": _walk(walk, "px", "cx"), "f32": f32}
    local = (f"{table.name}_lut" + ("_t" if table_arg else "") + _walk_tag(walk) + (f"_n{n}" if n else "")
             + (f"_s{scale}c{clamp:g}" if (scale, clamp) != (1, table.t_max) else "") + ("_i" if inline else "")
             + ("_f" if f32 else "") + ("_c1" if copies == 1 else "") + ("_s" if small and not table.full else ""))
    return make(local, "gather.act.cc.jinja", variables=variables, inline=inline, deps=(tab,), lanes=32, arg="ptr",
                ret="void", func="act",
                params={"table": table, "table_arg": table_arg, "walk": walk, "n": n, "scale": scale, "clamp": clamp,
                       "inline": inline, "f32": f32} | ({"copies": copies} if copies != 2 else {})
                | ({"small": True} if small and not table.full else {}),
                domain=dom, valid=(-3e38, 3e38), ulp=None if f32 else ulp,
                rel=_ACT_REL.get(table.name) if f32 else None)


def silu_tanh(*, walk: tuple[int, int] = (1, 32), inline: bool = True) -> Fn:
    """silu of N bf16 into y (a template ``<N>(x, y, zero)``, ``zero`` a runtime 0) in two pipelined passes through
    vtanh, one vexp2 chord and a fit for |x| < 2.5: ~0.6 cycles per element, up to 38 bf16 ULP below x = -2.5
    (``act(SILU)``: ~0.5 and 0.59 ULP everywhere, its table in L1). ``walk``: x's walk (``table_pass``)."""
    chord = exp2(32, bias=0, prec="chord", inline=True)
    small = make("silu_small", "gather.silu_small.cc.jinja", variables={}, inline=True, lanes=32, arg="bfloat16",
                 ret="bfloat16")
    local = "silu_tanh" + _walk_tag(walk) + ("_i" if inline else "")
    return make(local, "gather.silu_tanh.cc.jinja", inline=inline, deps=(chord, small), lanes=32, arg="ptr",
                ret="void", func="silu_tanh", params={"walk": walk, "zero": True}, domain=(2.5, 3e38),
                valid=(-3e38, 3e38), ulp=0.92,
                variables={"chord": chord.name, "small": small.name, "walk": walk, "wg": _walk(walk, "pg", "cg")})
