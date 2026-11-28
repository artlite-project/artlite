"""A softmax's P = 2^(s - m) in two passes over a bias-192 exp2's parts (``exp2(32, bias=192)``: ``reduce``, A -> P, f;
``poly``, P, f -> 2^x), each in a separate pipelined loop (in one loop the exp's constants spill; the decode's octets: II 8
+ 10, the prefill's 308 cycles per group): ``p_pass1`` stores P and f, ``p_pass2`` turns pairs of 32-lane records of
them into the MMUL's P panels (bfp16, or hi and lo panels: the exact split) and their row sums. Templates:
``softmax.*.cc.jinja``."""

from __future__ import annotations

from .fn import KM, Fn, check, make


def _e192(e192: Fn) -> None:
    if e192.func != "exp2" or e192.params.get("bias") != 192 or e192.lanes != 32:
        raise ValueError(f"kmath: {e192.name}: a 32-lane exp2 of bias 192 (its reduce / poly parts)")


def p_pass1(e192: Fn, *, layout: str = "split", track: bool = False, inline: bool = False) -> Fn:
    """Pass 1 of N steps of 32 scores A = s - m + 192 (fp32): ``e192``'s reduce, its P and f stored as bf16. A
    template ``<N>``: ``layout`` "split" ``(sl, pP, pF)``, P and f to two arrays of 32 N; "inplace" ``(s)``, a step's
    [P 32 | f 32] written over the step's 32 fp32 scores (P stored first). ``track``: returns whether every bf16(A) < 209 (s < m + 16:
    the row maxima need no raise)."""
    _e192(e192)
    check("layout", layout, ("split", "inplace"))
    if track and layout == "inplace":
        raise ValueError("kmath: p_pass1 track = True is the split layout's")
    local = f"pp1_{layout}" + ("_t" if track else "") + "__" + e192.name.removeprefix(KM + "_")
    return make(local + ("_i" if inline else ""), "softmax.pass1.cc.jinja", inline=inline, deps=(e192.parts["reduce"],),
                lanes=32, arg="ptr", ret="bool" if track else "void", func="p_pass1",
                params={"e192": e192, "layout": layout, "track": track},
                variables={"reduce": e192.parts["reduce"].name, "split": layout == "split", "track": track})


def p_pass2(e192: Fn, *, layout: str = "split", lsum: int | None = None, lo: int | None = None,
            rounding: bool = False, inline: bool = False) -> Fn:
    """Pass 2 of N steps, each two 32-lane records (P, f of ``p_pass1``) -> ``e192``'s poly -> one bfp16 P panel pushed
    at pp (``lo``: hi and lo panels, the lo ones from pp + ``lo`` bytes: the exact split on the MMUL). A template
    ``<N>``: ``layout`` "split" ``(pP, pF, [l,] pp)``, the records at pP / pF + 32 u; "inplace" ``(p0, p1, [l,] pp)``,
    the two records' [P 32 | f 32] steps of two streams (with ``lsum``, without ``lo``: the decode cores' form; without
    lsum its panels came out wrong on the NPU). ``lsum``: l += P per record, l0 at l and l1 at l + ``lsum`` (fp32, kept
    in registers over the pass). ``rounding``: sets conv_even (else the caller has)."""
    _e192(e192)
    check("layout", layout, ("split", "inplace"))
    if layout == "inplace" and (lsum is None or lo is not None):
        raise ValueError("kmath: p_pass2 inplace takes lsum and no lo (the decode cores' form)")
    local = (f"pp2_{layout}" + (f"_l{lsum}" if lsum is not None else "") + (f"_h{lo}" if lo is not None else "")
             + ("_r" if rounding else "") + "__" + e192.name.removeprefix(KM + "_"))
    return make(local + ("_i" if inline else ""), "softmax.pass2.cc.jinja", inline=inline, deps=(e192.parts["poly"],),
                lanes=32, arg="ptr", ret="void", func="p_pass2",
                params={"e192": e192, "layout": layout, "lsum": lsum, "lo": lo, "rounding": rounding},
                variables={"poly": e192.parts["poly"].name, "split": layout == "split", "lsum": lsum, "lo": lo,
                           "rounding": rounding})
