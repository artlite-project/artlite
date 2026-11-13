"""The prefill's choices of how a planar projection runs: its passes (``plan_planar`` / ``choose_planar_plan``: tiles per
pass, blocks of passes, the C columns of each), which GEMM operator a chunk of T tokens gets (``gemm_mac_rows``: GemmQ, or
GemmQd on 1 .. 3 MAC rows) and with how many tokens per MAC row (``planar_gemm``), by estimates of their time
(``planar_seconds``). The operators themselves: ``gemm.gemm_q`` and ``gemm.gemm_qd``."""

import itertools
from dataclasses import dataclass

from ..formats import FORMATS, KT, slot
from ..gemm.gemm_qd import GemmQdOperator, k_tile_cycles
from ..gemm.geometry import LINE, GemmSpec, L1Error, line_window
from ..operators.design import COLS


@dataclass(frozen=True)
class PlanarPlan:
    """Passes of a planar projection (``plan_planar``): ``tiles`` row tiles per pass of R rows; ``blocks``
    (first row tile, passes) per block of the column's row tiles (one block if the passes run uniformly
    over all of them); a pass writes ``width`` C columns (its rows, even), a block ``pitches[b]``."""

    tiles: int
    rows: int
    blocks: tuple[tuple[int, int], ...]
    width: int
    pitches: tuple[int, ...]
    region_rt: int  # row tiles of the column's blocks (the passes may read past them)

    @property
    def passes(self) -> int:
        return sum(n for _, n in self.blocks)

    def out_cols(self, col0s: list[int], pitches: list[int]) -> list[list[int]]:
        """Each column's pass outputs: block b's rows at col0s[b] + c pitches[b] (pitches[b] >= self.pitches[b])."""
        out = []
        for c in range(COLS):
            cols = []
            for b, nb in enumerate(self.block_passes):
                cols += [col0s[b] + c * pitches[b] + self.width * p for p in range(nb)]
            out.append(cols)
        return out

    @property
    def block_passes(self) -> list[int]:
        """Passes of each logical block (also when they run as one uniform block)."""
        return [p // self.width for p in self.pitches]

    @property
    def line_pitches(self) -> tuple[int, ...]:
        """Each block's columns with whole-line drains (``GemmQOperator``'s ``lines``, blocks from line boundaries): at
        least to the end of its last pass's window, ``line_window`` columns from that pass's line boundary."""
        carry = tuple((self.width * p % LINE, 0) for n in self.block_passes for p in range(n))
        win = line_window(carry, self.width)
        return tuple(max(pitch, self.width * (n - 1) // LINE * LINE + win)
                     for pitch, n in zip(self.pitches, self.block_passes, strict=True))


def plan_planar(block_rows: list[int], R: int, *, few_passes: bool = False, n_max: int = 128,
                fmt: str = "") -> PlanarPlan:
    """Passes over a column's consecutive blocks of R-row tiles (``block_rows`` rows each, the last tile of a
    block padded: e.g. a QKV projection's Q, K and V heads): one tile count per pass for all blocks, each block
    padded to whole passes (its last pass reads the next row tiles; their outputs land in the block's padding),
    the rows of a block contiguous (an odd row count per pass only for blocks of one pass).
    The count minimizes a port-bound estimate: per pass and k-tile max(A, D port: 9 cycles per MMUL column,
    MAC) at 64 tokens per MAC row, with the A strip refetched from DDR per pass when ``few_passes`` (down);
    ties: fewer blocks, then less padding. ``fmt``: the tile format's own limits (``Slot.pass_tiles``)."""
    nts = [-(-r // R) for r in block_rows]
    best = None
    for t in range(1, 17):
        n = -(-t * R // 16) * 16
        if n > n_max:
            break
        passes = [-(-nt // t) for nt in nts]
        if t * R % 2 and max(passes) > 1:  # (an odd row count packs a zero column: a gap after every pass)
            continue
        if fmt and not slot(fmt).pass_tiles(fmt, R, t):
            continue
        width = t * R + t * R % 2
        uniform = all(nt % t == 0 for nt in nts[:-1])
        blocks = ((0, sum(passes)),) if uniform else tuple(zip(itertools.accumulate([0] + nts[:-1]), passes))
        if len(blocks) > 4:  # (a chained weight fill per block and token chunk: at most 4 queued tasks)
            continue
        cost = sum(passes) * max(1212 if few_passes else 576, 9 * n, 8 * n)
        key = (cost, len(blocks), sum(passes) * width)
        if best is None or key < best[0]:
            best = (key, PlanarPlan(t, R, blocks, width, tuple(p * width for p in passes), sum(nts)))
    assert best is not None
    return best[1]


def choose_planar_plan(block_rows: list[int], R: int, *, few_passes: bool, T: int, fmt: str, K: int, kc_group: int,
                       tile_bytes: int, kc: int) -> PlanarPlan:
    """``plan_planar`` of a projection of T tokens on K inputs: from T = 256 on the passes have at most 80 rows (96
    tokens per MAC row fit L1: the dequant costs more per weight than the MAC, more tokens per weight pass), or up to
    128 rows when ``planar_seconds`` estimates them 15% faster."""
    if T < 256:
        return plan_planar(block_rows, R, few_passes=few_passes, fmt=fmt)
    plans = [plan_planar(block_rows, R, few_passes=few_passes, n_max=n, fmt=fmt) for n in (80, 128)]
    if plans[1] == plans[0]:
        return plans[0]

    def est(plan: PlanarPlan) -> float:  # (a GemmQd of the plan: the arguments that matter for L1 and the estimate)
        nb = len(plan.pitches)
        spec = GemmSpec(fmt=fmt, rows=R, M=T, K=K, N=COLS * sum(plan.pitches), tile_bytes=tile_bytes,
                        col_bytes=plan.region_rt * (K // kc) * tile_bytes, passes=plan.passes,
                        out_cols=plan.out_cols([0] * nb, list(plan.pitches)), kc_group=kc_group, kc=kc,
                        tiles=plan.tiles, region_rt=plan.region_rt, blocks=plan.blocks, c_pack=True)
        return planar_seconds(planar_gemm(spec.operator_kwargs(), T))

    return plans[1] if est(plans[1]) < 0.85 * est(plans[0]) else plans[0]


def planar_gemm(kw: dict[str, object], T: int) -> GemmQdOperator:
    """A planar GemmQd of T >= 256 tokens on 3 MAC rows with the tokens per MAC row that fit L1 and need the least
    time: a weight sweep per 3 m tokens, per k-tile the slower of the dequant and the MAC cores (3B Q6_K down T = 2048
    at 96 instead of 64: 7.39 instead of 9.24 ms, V 1.07 instead of 1.26)."""
    best = None
    for m in (96, 80, 64):
        try:
            cand = GemmQdOperator(mac_rows=3, m=m, **kw)
        except L1Error:
            continue
        sweeps = -(-T // (3 * m))
        cost = (sweeps * k_tile_cycles(cand), sweeps * m)  # (ties: fewer padded tokens)
        if best is None or cost < best[0]:
            best = (cost, cand)
    assert best is not None, "no tokens per MAC row fit L1"
    return best[1]


def planar_seconds(op: GemmQdOperator) -> float:
    """Estimated time of a planar GemmQd on 3 MAC rows: the slower of its compute (``k_tile_cycles`` per pass and
    k-tile of each weight sweep) and its DDR reads at 56 GB/s (the weights once per sweep, the A strip once or, when
    it does not stay in the memtile, once per pass). It ranks O and down plans as measured (it underestimates both by
    ~1.3x)."""
    sweeps = op.M // (op.nrows * op.m)
    compute = sweeps * op.passes * op.n_kc * (op.kc // KT) * k_tile_cycles(op) / 1.81e9
    weights = COLS * op.passes * op.pass_fill_words * 2  # (bytes of a sweep)
    a = op.M * op.K * 9 / 8 * (1 if op.flow.a_res else op.passes)
    return max(compute, (sweeps * weights + a) / 56e9)


def gemm_mac_rows(T: int, which: str, fmt: str, planar: bool = False) -> int:
    """MAC core rows of a GemmQd (one dequantizing core per column, ``gemm_qd``) for a prefill GEMM of
    T tokens on the decode tiles, or 0 for a GemmQ (pairs of cores split each k-tile's dequant). Measured on Llama-3.2-3B Q4_K:
    at T = 2048 pair GemmQ wins gate/up and O (6.72 / 1.43 vs 7.08 / 1.59 ms each alone; prefill of 2048 tokens:
    1B Q4_0 241 -> 233 ms, 3B 656 -> 647, 8B 1558 -> 1482), GemmQd QKV (2.45 vs 2.64); down (its A strip, refetched
    per weight pass) is faster on GemmQ's 4 x 64 tokens per weight pass, and at T = 256 GemmQ streams the weights once
    (GemmQd twice). At T <= 128 every GEMM is a GemmQ (GemmQd's smaller programs pay off when configurations write
    program memory by blockwrites; with the control packets of ``fused.ctrlpkt`` GemmQ's faster passes win: chunks of
    64 tokens 3B Q4_K_M 69.7 -> 62.6 ms, 8B 135.8 -> 119.5, of 128 tokens 1B Q4_0 -1.4 ms). Planar tiles: GemmQd only
    (2 or 3 MAC rows at T <= 128: no faster)."""
    if planar:
        return 1 if T <= 64 else 2 if T <= 128 else 3
    if not FORMATS[fmt].dequant or T <= 256 or which == "down":  # (BFP16: no dequant)
        return 0
    return 0 if which in ("gu", "o") else 3
