"""Prefill GEMM on the decode overlay's GGUF tiles: C = A @ W^T.

A: (M, K) bf16 activations (M = prompt tokens). W: tiles of the decode
overlay (``formats.pack_tile``: per-row scale planes, then the codes
of the super-blocks), read in place, so prefill and decode share one weight
file. C: (M, N) bf16. What a GEMM's shape makes of its passes, tiles and C objects is ``geometry``, its shim taps and
memtile walks ``dataflow``, its pacing ``pacing``, the streams both designs share ``design_parts``; this module holds
the kernel source, the L1 buffers and the design.

Work split: core row r takes tokens [m r, m r + m), column c
the tiles of its row range, in passes of 4 tiles (64 rows for R = 16). Per
pass the tiles are streamed per K chunk of 2048 as 9 memtile objects of 4 x
2 KB (the scale header, then super-blocks 0..7); each is cut into four 2 KB
L1 objects (k-tiles of 64 k).

The memtile does the transpose the MMUL needs: tile rows store 32-bit words
of 8 consecutive k (word (s, i) = k in [32 i + 8 s, +8) of a super-block), a
bfp16 B panel holds 8 rows x 8 k. The memtile S2MM writes each super-block as
[tile][i][s][row] and the MM2S reads it as [kt][tile][i % 2][s][row] (i = 2 kt
+ i % 2), i.e. every L1 k-tile arrives as 32-byte packed panels in the
MAC loop's B pop order (``mac.py``) [16-row block][k-step][8-row half][row][word].

The core rows of a column dequantize in pairs (rows 2 / 3, 4 / 5): the lower core of a pair converts the pass's
tiles 0, 1 of every k-tile, the upper one tiles 2, 3 (the memtile ``split``s each k-tile object between them), each
into bfp16 panels in its own memory (zip each code byte with 0x43 -> bf16 128 + q, times the expanded group scales
plus -136 d: exactly d (q - 8) in fp32, one rounding) that it hands to its partner through a FIFO (adjacent cores:
shared memory, no DMA). Both run the MAC loop (``mac.py`` split; A as bfp16 panels from DDR) over their own half,
then the partner's, so the pair never streams B from one memory at once, into the L1 C tile (bf16 between
k-tiles). The k-tiles alternate between two C tiles in different 16 KB regions (in one region the MAC loop's C loads
and stores conflict, ~90 per k-tile); the second one (``st``) also collects the Q4_K header at a K chunk's start and
is the SiLU epilogue's scratch (the sum is in the first after every even number of k-tiles). The scale header passes
through the same permutation; ``_hdr`` (Q4_K: ``_hcp`` + ``_hdr_q4k``) expands the core's two tiles' scales into a
workspace. (Every core converting whole k-tiles cost ~980 instead of ~800 cycles per k-tile: 9B qwen35 gate/up 2.19 ->
1.71 ms.) BFP16 tiles hold MMUL operands: the MAC reads the k-tile objects.

Tokens come in chunks of 4 m (m <= 64 per core row, the largest that fits L1
(``_size``); the weights stream once per chunk; the weights reach the cores
in 2 KB k-tile objects so that m = 64 fits); ``nrows = 2`` halves the program
images for short prompts (each op of a fused ELF is a reconfiguration). The core
buffers are pinned (``_l1_layout``): each stream of the MAC loop (A, the
dequantized half, the two C tiles) has its own 16 KB bank region (memory stalls per k-tile at m = 64 with
one C tile: 82 cycles instead of 222 unpinned). At m = 64 the A stream (4.6 KB per k-tile, ~580 cycles of a
64-bit DMA channel) is not the limit: the dequant + MAC take ~850 cycles per k-tile. A comes as bfp16ebs8
panels [M / 8][K / 8][72 B] (half the bytes of bf16, which would bound 32 tokens per core row: 128 KB
per pass; resident in the memtile up to ``dataflow.A_RESIDENT_MAX``, else refetched per pass), ``c_bfp16``
writes the gate/up SiLU output in the same layout for the down GEMM.

Tiles of 1024 columns (``kc=1024``: the Llama and Qwen images): a row tile's K chunks are consecutive tiles
(``kc_group``: in groups of the decode's activation objects for down). A Q4_K tile's header (1 KB)
is half a slice; it is read as a slice-sized unit starting 1 KB before the tile (``hdr_skip``: every
unit the memtile permutes is then one slice of the pass's tiles; ~10% more weight bytes). The
per-pass pattern [chunk][unit][tile][words] has more dimensions than a shim BD (3 + repeat), so the
weights of all passes come as one chained task per column and token chunk (``dataflow.Dataflow.b_chain``).
``gemm_qd.GemmQdOperator`` runs the dequant on one core per column instead of on every core; it
also reads every other dequantized tile through the planar gather (``planar``: Q5_K, Q6_K, Q2_K, Q3_K, and
Q4_0 / Q4_1 tiles of other than 16 rows or of 512 columns; ``tiles`` per pass, ``blocks``, ``c_pack``,
``region_rt``: see GemmQd).

Pacing (``pace``: the GEMM's class, ``pacing.PACE_CLASSES``): ``pacing``.
"""

from collections.abc import Iterator

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, ScratchpadParameter, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

from ..formats import KC, KT, kernel_template, slot
from ..operators import templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.chain import chained_fill
from ..operators.design import COLS, ROWS, SHIM_QUEUE, TILE_BDS, range32, times_i32
from ..operators.kernel_source import kernel_prefix
from ..operators.l1_layout import REGION, align64
from . import design_parts as DP
from .dataflow import Dataflow, drain_runs
from .geometry import B_DEPTH, L1_BUDGET, M_MAX, TILES, GemmSpec, Geometry, L1Error, L1Plan, expose
from .mac import MAC, render_mac
from .pacing import PACE, PACE_CAP, PACE_IDLE, run_paced

MACF = kernel_prefix("macf")  # (the fp32-C MAC helper: ``c_pairs``)

# the dequant kernels of the format (``gemm/<fmt>``) are its slot's fragment (``formats``)
KERNEL_TEMPLATE = kernel_template(templating.source("artlite/gemm/gemm_q.cc.jinja"))


class GemmQOperator(ArtcOperator):
    """C (M, N) = A (M, K) @ W^T for W in decode-overlay GGUF tiles: GemmQ reads the tiles of the tile-major
    permutation (``tile_major``: 16-row Q4_0 / Q4_1 tiles of 2048 or 1024 columns, Q4_K, BFP16); the planar
    path (``planar``: every other dequantized tile) is GemmQdOperator's.

    W is passed as the byte region of the decode weight buffer that holds the
    matrix (preceded by ``hdr_skip`` bytes: 1 KB for Q4_K tiles of 1024 columns): column c's
    tiles start at ``c * col_bytes``. ``out_cols[c][p]`` is the first output
    column of pass p of column c (4 row tiles = 4 R output columns, or 2 R
    with ``silu``: tiles of 8 gate + 8 up rows, output silu(gate) * up).
    ``kc``: the K columns of a tile (default 2048; the images' 1024, Gemma's 512).
    A row tile's K chunks are consecutive tiles in groups of ``kc_group`` (default: all of them), group G of row
    tile j at ``(G n_rt + j) kc_group`` (n_rt: row tiles of the column, ``region_rt``; ``rt_stride`` instead of
    kc_group: the first kc_group chunks of groups of rt_stride, a K part of the image's groups; the passes
    may read past them: a block padded to whole passes reads the next row tiles, of the next block or
    region, whose outputs land in the column's padding). ``blocks``: (first row tile, passes) of
    runs of passes (default one run from row tile 0): e.g. a QKV projection's Q, K and V row tiles,
    each padded to whole passes.
    """

    DQ_CORE = False  # (GemmQdOperator: one dequantizing core per column)
    p = kernel_prefix("gq")  # (the kernels' names: ``kernel_source``)

    def __init__(
        self,
        *,
        fmt: str = "Q4_0",
        rows: int = 16,
        M: int,
        K: int,
        N: int,
        tile_bytes: int,
        col_bytes: int,
        passes: int,
        out_cols: list[list[int]],
        silu: bool = False,
        act: str = "silu",
        nrows: int = ROWS,
        c_bfp16: bool = False,
        ablate: tuple[str, ...] = (),
        kc: int = KC,
        kc_group: int = 0,
        rt_stride: int = 0,
        tiles: int = 0,
        c_pack: bool = False,
        region_rt: int = 0,
        m: int = 0,
        blocks: tuple[tuple[int, int], ...] = (),
        a_cols: tuple[int, int] = (0, 0),
        k_rep: int = 1,
        c_pairs: bool = False,
        groups: int = 1,
        slots: int = 0,
        mc_param: str = "",
        mc_unit: int = 0,
        lines: bool = False,
        pace: int | None = None,
    ):
        """``tiles``: decode tiles per pass (default 64 rows' worth; planar tiles: any count, MMUL-padded
        to a multiple of 16 rows; with ``silu`` whole SiLU groups, ``formats.gu_tiles``, or half groups of
        tiles whose rows divide 64);
        ``c_pack``: C objects of the pass's rows only (row-major, packed from the MAC tile by an epilogue:
        the MMUL padding rows never leave the core; an odd row count gets one zero column more, whole
        32-bit words);
        ``lines``: C rows of whole 64-byte lines where they fit (a row that starts or ends inside a line makes the
        memory read and merge that line: on a machine without a cache in front of DRAM, Krackan Point, 15.6 instead of
        73.2 GB/s, ``artlite-probe``). Packed passes (``line_carry``): each pass writes ``win`` columns from the line
        boundary at or before its first column, the previous pass's last columns (kept in the C object, whose rows the
        memtile reads ``win`` columns of, into the next row) first, stale columns after its own (the next pass's, or
        padding: the caller leaves room for the last window, ``line_window``). Gate/up (``c_parts``): C objects of two
        32-output parts, drained as 576-byte rows;
        ``a_cols`` = (k0, width): A is the columns k0 .. k0 + K of panels ``width`` columns wide (a projection split
        over K: each part a GEMM of its chunks; default (0, K)). ``nrows``: even for dequantized tiles (pairs).
        ``k_rep``: A has k_rep K columns, the weights' K chunks stream k_rep times (BFP16 tiles: an MoE router's rows
        against x as [hi | lo] panels: K' = 2 K). ``c_pairs``: C in fp32, the sums of each pair of adjacent output
        columns (an MoE router's [hi, lo] rows of an expert: its logit), as row-major fp32 objects of n / 2 values (the
        C argument in bf16 units: two per value; ``out_cols`` count them). ``groups``: row-group tiles
        (``decode.image.Block.G``: a tile row holds G rows of kc / G columns, tile i's row r slot j = output row
        G R i + R j + r; A has K / G columns, the K chunks' parts of a slot one after the other): per pass G slots, each a
        GEMM of the A strip with its super-blocks (the strip resent per slot), C objects of ``slots`` slots
        (default: one, two for tiles of an odd row count; GemmQd only). ``mc_param``: the cores run as many token
        chunks as this scratchpad parameter says, in units of ``mc_unit`` tokens (a multiple of nrows m; default
        nrows m): the designs of any M are one (the runtime sequence: M / (nrows m) chunks). ``pace``: the design's pace
        class (``pacing.PACE_CLASSES``; None: unpaced), its level from the ``PACE`` parameter (``pacing``).

        Experiment switches (timing only, wrong results): ``ablate`` "dq" / "mm" compile the
        kernel bodies out ("hdr": GemmQd's Q4_K header expansion; "act": the gate/up epilogue), "ev_dq" / "ev_mm" /
        "ev_kt" / "ev_act" put event0/1 markers around dq, mm, dq..mm of a k-tile or the gate/up epilogue, "noperm"
        drops the memtile transposes, "nosplit" keeps GemmQd's C tiles whole (``csplit``)."""
        spec = GemmSpec(
            fmt=fmt, rows=rows, M=M, K=K, N=N, tile_bytes=tile_bytes, col_bytes=col_bytes, passes=passes,
            out_cols=out_cols, silu=silu, act=act, nrows=nrows, c_bfp16=c_bfp16, ablate=ablate, kc=kc,
            kc_group=kc_group, rt_stride=rt_stride, tiles=tiles, c_pack=c_pack, region_rt=region_rt, m=m,
            blocks=blocks, a_cols=a_cols, k_rep=k_rep, c_pairs=c_pairs, groups=groups, slots=slots,
            mc_param=mc_param, mc_unit=mc_unit, lines=lines, pace=pace)
        self.geo, self.plan = self._size(*Geometry.variants(spec, dq_core=self.DQ_CORE), m=m)
        self.flow = Dataflow.make(self.geo)
        super().__init__()

    def _size(self, plain: Geometry, lines: Geometry | None, *, m: int) -> tuple[Geometry, L1Plan]:
        """The geometry with the tokens per core row (m, or the largest that divides the prompt and fits L1, the weights
        stream once per chunk) and its L1 plan. ``lines``: the variant with whole-line drains, kept where it keeps the
        tokens per core row, the pinned L1 layout, the A strip resident in the memtile and the drains' queue."""

        def fit(g: Geometry) -> list[int]:
            return [m_ for m_ in ((m,) if m else (64, 48, 32, 16))
                    if (m or m_ <= M_MAX) and g.M % (g.nrows * m_) == 0
                    and self._l1_bytes(gm := g.with_m(m_), self._plan_l1(gm)) <= L1_BUDGET]

        geo, ms = (lines or plain), fit(lines or plain)
        if lines is not None:
            plain_ms = fit(plain)
            if not (ms and ms[:1] == plain_ms[:1] and self._lines_ok(plain.with_m(plain_ms[0]), lines.with_m(plain_ms[0]))):
                geo, ms = plain, plain_ms
        if not ms:
            raise L1Error(f"no token count per core row fits L1 ({m or 'any'})")
        geo = geo.with_m(ms[0])
        return geo, self._plan_l1(geo)

    def _plan_l1(self, g: Geometry) -> L1Plan:
        """The L1 choices of the geometry (m chosen): GemmQ has none."""
        return L1Plan()

    def _lines_ok(self, plain: Geometry, lines: Geometry) -> bool:
        """Whether the whole-line variant keeps what the plain one has: the drains' queue (``_drains_fit``), a pinned L1
        layout and a resident A strip (the geometries with the tokens per core row chosen)."""
        flow = Dataflow.make(plain)
        if lines.carry is not None and not self._drains_fit(flow, lines.carry):
            return False
        pinned = self._layout(plain, self._plan_l1(plain)) is not None
        return not ((pinned and self._layout(lines, self._plan_l1(lines)) is None)
                    or (flow.a_res and not Dataflow.make(lines).a_res))

    @staticmethod
    def _drains_fit(flow: Dataflow, carry: tuple[tuple[int, int], ...]) -> bool:
        """Whether whole-line windows (``carry``) keep a column's C drains within its shim channel's task queue
        (``SHIM_QUEUE``: two token chunks in flight, up to 2 drains each, or no more drains than without windows) and
        its tile's ``TILE_BDS`` BDs (the weight fills', an A strip's, the drains' per chunk in flight; ``_sequence``)."""
        cols = flow.c_out_cols(0)
        runs = len(drain_runs([o - cr for o, (cr, _) in zip(cols, carry, strict=True)]))
        if runs <= len(drain_runs(cols)):
            return True
        if runs > 2:
            return False
        chains = [flow.b_chain(c) for c in range(COLS)]
        depth = 1 if any(ch is not None and sum(map(len, ch)) > SHIM_QUEUE for ch in chains) else 2
        fills = max(sum(map(len, ch)) if ch is not None else len(flow.b_taps(c)) for c, ch in enumerate(chains))
        return depth * (fills + 1 + runs) <= TILE_BDS

    def _chunk_passes(self, prm: object, index: bool = False, step: int = 1) -> Iterator[object]:
        """The cores' loop over token chunks and passes: mc chunks, or ``mc_param``'s (prm: its handle); ``index``:
        yields the pass's index (else None); ``step``: one iteration per that many passes."""
        passes = self.passes // step
        if prm is None:
            if index:
                for _ in range_(self.mc):
                    for p in range_(passes):
                        yield p
                return
            for _ in range_(self.mc * passes):
                yield None
            return
        n, scale = prm.read(), (self.mc_unit or self.nrows * self.m) // (self.nrows * self.m)
        assert scale * self.nrows * self.m == (self.mc_unit or self.nrows * self.m), (self.mc_unit, self.nrows, self.m)
        for _ in range32(times_i32(n, scale) if scale > 1 else n):
            for p in range_(passes):
                yield p if index else None

    def _mc_prm(self) -> object:
        """``mc_param``'s handle for the workers (None without it)."""
        return ScratchpadParameter(self.mc_param, np.int32) if self.mc_param else None

    def _pace_vars(self) -> dict[str, object]:
        return {"pace": self.pace, "pace_idle": hex(sum(v << 8 * i for i, v in enumerate(PACE_IDLE))),
                "pace_cap": PACE_CAP}

    def pace_cores(self) -> list[tuple[int, int]]:
        """(column, row) of the cores a paced design paces: the one that sets the array's rate through the FIFOs
        (GemmQ: every core does the same work; column 0's first core: its A row reaches every column, its column's B
        every row)."""
        return [(0, 2)] if self.pace is not None else []

    def _pace_args(self) -> dict[tuple[int, int], tuple]:
        """Per paced core its worker argument ``pc`` (``run_paced``): the PACE parameter, its timestamp buffer, the
        kernels; other cores get None."""
        if self.pace is None:
            return {}
        prm = ScratchpadParameter(PACE, np.int32)
        st_ty = np.ndarray[(16,), np.dtype[np.int32]]
        k0 = Kernel(f"{self.p}_pace0", "kernel.o", [st_ty])
        k1 = Kernel(f"{self.p}_pace1", "kernel.o", [st_ty, np.int32])
        return {(c, r): (prm, Buffer(st_ty, name=f"gq_pace_{c}_{r}"), k0, k1) for c, r in self.pace_cores()}

    def _buffers(self, g: Geometry, plan: L1Plan) -> dict[str, tuple[int, int]]:
        """(count, bytes) of each core buffer with g.m tokens per core row: the FIFO objects A
        (one k-tile), B (one k-tile of the pass's tiles) and C, the MAC target ``acc`` (silu
        only), the scale workspace ``ws``, the core's half of a dequantized k-tile ``X`` (its FIFO to the partner) and
        ``st``, the second C tile (also the collected header, ``Slot.gemm_q_collects``, and the SiLU scratch); BFP16:
        A, B, C, ``acc`` and ``st``, the SiLU scratch. get_mlir_artifact asserts its buffer types against
        this."""
        m = g.m
        if g.pair:
            return {"A": (2, m * 72), "B": (B_DEPTH, 4 * g.b_l1_words),
                    "C": (2, m * g.n_out * 9 // 8 * g.c_parts if g.c_bfp16 else m * g.n_c * 2),
                    "acc": (1 if g.silu else 0, m * g.n * 2),
                    "ws": (1, 1024 * g.nsb * g.ws_planes * g.tq // TILES),
                    "st": (1, max(m * g.n * 2, 16 * g.b_l1_words if slot(g.fmt).gemm_q_collects else 0)),
                    "X": (2, g.n // 2 * KT * 9 // 8)}
        return {"A": (2, m * 72), "B": (B_DEPTH, 4 * g.b_l1_words),
                "C": (2, m * g.n_out * 9 // 8 * g.c_parts if g.c_bfp16 else m * g.n_c * 2),
                "acc": (1 if g.silu or g.c_pairs else 0, m * g.n * (4 if g.c_pairs else 2)),
                "st": (1, max(64, m * g.n if g.silu else 0))}

    def _l1_bytes(self, g: Geometry, plan: L1Plan) -> int:
        """Core data memory of the design with g.m tokens per core row (buffers and the 2 KB stack)."""
        return sum(n * b for n, b in self._buffers(g, plan).values()) + 2048

    def _l1_layout(self) -> dict[str, list[int]] | None:
        """The pinned layout of the chosen geometry (``_layout``)."""
        return self._layout(self.geo, self.plan)

    def _layout(self, g: Geometry, plan: L1Plan) -> dict[str, list[int]] | None:
        """Byte addresses of a core's buffers, or None (automatic placement) if they do not fit.

        Core data memory behaves like four 16 KB regions of two interleaved banks. The MAC
        loop streams A (the A objects), its B operands (the dequantized halves ``X``, or the B objects of BFP16
        tiles) and the C tile(s) every cycle: each gets its own region. The stack is at 0.
        """
        sizes = self._buffers(g, plan)
        target = "acc" if g.silu or g.c_pairs else "C"  # the MAC's C tile
        # (buffer, allowed regions in order of preference)
        if g.pair:  # the MAC streams X (or the partner's), A and the two C tiles: one region each; the dequant
            # reads B and ws and writes X: three regions; the C objects of the epilogue (cold) next to X
            order = [("X", [0]), ("A", [1]), (target, [2]), ("st", [3]), ("ws", [2, 3, 0])]
            order += [("C", [0, 3, 1])] if g.silu else []
            order += [("B", [3, 1, 0, 2])]
        else:
            order = [("st", [0]), ("A", [1]), (target, [2])] + ([("C", [0, 3, 1])] if g.silu or g.c_pairs else [])
            order += [("B", [3])]
        free = [REGION - 2048] + [REGION] * 3
        nxt = [2048, REGION, 2 * REGION, 3 * REGION]
        layout: dict[str, list[int]] = {}
        for name, regions in order:
            n, sz = sizes[name]
            need = n * align64(sz)
            r = next((r for r in regions if free[r] >= need), None)
            if r is None:
                return None
            layout[name] = [nxt[r] + i * align64(sz) for i in range(n)]
            free[r] -= need
            nxt[r] += need
        return layout

    def get_arg_spec(self) -> list[ArgSpec]:
        return [
            ArgSpec("in", (self.M * self.a_K * 9 // 16,), bfloat16),
            ArgSpec("in", (self.w_bytes // 2,), bfloat16),
            ArgSpec("out", (self.M * self.N * 9 // 16,) if self.c_bfp16 else (self.M, self.N), bfloat16),
        ]

    def get_kernel_src(self) -> str:
        # the MAC: B as the dequantized bfp16 panels, or BFP16 tiles
        mac = render_mac(k=KT, split=self.pair)
        if self.c_pairs:
            mac += render_mac(name=MACF, k=KT, f32=True)
        return KERNEL_TEMPLATE.render(p=self.p, mac=f"{MAC}<{self.m}, {self.n}>", m=self.m, n=self.n, nd=self.n, halves=1, k=KT, silu=self.silu, act=self.act,
                                  fmt=self.fmt,
                                  nsb=self.nsb, dqb=False,
                                  c_bfp16=self.c_bfp16, ablate=self.ablate, mac_src=mac,
                                  c_pack=False, carry=None, win=0, c_parts=self.c_parts, pingpong=False, planar=False,
                                  sxp=1, csplit=0, tq=self.tq, pair=self.pair,
                                  c_pairs=self.c_pairs, macf=f"{MACF}<{self.m}, {self.n}>", slots=1,
                                  **self._pace_vars())

    def _sequence(self, a_prods: list, b_prods: list, c_conses: list, ROWS: int) -> object:
        """The runtime sequence (A, W, C): weight fills (``b_taps``), A strips (``a_tap``) of the
        ``ROWS`` token rows and C drains (``c_taps``)."""

        def sequence(A, W, C, _):
            # per token chunk one task group: its B fills (one for all passes, a chained one, or <= 4
            # per-pass ones: 1B down), A strips and C drains (one drain BD per run of evenly spaced
            # pass outputs covers many C objects); at most 2 chunks in flight. A chunk's last drain
            # completing proves its fills completed, so its BDs are recycled then.
            flow = self.flow
            taps = [flow.b_taps(c) for c in range(COLS)]
            chains = [flow.b_chain(c) for c in range(COLS)]
            assert all(ch_ is not None for ch_ in chains) or max(len(t) for t in taps) <= SHIM_QUEUE, \
                "per-pass weight fills beyond a shim queue and no chained fill (passes > MAX_BD_ITER)"
            assert all(ch_ is None or len(ch_) <= SHIM_QUEUE for ch_ in chains), "chained fills beyond a shim queue"
            # a chained weight fill holds one BD per K chunk until its chunk's drains complete
            depth = 1 if any(ch_ is not None and sum(map(len, ch_)) > SHIM_QUEUE for ch_ in chains) else 2
            inflight: list[TaskGroup] = []
            for ch in range(self.mc):
                tg = TaskGroup()
                for c in range(COLS):
                    if chains[c] is not None:
                        for chain in chains[c]:
                            chained_fill(b_prods[c], W, chain, tg)
                        continue
                    for _, tap in taps[c]:
                        b_prods[c].fill(W, tap, group=tg, wait=False)
                for row in range(ROWS):
                    a_prods[row].fill(A, flow.a_tap(ch, row), group=tg, wait=False)
                for c in range(COLS):
                    if self.G > 1:
                        chained_fill(c_conses[c], C, flow.c_chain(ch, c), tg, wait=True)
                        continue
                    for tap in flow.c_taps(ch, c):
                        c_conses[c].drain(C, tap, group=tg, wait=True)
                inflight.append(tg)
                if len(inflight) >= depth:
                    inflight.pop(0).finish()
            for tg in inflight:
                tg.finish()

        return sequence

    def get_mlir_artifact(self) -> Module:
        ROWS = self.nrows  # shadows the module default
        m, n = self.m, self.n
        i32 = np.int32
        df = self.flow
        B_l2_ty = np.ndarray[(df.b_obj,), np.dtype[i32]]
        B_l1_ty = np.ndarray[(df.b_l1,), np.dtype[i32]]
        cb = self.c_bfp16
        A_l2_ty = np.ndarray[(df.a_obj,), np.dtype[bfloat16]]
        A_l1_ty = np.ndarray[(df.a_l1,), np.dtype[bfloat16]]
        no = self.n_out
        c_elems = df.c_l1
        C_l1_ty = np.ndarray[(c_elems,) if cb else (m, no), np.dtype[bfloat16]]
        C_l2_ty = np.ndarray[(ROWS * c_elems,) if cb else (ROWS * m, no), np.dtype[bfloat16]]
        acc_ty = np.ndarray[(m, n), np.dtype[np.float32 if self.c_pairs else bfloat16]]  # MAC target (silu, c_pairs)
        bufs = self._buffers(self.geo, self.plan)
        # the second C tile (dequant: also the collected header) and the SiLU epilogue's scratch (m x n / 2 bf16)
        st_ty = np.ndarray[(bufs["st"][1] // 2,), np.dtype[bfloat16]]

        p = self.p
        k_zero = Kernel(f"{p}_zerof" if self.c_pairs else f"{p}_zero", "kernel.o", [acc_ty])
        k_silu = Kernel(f"{p}_{self.act}", "kernel.o", [acc_ty, C_l1_ty, st_ty, np.int32,
                                                         *[np.int32] * (self.c_parts - 1)]) if self.silu else None
        pair = self.pair
        if pair:
            ws_ty = np.ndarray[(bufs["ws"][1] // 2,), np.dtype[bfloat16]]  # planes x nsb x 128 tq
            X_ty = np.ndarray[(bufs["X"][1],), np.dtype[np.uint8]]  # a core's dequantized half of a k-tile
            k_hdr, k_hdr2 = slot(self.fmt).gemm_q_header(
                self, lambda name, args: Kernel(f"{p}_{name}", "kernel.o", args), B_l1_ty, ws_ty, st_ty)
            k_dq = Kernel(f"{p}_dq", "kernel.o", [B_l1_ty, ws_ty, X_ty, np.int32, np.int32])
            k_mm = Kernel(f"{p}_mm", "kernel.o", [A_l1_ty, X_ty, X_ty, acc_ty, st_ty, np.int32, np.int32])
        else:
            k_hdr2 = None
            k_mm = Kernel(f"{p}_mmf" if self.c_pairs else f"{p}_mm", "kernel.o", [A_l1_ty, B_l1_ty, acc_ty])
        k_pairs = Kernel(f"{p}_pairs", "kernel.o", [acc_ty, C_l1_ty]) if self.c_pairs else None

        # B: shim -> memtile -> the column's cores (broadcast), k-tile objects in dequant / MAC order (pair: segment
        # h of each memtile object to the cores of half h, rows h, h + 2)
        if pair:
            b_l3 = [ObjectFifo(B_l2_ty, name=f"gqB_{c}", depth=2, from_stream_per_cons=df.b_wdims)
                    for c in range(COLS)]
            b_l1 = [b_l3[c].cons().split([df.b_obj // 2 * h for h in range(2)], tile=Tile(c, 1), depths=[B_DEPTH] * 2,
                                          obj_types=[B_l1_ty] * 2, names=[f"gqB1_{c}_{h}" for h in range(2)],
                                          to_stream=[df.b_rdims] * 2)
                    for c in range(COLS)]
        else:
            b_l3, b_l1 = DP.b_chain(df, B_l2_ty, B_l1_ty, B_DEPTH)
        # each core's dequantized half, consumed by its partner (adjacent cores: shared memory, no DMA)
        x_f = {(c, row): ObjectFifo(X_ty, name=f"gqX_{c}_{row}", depth=2) for c in range(COLS) for row in range(ROWS)
               } if pair else {}
        a_l3, a_l1 = DP.a_chain(df, A_l2_ty, A_l1_ty, ROWS)
        # C: the column's cores join in its memtile
        c_l3, c_l1 = DP.c_join(df, C_l2_ty, C_l1_ty, ROWS, 2)

        n_kc, planes = self.n_kc, self.hdr_units
        q4k = k_hdr2 is not None  # (Q4_K: the header slots collected in st, expanded per K chunk)

        silu = self.silu

        c_pairs = self.c_pairs

        parts = self.c_parts  # (C objects of two passes: their outputs in turn)

        def core_fn(pc, prm, bf, af, cf, zero, mm, st, *extra):  # BFP16: the k-tile objects are MMUL panels already
            run_paced(pc, self.pace, lambda b: core_item(prm, b, af, cf, zero, mm, st, *extra), bf)

        def core_item(prm, bf, af, cf, zero, mm, st, *extra):
            for _ in self._chunk_passes(prm, step=parts):  # token chunks x passes
                c = cf.acquire(1)
                for h in range(parts):
                    acc = extra[1] if silu or c_pairs else c
                    zero(acc)
                    for _ in range_(self.k_rep * n_kc * (self.kc // KT)):
                        b = bf.acquire(1)
                        a = af.acquire(1)
                        mm(a, b, acc)
                        af.release(1)
                        bf.release(1)
                    if silu:
                        extra[0](acc, c, st, 0, *([h] if parts == 2 else []))  # (0: the kernel's opaque runtime zero)
                    elif c_pairs:
                        extra[0](acc, c)
                cf.release(1)

        def pair_fn(pc, prm, bf, af, cf, xo, xi, zero, hdr, dq, mm, ws, st, h, *extra):
            run_paced(pc, self.pace, lambda b: pair_item(prm, b, af, cf, xo, xi, zero, hdr, dq, mm, ws, st, h, *extra), bf)

        def pair_item(prm, bf, af, cf, xo, xi, zero, hdr, dq, mm, ws, st, h, *extra):
            for _ in self._chunk_passes(prm, step=parts):  # token chunks x passes
                c = cf.acquire(1)
                for part in range(parts):
                    pass_fn(bf, af, xo, xi, zero, hdr, dq, mm, ws, st, h, c, part, *extra)
                cf.release(1)

        def pass_fn(bf, af, xo, xi, zero, hdr, dq, mm, ws, st, h, c, part, *extra):
            acc = extra[1] if silu else c
            zero(acc)
            for _ in range_(n_kc):
                for pl in range(planes):  # header planes, one k-tile slot per L1 object (this half's tiles)
                    for kt in range_(4):
                        b = bf.acquire(1)
                        hdr(b, st if q4k else ws, *([] if q4k else [pl]), kt)
                        bf.release(1)
                if q4k:
                    extra[-1](st, ws)
                for sb in range_(self.nsb):
                    for kt in range_(4):
                        b = bf.acquire(1)
                        x = xo.acquire(1)
                        dq(b, ws, x, sb, kt)
                        # (the partner may read x now; this core keeps reading it: the FIFO hands the object back
                        # for the next k-tile but one, after the partner's release)
                        xo.release(1)
                        bf.release(1)
                        a = af.acquire(1)
                        y = xi.acquire(1)
                        mm(a, x, y, acc, st, h, kt)  # (the C tiles acc and st in turn: an even count per chunk)
                        xi.release(1)
                        af.release(1)
            if silu:
                extra[0](acc, c, st, 0, *([part] if parts == 2 else []))  # (0: the kernel's opaque runtime zero)

        workers = []
        prm = self._mc_prm()
        pcs = self._pace_args()
        for row in range(ROWS):
            for c in range(COLS):
                pc = pcs.get((c, row + 2))
                st = Buffer(st_ty, name=f"gq_st_{c}_{row}")
                extra = ([k_silu, Buffer(acc_ty, name=f"gq_acc_{c}_{row}")] if self.silu else
                         [k_pairs, Buffer(acc_ty, name=f"gq_acc_{c}_{row}")] if self.c_pairs else [])
                if k_hdr2 is not None:
                    extra.append(k_hdr2)
                if pair:
                    h = row % 2
                    ws = Buffer(ws_ty, name=f"gq_ws_{c}_{row}")
                    args = [pc, prm, b_l1[c][h].cons(), a_l1[row].cons(), c_l1[row][c].prod(), x_f[c, row].prod(),
                            x_f[c, row ^ 1].cons(), k_zero, k_hdr, k_dq, k_mm, ws, st, h, *extra]
                    workers.append(Worker(pair_fn, args, tile=Tile(c, row + 2), stack_size=2048,
                                          dynamic_objfifo_lowering=True))
                    continue
                workers.append(Worker(core_fn, [pc, prm, b_l1[c].cons(), a_l1[row].cons(), c_l1[row][c].prod(), k_zero, k_mm,
                                                 st, *extra],
                                      tile=Tile(c, row + 2), stack_size=2048, dynamic_objfifo_lowering=True))

        module = DP.resolve(self, workers, (a_l3, b_l3, c_l3), ROWS, c_bfp16=cb)
        # the L1 budget / pinned layout describe these buffers
        actual = {"A": DP.nbytes(A_l1_ty), "B": DP.nbytes(B_l1_ty), "C": DP.nbytes(C_l1_ty), "acc": DP.nbytes(acc_ty),
                  "st": DP.nbytes(st_ty), **({"ws": DP.nbytes(ws_ty), "X": DP.nbytes(X_ty)} if pair else {})}
        assert actual == {k: b for k, (_, b) in bufs.items()}, (actual, bufs)
        layout = self._l1_layout()
        if layout is not None:
            DP.pin(module, layout, {"gqA1": "A", "gqB1": "B", "gqC1": "C", "gqX": "X"})
        return module


expose(GemmQOperator, Geometry, "geo")
