"""GemmQ with one dequantizing core per column: C = A @ W^T on the decode overlay's GGUF tiles.

GemmQOperator (gemm_q.py) dequantizes each k-tile of weights on the cores of a column: the four
cores hold different tokens but the same weights, so the dequant (and the Q4_K header expansion)
runs twice per weight (its pairs of cores split each k-tile). Here core row 2 of each column dequantizes the
column's weights once, into
bfp16ebs8 panels in the MAC's pop order, and broadcasts them (one k-tile object of n x 64
weights per k-tile) to the column's MAC cores, rows 3 .. 2 + ``mac_rows`` (``mac_rows`` x m
tokens per weight pass). A MAC core runs only the bfp16 MAC loop (II 32: one MMUL per cycle) and
the gate/up SiLU epilogue. Data movement otherwise as GemmQ (``dataflow``, ``b_taps``, ``a_tap``,
``c_taps``): the weights reach the dequantizing core instead of all cores, A per MAC row through
memtile 2 row, C joined from the MAC cores.

Planar tiles (``geometry.planar``: Q5_K, Q6_K, Q2_K, Q3_K, Q4_0 / Q4_1 of other than 16 rows or of 512 columns) of
R rows: the memtile gathers a k-tile's words of the pass's ``tiles`` x R rows from the tiles' 32-byte planes
(``TileFormat.planes``, plane- or row-major slices) into [plane][word][row] (word-major Q2_K / Q3_K tiles: [word]
[row]), zero-padded to the MMUL's n; the
header units (slice-sized, from ``hdr_skip`` bytes before the tile when the header is shorter) go through the
same gather: the dequantizing core collects their four k-tile slots (``_hcp``) and expands a super-block's
scales before its k-tiles (``_hdr_sb``; Q6_K: d once per K chunk, ``_hdr_d``; Q5_K: then per k-tile the two
groups' scales per MAC lane, ``_expand``). A column's projection runs in passes of whole tiles
(``prefill.planar.plan_planar``: blocks of passes, the last tile of a block ends in padding rows), ``c_pack``: the MAC cores
copy the pass's rows out of the MMUL tile in 16-byte items (``_pack``; the memtile drops the columns past the
pass's rows), or with ``lines`` behind the previous pass's last columns, so that each pass drains whole 64-byte lines
(``geometry.line_carry``). gate/up: a pass is one SiLU group (128 rows and its padding rows: dropped by the memtile's read
when every part of the tile maps row r to gathered row r, else skipped by the dequant) in two halves of 64 rows
(``halves``: two D objects per k-tile, a C tile of 64 columns each on the MAC cores, so that m = 96 fits L1: 3B
Q5_K at T = 256 one weight sweep instead of two, gate/up 1.88 -> 1.11 ms), or half a group when R divides 64
(``pingpong``)."""

import itertools
from dataclasses import replace

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ..formats import FORMATS, KC, KT, slot
from ..operators import kmath
from ..operators.design import COLS
from ..operators.kernel_source import kernel_prefix
from ..operators.l1_layout import REGION, align64, pin_l1_buffers
from .gemm_q import KERNEL_TEMPLATE, GemmQOperator
from .geometry import B_DEPTH, L1_BUDGET, Geometry, L1Plan, expose, pack_plan, planar
from .mac import MAC, render_mac
from .pacing import run_paced

D_DEPTH = 2  # dequantized k-tile objects (producer and each consumer)
DQ_ONLY = "dq_only"  # (a key of a layout that pins the dequantizing core alone: ``GemmQdOperator._dq_layout``)
MAC_STACK = 0x400  # a MAC core's stack (its deepest call: the gate/up epilogue, ~320 B with the control code)
# Planar MAC cores' buffers by region (``GemmQdOperator._planar_layout``): A0 / A1 / D0 / D1 the A and D objects, T0 / T1
# the C tiles the MAC accumulates in (acc and the second half's or the pingpong's accb; the C objects when there is no
# acc), the stack before region 0's; the other buffers first-fit after them. A window streams A, D and a C tile while
# the DMA writes the next A and D: no two streams of a window in one region, and the DMA's writes into few of them.
PLANAR_PLANS: dict[str, tuple[tuple[str, ...], ...]] = {
    "crossed": (("A0", "D1"), ("A1", "D0"), ("T0",), ()),  # one C tile
    "separate": (("A0", "A1"), ("D0", "D1"), ("T0",), ()),
    "g4": (("A0", "A1"), ("T0", "D1"), ("T1",), ("D0",)),  # two (half 0: D0 into acc, half 1: D1 into accb)
    "g1": (("A0", "A1"), ("T0",), ("T1",), ("D0", "D1")),
    "g1s": (("A0", "A1"), ("T0", "T1"), ("T0h", "T1h"), ("D0", "D1")),  # two, each in halves (``csplit``)
}


class GemmQdOperator(GemmQOperator):
    """C (M, N) = A (M, K) @ W^T on decode tiles of a dequantized format (Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / Q2_K /
    Q3_K), the dequant on one core per column (row 2), ``mac_rows`` MAC cores (rows 3 ..). Arguments as GemmQOperator
    (``nrows`` is ``mac_rows``); planar tiles: see the module docstring."""

    DQ_CORE = True
    p = kernel_prefix("gqd")
    mac_stack = MAC_STACK  # (the MAC cores' Worker stack: the pinned layouts start after it)

    def __init__(self, *, mac_rows: int = 3, m: int = 64, pingpong: bool | None = None, **kw: object):
        """``M``: the tokens the caller uses; the operator runs ``M`` padded to a multiple of
        mac_rows x m (its A and C arguments have that many rows; the padding rows are computed from
        whatever A holds there and never matter). Gate/up (``silu``): the MAC cores' k-tiles
        alternate between two C tiles in different L1 regions (``pingpong``) (in one region the MAC loop's C
        loads and stores conflict: 90 bank conflicts per k-tile; gate/up T = 2048 7.70 -> 7.50 ms). The planar MAC
        cores' L1 placement: the first of ``PLANAR_PLANS`` that fits, else aiecc's. ``pingpong``: as said (default as
        above)."""
        assert 1 <= mac_rows <= 3 and "nrows" not in kw and m > 0 and m % 16 == 0
        # (planar SiLU passes of a whole group: two C tiles of its halves, no pingpong)
        rows = kw.get("rows", 16)
        self._want_pingpong = bool(kw.get("silu")) and (not planar(kw.get("fmt", "Q4_0"), rows, kw.get("kc", KC))
                                                        or kw.get("tiles", 0) * rows == 64) if pingpong is None else pingpong
        self.T = kw.pop("M")
        step = mac_rows * m
        super().__init__(nrows=mac_rows, M=-(-self.T // step) * step, m=m, **kw)
        assert self.dequant, "GemmQd: dequantized tile formats"
        assert not self.pingpong or self.n_kc * (self.kc // KT) % 2 == 0
        # two C tiles (a pass's halves), each in halves 16 KB apart (``csplit``, plan g1s): the MAC loop's blocks
        # alternate regions, so a block's C stores never meet the next block's C loads (one region per tile: a
        # conflict per C access)
        if self.halves == 2 and self.planar and self.m % 32 == 0 and "nosplit" not in self.ablate:
            split = replace(self.plan, csplit=REGION)
            if self._planar_layout(self.geo, split) is not None:
                self.plan = split

    def pace_cores(self) -> list[tuple[int, int]]:
        """(column 0's dequantizing core and its first MAC core: whichever of the two kinds is the slower sets the rate,
        and a paced core with slack would idle into its waits)"""
        return [(0, 2), (0, 3)] if self.pace is not None else []

    def _buffers(self, g: Geometry, lp: L1Plan) -> dict[str, tuple[int, int]]:
        """(count, bytes) of the MAC core's buffers (the dequantizing core's: ``_dq_buffers``): A, the
        dequantized k-tile D (bfp16 panels), C, the MAC target ``acc`` and the SiLU scratch ``sx``
        (silu only)."""
        m, nm = g.m, g.nm
        return {"A": (2, m * 72), "D": (D_DEPTH, nm * KT * 9 // 8),
                "C": (lp.c_depth, m * g.n_out * 9 // 8 * g.c_parts if g.c_bfp16 else
                      (m * g.n_c + g.c_tail) * 2),
                "acc": ((2 if lp.pingpong or g.halves == 2 or g.slots == 2 else 1) if g.silu or g.c_pack
                        else 0, m * nm * 2),
                "sx": (1 if g.silu else 0, m * nm // lp.sxp),
                "lut": (1 if g.silu and g.act == "gelu" else 0, kmath.GELU.table_words().nbytes),
                "lutb": (1 if g.silu and g.act == "gelu" else 0, kmath.GELU.table_words().nbytes)}

    def _plan_l1(self, g: Geometry) -> L1Plan:
        """The L1 choices of the geometry (g.m chosen): the dequantizing core's weight objects (``_dq_pick``), the C
        depth (2 if it fits) and the SiLU scratch's parts (``sxp``: the epilogue in 2 parts of m / 2 rows if the whole
        one does not fit). (Trading the C depth for a pinned planar layout was not consistent: 1B Q3_K_L's O at
        T = 256 -20%, at T = 2048 +8%.)"""
        # (two C objects and the scratch in two parts beat one C object, 1.19 vs 1.23 ms on Gemma's gate/up at T = 256;
        # planar: of the first C depth in budget, the scratch's parts whose pinned layout fits, else the first in
        # budget (aiecc's placement))
        # (whole-line drains: one C object, which keeps the carry for the next pass, ``line_carry``, or holds two parts,
        # taken after the MAC loop)
        auto = ((1, 1), (1, 2)) if g.carry is not None or g.c_parts == 2 else \
            ((2, 1), (2, 2), (1, 1), (1, 2)) if g.silu else ((2, 1), (1, 1), (1, 2))
        base = L1Plan(b_depth=self._dq_pick(g), pingpong=self._want_pingpong)  # (the layouts place the weight objects too)
        cands = [replace(base, c_depth=cd, sxp=sxp) for cd, sxp in auto]
        fits = [c for c in cands if super()._l1_bytes(g, c) <= L1_BUDGET]
        pinned = [c for c in fits if g.planar and c.c_depth == fits[0].c_depth and self._planar_layout(g, c) is not None]
        return (pinned or fits or cands[-1:])[0]

    def _l1_bytes(self, g: Geometry, lp: L1Plan) -> int:
        """Data memory of the larger role (MAC core: ``_buffers``, dequantizing core: ``_dq_buffers``), the
        2 KB stacks included."""
        return max(super()._l1_bytes(g, lp), self._dq_bytes(g, lp.b_depth))

    def _dq_pick(self, g: Geometry) -> int:
        """The dequantizing core's weight objects: two, or one if two do not fit (e.g. Q6_K SiLU groups of 2048
        columns: its weights then arrive between dequant calls)."""
        return next((d for d in (B_DEPTH, 1) if self._dq_bytes(g, d) <= L1_BUDGET), 1)

    def _dq_bytes(self, g: Geometry, b_depth: int) -> int:
        """The dequantizing core's data memory with ``b_depth`` weight objects, stack included."""
        return sum(n * b for n, b in self._dq_buffers(g, b_depth).values()) + 2048

    def _dq_buffers(self, g: Geometry, b_depth: int) -> dict[str, tuple[int, int]]:
        """The dequantizing core's buffers: the weight objects B, its output objects D, and the format's
        (``Slot.qd_scratch``): the scale workspace ``ws``, the header slots ``hb`` (Q5_K: ``ex``)."""
        return {"B": (b_depth, 4 * g.b_l1_words), "D": (D_DEPTH, g.nm * KT * 9 // 8),
                **slot(g.fmt).qd_scratch(g)}

    def _layout(self, g: Geometry, lp: L1Plan) -> dict[str, list[int]] | None:
        """Byte addresses of both roles' buffers (D at the same addresses on both): each stream of
        the hot loops in its own 16 KB region (two interleaved banks) -- MAC core: A (1), D (2), the
        C tile (3), the stack and the epilogue's buffers (0); dequantizing core: B (1), D (2), the
        scale workspace (3), the header slots next to the stack (0); planar tiles: ``_planar_layout``. None: aiecc's
        placement."""
        if g.planar:
            return self._planar_layout(g, lp) or self._dq_layout(g, lp)
        if lp.c_depth == 1:
            return None
        mac, dq = self._buffers(g, lp), self._dq_buffers(g, lp.b_depth)

        def place(base: int, n: int, size: int) -> list[int]:
            step = align64(size)
            return [base + i * step for i in range(n)]

        lay = {"A": place(REGION, *mac["A"]), "D": place(2 * REGION, *mac["D"]), "B": place(REGION, *dq["B"]),
               "ws": place(3 * REGION, *dq["ws"]), "hb": place(0x0800, *dq["hb"])}
        lo = self.mac_stack  # region 0 after the stack: the MAC core's cold buffers
        if g.silu:  # the C tiles in regions 3 and 0, the SiLU scratch after the A objects
            lay["acc"], lay["accb"] = [3 * REGION], [lo]
            lo += align64(mac["acc"][1])
            lay["C"] = place(lo, *mac["C"])
            lo = lay["C"][-1] + align64(mac["C"][1])
            lay["sx"] = [lay["A"][-1] + align64(mac["A"][1])]
        else:  # the MAC writes C objects in turn: one in region 3, the other after the stack
            lay["C"] = [3 * REGION, lo]
        ends = {k: v[-1] + align64(dict(mac, **dq, accb=mac["acc"])[k][1]) for k, v in lay.items()}
        if any(e > (b // REGION + 1) * REGION for k, b, e in ((k, v[0], ends[k]) for k, v in lay.items())
               if not (k == "C" and not g.silu)):
            return None
        return lay

    def _planar_layout(self, g: Geometry, lp: L1Plan) -> dict[str, list[int]] | None:
        """Planar tiles: the MAC cores' buffers by ``PLANAR_PLANS`` (crossed, else separate, for one
        C tile, g4, else g1, for a pass's two halves, g1 for pingpong; g1s for C tiles in halves, ``csplit``), the other
        buffers first-fit in regions 3 .. 0; the dequantizing core's D objects
        at the MAC cores' addresses, its other buffers first-fit around them. None if nothing fits. Measured at T = 2048
        against aiecc's placement (alternating runs): 3B Q2_K down (Q3_K, m 64, n 128, crossed) 5.74 -> 5.15 ms (aiecc
        put A and D of one k-tile in one region: the MAC cores' memory stall 1.58 -> 0.62 ms), gate/up (g4) 7.41 -> 7.13,
        3B Q3_K_M gate/up 7.32 -> 6.99, 1B Q3_K_L 4.99 -> 4.78; 3B Q5_K (crossed) unchanged."""
        two = g.halves == 2 or lp.pingpong or g.slots == 2
        if lp.csplit:
            return self._place_plan(g, lp, "g1s", two)
        # (pingpong: both C tiles in every window, D alternating as A: g4's acc and D1 would meet)
        auto = ["g1"] if lp.pingpong else ["g4", "g1"] if two else ["crossed", "separate"]
        for name in auto:
            if ("T1" in sum(PLANAR_PLANS[name], ())) == two and (lay := self._place_plan(g, lp, name, two)) is not None:
                return lay
        return None

    def _place_plan(self, g: Geometry, lp: L1Plan, plan: str, two: bool) -> dict[str, list[int]] | None:
        """The MAC cores' buffers by ``PLANAR_PLANS[plan]`` and the dequantizing core's around them."""
        m, mac = g.m, self._buffers(g, lp)
        tiles = [("acc", 0), ("accb", 0)] if two else [("acc", 0)] if mac["acc"][0] else [("C", 0)]
        sizes = {k: align64(v[1]) for k, v in {**mac, "accb": mac["acc"]}.items()}
        if lp.csplit:  # (the tiles' first halves acc, accb; their second halves acch, accbh)
            sizes.update(dict.fromkeys(("acc", "accb", "acch", "accbh"), align64(m * g.nm)))
        nxt = [self.mac_stack, REGION, 2 * REGION, 3 * REGION]  # (after the stack)
        lay: dict[str, list[int]] = {}

        def put(key: str, i: int, regions: list[int]) -> bool:
            # (a C object with whole-line rows, ``lines``, may start in the free end of a region and fill the next,
            # empty one: its DMA reads meet that region's stream only in its first part)
            span = key == "C" and g.carry is not None and sizes[key] > REGION
            for r in regions:
                end = (r + 2 if span and r < 3 and nxt[r + 1] == (r + 1) * REGION else r + 1) * REGION
                if nxt[r] + sizes[key] <= end:
                    lay[key] = lay.get(key, []) + [0] * (i + 1 - len(lay.get(key, [])))
                    lay[key][i] = nxt[r]
                    if end > (r + 1) * REGION:
                        nxt[r + 1] = nxt[r] + sizes[key]
                        nxt[r] = (r + 1) * REGION
                    else:
                        nxt[r] += sizes[key]
                    return True
            return False

        named = {"A0": ("A", 0), "A1": ("A", 1), "D0": ("D", 0), "D1": ("D", 1), "T0": tiles[0], "T1": tiles[-1],
                 "T0h": ("acch", 0), "T1h": ("accbh", 0)}
        cold = ([("C", i) for i in range(mac["C"][0]) if ("C", i) not in tiles] + [("sx", 0)] * mac["sx"][0]
                + [("lut", 0)] * mac["lut"][0])
        if not (all(put(*named[x], [r]) for r, members in enumerate(PLANAR_PLANS[plan]) for x in members)
                and all(put(key, i, [3, 2, 1, 0]) for key, i in cold)):
            return None
        if mac["lutb"][0] and not put("lutb", 0, [r for r in (3, 2, 1, 0) if r != lay["lut"][0] // REGION]):
            return None  # (the table's second copy in another region than the first)
        if lp.csplit and any(lay[h][0] - lay[t][0] != lp.csplit for t, h in (("acc", "acch"), ("accb", "accbh"))):
            return None
        return self._place_dq(g, lp, lay, {"D": lay["D"]})

    def _dq_layout(self, g: Geometry, lp: L1Plan) -> dict[str, list[int]] | None:
        """A planar GEMM of a gather table's format whose MAC cores fit no plan: the dequantizing core alone pinned (the
        MAC cores aiecc's): its weight objects in regions 2 and 3 (the DMA writes them), the D objects in region 1, the
        rest first-fit as ``_place_dq``; weight objects wider than a region: the table's copies at the two ends of its
        memory, the rest in between (in one region every gather waits a cycle: aiecc put IQ4_XS's two in region 3, its
        dequant 37% in memory stalls). None if it does not fit; ``DQ_ONLY`` marks it."""
        dq = self._dq_buffers(g, lp.b_depth)
        if "tab" not in dq:
            return None
        b_size, d_size = (align64(dq[k][1]) for k in ("B", "D"))
        if lp.b_depth == 2 and b_size <= REGION and 2 * d_size <= REGION:
            lay = self._place_dq(g, lp, {}, {"D": [REGION, REGION + d_size], "B": [2 * REGION, 3 * REGION]})
            return None if lay is None else {**lay, DQ_ONLY: []}
        # (weight objects wider than a region: the table's copies at the two ends, region 0 after the stack and the top
        # of region 3, everything else in between)
        size_of = {k: align64(v[1]) for k, v in dq.items()}
        tabb = 0x10000 - size_of["tabb"]
        lay, at = {"tab": [0x0800], "tabb": [tabb]}, 0x0800 + size_of["tab"]
        for key in (k for k in dq if k not in ("tab", "tabb")):
            for _ in range(dq[key][0]):
                lay.setdefault(key, []).append(at)
                at += size_of[key]
        if at > tabb or size_of["tab"] > REGION - 0x0800:
            return None
        return {**lay, DQ_ONLY: []}

    def _place_dq(self, g: Geometry, lp: L1Plan, lay: dict[str, list[int]],
                  fixed: dict[str, list[int]]) -> dict[str, list[int]] | None:
        """The dequantizing core's buffers into ``lay``: ``fixed`` (D: the MAC cores' addresses, or B and D) where
        given, the others first-fit in the gaps around them; a gather table's two copies each within one region,
        "tabb" in another region than "tab", both out of the D objects' regions where they fit (the DMA reading a D
        object meets the gathers: an IQ3_XXS expert gate / up 360 -> 347 us per chunk), the core's other buffers then
        within one region where they fit. None if something does not fit."""
        dq = self._dq_buffers(g, lp.b_depth)
        size_of = {k: align64(v[1]) for k, v in dq.items()}
        lay = {**lay, **fixed}
        edges = [0x0800] + [e for k, addrs in fixed.items() for a in sorted(addrs) for e in (a, a + size_of[k])]
        edges = [0x0800] + sorted(edges[1:]) + [0x10000]
        free = [[lo, hi] for lo, hi in zip(edges[::2], edges[1::2]) if hi > lo]
        tables = "tab" in dq

        def regions(lo: int, size: int) -> set[int]:
            return set(range(lo // REGION, (lo + size - 1) // REGION + 1))

        def fit(size: int, one: bool, avoid: set[int]) -> tuple[list[int], int] | None:
            """(gap, start) of the lowest place for a buffer (one: within one region; avoid: in none of these)."""
            for iv in free:
                lo = iv[0]
                while lo + size <= iv[1]:
                    if (not one or len(regions(lo, size)) == 1) and not regions(lo, size) & avoid:
                        return iv, lo
                    lo = (lo // REGION + 1) * REGION
            return None

        d_regions = {r for a in lay["D"] for r in regions(a, size_of["D"])}
        for key in (k for k in dq if k not in fixed):
            n, size = dq[key][0], size_of[key]
            for _ in range(n):
                if key in ("tab", "tabb"):
                    other = regions(lay["tab"][0], size) if key == "tabb" else set()
                    found = fit(size, True, other | d_regions) or fit(size, True, other)
                else:
                    found = (tables and fit(size, True, set())) or fit(size, False, set())
                if found is None:
                    return None
                iv, lo = found
                if lo > iv[0]:  # (the gap before it stays free)
                    free.insert(free.index(iv), [iv[0], lo])
                lay.setdefault(key, []).append(lo)
                iv[0] = lo + size
        return lay


    def get_kernel_src(self) -> str:
        # (without forcing the post-RA pipeliner: II 33 per block instead of 34)
        mac = render_mac(k=KT) + (render_mac(name=f"{MAC}s", k=KT, pats=True) if self.csplit else "")
        return KERNEL_TEMPLATE.render(p=self.p, mac=f"{MAC}<{self.m}, {self.nm}>", macs=f"{MAC}s<{self.m}, {self.nm}>",
                                  csplit=self.csplit, m=self.m, n=self.nm, nd=self.n, halves=self.halves, k=KT, silu=self.silu,
                                  act=self.act, fmt=self.fmt,
                                  nsb=self.nsb,
                                  dqb=True, c_bfp16=self.c_bfp16, ablate=self.ablate,
                                  mac_src=mac, c_pack=self.c_pack, carry=self.carry, win=self.win, c_parts=self.c_parts,
                                  n_out=self.n_out, n_c=self.n_c, R=self.R, tiles=self.tiles,
                                  TR=self.TR, pingpong=self.pingpong, planar=self.planar, hu=self.hdr_units,
                                  hp0=self.hdr_plane0, hs=self.hdr_slot_words, sxp=self.sxp, tq=self.tq, pair=False,
                                  us=self.us, c_pairs=False, macf="", slots=self.slots,
                                  pack_plan=pack_plan(self.nm, self.n_out, self.n_c, self.tiles, self.R)
                                  if self.c_pack and self.slots == 2 else None,
                                  **self._pace_vars(), **slot(self.fmt).qd_template_vars(self))

    def get_mlir_artifact(self) -> Module:
        MR = self.nrows  # MAC rows
        m, n, halves = self.m, self.nm, self.halves  # (n: the MAC's columns)
        i32, u8 = np.int32, np.uint8
        df = self.flow
        cb = self.c_bfp16
        B_l2_ty = np.ndarray[(df.b_obj,), np.dtype[i32]]
        B_l1_ty = np.ndarray[(df.b_l1,), np.dtype[i32]]
        D_ty = np.ndarray[(n * KT * 9 // 8,), np.dtype[u8]]
        A_l2_ty = np.ndarray[(df.a_obj,), np.dtype[bfloat16]]
        A_l1_ty = np.ndarray[(df.a_l1,), np.dtype[bfloat16]]
        c_elems = df.c_l1
        flat = cb or self.c_tail  # (whole-line windows: rows and the tail the memtile's last window reads)
        C_l1_ty = np.ndarray[(c_elems,) if flat else (m, self.n_c), np.dtype[bfloat16]]
        C_l2_ty = np.ndarray[(MR * c_elems,) if flat else (MR * m, self.n_c), np.dtype[bfloat16]]
        acc_ty = np.ndarray[(m // 2 if self.csplit else m, n), np.dtype[bfloat16]]  # (csplit: a tile's first half)
        sx_ty = np.ndarray[(m * n // 2 // self.sxp,), np.dtype[bfloat16]]
        dqb = self._dq_buffers(self.geo, self.plan.b_depth)
        ws_ty = np.ndarray[(dqb["ws"][1] // 2,), np.dtype[bfloat16]]
        hb_ty = np.ndarray[(dqb["hb"][1] // 2,), np.dtype[bfloat16]]

        p = self.p
        k_zero = Kernel(f"{p}_zero", "kernel.o", [acc_ty])
        gelu = self.silu and self.act == "gelu"
        lut_ty = np.ndarray[(kmath.GELU.table_words().size,), np.dtype[np.uint32]]
        k_silu = Kernel(f"{p}_{self.act}", "kernel.o", [acc_ty, C_l1_ty, sx_ty, *([lut_ty] * 2 if gelu else [np.int32]),
                                                         *[np.int32] * (self.c_parts - 1)]) if self.silu else None
        # the table's two copies (gelu): in two regions where the layout is pinned (``_place_plan``); else where aiecc
        # puts them (a mem_bank for them reshuffled aiecc's placement of the MAC loop's streams: memory stall +42%)
        lay = self._l1_layout()
        dk = slot(self.fmt).qd_kernels(self, lambda name, args: Kernel(f"{p}_{name}", "kernel.o", args),
                                       B_l1_ty=B_l1_ty, ws_ty=ws_ty, hb_ty=hb_ty, D_ty=D_ty)
        k_mm = (Kernel(f"{p}_mm2", "kernel.o", [A_l1_ty, D_ty, acc_ty, acc_ty]) if self.pingpong else
                Kernel(f"{p}_mm", "kernel.o", [A_l1_ty, D_ty, acc_ty]))
        k_pack = (Kernel(f"{p}_pack2", "kernel.o", [acc_ty, acc_ty, C_l1_ty]) if self.slots == 2 else
                  Kernel(f"{p}_pack", "kernel.o", [acc_ty, C_l1_ty, *([np.int32] if self.carry else [])])) \
            if self.c_pack else None

        # B: shim -> memtile -> the column's dequantizing core (k-tile objects in dequant order)
        b_l3 = [ObjectFifo(B_l2_ty, name=f"gqB_{c}", depth=2, from_stream_per_cons=df.b_wdims)
                for c in range(COLS)]
        b_l1 = [b_l3[c].cons().forward(obj_type=B_l1_ty, name=f"gqB1_{c}", depth=self.b_depth, tile=Tile(c, 1),
                                        to_stream=df.b_rdims)
                for c in range(COLS)]
        # D: the dequantizing core -> the column's MAC cores (broadcast)
        d_f = [ObjectFifo(D_ty, name=f"gqD_{c}", depth=D_DEPTH) for c in range(COLS)]
        # A: one strip per MAC row in memtile (2 row, 1), broadcast to the 8 columns
        a_l3 = [ObjectFifo(A_l2_ty, name=f"gqA_{row}", depth=1 if df.a_res else 2,
                           from_stream_per_cons=df.a_wdims)
                for row in range(MR)]
        a_l1 = [a_l3[row].cons().forward(obj_type=A_l1_ty, name=f"gqA1_{row}", depth=2, tile=Tile(2 * row, 1),
                                          repeat_count=df.a_repeat if df.a_repeat > 1 else None)
                for row in range(MR)]
        # C: the column's MAC cores join in its memtile
        c_l3 = []
        c_l1: list[list[ObjectFifo]] = [[] for _ in range(MR)]
        for c in range(COLS):
            f = ObjectFifo(C_l2_ty, name=f"gqC_{c}", depth=2, to_stream=df.c_dims)
            c_l3.append(f)
            parts = f.prod().join([c_elems * i for i in range(MR)], obj_types=[C_l1_ty] * MR,
                                  names=[f"gqC1_{c}_{row}" for row in range(MR)], depths=[self.c_depth] * MR,
                                  tile=Tile(c, 1))
            for row in range(MR):
                c_l1[row].append(parts[row])

        n_kc, planes, nsb = self.n_kc, self.hdr_units, self.nsb
        silu = self.silu
        # hg > 1: runs (chunks per header unit, units) of a chunk group's header groups (the same in every group)
        runs = [(r, len(list(it))) for r, it in itertools.groupby(r for k0, r in self.hdr_groups if k0 < self.kc_group)]

        def dq_tiles(bf, dfo, dq, ws, ex, sb):  # the k-tiles of super-block sb
            for kt in range_(4):
                b = bf.acquire(1)
                for h in range(halves):  # (halves: the k-tile's rows as two D objects)
                    d = dfo.acquire(1)
                    dq(b, ws, d, sb, kt, *([h] if halves > 1 else []), *ex)
                    dfo.release(1)
                bf.release(1)

        def dq_fn(pc, prm, bf, dfo, hdr, dq, ws, hb, *extra):  # extra: dk.sb, dk.chunk, the ex buffer (those present)
            run_paced(pc, self.pace, lambda b: dq_item(prm, b, dfo, hdr, dq, ws, hb, *extra), bf)

        def dq_item(prm, bf, dfo, hdr, dq, ws, hb, *extra):
            ex = extra[-1:] if dk.ex_ty is not None else ()
            for _ in self._chunk_passes(prm):  # token chunks x passes: the weights stream once per chunk
                if self.hg > 1:  # a header unit per group of K chunks, then its chunks
                    for _ in range_(n_kc // self.kc_group):
                        for r, n_units in runs:
                            for _ in range_(n_units):
                                for kt in range_(4):
                                    b = bf.acquire(1)
                                    hdr(b, hb, 0, kt)
                                    bf.release(1)
                                for q in range_(r):
                                    for sb in range_(nsb):
                                        extra[0](hb, ws, sb, q)
                                        dq_tiles(bf, dfo, dq, ws, ex, sb)
                else:
                    for _ in range_(n_kc):
                        for pl in range(planes):  # header units, one k-tile slot per L1 object
                            for kt in range_(4):
                                b = bf.acquire(1)
                                hdr(b, hb if dk.collect else ws, *([pl] if dk.unit else []), kt)
                                bf.release(1)
                        if dk.chunk is not None:  # (Q6_K: d of the rows) once per K chunk
                            extra[1 if dk.sb is not None else 0](hb, ws)
                        for sb in range_(nsb):
                            if dk.sb is not None:  # (the super-block's records, just before its k-tiles)
                                extra[0](hb, ws, sb)
                            dq_tiles(bf, dfo, dq, ws, ex, sb)

        def units_fn(pc, prm, bf, dfo, sbk, dq, ws, hb, *ex):  # (super-block units: a weight object per unit; ex: ``ex``, tables)
            run_paced(pc, self.pace, lambda b: units_item(prm, b, dfo, sbk, dq, ws, hb, *ex), bf)

        def units_item(prm, bf, dfo, sbk, dq, ws, hb, *ex):
            for _ in self._chunk_passes(prm):
                for _ in range_(n_kc):
                    for sb in range_(nsb):
                        b = bf.acquire(1)
                        sbk(b, ws, hb, sb)
                        for kt in range_(4):
                            for h in range(halves):
                                d = dfo.acquire(1)
                                dq(b, ws, d, sb, kt, *([h] if halves > 1 else []), *ex)
                                dfo.release(1)
                        bf.release(1)

        c_pack = self.c_pack
        lines = self.carry is not None

        pingpong = self.pingpong

        def mac_fn(pc, prm, dfi, af, cf, zero, mm, *extra):
            run_paced(pc, self.pace, lambda d: mac_item(prm, d, af, cf, zero, mm, *extra), dfi)

        def mac_item(prm, dfi, af, cf, zero, mm, *extra):
            act_args = tuple(extra[-2:]) if gelu else (0,)  # (gelu: the table copies; silu: the kernel's opaque runtime zero)
            if self.G > 1:  # row groups: per pass vp C objects of ``slots`` slots, each over the whole A strip
                for _ in self._chunk_passes(prm):
                    for _ in range_(self.vp):
                        c = cf.acquire(1)
                        for s_ in range(self.slots):
                            acc = extra[1 + s_] if c_pack else c
                            zero(acc)
                            for _ in range_(self.KA // KT):
                                d = dfi.acquire(1)
                                a = af.acquire(1)
                                mm(a, d, acc)
                                af.release(1)
                                dfi.release(1)
                        if c_pack:
                            extra[0](*extra[1:1 + self.slots], c)
                        cf.release(1)
                return
            if halves > 1:  # one A k-tile for both halves' D objects, a C tile each; their outputs as two C objects
                accs = (extra[1], extra[3])
                for _ in self._chunk_passes(prm):
                    for acc in accs:
                        zero(acc)
                    for _ in range_(n_kc * (self.kc // KT)):
                        a = af.acquire(1)
                        for acc in accs:
                            d = dfi.acquire(1)
                            mm(a, d, acc)
                            dfi.release(1)
                        af.release(1)
                    if self.c_parts == 2:  # (both halves in one C object: 576-byte rows)
                        c = cf.acquire(1)
                        for h, acc in enumerate(accs):
                            extra[0](acc, c, extra[2], *act_args, h)
                        cf.release(1)
                        continue
                    for acc in accs:
                        c = cf.acquire(1)
                        extra[0](acc, c, extra[2], *act_args)
                        cf.release(1)
                return

            def mac_loop(acc: object) -> None:  # (a pass's k-tiles into its C tile)
                zero(acc)
                if pingpong:  # acc -> accb -> acc: an even k-tile count ends in acc
                    for _ in range_(n_kc * (self.kc // KT) // 2):
                        for src, dst in ((acc, extra[3]), (extra[3], acc)):
                            d = dfi.acquire(1)
                            a = af.acquire(1)
                            mm(a, d, src, dst)
                            af.release(1)
                            dfi.release(1)
                else:
                    for _ in range_(n_kc * (self.kc // KT)):
                        d = dfi.acquire(1)
                        a = af.acquire(1)
                        mm(a, d, acc)
                        af.release(1)
                        dfi.release(1)

            if silu and self.c_parts == 2:  # (pairs of passes in one C object, taken after the first one's MAC loop:
                # the previous object's drain done by then)
                for _ in self._chunk_passes(prm, step=2):
                    mac_loop(extra[1])
                    c = cf.acquire(1)
                    extra[0](extra[1], c, extra[2], *act_args, 0)
                    mac_loop(extra[1])
                    extra[0](extra[1], c, extra[2], *act_args, 1)
                    cf.release(1)
                return
            for p_ in self._chunk_passes(prm, index=lines):
                if not lines:  # (whole-line drains: the C object after the MAC loop, its last drain done by then)
                    c = cf.acquire(1)
                acc = extra[1] if silu or c_pack else c
                mac_loop(acc)
                if silu:
                    extra[0](acc, c, extra[2], *act_args)
                elif lines:
                    c = cf.acquire(1)
                    extra[0](acc, c, p_)
                elif c_pack:
                    extra[0](acc, c)
                cf.release(1)

        workers = []
        prm = self._mc_prm()
        pcs = self._pace_args()
        for c in range(COLS):
            ws = Buffer(ws_ty, name=f"gq_ws_{c}")
            hb = Buffer(hb_ty, name=f"gq_hb_{c}")
            ex_buf = [Buffer(dk.ex_ty, name=f"gq_ex_{c}")] if dk.ex_ty is not None else []
            if dk.tables is not None:  # (a lookup table's two copies: ``_place_plan`` puts them in two regions)
                ex_buf += [Buffer(dk.tables[0], name=f"gq_{key}_{c}", initial_value=dk.tables[1]) for key in ("tab", "tabb")]
            if dk.units:
                workers.append(Worker(units_fn, [pcs.get((c, 2)), prm, b_l1[c].cons(), d_f[c].prod(), dk.sb, dk.dq, ws, hb,
                                                 *ex_buf],
                                      tile=Tile(c, 2), stack_size=2048, dynamic_objfifo_lowering=True))
            else:
                workers.append(Worker(dq_fn, [pcs.get((c, 2)), prm, b_l1[c].cons(), d_f[c].prod(), dk.hdr, dk.dq, ws, hb,
                                              *([dk.sb] if dk.sb is not None else []),
                                              *([dk.chunk] if dk.chunk is not None else []), *ex_buf],
                                      tile=Tile(c, 2), stack_size=2048, dynamic_objfifo_lowering=True))
            for row in range(MR):
                extra = ([k_silu, Buffer(acc_ty, name=f"gq_acc_{c}_{row}"), Buffer(sx_ty, name=f"gq_sx_{c}_{row}")]
                         if silu else [k_pack, Buffer(acc_ty, name=f"gq_acc_{c}_{row}")] if c_pack else [])
                if pingpong or halves > 1 or self.slots == 2:
                    extra.append(Buffer(acc_ty, name=f"gq_accb_{c}_{row}"))
                if self.csplit:  # (the tiles' second halves: the kernels reach them 16 KB on)
                    extra += [Buffer(acc_ty, name=f"gq_{h}_{c}_{row}") for h in ("acch", "accbh")]
                if gelu:
                    extra += [Buffer(lut_ty, name=f"gq_{key}_{c}_{row}", initial_value=kmath.GELU.table_words())
                              for key in ("lut", "lutb")]
                workers.append(Worker(mac_fn, [pcs.get((c, 3 + row)), prm, d_f[c].cons(), a_l1[row].cons(),
                                               c_l1[row][c].prod(), k_zero,
                                               k_mm, *extra],
                                      tile=Tile(c, 3 + row), stack_size=self.mac_stack, dynamic_objfifo_lowering=True))

        A_ty = np.ndarray[(self.M * self.a_K * 9 // 16,), np.dtype[bfloat16]]
        W_ty = np.ndarray[(self.w_bytes // 2,), np.dtype[bfloat16]]
        C_ty = np.ndarray[(self.M * self.N * 9 // 16 if cb else self.M * self.N,), np.dtype[bfloat16]]
        a_prods = [a_l3[row].prod(tile=Tile(2 * row, 0)) for row in range(MR)]
        b_prods = [b_l3[c].prod(tile=Tile(c, 0)) for c in range(COLS)]
        c_conses = [c_l3[c].cons(tile=Tile(c, 0)) for c in range(COLS)]
        sequence = self._sequence(a_prods, b_prods, c_conses, MR)
        rt = Runtime(sequence, [A_ty, W_ty, C_ty, [a_prods, b_prods, c_conses]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        layout = lay
        if layout is not None:
            roles = {"gqA1": "A", "gqB1": "B", "gqC1": "C", "gqD": "D"}

            def worker_buffer(name: str) -> int | None:  # gq_<role>_<col>[_<row>]
                role = name.split("_")[1] if name.startswith("gq_") else None
                return layout[role][0] if role in layout else None

            def fifo(name: str, depth: int) -> list[int] | None:  # (dq-only: not the MAC cores' D pools)
                return None if DQ_ONLY in layout and "__cons" in name and name.startswith("gqD_") else \
                    layout.get(roles.get(name.split("_")[0], ""))

            pin_l1_buffers(module, fifo=fifo, buffer=worker_buffer)
        return module


def k_tile_cycles(op: GemmQdOperator) -> float:
    """Estimated cycles per k-tile of a GemmQd pass on planar tiles: the slowest of the dequantizing core, the MAC
    cores (MMUL m n / 8, A and D ports 9 m and 9 n: 72 B per token or row at 8 B per cycle) and the column's weight
    stream (its tiles' bytes per k-tile at ~3.9 B per cycle: 56 GB/s over 8 columns at 1.81 GHz). The dequant:
    ``TileFormat.dq_row_cycles`` (measured order: Q4 nibbles ~6, Q5_K ~10.5 with its scale expansion, Q6_K ~12.5)."""
    m, n = op.m, op.nm * op.halves  # (halves: the dequant converts, the MAC cores multiply both)
    weights = op.tiles * op.tile_bytes / (4 * op.nsb)  # (a tile: nsb super-blocks of 4 k-tiles)
    return max(FORMATS[op.fmt].dq_row_cycles * n, m * n / 8, 9 * m, 9 * n, weights / 3.87)


expose(GemmQdOperator, L1Plan, "plan")
