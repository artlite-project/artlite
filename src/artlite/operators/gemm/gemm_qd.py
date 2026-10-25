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

Planar tiles (``gemm_q.planar``: Q5_K, Q6_K, Q2_K, Q3_K, Q4_0 / Q4_1 of other than 16 rows or of 512 columns) of
R rows: the memtile gathers a k-tile's words of the pass's ``tiles`` x R rows from the tiles' 32-byte planes
(``TileFormat.planes``, plane- or row-major slices) into [plane][word][row] (word-major Q2_K / Q3_K tiles: [word]
[row]), zero-padded to the MMUL's n; the
header units (slice-sized, from ``hdr_skip`` bytes before the tile when the header is shorter) go through the
same gather: the dequantizing core collects their four k-tile slots (``_hcp``) and expands a super-block's
scales before its k-tiles (``_hdr_sb``; Q6_K: d once per K chunk, ``_hdr_d``; Q5_K: then per k-tile the two
groups' scales per MAC lane, ``_expand``). A column's projection runs in passes of whole tiles
(``plan_planar``: blocks of passes, the last tile of a block ends in padding rows), ``c_pack``: the MAC cores
copy the pass's rows out of the MMUL tile in 16-byte items (``_pack``; the memtile drops the columns past the
pass's rows). gate/up: a pass is one SiLU group (128 rows and its padding rows: dropped by the memtile's read
when every part of the tile maps row r to gathered row r, else skipped by the dequant) in two halves of 64 rows
(``halves``: two D objects per k-tile, a C tile of 64 columns each on the MAC cores, so that m = 96 fits L1: 3B
Q5_K at T = 256 one weight sweep instead of two, gate/up 1.88 -> 1.11 ms), or half a group when R divides 64
(``pingpong``)."""

import itertools
from dataclasses import dataclass

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from .. import kmath
from ..design import COLS
from ..formats import FORMATS, KC, slot
from ..kernel_source import kernel_prefix
from ..l1_layout import pin_l1_buffers
from .gemm_q import _KERNEL_TPL, B_DEPTH, KT, L1_BUDGET, GemmQOperator, L1Error, pack_plan, planar
from .mac import MAC, render_mac

D_DEPTH = 2  # dequantized k-tile objects (producer and each consumer)
REGION = 0x4000  # a core's data memory: four 16 KB regions of two interleaved banks
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
        assert 1 <= mac_rows <= 3 and "nrows" not in kw and m % 16 == 0
        # (planar SiLU passes of a whole group: two C tiles of its halves, no pingpong)
        rows = kw.get("rows", 16)
        self.pingpong = bool(kw.get("silu")) and (not planar(kw.get("fmt", "Q4_0"), rows, kw.get("kc", KC))
                                                  or kw.get("tiles", 0) * rows == 64) if pingpong is None else pingpong
        # MAC cores' C objects (1: the next pass waits for the drain): 2 if they fit L1, else 1 (``_l1_bytes``)
        self.c_depth = 2
        self.T = kw.pop("M")
        step = mac_rows * m
        self.csplit = 0
        super().__init__(nrows=mac_rows, M=-(-self.T // step) * step, m=m, **kw)
        assert self.dequant, "GemmQd: dequantized tile formats"
        assert not self.pingpong or self.n_kc * (self.kc // KT) % 2 == 0
        # two C tiles (a pass's halves), each in halves 16 KB apart (``csplit``, plan g1s): the MAC loop's blocks
        # alternate regions, so a block's C stores never meet the next block's C loads (one region per tile: a
        # conflict per C access)
        if self.halves == 2 and self.planar and self.m % 32 == 0 and "nosplit" not in self.ablate:
            self.csplit = REGION
            if self._planar_layout(self.m) is None:
                self.csplit = 0

    def _buffers(self, m: int) -> dict[str, tuple[int, int]]:
        """(count, bytes) of the MAC core's buffers (the dequantizing core's: ``_dq_buffers``): A, the
        dequantized k-tile D (bfp16 panels), C, the MAC target ``acc`` and the SiLU scratch ``sx``
        (silu only)."""
        nm = self.nm
        return {"A": (2, m * 72), "D": (D_DEPTH, nm * KT * 9 // 8),
                "C": (self.c_depth, m * self.n_out * 9 // 8 if self.c_bfp16 else m * self.n_c * 2),
                "acc": ((2 if self.pingpong or self.halves == 2 or self.slots == 2 else 1) if self.silu or self.c_pack
                        else 0, m * nm * 2),
                "sx": (1 if self.silu else 0, m * nm // self.sxp),
                "lut": (1 if self.silu and self.act == "gelu" else 0, kmath.GELU.table_words().nbytes),
                "lutb": (1 if self.silu and self.act == "gelu" else 0, kmath.GELU.table_words().nbytes)}

    def _l1_bytes(self, m: int) -> int:
        """Data memory of the larger role (MAC core: ``_buffers``, dequantizing core: ``_dq_buffers``), the
        2 KB stacks included; picks the C depth (2 if it fits) and the SiLU scratch's parts (``sxp``: the
        epilogue in 2 parts of m / 2 rows if the whole one does not fit). (Trading the C depth for a pinned planar
        layout was not consistent: 1B Q3_K_L's O at T = 256 -20%, at T = 2048 +8%.)"""
        # (two C objects and the scratch in two parts beat one C object, 1.19 vs 1.23 ms on Gemma's gate/up at T = 256;
        # planar: of the first C depth in budget, the scratch's parts whose pinned layout fits, else the first in
        # budget (aiecc's placement))
        auto = ((2, 1), (2, 2), (1, 1), (1, 2)) if self.silu else ((2, 1), (1, 1), (1, 2))
        dq = self._dq_bytes()  # (picks the dequantizing core's weight objects first: the layouts place them too)
        fits = [(cd, sxp) for cd, sxp in auto if self._try(m, cd, sxp) <= L1_BUDGET]
        pinned = [c for c in fits if self.planar and c[0] == fits[0][0] and self._try(m, *c)
                  and self._planar_layout(m) is not None]
        self._try(m, *(pinned or fits or auto[-1:])[0])
        return max(super()._l1_bytes(m), dq)

    def _try(self, m: int, c_depth: int, sxp: int) -> int:
        """Set the C depth and the scratch's parts; the MAC core's data memory with them."""
        self.c_depth, self.sxp = c_depth, sxp
        return super()._l1_bytes(m)

    def _dq_bytes(self) -> int:
        """The dequantizing core's data memory, stack included; picks its weight objects (two, or one if two do not
        fit, e.g. Q6_K SiLU groups of 2048 columns: its weights then arrive between dequant calls)."""
        for self.b_depth in (B_DEPTH, 1):
            dq = sum(n * b for n, b in self._dq_buffers().values()) + 2048
            if dq <= L1_BUDGET:
                break
        return dq

    def _dq_buffers(self) -> dict[str, tuple[int, int]]:
        """The dequantizing core's buffers: the weight objects B, its output objects D, and the format's
        (``Slot.qd_scratch``): the scale workspace ``ws``, the header slots ``hb`` (Q5_K: ``ex``)."""
        return {"B": (self.b_depth, 4 * self.b_l1_words), "D": (D_DEPTH, self.nm * KT * 9 // 8),
                **slot(self.fmt).qd_scratch(self)}

    @property
    def _hdr_plane0(self) -> int:
        """The first plane of a plane-major header unit with header data (the earlier ones hold the previous
        tile's codes: ``hdr_skip``)."""
        return self.hdr_skip // (32 * self.R) if FORMATS[self.fmt].plane_major else 0

    @property
    def _hdr_slot_words(self) -> int:
        """Words of a collected header slot (planes ``_hdr_plane0`` .. of a k-tile object)."""
        return 2 * (FORMATS[self.fmt].planes - self._hdr_plane0) * self.n

    def _l1_layout(self) -> dict[str, list[int]] | None:
        """Byte addresses of both roles' buffers (D at the same addresses on both): each stream of
        the hot loops in its own 16 KB region (two interleaved banks) -- MAC core: A (1), D (2), the
        C tile (3), the stack and the epilogue's buffers (0); dequantizing core: B (1), D (2), the
        scale workspace (3), the header slots next to the stack (0); planar tiles: ``_planar_layout``. None: aiecc's
        placement."""
        if self.planar:
            return self._planar_layout(self.m)
        if self.c_depth == 1:
            return None
        mac, dq = self._buffers(self.m), self._dq_buffers()

        def place(base: int, n: int, size: int) -> list[int]:
            step = (size + 63) // 64 * 64
            return [base + i * step for i in range(n)]

        lay = {"A": place(0x4000, *mac["A"]), "D": place(0x8000, *mac["D"]), "B": place(0x4000, *dq["B"]),
               "ws": place(0xC000, *dq["ws"]), "hb": place(0x0800, *dq["hb"])}
        lo = self.mac_stack  # region 0 after the stack: the MAC core's cold buffers
        if self.silu:  # the C tiles in regions 3 and 0, the SiLU scratch after the A objects
            lay["acc"], lay["accb"] = [0xC000], [lo]
            lo += (mac["acc"][1] + 63) // 64 * 64
            lay["C"] = place(lo, *mac["C"])
            lo = lay["C"][-1] + (mac["C"][1] + 63) // 64 * 64
            lay["sx"] = [lay["A"][-1] + (mac["A"][1] + 63) // 64 * 64]
        else:  # the MAC writes C objects in turn: one in region 3, the other after the stack
            lay["C"] = [0xC000, lo]
        ends = {k: v[-1] + (dict(mac, **dq, accb=mac["acc"])[k][1] + 63) // 64 * 64 for k, v in lay.items()}
        if any(e > (b // 0x4000 + 1) * 0x4000 for k, b, e in ((k, v[0], ends[k]) for k, v in lay.items())
               if not (k == "C" and not self.silu)):
            return None
        return lay

    def _planar_layout(self, m: int) -> dict[str, list[int]] | None:
        """Planar tiles: the MAC cores' buffers by ``PLANAR_PLANS`` (crossed, else separate, for one
        C tile, g4, else g1, for a pass's two halves, g1 for pingpong; g1s for C tiles in halves, ``csplit``), the other
        buffers first-fit in regions 3 .. 0; the dequantizing core's D objects
        at the MAC cores' addresses, its other buffers first-fit around them. None if nothing fits. Measured at T = 2048
        against aiecc's placement (alternating runs): 3B Q2_K down (Q3_K, m 64, n 128, crossed) 5.74 -> 5.15 ms (aiecc
        put A and D of one k-tile in one region: the MAC cores' memory stall 1.58 -> 0.62 ms), gate/up (g4) 7.41 -> 7.13,
        3B Q3_K_M gate/up 7.32 -> 6.99, 1B Q3_K_L 4.99 -> 4.78; 3B Q5_K (crossed) unchanged."""
        two = self.halves == 2 or self.pingpong or self.slots == 2
        if self.csplit:
            return self._place_plan(m, "g1s", two)
        # (pingpong: both C tiles in every window, D alternating as A: g4's acc and D1 would meet)
        auto = ["g1"] if self.pingpong else ["g4", "g1"] if two else ["crossed", "separate"]
        for plan in auto:
            if ("T1" in sum(PLANAR_PLANS[plan], ())) == two and (lay := self._place_plan(m, plan, two)) is not None:
                return lay
        return None

    def _place_plan(self, m: int, plan: str, two: bool) -> dict[str, list[int]] | None:
        mac, dq = self._buffers(m), self._dq_buffers()
        tiles = [("acc", 0), ("accb", 0)] if two else [("acc", 0)] if mac["acc"][0] else [("C", 0)]
        sizes = {k: (v[1] + 63) // 64 * 64 for k, v in {**mac, "accb": mac["acc"]}.items()}
        if self.csplit:  # (the tiles' first halves acc, accb; their second halves acch, accbh)
            sizes.update(dict.fromkeys(("acc", "accb", "acch", "accbh"), (m * self.nm + 63) // 64 * 64))
        nxt = [self.mac_stack, REGION, 2 * REGION, 3 * REGION]  # (after the stack)
        lay: dict[str, list[int]] = {}

        def put(key: str, i: int, regions: list[int]) -> bool:
            for g in regions:
                if nxt[g] + sizes[key] <= (g + 1) * REGION:
                    lay[key] = lay.get(key, []) + [0] * (i + 1 - len(lay.get(key, [])))
                    lay[key][i] = nxt[g]
                    nxt[g] += sizes[key]
                    return True
            return False

        named = {"A0": ("A", 0), "A1": ("A", 1), "D0": ("D", 0), "D1": ("D", 1), "T0": tiles[0], "T1": tiles[-1],
                 "T0h": ("acch", 0), "T1h": ("accbh", 0)}
        cold = ([("C", i) for i in range(mac["C"][0]) if ("C", i) not in tiles] + [("sx", 0)] * mac["sx"][0]
                + [("lut", 0)] * mac["lut"][0])
        if not (all(put(*named[x], [g]) for g, members in enumerate(PLANAR_PLANS[plan]) for x in members)
                and all(put(key, i, [3, 2, 1, 0]) for key, i in cold)):
            return None
        if mac["lutb"][0] and not put("lutb", 0, [g for g in (3, 2, 1, 0) if g != lay["lut"][0] // REGION]):
            return None  # (the table's second copy in another region than the first)
        if self.csplit and any(lay[h][0] - lay[t][0] != self.csplit for t, h in (("acc", "acch"), ("accb", "accbh"))):
            return None
        # the dequantizing core: first-fit in the gaps around its D objects; a gather table's two copies each within one
        # region, "tabb" in another region than "tab", both out of the D objects' regions where they fit (the DMA
        # reading a D object meets the gathers: an IQ3_XXS expert gate / up 360 -> 347 us per chunk), the core's other
        # buffers then within one region where they fit
        d_size = (dq["D"][1] + 63) // 64 * 64
        edges = [0x0800] + [e for a in sorted(lay["D"]) for e in (a, a + d_size)] + [0x10000]
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

        d_regions = {r for a in lay["D"] for r in regions(a, d_size)}
        for key in (k for k in dq if k != "D"):
            n, size = dq[key][0], (dq[key][1] + 63) // 64 * 64
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
        return _KERNEL_TPL.render(p=self.p, mac=f"{MAC}<{self.m}, {self.nm}>", macs=f"{MAC}s<{self.m}, {self.nm}>",
                                  csplit=self.csplit, m=self.m, n=self.nm, nd=self.n, halves=self.halves, k=KT, silu=self.silu,
                                  act=self.act, fmt=self.fmt,
                                  nsb=self.nsb,
                                  dqb=True, c_bfp16=self.c_bfp16, ablate=self.ablate,
                                  mac_src=mac, c_pack=self.c_pack, n_out=self.n_out, n_c=self.n_c, R=self.R, tiles=self.tiles,
                                  TR=self.TR, pingpong=self.pingpong, planar=self.planar, hu=self.hdr_units,
                                  hp0=self._hdr_plane0, hs=self._hdr_slot_words, sxp=self.sxp, tq=self.tq, pair=False,
                                  us=self.us, c_pairs=False, macf="", slots=self.slots,
                                  pack_plan=pack_plan(self.nm, self.n_out, self.n_c, self.tiles, self.R)
                                  if self.c_pack and self.slots == 2 else None,
                                  **slot(self.fmt).qd_template_vars(self))

    def get_mlir_artifact(self) -> Module:
        MR = self.nrows  # MAC rows
        m, n, halves = self.m, self.nm, self.halves  # (n: the MAC's columns)
        i32, u8 = np.int32, np.uint8
        df = self.dataflow()
        cb = self.c_bfp16
        B_l2_ty = np.ndarray[(df.b_obj,), np.dtype[i32]]
        B_l1_ty = np.ndarray[(df.b_l1,), np.dtype[i32]]
        D_ty = np.ndarray[(n * KT * 9 // 8,), np.dtype[u8]]
        A_l2_ty = np.ndarray[(df.a_obj,), np.dtype[bfloat16]]
        A_l1_ty = np.ndarray[(df.a_l1,), np.dtype[bfloat16]]
        c_elems = df.c_l1
        C_l1_ty = np.ndarray[(c_elems,) if cb else (m, self.n_c), np.dtype[bfloat16]]
        C_l2_ty = np.ndarray[(MR * c_elems,) if cb else (MR * m, self.n_c), np.dtype[bfloat16]]
        acc_ty = np.ndarray[(m // 2 if self.csplit else m, n), np.dtype[bfloat16]]  # (csplit: a tile's first half)
        sx_ty = np.ndarray[(m * n // 2 // self.sxp,), np.dtype[bfloat16]]
        dqb = self._dq_buffers()
        ws_ty = np.ndarray[(dqb["ws"][1] // 2,), np.dtype[bfloat16]]
        hb_ty = np.ndarray[(dqb["hb"][1] // 2,), np.dtype[bfloat16]]

        p = self.p
        k_zero = Kernel(f"{p}_zero", "kernel.o", [acc_ty])
        gelu = self.silu and self.act == "gelu"
        lut_ty = np.ndarray[(kmath.GELU.table_words().size,), np.dtype[np.uint32]]
        k_silu = Kernel(f"{p}_{self.act}", "kernel.o", [acc_ty, C_l1_ty, sx_ty, *([lut_ty] * 2 if gelu else [np.int32])]) \
            if self.silu else None
        # the table's two copies (gelu): in two regions where the layout is pinned (``_place_plan``); else where aiecc
        # puts them (a mem_bank for them reshuffled aiecc's placement of the MAC loop's streams: memory stall +42%)
        lay = self._l1_layout()
        dk = slot(self.fmt).qd_kernels(self, lambda name, args: Kernel(f"{p}_{name}", "kernel.o", args),
                                       B_l1_ty=B_l1_ty, ws_ty=ws_ty, hb_ty=hb_ty, D_ty=D_ty)
        k_mm = (Kernel(f"{p}_mm2", "kernel.o", [A_l1_ty, D_ty, acc_ty, acc_ty]) if self.pingpong else
                Kernel(f"{p}_mm", "kernel.o", [A_l1_ty, D_ty, acc_ty]))
        k_pack = (Kernel(f"{p}_pack2", "kernel.o", [acc_ty, acc_ty, C_l1_ty]) if self.slots == 2 else
                  Kernel(f"{p}_pack", "kernel.o", [acc_ty, C_l1_ty])) if self.c_pack else None

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

        def dq_fn(prm, bf, dfo, hdr, dq, ws, hb, *extra):  # extra: dk.sb, dk.chunk, the ex buffer (those present)
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

        def units_fn(prm, bf, dfo, sbk, dq, ws, hb, *ex):  # (super-block units: a weight object per unit; ex: ``ex``, tables)
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

        pingpong = self.pingpong

        def mac_fn(prm, dfi, af, cf, zero, mm, *extra):
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
                    for acc in accs:
                        c = cf.acquire(1)
                        extra[0](acc, c, extra[2], *act_args)
                        cf.release(1)
                return
            for _ in self._chunk_passes(prm):
                c = cf.acquire(1)
                acc = extra[1] if silu or c_pack else c
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
                if silu:
                    extra[0](acc, c, extra[2], *act_args)
                elif c_pack:
                    extra[0](acc, c)
                cf.release(1)

        workers = []
        prm = self._mc_prm()
        for c in range(COLS):
            ws = Buffer(ws_ty, name=f"gq_ws_{c}")
            hb = Buffer(hb_ty, name=f"gq_hb_{c}")
            ex_buf = [Buffer(dk.ex_ty, name=f"gq_ex_{c}")] if dk.ex_ty is not None else []
            if dk.tables is not None:  # (a lookup table's two copies: ``_place_plan`` puts them in two regions)
                ex_buf += [Buffer(dk.tables[0], name=f"gq_{key}_{c}", initial_value=dk.tables[1]) for key in ("tab", "tabb")]
            if dk.units:
                workers.append(Worker(units_fn, [prm, b_l1[c].cons(), d_f[c].prod(), dk.sb, dk.dq, ws, hb, *ex_buf],
                                      tile=Tile(c, 2), stack_size=2048, dynamic_objfifo_lowering=True))
            else:
                workers.append(Worker(dq_fn, [prm, b_l1[c].cons(), d_f[c].prod(), dk.hdr, dk.dq, ws, hb,
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
                workers.append(Worker(mac_fn, [prm, d_f[c].cons(), a_l1[row].cons(), c_l1[row][c].prod(), k_zero,
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

            pin_l1_buffers(module, fifo=lambda name, depth: layout.get(roles.get(name.split("_")[0], "")),
                           buffer=worker_buffer)
        return module





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
        kw = {"fmt": fmt, "rows": R, "M": T, "K": K, "N": COLS * sum(plan.pitches), "tile_bytes": tile_bytes,
              "col_bytes": plan.region_rt * (K // kc) * tile_bytes, "passes": plan.passes,
              "out_cols": plan.out_cols([0] * nb, list(plan.pitches)), "kc_group": kc_group, "kc": kc,
              "tiles": plan.tiles, "region_rt": plan.region_rt, "blocks": plan.blocks, "c_pack": True}
        return planar_seconds(planar_gemm(kw, T))

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
    a = op.M * op.K * 9 / 8 * (1 if op.dataflow().a_res else op.passes)
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


def k_tile_cycles(op: GemmQdOperator) -> float:
    """Estimated cycles per k-tile of a GemmQd pass on planar tiles: the slowest of the dequantizing core, the MAC
    cores (MMUL m n / 8, A and D ports 9 m and 9 n: 72 B per token or row at 8 B per cycle) and the column's weight
    stream (its tiles' bytes per k-tile at ~3.9 B per cycle: 56 GB/s over 8 columns at 1.81 GHz). The dequant:
    ``TileFormat.dq_row_cycles`` (measured order: Q4 nibbles ~6, Q5_K ~10.5 with its scale expansion, Q6_K ~12.5)."""
    m, n = op.m, op.nm * op.halves  # (halves: the dequant converts, the MAC cores multiply both)
    weights = op.tiles * op.tile_bytes / (4 * op.nsb)  # (a tile: nsb super-blocks of 4 k-tiles)
    return max(FORMATS[op.fmt].dq_row_cycles * n, m * n / 8, 9 * m, 9 * n, weights / 3.87)
