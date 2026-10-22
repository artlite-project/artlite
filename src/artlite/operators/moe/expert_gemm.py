"""The experts' GEMMs of an MoE layer's prefill: per column a work list of chunks, each ``CHUNK`` = 3 x 64 tokens of
one expert (``ExpertGemmOperator``, a GemmQd per column: the dequantizing core at row 2, MAC cores at rows 3 .. 5).

Column c runs n_c chunks (scratchpad parameter ``N_PARAM`` c, read by its cores). Chunk k reads its A rows from the
column's k-th slot of the A argument (``kind`` "gu": the gathered inputs' bfp16 panels, hidden wide; "down": the gate /
up GEMM's H panels) and writes the k-th slot of C, and streams the weights of its expert, whose slab addresses the
column's control blocks hold (``ctl``: the host writes them per layer, ``plan``). Most chunks hold fewer valid rows than
CHUNK (an expert's pairs: a median of ~20 at 2048 tokens), so the rows that move are a chunk's 64-row strips with valid
rows: MAC row 0's always, rows 1 and 2's together when the chunk has more than 64 (``active``; the column's last chunk
always: its drains carry the tokens the sequence waits for). The control stream's block k (``chunk_template``):

- the weights: the column's MM2S1 is packet-switched (``ctl.XID`` to the memtile; ``ctl.CID`` to the shim's
  TileControl); block k pushes chunk k's weight tasks (``w_chains``, at the expert's slab: "gu" a task per expert half,
  "down" one task of both halves' tiles: the halves are the K chunks of its row-group tiles), then block k + 1;
- A: one task of 3 BDs on MM2S0 (``A_BDS``), a strip each into the column's memtile (one object of the 3 strips:
  resident for the chunk's passes, its parts resent per pass and slot to their rows): an inactive strip one marker
  panel (``MARKER``: all ones, the column's control buffer's last words), which the memtile's S2MM, in finish-on-TLAST
  mode, takes as the whole strip; a MAC core that finds the marker in its strip's first panel (``chk``) computes on
  whatever the object holds and writes no C;
- C: row 0's objects straight to the shim's S2MM0, rows 1 and 2's joined in the memtile to S2MM1; per chunk a drain
  task on each channel that has objects (``drain_taps``: a BD iterating over the chunk's C objects, BD set k % 2; block
  k + 2 waits for chunk k's drains by a shim lock, ``LOCKS``, which their BDs release once per object), the last chunk's
  with a token. The down's chunk slot is pass-major (``y_layout``): its C objects written whole, one after the other
  (per token a C object holds ``vp`` runs of the pass's outputs).

"gu": the halves' gate / up rows of the slab ("exp_gu": per half 32 row tiles x 2 K chunks, ``overlay.image.gate_up_rows``),
passes of ``GU_ROWS`` rows (64 outputs) in the halves' order: H = silu(gate) up as bfp16 panels of the expert's width.
"down": the slab's "exp_down" (all hidden rows over a half's columns, row groups of 4; Q4_K or Q6_K), the halves as
two K chunks one slab stride apart (A: the expert's H, 512 columns), two slots per C object (``GemmQOperator.slots``).
"""

import functools

import numpy as np
import torch
from aie.dialects import arith
from aie.dialects._aiex_ops_gen import npu_sync as _npu_sync
from aie.dialects.aie import T
from aie.dialects.aiex import _as_i32, npu_maskwrite32, npu_write32
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import if_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ... import txn
from ..base import JINJA_ENV, ArgSpec
from ..design import COLS, range32, stream_walk
from ..formats import KT, slot
from ..gemm.gemm_q import planar
from ..gemm.gemm_qd import D_DEPTH, GemmQdOperator
from ..l1_layout import pin_l1_buffers
from . import ctl

MAC_ROWS, M = 3, 64
CHUNK = MAC_ROWS * M  # tokens of a chunk
N_PARAM = "eg_n"  # (+ the column: its chunks, from bit 8 its gathered row groups)
BLOCK_BDS = (14, 15)  # the control blocks' shim BDs (the sequence's own, block 0's fill: the low ids)
W_BDS = (9, 10, 11, 12, 13)  # the weight tasks' BDs
A_BDS = (1, 2, 3)  # a chunk's A strips (MM2S0)
C_BDS = ((4, 5), (6, 7))  # per BD set (chunk parity): row 0's drain (S2MM0), rows 1 and 2's (S2MM1)
LOCKS = (2, 3)  # the drain sets' shim locks
A_CH = 0  # the A object's memtile S2MM channel (finish-on-TLAST)
MEM_S2MM_CTRL = 0xA0600  # + 8 channel: a memtile's DMA_S2MM_<ch>_Ctrl (AM029; FoT_Mode bits 17:16, 1: no counts)
MARKER = 32  # words of a column's control buffer at its end, all ones (a panel of them: an inactive strip)


_CHK = JINJA_ENV.from_string(r"""
extern "C" {
// fl[0] = 1 if the A object's first panel is the marker (all ones: no finite panel has exponents 0xFF), else 0
void {{ p }}_chk(const bfloat16 *__restrict a, int32_t *__restrict fl)
{
    const int32_t *__restrict w = (const int32_t *)a;  // (the panel: 16 words of mantissas, 2 of exponents)
    fl[0] = aie::equal(aie::load_v<16>(w), aie::broadcast<int32_t, 16>(-1)) && w[16] == -1 && w[17] == -1;
}
}
""")


class ExpertGemmOperator(GemmQdOperator):
    """Per column n_c chunks of CHUNK tokens of an expert (module docstring): ``kind`` "gu" / "down", the slab block's
    tile format ``fmt`` of R rows, ``row_tiles`` per expert half (G: its row groups), K the inputs (gu: the hidden
    size), tiles of tile_bytes over kc columns (gu: K chunks in groups of kc_group); ``cap``: chunks per column at most
    (the A and C arguments' slots); ``out``: C's columns per token (gu: the expert's width; down: the hidden size; C's
    rows ``N`` wide: whole passes, Q6_K's 47 tiles of 44 rows as 12 passes of 4: 2112, the last tile's rows the next
    half's bytes, never read). ``of``: from a layer's slab region."""

    GU_ROWS = 128  # rows of a gate / up pass (128: its A strip resent half as often as at 64)

    def __init__(self, *, kind: str, fmt: str, R: int, row_tiles: int, G: int = 1, K: int, tile_bytes: int, kc: int,
                 kc_group: int = 1, cap: int, out: int):
        assert kind in ("gu", "down")
        self.kind, self.cap = kind, cap
        self._templates: dict[tuple, tuple[np.ndarray, list]] = {}
        if kind == "gu":
            tp = self.GU_ROWS // R  # (tiles per pass)
            ph = row_tiles // tp  # passes per half
            kw = {"fmt": fmt, "rows": R, "K": K, "N": out, "passes": 2 * ph,
                  "out_cols": [[self.GU_ROWS // 2 * p for p in range(2 * ph)]] * COLS, "kc_group": kc_group,
                  "region_rt": row_tiles, "silu": True, "act": "silu", "c_bfp16": True,
                  "blocks": ((0, 2 * ph),), "tiles": tp, "pingpong": self.GU_ROWS == 64}
            self.ph = ph
        else:
            tiles = 4
            passes = -(-row_tiles // tiles)
            kw = {"fmt": fmt, "rows": R, "K": 2 * kc, "N": max(out, passes * tiles * G * R), "passes": passes,
                  "out_cols": [[tiles * G * R * p for p in range(passes)]] * COLS, "kc_group": 1,
                  "region_rt": row_tiles, "rt_stride": 1, "groups": G,
                  "blocks": ((0, passes),), "c_pack": True, "slots": 2}
            if planar(fmt, R, kc):
                kw["tiles"] = tiles
            self.ph = passes
        super().__init__(mac_rows=MAC_ROWS, m=M, M=CHUNK, tile_bytes=tile_bytes, col_bytes=0, kc=kc, **kw)
        assert self.mc == 1 and self.dataflow().b_obj > 0
        if kind == "down":
            assert self.slots == 2 or self.G * self.R % 2 == 0, "C runs of whole words"

    @classmethod
    def of(cls, kind: str, region: object, *, cap: int, out: int) -> "ExpertGemmOperator":
        """The operator of a layer's slab region ("exp_gu" / "exp_down")."""
        (b,) = region.blocks
        return cls(kind=kind, fmt=b.fmt, R=b.R, row_tiles=b.tiles, G=b.G, K=region.K, tile_bytes=region.tile_bytes,
                   kc=region.kc, kc_group=region.kc_group, cap=cap, out=out)

    def get_kernel_src(self) -> str:
        return super().get_kernel_src() + _CHK.render(p=self.p)

    def _dq_buffers(self) -> dict[str, tuple[int, int]]:
        """(the down: a header slot set and scale workspace per half)"""
        out = super()._dq_buffers()
        if self.kind == "down":
            out["ws"], out["hb"] = (2, out["ws"][1]), (2, out["hb"][1])
        return out

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def a_slot(self) -> int:
        """bf16 units of a chunk's A (the 3 rows' tokens: panels of KA columns)."""
        return CHUNK * self.KA * 9 // 16

    @property
    def c_slot(self) -> int:
        """bf16 units of a chunk's C."""
        return CHUNK * self.N * 9 // 16 if self.c_bfp16 else CHUNK * self.N

    @functools.cached_property
    def b0_words(self) -> int:
        """Words of block 0: one object of the weight channel's ObjectFifo."""
        return self.dataflow().b_obj

    @functools.cached_property
    def cb_words(self) -> int:
        """32-bit words of a column's control stream at most (``plan``)."""
        per = (2 * len(W_BDS) + 2 * len(A_BDS) + 2 * 2) * 6 + 4 * 3 + ctl.NEXT_WORDS
        return -(-(self.b0_words + self.cap * per + MARKER) // 16) * 16

    @property
    def a_strip(self) -> int:
        """bf16 units of a MAC row's A strip (its 64 tokens' panels; a chunk slot's group-major rows)."""
        return M * self.KA * 9 // 16

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * self.cb_words * 2,), torch.bfloat16),
                ArgSpec("in", (COLS * self.cap * self.a_slot,), torch.bfloat16),
                ArgSpec("out", (COLS * self.cap * self.c_slot,), torch.bfloat16)]

    # ------------------------------------------------------------------
    # Host side: a chunk's tasks
    # ------------------------------------------------------------------

    def w_chains(self, half_bytes: int) -> list[list[TensorAccessPattern]]:
        """A chunk's weight tasks as chains of taps (bf16 units from its expert's slab, the second half half_bytes
        after the first): "gu" per half one chain (a BD per K chunk, [pass, unit, tile, words]); "down" one chain: the
        header units of both halves, then per slot its super-blocks of both halves ([pass, half, tile, words])."""
        tw, (nsl, sw) = self.tile_bytes // 2, self._units
        nt, rs = self.tiles, self.rt_stride
        if self.kind == "gu":
            g = self.kc_group
            return [[TensorAccessPattern((1, 1 << 30), h * half_bytes // 2 + k * tw, [self.ph, nsl, nt, sw],
                                         [nt * rs * tw, sw, rs * tw, 1]) for k in range(g)] for h in range(2)]
        hu, spg = self.hdr_units, self.nsb // self.G
        assert hu == 1 and spg == 1, (hu, spg)
        half = half_bytes // 2
        chain = [TensorAccessPattern((1, 1 << 30), 0, [self.ph, 2, nt, sw], [nt * rs * tw, half, rs * tw, 1])]
        chain += [TensorAccessPattern((1, 1 << 30), (hu + j) * sw, [self.ph, 2, nt, sw], [nt * rs * tw, half, rs * tw, 1])
                  for j in range(self.G)]
        return [chain]

    def a_taps(self) -> list[TensorAccessPattern]:
        """A chunk's A strips (bf16 units from its slot): MAC row r's 64 tokens, [k-tile, token group, 8 k-steps' panels]
        (the memtile's walk lays them out in the MAC's pop order)."""
        aK = self.KA
        return [TensorAccessPattern((1, 1 << 30), r * self.a_strip, [1, aK // KT, M // 8, 8 * 36],
                                    [0, 8 * 36, aK // 8 * 36, 1]) for r in range(MAC_ROWS)]

    def drain_taps(self) -> tuple[TensorAccessPattern, TensorAccessPattern]:
        """A chunk's C drains (bf16 units from its slot), row 0's and rows 1 and 2's: a BD iterating over the chunk's C
        objects (per pass and object ``vp``): "gu" H panels [group][N / 8][36] (a pass: its 4 panels of the rows'
        groups), "down" the pass-major ``y_layout`` (an object: its rows' tokens whole)."""
        if self.kind == "gu":
            g, w = self.N // 8 * 36, self.n_out // 8 * 36  # (a token group's panels, a pass's)
            return tuple(TensorAccessPattern((1, 1 << 30), g0 * g, [self.passes, 1, n, w], [w, 0, g, 1])
                         for g0, n in ((0, M // 8), (M // 8, 2 * M // 8)))
        obj = CHUNK * self.n_out
        return tuple(TensorAccessPattern((1, 1 << 30), t0 * self.n_out, [self.passes * self.vp, 1, 1, n * self.n_out],
                                         [obj, 0, 0, 1]) for t0, n in ((0, M), (M, 2 * M)))

    def chunk_template(self, parity: int, last: bool, active: bool, half_bytes: int) -> ctl.Template:
        """A chunk's block body (its packets) and its address slots (``ctl.template``; bases: 0 the expert's slab, 1
        its C slot, 2 its A slot, 3 the marker panel): the weight tasks (``w_chains``), the A strips (rows 1 and 2's
        the marker unless ``active``), the drains in set ``parity`` (rows 1 and 2's if ``active``; the last chunk's
        pushes with a token)."""
        key = (parity, last, active, half_bytes)
        if key in self._templates:
            return self._templates[key]
        out: list[int] = []
        slots: list[tuple[int, int, int, int]] = []

        def bd(bd_id: int, words: list[int], base: int, offset: int) -> None:
            slots.append((len(out) + 3, len(out) + 4, base, offset))
            out.extend(ctl.bd_packets(bd_id, words))

        ids_w = iter(W_BDS)
        for chain in self.w_chains(half_bytes):
            ids = [next(ids_w) for _ in chain]
            for i, (tap, b) in enumerate(zip(chain, ids)):
                bd(b, ctl.bd_words(0, tap, packet=ctl.XID, next_bd=ids[i + 1] if i + 1 < len(ids) else None), 0,
                   2 * tap.offset - self.hdr_skip)
            out.extend(ctl.push(txn.MM2S, 1, ids[0], repeat=chain[0].sizes[0]))
        marker = TensorAccessPattern((1, 36), 0, [1, 1, 1, 36], [0, 0, 0, 1])
        for r, tap in enumerate(self.a_taps()):
            nxt = A_BDS[r + 1] if r + 1 < len(A_BDS) else None
            if r == 0 or active:
                bd(A_BDS[r], ctl.bd_words(0, tap, next_bd=nxt), 2, 2 * tap.offset)
            else:
                bd(A_BDS[r], ctl.bd_words(0, marker, next_bd=nxt), 3, 0)
        out.extend(ctl.push(txn.MM2S, 0, A_BDS[0]))
        for ch, tap in enumerate(self.drain_taps()[:1 + active]):
            bd(C_BDS[parity][ch], ctl.bd_words(0, tap, release=(LOCKS[parity], 1)), 1, 2 * tap.offset)
            out.extend(ctl.push(txn.S2MM, ch, C_BDS[parity][ch], repeat=tap.sizes[0], token=last))
        self._templates[key] = ctl.template(out, slots)
        return self._templates[key]

    def stream(self, experts: np.ndarray, active: np.ndarray, *, cb: int, slab: int, slab_stride: int,
               c_address: int, a_address: int, out: np.ndarray | None = None) -> np.ndarray:
        """A column's control stream (uint32 words at device address ``cb``; into ``out``, else a new array) for its
        chunks of ``experts`` (chunk k: expert experts[k], its slab region's first half at slab + 2 e slab_stride;
        its C slot at c_address + k slot, its A slot at a_address + k slot; ``active`` (n): rows 1 and 2 hold valid
        rows): the chunks' templates (``chunk_template``) as the blocks of ``ctl.assemble`` (chunk k's drains in set
        k % 2, block k's BD acquiring its set's lock for chunk k - 2's drains); the marker panel at the end."""
        out = np.empty(self.cb_words, np.uint32) if out is None else out
        return self.streams([experts], [active], cb=[cb], slab=slab, slab_stride=slab_stride, c_address=[c_address],
                            a_address=[a_address], out=[out])[0]

    def streams(self, experts: list[np.ndarray], active: list[np.ndarray], *, cb: list[int], slab: int,
                slab_stride: int, c_address: list[int], a_address: list[int], out: list[np.ndarray]
                ) -> list[np.ndarray]:
        """``stream`` of several columns (their blocks computed together): per column its experts, active flags,
        addresses and output array (the column's whole control buffer: the marker panel at its end); returns each
        column's words."""
        n = np.array([len(e) for e in experts])
        assert ((1 <= n) & (n <= self.cap)).all()
        e = np.concatenate(experts).astype(np.int64)
        col = np.repeat(np.arange(len(n)), n)
        k = np.arange(e.size) - np.repeat(np.cumsum(n) - n, n)  # (a chunk's index in its column)
        last = k == n[col] - 1
        act = np.concatenate(active).astype(bool) | last
        ts = [self.chunk_template(p, la, ac, slab_stride) for p in (0, 1) for la in (False, True) for ac in (False, True)]
        blocks = (4 * (k % 2) + 2 * last + act).astype(np.int32)
        marker = [x + 4 * (self.cb_words - MARKER) for x in cb]
        bases = np.stack([slab + 2 * e * slab_stride, np.asarray(c_address, np.int64)[col] + 2 * self.c_slot * k,
                          np.asarray(a_address, np.int64)[col] + 2 * self.a_slot * k,
                          np.asarray(marker, np.int64)[col]], 1)
        # (block k acquires its set's lock for chunk k - 2's drains: an object each of row 0's, of rows 1 and 2's if
        # active; chunks 0 and 1 the lock's initial 1)
        objs = self.passes * self.vp
        rel = objs * (1 + act)
        val = np.where(k >= 2, np.roll(rel, 2), 1)
        locks = np.array(LOCKS, np.int64)[k % 2]
        ends = np.cumsum(n)
        res = []
        for c, (a, b) in enumerate(zip([0, *ends[:-1]], ends)):
            w = ctl.assemble(out[c], ts, blocks[a:b], bases[a:b], base=cb[c], b0_words=self.b0_words, bds=BLOCK_BDS,
                             acquire=(locks[a:b], val[a:b]))
            assert w <= self.cb_words - MARKER, (w, self.cb_words)
            out[c][self.cb_words - MARKER:self.cb_words] = 0xFFFFFFFF
            res.append(out[c][:w])
        return res

    # ------------------------------------------------------------------
    # The down's outputs (Y)
    # ------------------------------------------------------------------

    @property
    def y_layout(self) -> tuple[int, int, int, int]:
        """(down) A chunk slot of Y: (passes, vp, tiles, run): per pass p and C object v the chunk's tokens' runs of
        tiles x run values (tile t: run values of the object's slots, ``GemmQOperator.slots``); a token's output o =
        p tiles vp run + t vp run + v run + r is at ((p vp + v) CHUNK + token) tiles run + t run + r."""
        assert self.kind == "down"
        return self.passes, self.vp, self.tiles, self.slots * self.R

    def y_offset(self, pos: np.ndarray) -> np.ndarray:
        """bf16 offsets in Y of the outputs of rows ``pos`` (``plan.Plan.pos``: chunk slot CHUNK g + token j)."""
        g, j = np.divmod(np.asarray(pos, np.int64), CHUNK)
        return g * self.c_slot + j * self.n_out

    def y_rows(self, y: np.ndarray) -> np.ndarray:
        """(host) Chunk slots of Y (..., c_slot) -> their tokens' outputs in order (..., CHUNK, N)."""
        passes, vp, tiles, run = self.y_layout
        v = y.reshape(*y.shape[:-1], passes, vp, CHUNK, tiles, run)
        nd = v.ndim
        order = (*range(nd - 5), nd - 3, nd - 5, nd - 2, nd - 4, nd - 1)
        return v.transpose(order).reshape(*y.shape[:-1], CHUNK, self.N)

    # ------------------------------------------------------------------
    # The design
    # ------------------------------------------------------------------

    def get_mlir_artifact(self) -> Module:
        MR, m, n, cap = self.nrows, self.m, self.nm, self.cap
        i32, u8 = np.int32, np.uint8
        df = self.dataflow()
        cb, down = self.c_bfp16, self.kind == "down"
        B_l2_ty = np.ndarray[(df.b_obj,), np.dtype[i32]]
        B_l1_ty = np.ndarray[(df.b_l1,), np.dtype[i32]]
        D_ty = np.ndarray[(n * KT * 9 // 8,), np.dtype[u8]]
        strip = m * self.KA * 9 // 16  # (a row's A strip, bf16 units)
        A_l2_ty = np.ndarray[(MR * strip,), np.dtype[bfloat16]]
        A_l1_ty = np.ndarray[(df.a_l1,), np.dtype[bfloat16]]
        c_elems = df.c_l1
        C_l1_ty = np.ndarray[(c_elems,) if cb else (m, self.n_c), np.dtype[bfloat16]]
        C12_ty = np.ndarray[((MR - 1) * c_elems,) if cb else ((MR - 1) * m, self.n_c), np.dtype[bfloat16]]
        acc_ty = np.ndarray[(m, n), np.dtype[bfloat16]]
        sx_ty = np.ndarray[(m * n // 2 // self.sxp,), np.dtype[bfloat16]]
        dqb = self._dq_buffers()
        ws_ty = np.ndarray[(dqb["ws"][1] // 2,), np.dtype[bfloat16]]
        hb_ty = np.ndarray[(dqb["hb"][1] // 2,), np.dtype[bfloat16]]

        p = self.p
        k_zero = Kernel(f"{p}_zero", "kernel.o", [acc_ty])
        k_silu = Kernel(f"{p}_silu", "kernel.o", [acc_ty, C_l1_ty, sx_ty, np.int32]) if self.silu else None
        dk = slot(self.fmt).qd_kernels(self, lambda name, args: Kernel(f"{p}_{name}", "kernel.o", args),
                                       B_l1_ty=B_l1_ty, ws_ty=ws_ty, hb_ty=hb_ty, D_ty=D_ty)
        k_mm = (Kernel(f"{p}_mm2", "kernel.o", [A_l1_ty, D_ty, acc_ty, acc_ty]) if self.pingpong else
                Kernel(f"{p}_mm", "kernel.o", [A_l1_ty, D_ty, acc_ty]))
        k_pack = (Kernel(f"{p}_pack2", "kernel.o", [acc_ty, acc_ty, C_l1_ty]) if self.slots == 2 else
                  Kernel(f"{p}_pack", "kernel.o", [acc_ty, C_l1_ty])) if self.c_pack else None
        pn = [ScratchpadParameter(f"{N_PARAM}{c}", i32) for c in range(COLS)]
        fl_ty = np.ndarray[(16,), np.dtype[i32]]
        k_chk = Kernel(f"{p}_chk", "kernel.o", [A_l1_ty, fl_ty])

        # B: the column's packet-switched MM2S1 -> memtile -> the dequantizing core
        b_l3 = [ObjectFifo(B_l2_ty, name=f"egB_{c}", depth=2, from_stream_per_cons=df.b_wdims, packet=True,
                           packet_id=ctl.XID) for c in range(COLS)]
        b_l1 = [b_l3[c].cons().forward(obj_type=B_l1_ty, name=f"egB1_{c}", depth=self.b_depth, tile=Tile(c, 1),
                                        to_stream=df.b_rdims) for c in range(COLS)]
        d_f = [ObjectFifo(D_ty, name=f"egD_{c}", depth=D_DEPTH) for c in range(COLS)]
        # A: the column's 3 strips as one memtile object (the strips' panels re-laid per strip), split to the rows,
        # each part resent per pass and slot
        # (a split writes its input per part: the walk of one strip)
        a_wdims = [((self.KA // KT) * (m // 16), 8 * 72), (2, 36), (8, 72), (36, 1)]
        a_l3 = [ObjectFifo(A_l2_ty, name=f"egA_{c}", depth=1, from_stream_per_cons=stream_walk(strip, a_wdims))
                for c in range(COLS)]
        rep = self.passes * self.G
        a_l1 = [a_l3[c].cons(channel=A_CH).split([strip * r for r in range(MR)], tile=Tile(c, 1), depths=[2] * MR,
                                      obj_types=[A_l1_ty] * MR, names=[f"egA1_{c}_{r}" for r in range(MR)],
                                      repeat_counts=[rep] * MR if rep > 1 else None) for c in range(COLS)]
        # C: row 0's objects straight to the shim, rows 1 and 2's joined in the memtile
        assert df.c_dims is None, "C objects as the slot lays them out (no memtile walk)"
        c0 = [ObjectFifo(C_l1_ty, name=f"egC0_{c}", depth=self.c_depth) for c in range(COLS)]
        c12, c_l1 = [], [list(c0), [], []]
        for c in range(COLS):
            f = ObjectFifo(C12_ty, name=f"egC_{c}", depth=2)
            c12.append(f)
            parts = f.prod().join([c_elems * i for i in range(MR - 1)], obj_types=[C_l1_ty] * (MR - 1),
                                  names=[f"egC1_{c}_{r}" for r in range(1, MR)], depths=[self.c_depth] * (MR - 1),
                                  tile=Tile(c, 1))
            for r in range(1, MR):
                c_l1[r].append(parts[r - 1])

        passes, planes, nsb = self.passes, self.hdr_units, self.nsb
        two = down  # (the down: the halves' headers both held, ``hb`` / ``ws`` per half)

        def dq_tiles(bf, dfo, dq, ws, sb):
            for kt in range_(4):
                b = bf.acquire(1)
                d = dfo.acquire(1)
                dq(b, ws, d, sb, kt)
                dfo.release(1)
                bf.release(1)

        has_sb, has_chunk = dk.sb is not None, dk.chunk is not None

        def dq_fn(bf, dfo, prm, hdr, dq, *rest):
            sbk = rest[0] if has_sb else None
            chunk = rest[int(has_sb)] if has_chunk else None
            ws0, hb0, *wb = rest[int(has_sb) + int(has_chunk):]
            ws1, hb1 = wb if two else (None, None)
            nk = prm.read() & 0xFF  # (the column's chunks: ``N_PARAM``)
            for _ in range32(nk):
                for _ in range_(passes):
                    if not two:
                        for _ in range(self.n_kc):
                            for pl in range(planes):
                                for kt in range_(4):
                                    b = bf.acquire(1)
                                    hdr(b, hb0 if dk.collect else ws0, *([pl] if dk.unit else []), kt)
                                    bf.release(1)
                            if chunk is not None:
                                chunk(hb0, ws0)
                            for sb in range_(nsb):
                                if sbk is not None:
                                    sbk(hb0, ws0, sb)
                                dq_tiles(bf, dfo, dq, ws0, sb)
                        continue
                    for hb in (hb0, hb1):  # (the halves' header units, in turn)
                        for kt in range_(4):
                            b = bf.acquire(1)
                            hdr(b, hb, kt)
                            bf.release(1)
                    if chunk is not None:
                        chunk(hb0, ws0)
                        chunk(hb1, ws1)
                    for sb in range_(nsb):  # (G = nsb: a slot per super-block, both halves')
                        for hb, ws in ((hb0, ws0), (hb1, ws1)):
                            sbk(hb, ws, sb)
                            dq_tiles(bf, dfo, dq, ws, sb)

        c_pack, silu, pingpong = self.c_pack, self.silu, self.pingpong

        assert c_pack or silu, "C written by an epilogue after the k-tiles"
        n_kt = self.KA // KT

        def mac_fn(dfi, af, cf, zero, mm, prm, chk, fl, *extra):
            def tile(src, dst, first):  # (a k-tile: the first of a slot checks its strip for the marker)
                d = dfi.acquire(1)
                a = af.acquire(1)
                if first:
                    chk(a, fl)
                mm(a, d, src, dst) if dst is not None else mm(a, d, src)
                af.release(1)
                dfi.release(1)

            nk = prm.read() & 0xFF  # (the column's chunks: ``N_PARAM``)
            for _ in range32(nk):
                for _ in range_(passes):
                    for _ in range_(self.vp):
                        for s_ in range(self.slots):
                            acc = extra[1 + s_] if c_pack else extra[1]
                            zero(acc)
                            if pingpong:  # (acc -> accb -> acc: an even k-tile count ends in acc)
                                tile(acc, extra[3], True)
                                tile(extra[3], acc, False)
                                for _ in range_(n_kt // 2 - 1):
                                    tile(acc, extra[3], False)
                                    tile(extra[3], acc, False)
                                continue
                            tile(acc, None, True)
                            for _ in range_(n_kt - 1):
                                tile(acc, None, False)
                        with if_(arith.cmpi(arith.CmpIPredicate.eq, fl[0], arith.constant(T.i32(), 0))):
                            c = cf.acquire(1)
                            if silu:
                                extra[0](extra[1], c, extra[2], 0)
                            else:
                                extra[0](*extra[1:1 + self.slots], c)
                            cf.release(1)

        workers = []
        for c in range(COLS):
            bufs = [Buffer(ws_ty, name=f"eg_ws_{c}"), Buffer(hb_ty, name=f"eg_hb_{c}")]
            if two:
                bufs += [Buffer(ws_ty, name=f"eg_ws1_{c}"), Buffer(hb_ty, name=f"eg_hb1_{c}")]
            kerns = [k for k in (dk.sb, dk.chunk) if k is not None]
            workers.append(Worker(dq_fn, [b_l1[c].cons(), d_f[c].prod(), pn[c], dk.hdr, dk.dq, *kerns, *bufs],
                                  tile=Tile(c, 2), stack_size=2048, dynamic_objfifo_lowering=True))
            for r in range(MR):
                if silu:
                    extra = [k_silu, Buffer(acc_ty, name=f"eg_acc_{c}_{r}"), Buffer(sx_ty, name=f"eg_sx_{c}_{r}")]
                    if pingpong:
                        extra.append(Buffer(acc_ty, name=f"eg_accb_{c}_{r}"))
                elif c_pack:
                    extra = [k_pack] + [Buffer(acc_ty, name=f"eg_acc{'b' if s_ else ''}_{c}_{r}")
                                        for s_ in range(self.slots)]
                workers.append(Worker(mac_fn, [d_f[c].cons(), a_l1[c][r].cons(), c_l1[r][c].prod(), k_zero, k_mm, pn[c],
                                               k_chk, Buffer(fl_ty, name=f"eg_fl_{c}_{r}"), *extra],
                                      tile=Tile(c, 3 + r), stack_size=2048, dynamic_objfifo_lowering=True))

        n_cb, n_a, n_c = COLS * self.cb_words * 2, COLS * cap * self.a_slot, COLS * cap * self.c_slot

        def seq(CB, A, C, bh, ah, ch0, ch12):
            sync_parameters()
            for c in range(COLS):
                for reg, val in ctl.ROUTE:
                    npu_write32(reg, val, column=c, row=0)
                for lk in LOCKS:  # (chunks 0 and 1 acquire 1: ``streams``)
                    npu_write32(txn.SHIM_LOCK0 + 16 * lk, 1, column=c, row=0)
                for ch in (0, 1):  # (the drains' tokens to the controller)
                    npu_maskwrite32(txn.CTRL[txn.S2MM][ch], txn.CONTROLLER_ID << 8, 0x1F00, column=c, row=0)
                npu_maskwrite32(MEM_S2MM_CTRL + 8 * A_CH, 1 << 16, 3 << 16, column=c, row=1)  # (A: finish on TLAST)
            tg = TaskGroup()
            for c in range(COLS):
                bh[c].fill(CB, TensorAccessPattern((n_cb,), 2 * c * self.cb_words, [1, 1, 1, 2 * self.b0_words],
                                                   [0, 0, 0, 1]), packet=(0, ctl.CID), group=tg)
            tg.finish()
            for c in range(COLS):  # (the last chunk's drains: their tokens)
                for ch in (0, 1):
                    _npu_sync(_as_i32(c), _as_i32(0), _as_i32(txn.S2MM), _as_i32(ch), _as_i32(1), _as_i32(1))

        bf = np.dtype[bfloat16]
        rt = Runtime(seq, [np.ndarray[(n_cb,), bf], np.ndarray[(n_a,), bf], np.ndarray[(n_c,), bf],
                           [b_l3[c].prod(tile=Tile(c, 0), channel=1) for c in range(COLS)],
                           [a_l3[c].prod(tile=Tile(c, 0), channel=0) for c in range(COLS)],
                           [c0[c].cons(tile=Tile(c, 0), channel=0) for c in range(COLS)],
                           [c12[c].cons(tile=Tile(c, 0), channel=1) for c in range(COLS)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        layout = self._l1_layout()
        if layout is not None:
            roles = {"egA1": "A", "egB1": "B", "egC1": "C", "egC0": "C", "egD": "D"}

            def worker_buffer(name: str) -> int | None:  # eg_<role>_<col>[_<row>] (ws1, hb1: the second half's)
                role = name.split("_")[1] if name.startswith("eg_") else None
                i = int(role in ("ws1", "hb1"))
                role = role[:-1] if i else role
                return layout[role][i] if role in layout and i < len(layout[role]) else None

            pin_l1_buffers(module, fifo=lambda name, depth: layout.get(roles.get(name.split("_")[0], "")),
                           buffer=worker_buffer)
        return module
