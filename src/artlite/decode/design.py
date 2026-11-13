"""The decode overlay shared by the models run on host-built transactions (qwen35, gemma4, llama): one design whose
cores interpret their inputs, run by host-built transactions (``artlite.overlay``), computing up to ``SLOTS`` (4) tokens
per stream of the weights (8: "Designs of 8 slots"). A model's design (``DecodeOverlayOperator`` subclass) adds its own
cores (qwen35: the GDN and attention cores; gemma4, llama: the attention cores) to the main and helper cores of every
column:

- main (c, 4): the GEMV phases: X objects, its half of every weight tile (``gemv``: super-blocks 0, 1), residuals,
  outputs (C), with ``msg`` a phase's rows to another core (M), with ``relay`` another core's outputs (R) to C;
- helper (c, 3): by commands of the main core (``CMD``, through shared memory, answered on ``RES``): a phase's input
  objects (read from the main core's memory: the slots' x panels into its own, the sum panels into the main
  core's (``gemv``), the norms' 1 / rms), the other half of every tile (super-blocks 2, 3, read
  from the main core's A object), its partial rows back. (Program memory: each core holds the GEMV kernels of every
  format, so the rest of the work is split between them.)

Streams (shim channels fixed: ``ENDPOINTS``): A (weight tiles, norm-weight objects) shim MM2S 0 -> memtile -> main;
X (activation objects, other cores' objects) shim MM2S 1 -> memtile -> main (and the model's cores); C (main's outputs,
objects of ``C_LEN`` bf16) -> shim S2MM 0; D (a model core's outputs) -> shim S2MM 1.

X objects (``XB`` bytes: ``META`` metadata words, then a payload; a phase's input in DDR, its layout, the main core's
program per object): ``decode.protocol``, which declares the words, flags and ids the cores and the host share.

The cores' kernel source: ``design.cc.jinja`` (and ``gemv.py``).

MoE (``moe``: an MoE model's FFN; its transaction: ``qwen35.overlay``): the X stream is packet-switched (``XID``),
its shim channel also carrying control packets into the shim's own TileControl (``moe.route``); the decode's K groups
stop at ``max_k`` (2048), and the x / sum panels have one chunk more, the side chunk. Per layer:

- the input phase (F_NORM) has blocks of their own kinds (the block word's B_*): the shared expert's gate / up (B_GU),
  the router's [hi, lo] rows (B_PAIR: their sums go out, fp32), the shared expert's gate (B_PAIR, B_KEEP: its logit
  stays in the state, S_SG);
- the route object (F_ROUTE: the 8 columns' logits as its payload): the helper routes (``moe.route``), the main core
  emits the control packets and the expert object's metadata (rbuf);
- the shared expert's down (G 2 row groups; F_WS: times the gate's sigmoid; its panels in the side chunk, M_XC0);
- the expert object (F_EXPERT, written by the column itself: its items, M_NT their A objects for the other cores): the
  main core's sub-phases (S_NSUB; set up by the helper from the plan's templates, ``hsub``): per item and half, gate /
  up on the input phase's panels and 1 / rms (F_HOUT: its h as [h | h] strips in the hold buffer), down on two strips
  of the hold buffer in the side chunk (G 4 row groups: a tile's 4 blocks of R rows are rows of their own; F_WS: the
  item's weights; F_ACCUM: into acc, fp32 [slot][hidden] in a model core's scratch, ``_acc``; F_ACC0 stores; F_EMITA:
  acc out, per 128 rows and slot);
- the combine (F_COMB): its input objects are, per slot and half of the column's residual slice, the 8 columns'
  partials of it, which the helper sums and adds into the residual (K_ACC); the residual then goes out.

  A slot's partials are exact sums: every expert half's rows go onto the slot's grid (2^-14 of its rms' binade, from the
  input phase's 1 / rms, the same on every column) before they add into acc, and the combine sums the columns' (fp32)
  before it adds them into the residual, so that a slot's FFN output does not depend on how the union of the step's
  experts is dealt to the columns (``moe.route``), nor on the other slots of the step.

Designs of 8 slots (``slots=8``: a dense model whose formats run at 8 slots, ``gemv``): x per k block one panel of the
8 slots' single bfp16 rows (``gemv``'s "Designs of 8 slots"; the sum panels a hi and a lo panel each), an input object
of ``obj_k`` 256 k ([k block][slot 8][8] bf16; a strip of the helper's is a super-block), a norm-weight object over 16 of
them, a row tile's outputs [g][slot 8][r 8]. The residual goes out as the next input's x lies ([k block][slot SP][8]: a column's
whole rows in DDR, where a slot's objects scattered 16-byte rows, ~22 us a phase at S 8). L1 (the main core's own regions are its 4-slot design's at twice the
slots, but the residual): the main core's workspace holds the residual of slots 0 .. 3, the helper's memory that of 4
.. 7 (``hres``: the main core's south neighbour), the helper's scratch (its GEMV scratch and a K group's 16-k sums)
and the sum of squares lie in the core south of it (``HS8_ROW``: the model's core there leaves it room), the main core's SiLU
table in the helper's memory (``SILU8_AT``): the main core holds a second X object, so that the next object comes in while
the helper converts this one's strips (at 8 rows per k block the objects double). With gather tables (paged formats, tiles
of more than 16 rows, gate / up windows) every byte counts: the cores' buffers pinned (``l1_8``), one X object, one SiLU
copy, smaller stacks (``STACK8``: the pages' frames counted, ``operators.checks``), the residual's pitch 16 past a row.
Their memtiles' channels leave a side 14 BDs: each core half's pages have a channel of their own (``pg_split``). No moe,
post, packed designs.

Shared A objects (``a_bcast``): a fill on column c's A stream with a pair's packet id reaches the memtile of column c and of
the pair's other column, whose cores take the objects as their own, one read of DDR for both. A stream's shim -> memtile
segment is packet-switched for it (so that a memtile input takes two sources), the memtile's A channel pinned (``A_MT``); past
the memtiles nothing changes, and every other fill stays on its column (the stream's own id). Two sources keep no order
between them: a shared fill follows a sync that drained what the columns' objects before it fed, the columns' next
fills wait for it (``Sequence.wait_fills``): the flows never meet, which the router cannot see (it refuses flows that share
receivers: ``--allow-deadlock-prone-routing``). On a shim's switch the packet flows through it take arbiters and msels from 0
up (5 arbiters where several pass), so the MoE control route's msel stays above them (``moe.route.check_route`` refuses a
configuration where a stream uses its arbiter and msel: that stream's words would reach the shim's TileControl).

Broadcast inputs (``bcast``): a phase's [metadata object | x] are the same bytes for every column, and the cores of a column
take the same objects, so one shim (``BC_COL``) reads them once for all (a fill with the packet id ``BC_ID``): a packet flow to
every column's X consumers, on the channels the X objects of a column (residual slices, a model core's objects) reach from their
memtile, packet-switched past it too. Their flows share receivers, which the router refuses (arbiters held across
switchboxes): they never overlap here, each broadcast comes after a sync on the drains of what ran before. Only a step's first
phase comes after local objects with no drains in between: its inputs stay per column (``ModelOverlay._inputs``).

Paged formats (``paged``: a model whose formats' kernels do not fit the cores together): their kernels are code pages
(``paging``, the memtile transport): per column the main and helper cores request a format's page for their half
before a tile of it (``gemv.render(paged=)``) from a store in their memtile (``PG_*``: its channel, BDs, bytes, or a store
per core half: ``pg_split``; the pager's state words and wake lock on both cores), which ``page_store`` links and the device
loads before a dispatch
whenever it is stale (``artlite.overlay.OverlayDevice.set_prologue``: first, then after another hardware context ran).
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, Lock, ObjectFifo, Packet, Port, Program, Runtime, Transport, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.dataflow.objectfifo import ObjectFifoLink
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from .. import txn
from ..formats import FORMATS, TAB, TAB_BYTES, slot
from ..moe import route as moe_k
from ..operators import kmath, paging, templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS, linear_tap, packet_flow
from ..operators.kernel_source import kernel_prefix
from ..operators.l1_layout import pin_l1_buffers
from ..operators.util import Attr, regions
from ..operators.words import Kind
from . import gemv, protocol
from .protocol import (
    A_MT,
    AB_ID,
    AID,
    BC_COL,
    BC_ID,
    C_LEN,
    CMD_WORDS,
    ITEM_WORDS,
    MAX_K,
    META,
    MOE,
    MOE_STATE,
    NBLK,
    NORM_K,
    PG_BD,
    PG_CH,
    PG_PAGE_ID,
    PG_REQ_ID,
    PG_STATE,
    PG_STORE,
    PG_STORE_BYTES,
    PG_WAKE,
    PHASE,
    POST_K,
    POST_STATE,
    S_NASKIP,
    S_NE,
    S_NIN,
    S_NLS,
    S_NM,
    S_NNO,
    S_NPS,
    S_NR,
    S_NSKIP,
    S_NSUB,
    S_NT,
    S_NXN,
    S_NXP,
    S_NXS1,
    S_WORDS,
    SLOTS,
    STATE,
    TOK_STATE,
    TOKEN,
    WINS,
    X_CH,
    XB,
    XC_ID,
    XID,
    XMETA,
    XPAY,
)

_SRC = templating.load("artlite/decode/design.cc.jinja")  # (the main and helper cores' kernels)

WS8_AT = 0x4000  # (8 slots) the main core's workspace (region 1 on; then the sum panels)
# (8 slots) the main core's SiLU table, its two copies in regions 0, 3 of the helper's memory: the room of a second X object
SILU8_AT = (0x0800, 0xF9C0)
# (8 slots with gather tables: ``l1_8``) the main core's table copies in regions 1 (its workspace after it) and 3, the
# helper's in regions 0 and 3 (its scratch is in the core south of it)
TAB8 = ((0x74000, 0x7F780), (0x70400, 0x7F780))
STACK8 = (1536, 1024)  # (8 slots with gather tables) the main and helper cores' stacks (their deepest calls, pages included)


class DecodeOverlayOperator(ArtcOperator):
    """The shared decode overlay design (module docstring): the main and helper cores of every column. ``hidden``: the
    model's hidden size (the norm length, the residual slices; one of part of a K chunk: the phases reading it take
    whole chunks of zero-padded weights, their inputs' padding k zero, which the helper's norm squares); ``rows``: {tile format: rows per tile} of the formats
    the phases use; ``tile_bytes``: the A object size; ``nsb``: super-blocks per tile (its K chunk). The runtime
    sequence (W, ACT, OUT) is a probe that issues one task per stream (``check_endpoints``; never dispatched): the
    host runs the design with transactions.

    A model's subclass sets ``PREFIX`` (kernel and object names), ``msg`` / ``relay`` (the main core's M / R streams)
    and adds its cores: ``_extra_src`` (their kernel source), ``_extra_workers`` (their workers and D streams)."""

    PREFIX = "ovd"
    # (a GEMV dispatcher of core half H, ``gemv``'s ``{p}_mmv<H>``, calls its pages' entries ``{p}_pgf<H>_<format>``
    # through pointers: they are on its stack)
    INDIRECT_CALLS = ((r"_mmv_[0-9a-f]{8}ILi([01])E", r"_pgf\1_"),)
    A_DEPTH, X_DEPTH = 8, 2  # memtile objects of the A and X streams
    # (the A stream's L1 objects: the main core's; A_FWD, the forward's own depth, differs where a model core holds two of
    # them at once: MLIR-AIE then takes each end's depth at its word instead of its largest acquire + 1)
    A_L1, A_FWD = 2, 2
    STACK, HELPER_STACK = 2048, 2048
    MAIN_ROW, HELPER_ROW = 4, 3
    HS8_ROW = 2  # (8 slots: the core whose memory holds the helper's scratch, its south neighbour: a model core there
    # leaves it room)
    SHM = {1: -0x10000, -1: -0x30000}[MAIN_ROW - HELPER_ROW]  # the main core's memory as the helper sees it
    ENDPOINTS: ClassVar[dict[str, tuple[int, int]]] = {"A": (txn.MM2S, 0), "X": (txn.MM2S, 1), "C": (txn.S2MM, 0),
                                                       "D": (txn.S2MM, 1)}
    BD_BASE: ClassVar[dict[str, int]] = {"A": 0, "X": 4, "C": 8, "D": 12}
    msg = relay = False
    # gate: a model's attention whose gates the main core applies (its source: ``{p}_gbegin`` for begin, ``{p}_mgate`` on
    # each skipped X object, ``{p}_mrelay`` for the R objects, R_LEN bf16 elements each: o as fp32, records as bf16)
    gate = False
    R_LEN = C_LEN
    LENDS_WS = False  # (a model core uses the main core's workspace: _Streams.ws, placed on the main core's tile)
    # help_flag: an object of another core whose work the helper shares: the main core's begin commands it (K_HELP:
    # the count at metadata word help_word), then the helper runs ``_help``'s kernel that many times
    help_flag, help_word = 0, 0
    ACC_AT = 0  # (moe) acc's byte offset in ``_acc``'s buffer
    GATE_STATE: ClassVar[tuple[Kind, ...]] = ()  # (gate) the state words (``protocol.STATE``'s kinds) the model's gate code adds
    META_KINDS: ClassVar[tuple[Kind, ...]] = ()  # the kinds of X objects (``protocol.XMETA``) the model's cores add to the design's

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int = 9280, eps: float = 1e-6, nsb: int = 4,
                 act: str = "silu", post: bool = False, pli: bool = False, packed: bool = False, tok: bool = False,
                 gu_win: bool = False, paged: tuple[str, ...] = (), moe: tuple[int, int] | None = None,
                 max_k: int = MAX_K, win: str = "", slots: int = SLOTS, bcast: bool = False,
                 a_bcast: tuple[tuple[int, int], ...] = (), pairs: bool = False):
        """act: the gate/up activation ("silu", "gelu"); post: post-norm residual updates (F_POST, F_VOUT); pli: the
        F_PLI epilogue (gelu(y) times per-layer inputs); packed: the phases' inputs hold the step's slots only
        (``packed_slots``), not 4; tok: token objects (F_TOK: the main cores skip an attention's cache objects by the
        count of its token, M_NASKIP < 0; ``post`` designs have them); gu_win: the gate/up phases' tiles hold other
        than 16 rows (their [8 gate | 8 up] groups cross tiles: a window per slot); paged: formats whose kernels are
        code pages (module docstring); moe: (experts, experts used per token) of the MoE layers' phases (module
        docstring "MoE": the X stream packet-switched, the router's blocks, row-group tiles, the expert's sub-phases, the
        combine); max_k: the K of a phase's group at most (its x panels in the helper's memory; moe: one more chunk,
        the side chunk); slots: the token slots of a step (``gemv``'s x panels; 8: module docstring "Designs of 8
        slots"); bcast: phase inputs on one broadcast stream (module docstring "Broadcast inputs"); a_bcast: pairs of columns
        (c, d): a fill of column c's A stream with the packet id AB_ID + i (i: the pair's index) reaches both columns'
        memtiles, the same objects in both columns (module docstring "Shared A objects"); pairs: a dense design's blocks
        of plane pairs (B_PAIR: an F32 matrix's [hi, lo] rows, their sums out as fp32; an MoE design has them anyway)."""
        assert slots in (4, 8) and (slots == 4 or not (post or pli or tok or moe)), slots
        # (a ``post`` design has TOK_STATE's words too: S_NAS is always declared)
        STATE.together(*([POST_STATE] if post else []), *([TOK_STATE] if tok or post else []),
                       *([MOE_STATE] if moe else []), *(self.GATE_STATE if self.gate else ()))
        XMETA.consts(PHASE, TOKEN, MOE, *self.META_KINDS)  # (the design's words and flags: each name and bit once)
        self.slots, self.bcast = slots, bcast
        assert not (bcast and moe) and (slots == 8 or not bcast), "bcast: designs of 8 slots (placed: ``pin_l1_buffers``)"
        if bcast:
            self.PACKET, self.AIECC_FLAGS = {"X": XID}, ("--allow-deadlock-prone-routing",)  # (module docstring)
        self.obj_k = XPAY // (2 * slots)  # (k of an input object)
        self.nx_per_norm = NORM_K // self.obj_k  # (input objects per norm-weight object)
        self.chunk = gemv.Chunk(nsb, slots)
        self.moe = moe is not None
        self.pairs = pairs
        self.experts, self.top = moe or (0, 0)
        assert max_k % self.chunk.k == 0 and max_k <= MAX_K
        self.max_k = max_k
        self.max_g = max_k // self.chunk.k + int(self.moe)  # K chunks of a phase's group (moe: and the side chunk)
        assert hidden % (COLS * C_LEN) == 0 and hidden <= max_k
        assert not moe or not (post or pli or packed or tok) and act == "silu" and nsb == 4
        assert all(gemv.tile_rows_of(f, tile_bytes, nsb) == r for f, r in rows.items()), rows
        assert all(gemv.supported(f, nsb, slots) for f in rows), (rows, slots)
        assert act in ("silu", "gelu") and (not pli or act == "gelu")
        self.hidden, self.rows, self.tile_bytes, self.eps = hidden, dict(sorted(rows.items())), tile_bytes, eps
        self.act, self.post, self.pli, self.packed, self.tok, self.gu_win = act, post, pli, packed, tok or post, gu_win
        self.paged = tuple(sorted(paged))
        assert set(self.paged) <= set(rows), (paged, rows)
        # (the formats with a gather table: one table slot per core, TAB; a resident format's table resident, else every
        # one in its format's page)
        self.tables = [f for f in self.rows if slot(f).mmv_table(f) is not None]
        self.tab = TAB8 if slots == 8 and self.tables else TAB  # (the tables' copies per core half)
        self.stacks = STACK8 if slots == 8 and self.tables else (self.STACK, self.HELPER_STACK)  # (main, helper)
        assert len([f for f in self.tables if f not in self.paged]) <= 1 and \
            (set(self.tables) <= set(self.paged) or len(self.tables) == 1), (self.tables, self.paged)
        assert not gu_win or act == "silu"
        self.rowsplit = sorted(f for f in rows if gemv.row_split(f, nsb))
        # (the x panels as [core half][chunk]: the two cores' panels of a tile in different L1 regions of the helper's
        # memory, not one chunk's 128 panels in a row: their reads at once conflict; a 512-k input strip is a half)
        self.xh = nsb == 4 and not self.rowsplit and not post
        # (tiles of more than 16 rows, or an MoE design's gate / up windows)
        self.wide = any(r > 16 for r in rows.values()) or (moe is not None and gu_win)
        self.yt = (128 if self.wide else 64) * slots // 4  # (floats of a row tile's outputs)
        # (floats of the helper's partial rows: its row groups of a tile in the super-block split, or half a row-split's)
        # (floats of the helper's partial rows: [g][slot][r] of its row groups of a tile: 4 at 4 slots, the formats' at 8)
        wide_r = max([r for f, r in rows.items() if r > 16 and f not in self.rowsplit], default=0)
        self.yn = (8 * slots * (4 if slots == 4 else -(-wide_r // 8))) if wide_r else gemv.y_floats(slots)
        self.yb = self.yt  # (floats of a tile's row block: 16 rows [g 2][slot][r], 32 rows [g 4][slot][r])
        if self.moe:  # (row groups: a tile's 4 blocks, the helper's two; tiles of more than 16 rows: blocks of 32 rows,
            # their gate / up rows through the windows, ``gu_win``)
            self.yb = 128 if self.wide else 64
            self.yt, self.yn = 4 * self.yb, 2 * self.yb
            self.gu_win = self.gu_win or self.yb == 128
            self.PACKET, self.BD_COUNT = {"X": XID}, {"A": 3}  # (shim BD 3: the experts' tasks, ``moe``)
        self.a_bcast = a_bcast
        if a_bcast:
            self.PACKET, self.AIECC_FLAGS = getattr(self, "PACKET", {}) | {"A": AID}, ("--allow-deadlock-prone-routing",)
        self.win = win  # (a stats build: the event window of a region, WINS)
        self.ablate: tuple[str, ...] = ()  # (a timing build: work compiled out, results wrong; "strip": the input strips)
        self.h8 = hidden // COLS
        # (fp32 per slot: the window adds reach 32 past a row's 16-aligned base, 16 past the row's end; 8 slots with gather
        # tables: just that, ``l1_8``)
        self.res_pitch = -(-(self.h8 + (16 if slots == 8 and self.tables else 32)) // 16) * 16
        self.res_slots = min(slots, 4)  # (the slots of the residual in the main core's workspace; the others' in hres)
        super().__init__()

    @property
    def x_l1_depth(self) -> int:
        """The main core's X objects: two in a design of 8 slots (the next object comes in while the helper converts this
        one's strips), one with gather tables (``l1_8``: their room)."""
        return 2 if self.slots == 8 and not self.tables else 1

    def l1_8(self) -> dict[str, int]:
        """(8 slots with gather tables) The pinned buffers (local addresses) of the main core (``m_*``): A object 0 and the
        X object in region 0 after its stack, table copy 0 at region 1's start, the workspace (its scratch in region 2) and
        the sum panels after it, A object 1 and table copy 1 in region 3; of the helper (``h_*``): table copy 0 after its
        stack, the x panels, slots 4 .. 7's residual (``hres``), its partial rows (one object), the SiLU table (one copy,
        the main core gathers it twice) and ``hst``, table copy 1 (its scratch is in the core south of it: ``HS8_ROW``).
        The rest (statics, C and command objects) goes into the gaps: no room is left for a second X object or SiLU copy
        (``x_l1_depth``, ``silu8_addresses``)."""
        W, (mt, ht) = self.workspace(), self.tab

        def up(n: int) -> int:
            return -(-n // 64) * 64

        out = {"m_a0": self.stacks[0], "m_ws": mt[0] - 0x70000 + TAB_BYTES}
        out["m_x0"] = out["m_a0"] + up(self.tile_bytes)
        out["m_xs"] = out["m_ws"] + W["size"]
        out["m_a1"] = out["m_xs"] + up(self.xs_bytes)
        assert out["m_x0"] + XB + 4 * 2 * C_LEN <= mt[0] - 0x70000, out  # (and the C and command objects)
        assert out["m_a1"] + self.tile_bytes <= mt[1] - 0x70000, out
        assert ht[0] - 0x70000 >= self.stacks[1], (ht, self.stacks)
        out["h_xp"] = ht[0] - 0x70000 + TAB_BYTES
        out["h_hres"] = out["h_xp"] + up(self.xp_bytes)
        out["h_res"] = out["h_hres"] + up(self.hres_bytes)
        out["h_silu"] = out["h_res"] + up(4 * self.yn)
        out["h_hst"] = out["h_silu"] + up(4 * kmath.SILU.table_words().size)
        assert out["h_hst"] + 16 <= ht[1] - 0x70000, out
        return out

    def silu8_addresses(self) -> list[int]:
        """(8 slots) The SiLU table's two copies as the main core reads them in the helper's memory (at 0x40000), with
        gather tables its one copy twice (``l1_8``)."""
        if self.slots != 8:
            return [0, 0]
        return [0x40000 + self.l1_8()["h_silu"]] * 2 if self.tables else [0x40000 + a for a in SILU8_AT]

    @property
    def xp_bytes(self) -> int:
        """The x panels of a K group (in the helper's memory)."""
        return self.max_g * self.chunk.xp

    @property
    def xs_bytes(self) -> int:
        """The sum panels of a K group (in the main core's memory)."""
        return self.max_g * self.chunk.xs

    @property
    def hs_bytes(self) -> int:
        """The helper's scratch: its GEMV scratch and a K group's 16-k sums (fp32: 8 slots' 2 MAX_K bytes)."""
        return max(gemv.scr_bytes(self.slots), MAX_K * self.slots // 4)

    @property
    def hres_bytes(self) -> int:
        """(8 slots) The residual of slots 4 .. 7 (fp32, in the helper's memory: ``hres``)."""
        return 4 * (self.slots - self.res_slots) * self.res_pitch

    def workspace(self) -> dict[str, int]:
        """Byte offsets of the main core's workspace regions (and "size")."""
        S = self.slots
        sizes = [("st", 4 * S_WORDS), ("res", 4 * self.res_slots * self.res_pitch), ("inv", 4 * 8 * S),
                 ("yt", 4 * self.yt), ("stg", 2 * S * 2 * C_LEN), ("gs", 2 * 16 * S), ("scr", gemv.scr_bytes(S))]
        if self.post:
            sizes += [("pend", 4 * S * self.res_pitch), ("ivv", 4 * 32)]
        if self.pli:  # ([g 4][slot][r 8] bf16: p, then pli; e; [slot][8] fp32 sums, then the slots' 1 / rms(p) [slot][r 8]
            # bf16 hi | lo)
            sizes += [("pli", 2 * 2 * 4 * S * 8 + 4 * 32)]
        if self.gu_win:  # (a window of 64 rows per slot for tiles of more than 16 rows, else 32)
            sizes += [("gw", 4 * S * (64 if self.wide else 32))]
        if self.moe:  # (the slots' scales of F_WS, the hold buffer ([h | h] of a half: [k block][slot][8] bf16), the
            # routing's C objects (control packets, the expert object's metadata), the expert object's plan)
            sizes += [("wsc", 4 * 32), ("hold", 2 * S * self.obj_k), ("rbuf", 2 * C_LEN * moe_k.RBUF_OBJECTS),
                      ("plan", 4 * moe_k.PLAN_WORDS)]
        out = regions(sizes)
        assert self.slots == 8 or MAX_K + 2048 <= self.hs_bytes  # (the 16-k sums of a K group, fp32, and hpstrip's nb
        # in the helper's scratch)
        return out

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * self.tile_bytes // 2,), bfloat16),
                ArgSpec("in", (COLS * XB // 2,), bfloat16),
                ArgSpec("out", (2 * COLS * C_LEN,), bfloat16)]

    def _extra_src(self, p: str) -> str:
        """The kernel source of the model's cores."""
        return ""

    @property
    def pfx(self) -> str:
        """The placeholder prefix of the kernels (``kernel_source``), under ``PREFIX``."""
        return kernel_prefix(self.PREFIX)

    def get_kernel_src(self) -> str:
        p = self.pfx
        consts = protocol.consts()
        fmts = list(self.rows)
        assert not self.xh or self.obj_k * 2 * self.slots // 4 == self.chunk.k
        src = gemv.render(f"{p}m", fmts, tile_bytes=self.tile_bytes, nsb=self.chunk.nsb, paged=self.paged,
                          pg=self._page_spec() if self.paged else None, rg=self.moe, yb=self.yb, slots=self.slots,
                          tab=self.tab)
        src += self._extra_src(p)
        gv = {"NSB": self.chunk.nsb, "XP_CHUNK": self.chunk.xp, "XS_CHUNK": self.chunk.xs,
              "XS_SLOT": gemv.xs_slot(self.slots),
              "XPH": self.max_g * self.chunk.xp // 2, "XPHC": self.chunk.xp // 2,
              "PANEL": gemv.PANEL,
              "Y_FLOATS": gemv.Y_FLOATS}
        return src + _SRC.render(p=p, mp=f"{p}m", w_=Attr(self.workspace()), H=self.hidden, H8=self.h8, SHM=self.SHM,
                                 ACC_AT=self.ACC_AT,
                                 WIN=WINS.get(self.win, 0),
                                 RES_PITCH=self.res_pitch, eps=repr(self.eps), ple_eps=repr(self.eps * self.hidden),
                                 META=META, C_LEN=C_LEN, NBLK=NBLK,
                                 SLOTS=self.slots, XPAY=XPAY, OBJ_K=self.obj_k, NX_PER_NORM=self.nx_per_norm,
                                 RES_SLOTS=self.res_slots, wide=self.wide, gv=Attr(gv),
                                 rows=list(self.rows.items()), fid={n: f.fid for n, f in FORMATS.items()},
                                 rowsplit=bool(self.rowsplit), rows_split=self.rowsplit,
                                 helper_y=[(f, gemv.helper_y(f, self.chunk.nsb)) for f in self.rowsplit
                                           if gemv.helper_y(f, self.chunk.nsb)],
                                 YT=self.yt, YB=self.yb, YN=self.yn, act=self.act, post=self.post, pli=self.pli, tok=self.tok,
                                 gu_win=self.gu_win, POST_K=POST_K, xh=self.xh,
                                 COLS=COLS, paged=bool(self.paged),
                                 packed=self.packed, MAX_K=MAX_K, ablate=self.ablate, SILU8=self.silu8_addresses(),
                                 moe=self.moe, pairs=self.pairs, gate=self.gate, help_flag=self.help_flag, help_word=self.help_word,
                                 **self._moe_vars(p, consts), **consts)

    def _moe_vars(self, p: str, consts: dict[str, int]) -> dict[str, object]:
        """(moe) The template variables of the MoE paths: the routing's source (``moe.route``), the side chunk, the
        plan's layout."""
        if not self.moe:
            return {}
        route = moe_k.render(p, consts | {"META": META, "SLOTS": self.slots, "ITEM_WORDS": ITEM_WORDS,
                                          "w_": Attr(self.workspace())}, experts=self.experts, top=self.top, cols=COLS,
                             a_packet=AID if self.a_bcast else None)
        return {"route": route, "SIDE": self.max_g - 1, "PLAN_SUBS": moe_k.PLAN_SUBS, "PLAN_META": moe_k.PLAN_META,
                "ITEM_WORDS": ITEM_WORDS}

    @property
    def pg_split(self) -> bool:
        """Whether each core half's pages have a memtile channel of their own (``pg_channels``): a design of 8 slots, whose
        memtile channels take BDs 0-9 and 24-33, leaves no side room for the BDs of both halves."""
        return self.slots == 8

    @property
    def pg_channels(self) -> tuple[int, int]:
        """The memtile MM2S channel of each core half's pages: PG_CH (odd: BDs 24-47) for up to 4 paged formats or an MoE
        design, else 4 (even: BDs 0-23; a dense design's memtile channels take BDs 0-3 and 24-39 of them); ``pg_split``: 4
        and 5."""
        if self.pg_split:
            return 4, 5
        ch = PG_CH if len(self.paged) <= 4 or self.moe else 4
        return ch, ch

    @property
    def pg_store(self) -> tuple[int, int, int]:
        """(first BD, store address, bytes) of the page store: PG_BD, PG_STORE, PG_STORE_BYTES for up to 4 paged
        formats; of more, a BD per format and core half up to the channel's last (``pg_channels``' parity: 47 or 23), and
        4 KB per page (pages of ~3-4 KB), 12 KB more per paged format with a gather table (its two copies in the page of
        each core half: ~6 KB of control packets each), the store ending at the memtile's end (``pg_split``: the first BD
        of the main cores' pages; the helpers' end at BD 47, their bytes the second half)."""
        n = len(self.paged)
        if n <= 4 and not self.tables and not self.pg_split:
            return PG_BD, PG_STORE, PG_STORE_BYTES
        size = max(0xC000, 0x2000 * n) + 0x3000 * len([f for f in self.tables if f in self.paged])
        ch = self.pg_channels[0]
        return 24 * (ch % 2) + 24 - (1 if self.pg_split else 2) * n, 0x80000 - size, size

    def pg_halves(self) -> tuple[tuple[int, int, int, int], ...]:
        """Per page store (``pg_split``: per core half) (channel, first BD, store address, bytes)."""
        bd0, base, size = self.pg_store
        if not self.pg_split:
            return ((self.pg_channels[0], bd0, base, size),)
        n = len(self.paged)
        return ((4, bd0, base, size // 2), (5, 48 - n, base + size // 2, size - size // 2))

    def _page_spec(self) -> paging.PageSpec:
        """The pager of the paged formats (main core: half 0, helper: half 1)."""
        (h0, c0), (h1, c1) = (paging.request(stream_id=i, channel=ch) for i, ch in zip(PG_REQ_ID, self.pg_channels))
        bd0 = self.pg_store[0]
        bd1 = self.pg_halves()[1][1] if self.pg_split else bd0 + len(self.paged)
        return paging.PageSpec(state=0x70000 + PG_STATE, wake=PG_WAKE, hdr=(h0, h1), bd0=(bd0, bd1), ctl=(c0, c1))

    def page_store(self) -> tuple[paging.Store, ...] | None:
        """The compiled design's page stores (the paged formats' pages of the main cores, then of the helpers; ``pg_split``:
        a store per core half), checked against the design's configuration; None without paged formats."""
        if not self.paged:
            return None
        wd = os.path.dirname(self._program.insts_path)
        fp = f"{self.pfx}m"

        def pager(h: int, row: int, stack: int) -> paging.Pager:
            pages = gemv.page_entries(fp, self.paged, h)
            return paging.link(wd, pages=pages, pager=f"{fp}_pg_request",
                               calls={f"{fp}_mmv": tuple(e for es in pages.values() for e in es)},
                               roots=(f"{self.pfx}_tile",) if h == 0 else (f"{self.pfx}_htask",), stack=stack,
                               cores=[(c, row) for c in range(COLS)], state=PG_STATE, wake=PG_WAKE,
                               stream_id=PG_PAGE_ID[h])
        pagers = []
        for h, (row, stack) in enumerate(zip((self.MAIN_ROW, self.HELPER_ROW), self.stacks)):
            pg = pager(h, row, stack)
            data = {f: tuple((a - 0x70000, slot(f).mmv_table(f)) for a in self.tab[h]) for f in self.paged if f in self.tables}
            pagers.append(dataclasses.replace(pg, data=data) if data else pg)
        halves = self.pg_halves()
        groups = [tuple(pagers)] if len(halves) == 1 else [(pg,) for pg in pagers]
        stores = tuple(paging.Store(base=base, size=size, channel=ch, bd0=bd0, pagers=g)
                       for (ch, bd0, base, size), g in zip(halves, groups))
        for store in stores:
            store.check(wd, range(COLS))
        return stores

    def page_prologue(self, stores: tuple[paging.Store, ...]) -> np.ndarray:
        """The transaction that makes the pages loadable: the stores and their BDs in every memtile, every paging core's
        state words cleared (a configuration rewrites the slot's start with the startup code)."""
        t = txn.Txn()
        for store in stores:
            store.load(t, range(COLS))
        for c in range(COLS):
            for row in (self.MAIN_ROW, self.HELPER_ROW):
                t.write32(txn.tile_address(c, row, PG_STATE), 0)
        return t.words()

    @property
    def d_cols(self) -> list[int]:
        """The columns whose model cores drain a D stream."""
        return []

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        """The model cores' workers and their D streams (column, ObjectFifo, in ``d_cols`` order), on the shared
        streams ``s`` (their locks: ``_locks``)."""
        return [], []

    def _acc_type(self) -> type:
        """(moe) The type of ``_acc``'s buffers."""
        raise NotImplementedError

    def _acc(self, c: int) -> Buffer:
        """(moe) Column c's memory of acc (fp32 [slot][hidden] at ``ACC_AT``, a window of 32 past its end): scratch of a
        model core that the main core reaches (north of it) and that the MoE phases leave idle."""
        raise NotImplementedError

    def _pins(self) -> dict[str, int]:
        """L1 addresses of the model cores' buffers (their symbol names)."""
        return {}

    def _x_rows(self) -> tuple[int, ...]:
        """(bcast) The rows of the model's cores that take the X stream."""
        return ()

    def _help(self, c: int) -> tuple[Kernel, Buffer]:
        """(help_flag) Column c's helper's model kernel (kernel(buffer, the x panels)) and its buffer."""
        raise NotImplementedError

    def get_mlir_artifact(self) -> Module:
        p, P = self.pfx, self.PREFIX
        bf, u8, i32, f32 = np.dtype[bfloat16], np.dtype[np.uint8], np.dtype[np.int32], np.dtype[np.float32]
        a_ty = np.ndarray[(self.tile_bytes,), u8]
        x_ty = np.ndarray[(XB // 2,), bf]
        c_ty = np.ndarray[(C_LEN,), bf]
        rr_ty = np.ndarray[(self.R_LEN,), bf]  # (an R object)
        cmd_ty = np.ndarray[(CMD_WORDS,), i32]
        r_ty = np.ndarray[(self.yn,), f32]
        W = self.workspace()
        ws_ty = np.ndarray[(W["size"] // 4,), i32]
        xp_ty = np.ndarray[(self.xp_bytes,), u8]
        xs_ty = np.ndarray[(self.xs_bytes,), u8]
        hs_ty = np.ndarray[(self.hs_bytes,), u8]
        sq_ty = np.ndarray[(8 * self.slots,), f32]
        hres_ty = np.ndarray[(max(self.hres_bytes, 4) // 4,), f32]  # (8 slots: slots 4 .. 7's residual)
        s8 = self.slots == 8
        i_st = W["st"] // 4

        k_begin = Kernel(f"{p}_begin", "kernel.o", [x_ty, ws_ty, cmd_ty, np.int32])
        k_nbegin = Kernel(f"{p}_nbegin", "kernel.o", [a_ty, ws_ty])
        k_strip_n = Kernel(f"{p}_strip_n", "kernel.o", [x_ty, a_ty, ws_ty, cmd_ty])
        k_strip = Kernel(f"{p}_strip", "kernel.o", [x_ty, ws_ty, cmd_ty])
        k_input = Kernel(f"{p}_input", "kernel.o", [ws_ty, cmd_ty, xs_ty])
        k_inv = Kernel(f"{p}_inv", "kernel.o", [r_ty, ws_ty])
        k_cmd = Kernel(f"{p}_cmd", "kernel.o", [a_ty, ws_ty, cmd_ty, xs_ty])
        k_tile = Kernel(f"{p}_tile", "kernel.o", [a_ty, ws_ty, xp_ty, xs_ty])
        k_collect = Kernel(f"{p}_collect", "kernel.o", [r_ty, ws_ty, *[hres_ty] * s8])
        k_emit = Kernel(f"{p}_emit", "kernel.o", [ws_ty, c_ty, *[hres_ty] * s8])
        k_relay = (Kernel(f"{p}_mrelay", "kernel.o", [rr_ty, c_ty, ws_ty, xp_ty]) if self.gate else
                   Kernel(f"{p}_relay", "kernel.o", [c_ty, c_ty]))
        k_xgate = Kernel(f"{p}_mgate", "kernel.o", [x_ty, ws_ty, xp_ty]) if self.gate else None
        hst_ty = np.ndarray[(4,), i32]  # (help_flag: the helper's count of model work)
        k_htask = Kernel(f"{p}_htask", "kernel.o", [cmd_ty, r_ty, hs_ty, sq_ty, xp_ty, *[hres_ty] * s8,
                                                    *[hst_ty] * bool(self.help_flag)])
        k_psums = Kernel(f"{p}_psums", "kernel.o", [x_ty, ws_ty]) if self.post else None
        if self.moe:  # (acc: the experts' partials of the column, fp32 on the slots' grids, in ``_acc``'s buffer)
            aw_ty = self._acc_type()
            k_collect = Kernel(f"{p}_collect", "kernel.o", [r_ty, ws_ty, aw_ty])
            k_emit = Kernel(f"{p}_emit", "kernel.o", [ws_ty, c_ty, aw_ty])
            k_sub = Kernel(f"{p}_sub", "kernel.o", [ws_ty, cmd_ty])
            k_lstrip = Kernel(f"{p}_lstrip", "kernel.o", [ws_ty, cmd_ty])

        mr, hr = self.MAIN_ROW, self.HELPER_ROW
        akw, amt = ({"transport": Transport.dma(Packet(AID))}, {"port": Port.dma(A_MT)}) if self.a_bcast else ({}, {})
        a_l3 = [ObjectFifo(a_ty, name=f"{P}_A{c}", depth=self.A_DEPTH, **akw) for c in range(COLS)]
        a_l1 = [a_l3[c].cons(depth=self.A_DEPTH, **amt).forward(tile=Tile(c, 1), depth=self.A_FWD, name=f"{P}_a{c}")
                for c in range(COLS)]
        # (moe: the X stream packet-switched, its channel also carries the control packets into the shim's TileControl)
        x_l3 = [ObjectFifo(x_ty, name=f"{P}_X{c}", depth=self.X_DEPTH,
                           **({"transport": Transport.dma(Packet(XID))} if self.moe or self.bcast else {})) for c in range(COLS)]
        if self.bcast:  # (past the memtile too: the broadcast shares the cores' channels)
            x_l1 = [ObjectFifo(x_ty, name=f"{P}_x{c}", depth=1, transport=Transport.dma(Packet(XC_ID))) for c in range(COLS)]
            for c in range(COLS):
                ObjectFifoLink(x_l3[c].cons(depth=self.X_DEPTH), x_l1[c].prod(), Tile(c, 1))
        else:
            x_l1 = [x_l3[c].cons(depth=self.X_DEPTH).forward(tile=Tile(c, 1), depth=1, name=f"{P}_x{c}")
                    for c in range(COLS)]
        x_kw = {"port": Port.dma(X_CH)} if self.bcast else {}  # (a consumer's X channel: the broadcast's route names it)
        c_fifos = [ObjectFifo(c_ty, name=f"{P}_C{c}", depth=2) for c in range(COLS)]
        cmds = [ObjectFifo(cmd_ty, name=f"{P}_cmd{c}", depth=2) for c in range(COLS)]
        m_fifos = [ObjectFifo(c_ty, name=f"{P}_M{c}", depth=2) for c in range(COLS)] if self.msg else []
        r_fifos = [ObjectFifo(rr_ty, name=f"{P}_R{c}", depth=2) for c in range(COLS)] if self.relay else []
        ress = [ObjectFifo(r_ty, name=f"{P}_res{c}", depth=1 if s8 and self.tables else 2) for c in range(COLS)]

        def main_fn(a_f, x_f, c_f, cmd_f, res_f, *rest):
            rest = list(rest)
            if s8:  # (the SiLU table's copies: placed, read through their addresses)
                rest = rest[:-(1 if self.tables else 2)]
            if self.tables:  # (the gather tables: placed, read through their addresses)
                rest = rest[:-2]
            if self.paged:  # (the pager's state words and the column's page store: placed, used by no code here)
                rest = rest[:-2]
            m_f = rest.pop(0) if self.msg else None
            r_f = rest.pop(0) if self.relay else None
            if self.moe:
                sub, lstrip, acc, *rest = rest
            elif s8:
                acc, *rest = rest  # (8 slots: hres)
            xgate = rest.pop(0) if self.gate else None
            begin, nbegin, strip_n, strip, inp, inv, tcmd, tile, collect, emit, relay, *psums, ws, xp, xs, col = rest
            ea = (acc,) if self.moe or s8 else ()  # (collect's and emit's last argument: moe's acc buffer, 8 slots' hres)

            def command(call: object) -> None:  # a strip of the input object: the helper's command
                c = cmd_f.acquire(1)
                call(c)
                cmd_f.release(1)

            def answer() -> None:
                res_f.acquire(1)
                res_f.release(1)

            def strips(call: object) -> None:  # (packed: the object's strips, the next command before an answer)
                command(call)
                if self.packed:
                    for _ in range_(ws[i_st + S_NXS1]):
                        command(call)
                        answer()
                answer()

            def emits() -> None:
                for _ in range_(ws[i_st + S_NE]):
                    o = c_f.acquire(1)
                    emit(ws, o, *ea)
                    c_f.release(1)
                if m_f is not None:
                    for _ in range_(ws[i_st + S_NM]):
                        o = m_f.acquire(1)
                        emit(ws, o, *ea)
                        m_f.release(1)

            for _ in forever_():
                x = x_f.acquire(1)
                c = cmd_f.acquire(1)
                begin(x, ws, c, col)  # (a phase's metadata: the helper's command fills our state)
                cmd_f.release(1)
                answer()
                x_f.release(1)
                if psums:
                    for _ in range_(ws[i_st + S_NPS]):  # (a header: the partial sums object)
                        x = x_f.acquire(1)
                        psums[0](x, ws)
                        x_f.release(1)
                for _ in range_(ws[i_st + S_NNO]):  # the input under its norm-weight objects (the helper's work)
                    a = a_f.acquire(1)
                    nbegin(a, ws)
                    for _ in range_(ws[i_st + S_NXN]):
                        x = x_f.acquire(1)
                        strips(lambda c, x=x, a=a: strip_n(x, a, ws, c))
                        x_f.release(1)
                    a_f.release(1)
                for _ in range_(ws[i_st + S_NXP]):  # or without
                    x = x_f.acquire(1)
                    strips(lambda c, x=x: strip(x, ws, c))
                    x_f.release(1)
                def inputs() -> None:  # the sum panels, the 1 / rms
                    for _ in range_(ws[i_st + S_NIN]):
                        c = cmd_f.acquire(1)
                        inp(ws, c, xs)
                        cmd_f.release(1)
                        r = res_f.acquire(1)
                        inv(r, ws)
                        res_f.release(1)

                def tiles() -> None:
                    for _ in range_(ws[i_st + S_NT]):
                        a = a_f.acquire(1)
                        c = cmd_f.acquire(1)
                        tcmd(a, ws, c, xs)
                        cmd_f.release(1)
                        tile(a, ws, xp, xs)
                        r = res_f.acquire(1)
                        collect(r, ws, *ea)
                        res_f.release(1)
                        a_f.release(1)
                        emits()  # (after the last tile: the phase's last outputs too)

                inputs()
                if psums or self.moe:
                    emits()  # (a header: the updated residual slice; moe: the routing's objects, the combine's residual)
                tiles()
                if self.moe:
                    for _ in range_(ws[i_st + S_NSUB]):  # an expert object's sub-phases: the helper sets each up
                        command(lambda c: sub(ws, c))
                        answer()
                        for _ in range_(ws[i_st + S_NLS]):  # (a down's input: strips from the hold buffer)
                            command(lambda c: lstrip(ws, c))
                            answer()
                        inputs()
                        tiles()
                for _ in range_(ws[i_st + S_NSKIP]):  # the other cores' X objects (gate: a q object's gates)
                    x = x_f.acquire(1)
                    if xgate is not None:
                        xgate(x, ws, xp)
                    x_f.release(1)
                for _ in range_(ws[i_st + S_NASKIP]):  # and A objects
                    a_f.acquire(1)
                    a_f.release(1)
                if r_f is not None:
                    for _ in range_(ws[i_st + S_NR]):  # the model core's outputs -> C
                        r = r_f.acquire(1)
                        o = c_f.acquire(1)
                        if self.gate:
                            relay(r, o, ws, xp)
                        else:
                            relay(r, o)
                        c_f.release(1)
                        r_f.release(1)

        def helper_fn(cmd_f, res_f, htask, hs, sq, xp, *rest):
            eh = rest[:1] if s8 else ()  # (8 slots: hres; else the pager's state words, the grid tables: placed)
            hp = rest[1:4] if self.help_flag else ()  # (hst, the model's kernel, its buffer)
            for _ in forever_():
                c = cmd_f.acquire(1)
                r = res_f.acquire(1)
                htask(c, r, hs, sq, xp, *eh, *hp[:1])
                res_f.release(1)
                cmd_f.release(1)
                if hp:
                    hst, hk, hb = hp
                    for _ in range_(hst[0]):  # (the model core's work, K_HELP)
                        hk(hb, xp)

        workers, pg_flows, pg_locks, pg_main, pg_help = [], [], [], {}, {}
        self._locks: list[Lock] = []  # (the model's: ``_extra_workers``)
        self._flows: list = []  # (the model's packet flows: ``_extra_workers``)
        xps = []
        m_ws, m_xs, sqs = [], [], []
        st_ty = np.ndarray[(paging.STATE_WORDS,), i32]
        for c in range(COLS):  # (paged formats: per core its state words and wake lock, per column its store and routes)
            if not self.paged:
                break
            mem, main, helper = Tile(c, 1), Tile(c, mr), Tile(c, hr)
            store = Buffer(np.ndarray[(self.pg_store[2] // 4,), i32], tile=mem, name=f"{P}_pgstore{c}",
                           address=self.pg_store[1])
            pg_main[c] = [Buffer(st_ty, tile=main, name=f"{P}_pgst{c}_{mr}", address=PG_STATE), store]
            pg_help[c] = [Buffer(st_ty, tile=helper, name=f"{P}_pgst{c}_{hr}", address=PG_STATE)]
            pg_locks += [Lock(main, lock_id=PG_WAKE, init=0, name=f"{P}_pgw{c}_{mr}"),
                         Lock(helper, lock_id=PG_WAKE, init=0, name=f"{P}_pgw{c}_{hr}")]
            for h, t in enumerate((main, helper)):
                pg_flows += [paging.page_flow(mem, t, channel=self.pg_channels[h], stream_id=PG_PAGE_ID[h]),
                             paging.request_flow(t, mem, stream_id=PG_REQ_ID[h])]
        grids = {}  # (the gather tables: per core two copies at TAB, used through their addresses only; ``tables``)
        if self.tables:
            t_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
            res = [f for f in self.tables if f not in self.paged]
            init = np.zeros(TAB_BYTES // 4, np.uint32)
            if res:  # (a resident table; else every table comes with its format's page)
                tab = slot(res[0]).mmv_table(res[0])
                init[:tab.size] = tab
            for c in range(COLS):
                for h, r in enumerate((mr, hr)):
                    grids[c, h] = [Buffer(t_ty, tile=Tile(c, r), name=f"{P}_grid{c}_{r}{i}",
                                          address=self.tab[h][i] - 0x70000, initial_value=init)
                                   for i in range(2)]
        def silu_tab(c: int) -> list[Buffer]:  # (placed, read through their addresses)
            if not s8:
                return []
            words = kmath.SILU.table_words()
            at = [self.l1_8()["h_silu"]] if self.tables else list(SILU8_AT)
            return [Buffer(np.ndarray[(words.size,), np.dtype[np.uint32]], name=f"{P}_msilu{c}_{i}", tile=Tile(c, hr), address=a,
                           initial_value=words) for i, a in enumerate(at)]

        # (paged: the main cores take COLS + their column, not a constant 0 in column 0, which can give its control code
        # another size: every core's resident kernels must lie where the pages were linked, ``paging.link``)
        for c in range(COLS):
            ws = Buffer(ws_ty, name=f"{P}_ws{c}", **({"tile": Tile(c, mr)} if self.LENDS_WS else {}))
            m_ws.append(ws)
            moe_args = [k_sub, k_lstrip, self._acc(c)] if self.moe else []
            hres = [Buffer(hres_ty, name=f"{P}_hres{c}", tile=Tile(c, hr))] if s8 else []
            xp = Buffer(xp_ty, name=f"{P}_xp{c}", tile=Tile(c, hr))  # (the x panels in the helper's memory)
            xps.append(xp)
            xs = Buffer(xs_ty, name=f"{P}_xs{c}", tile=Tile(c, mr))
            m_xs.append(xs)
            hs = Buffer(hs_ty, name=f"{P}_hs{c}", tile=Tile(c, self.HS8_ROW if s8 else hr))
            sq = Buffer(sq_ty, name=f"{P}_sq{c}", tile=Tile(c, self.HS8_ROW if s8 else hr))
            sqs.append(sq)
            workers.append(Worker(main_fn, [a_l1[c].cons(depth=self.A_L1), x_l1[c].cons(depth=self.x_l1_depth, **x_kw), c_fifos[c].prod(), cmds[c].prod(),
                                            ress[c].cons(), *([m_fifos[c].prod()] if self.msg else []),
                                            *([r_fifos[c].cons()] if self.relay else []), *moe_args, *hres,
                                            *([k_xgate] if self.gate else []), k_begin, k_nbegin,
                                            k_strip_n,
                                            k_strip, k_input, k_inv, k_cmd, k_tile, k_collect, k_emit, k_relay,
                                            *([k_psums] if self.post else []), ws, xp, xs,
                                            (COLS if self.paged else 0) + c, *pg_main.get(c, []),
                                            *grids.get((c, 0), []), *silu_tab(c)],
                                  tile=Tile(c, mr), stack_size=self.stacks[0], dynamic_objfifo_lowering=True,
                                  while_true=False))
            help_args = []
            if self.help_flag:
                hk, hb = self._help(c)
                help_args = [Buffer(hst_ty, name=f"{P}_hst{c}", tile=Tile(c, hr), initial_value=np.zeros(4, np.int32)),
                             hk, hb]
            workers.append(Worker(helper_fn, [cmds[c].cons(), ress[c].prod(), k_htask, hs, sq, xp, *hres, *help_args,
                                              *pg_help.get(c, []), *grids.get((c, 1), [])],
                                  tile=Tile(c, hr),
                                  stack_size=self.stacks[1], dynamic_objfifo_lowering=True, while_true=False))
        s = _Streams(p=p, a_ty=a_ty, x_ty=x_ty, c_ty=c_ty, r_ty=rr_ty, a_l1=a_l1, x_l1=x_l1, m_fifos=m_fifos, r_fifos=r_fifos,
                     ws=m_ws, xs=m_xs, xp=xps, sq=sqs, x_kw=x_kw)
        extra, d_fifos = self._extra_workers(s)
        workers += extra
        a_prod = [f.prod(tile=Tile(c, 0), port=Port.dma(self.ENDPOINTS["A"][1])) for c, f in enumerate(a_l3)]
        x_prod = [f.prod(tile=Tile(c, 0), port=Port.dma(self.ENDPOINTS["X"][1])) for c, f in enumerate(x_l3)]
        c_cons = [f.cons(tile=Tile(c, 0), port=Port.dma(self.ENDPOINTS["C"][1])) for c, f in enumerate(c_fifos)]
        d_cons = [f.cons(tile=Tile(c, 0), port=Port.dma(self.ENDPOINTS["D"][1])) for c, f in d_fifos]
        d_len = [math.prod(f.shape) * np.dtype(f.dtype).itemsize // 2 for _, f in d_fifos]  # (whole objects)
        a_n, x_n, c_n = COLS * self.tile_bytes // 2, COLS * XB // 2, COLS * C_LEN + sum(d_len)

        def seq(W_, ACT, OUT, a_h, x_h, c_h, d_h):  # the probe: one task per stream, in ENDPOINTS order
            for c in range(COLS):
                a_h[c].fill(W_, linear_tap(a_n, c * self.tile_bytes // 2, self.tile_bytes // 2))
            for c in range(COLS):
                x_h[c].fill(ACT, linear_tap(x_n, c * XB // 2, XB // 2))
            for c in range(COLS):
                c_h[c].drain(OUT, linear_tap(c_n, c * C_LEN, C_LEN), wait=True)
            for c in range(len(d_h)):
                d_h[c].drain(OUT, linear_tap(c_n, COLS * C_LEN + sum(d_len[:c]), d_len[c]), wait=True)

        rt = Runtime(seq, [np.ndarray[(a_n,), bf], np.ndarray[(x_n,), bf], np.ndarray[(c_n,), bf],
                           a_prod, x_prod, c_cons, d_cons])
        for f in pg_flows + self._flows:
            rt.add_flow(f)
        for lk in pg_locks + self._locks:
            rt.add_lock(lk)
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        at = self._pins()
        pools: dict[str, list[int]] = {}
        if s8 and self.tables:  # (``l1_8``)
            L = self.l1_8()
            for c in range(COLS):
                at |= {f"{P}_ws{c}": L["m_ws"], f"{P}_xs{c}": L["m_xs"], f"{P}_xp{c}": L["h_xp"], f"{P}_hres{c}": L["h_hres"],
                       f"{P}_hst{c}": L["h_hst"]}
                pools |= {f"{P}_a{c}__cons0": [L["m_a0"], L["m_a1"]], f"{P}_x{c}__cons0": [L["m_x0"]], f"{P}_res{c}": [L["h_res"]]}
        elif s8:  # (the workspace and the sum panels in the main core's regions 1, 2: the A buffers in regions 0, 3)
            at |= {f"{P}_ws{c}": WS8_AT for c in range(COLS)} | {f"{P}_xs{c}": WS8_AT + W["size"] for c in range(COLS)}
            assert WS8_AT + W["size"] + self.xs_bytes + self.tile_bytes + XB + 4 * 2 * C_LEN <= 0x10000
        if at or pools:
            pin_l1_buffers(module, fifo=lambda name, depth: pools.get(name), buffer=at.get)
        if self.bcast:  # (on the placed design: the placer takes a flow from a shim channel an ObjectFifo uses for a channel more)
            packet_flow(module, BC_ID, (BC_COL, 0), self.ENDPOINTS["X"][1],
                        [(c, r, X_CH) for c in range(COLS) for r in (mr, *self._x_rows())])
        if self.a_bcast and not s8:  # (placed: ``packet_flow``)
            pin_l1_buffers(module, fifo=lambda name, depth: None)
        for i, (c, d) in enumerate(self.a_bcast):
            packet_flow(module, AB_ID + i, (c, 0), self.ENDPOINTS["A"][1], [(c, 1, A_MT), (d, 1, A_MT)])
        return module

    def check_endpoints(self, insts: np.ndarray) -> None:
        """Assert that aiecc gave every stream the shim channel ``ENDPOINTS`` names (the probe's queue pushes)."""
        queue = {reg: (d, ch) for d in (txn.S2MM, txn.MM2S) for ch, reg in enumerate(txn.QUEUE[d])}
        got = [(f["addr"] >> 25, *queue[f["addr"] & 0xFFFFF]) for name, f in txn.decode(insts)
               if name == "write32" and (f["addr"] & 0xFFFFF) in queue]
        want = [(c, *self.ENDPOINTS[s]) for s in "AXC" for c in range(COLS)]
        want += [(c, *self.ENDPOINTS["D"]) for c in self.d_cols]
        assert got == want, f"shim channels differ from ENDPOINTS: {got}"





@dataclass
class _Streams:
    """What a model's cores take of the shared design (``DecodeOverlayOperator._extra_workers``)."""

    p: str  # the kernel prefix
    a_ty: object
    x_ty: object
    c_ty: object
    r_ty: object  # an R object
    a_l1: list  # per column the A stream's L1 ObjectFifo (broadcast: a model core may consume it)
    x_l1: list
    m_fifos: list  # the main cores' M streams (msg)
    r_fifos: list  # the R streams to the main cores (relay)
    ws: list  # per column the main core's workspace and sum panels (``workspace``, xs: a model core may use the regions
    xs: list  # idle while its objects run; LENDS_WS)
    xp: list  # per column the helper's x panels (a model core may use them while its objects run)
    sq: list  # per column the input's sums of squares of a normed phase, the helper's (8 slots: in the HS8_ROW core's memory)
    x_kw: dict  # the arguments of a consumer's handle of x_l1
