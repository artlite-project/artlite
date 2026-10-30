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

X objects (``XB`` bytes): ``META`` metadata words, then a payload of ``XPAY`` bytes. A phase's input of K elements
per slot lives in DDR as ``[metadata object | x]`` (``region_bytes(K)``), x slot-interleaved: [k block K / 8][slot
4][8] bf16 (unused slots zero; ``packed`` designs: the step's slots only, [k block][slot SP][8], ``packed_slots``: an
input object then holds 4 / SP strips of ``OBJ_K`` k); one shim BD reads it as the metadata object and the input
objects, each the ``XB`` bytes that end with its part of x (``input_fill``: its first ``4 META`` bytes are the
previous ones' end, unread). Outputs go out per slot: whenever every slot has ``C_LEN`` new values, one C object per slot
(slot 0 .. S - 1), so that one drain BD scatters a column's outputs (into the next input's interleaved layout or by
slot).

The main core's program, per X object (``_begin``):

- without ``F_MAIN``: an object of another core (``M_SKIP`` more X objects, ``M_NASKIP`` A objects belong to it;
  ``M_NREL`` R objects to relay);
- ``F_LOADR``: the payload holds slot ``M_SLOT``'s part of the residual stream (fp32, hidden / 8 values: the column's
  rows of every output projection), kept in the main core's workspace across layers;
- else a phase's metadata object: ``M_NX`` input objects follow; with ``F_NORM`` they arrive under norm-weight A
  objects (``nbegin``: each holds the weights of ``NX_PER_NORM`` input objects, the input is multiplied by them,
  and the outputs are scaled by the slots' 1 / rms), then ``M_NT`` weight tiles (``M_NRT`` row tiles x ``M_G`` K
  chunks, a row tile's chunks in a row; blocks of row tiles ``M_B``: end row tile, format (| ``B_F32``), valid
  rows). Per tile the main core posts the helper's command, runs its half, adds the helper's partial rows; a
  complete row tile's rows per slot: ``F_RESID`` into the residual (fp32), ``F_GU`` (rows 0 .. 7 gate, 8 .. 15 up) ->
  bf16(act(bf16(gate))) up (``kmath.act``'s SiLU table), else the rows (bf16; ``B_F32``: fp32 as two bf16); ``F_EMITR``:
  after the tiles the residual goes out (bf16).

Metadata words from ``M_MODEL`` on are the model cores'. The cores' kernel source: ``kernels.py`` (and ``gemv.py``).

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
  item's weights; F_ACCUM: into acc, bf16 [slot][hidden] in the helper's memory; F_ACC0 stores; F_EMITA: acc out);
- the combine (F_COMB): its input objects are the 8 columns' partials of a slot's residual slice, which the helper adds
  into the residual (K_ACC); the residual then goes out.

Designs of 8 slots (``slots=8``: a dense model whose formats run at 8 slots, ``gemv``): x per k block one panel of the
8 slots' single bfp16 rows (``gemv``'s "Designs of 8 slots"; the sum panels a hi and a lo panel each), an input object
of ``obj_k`` 256 k ([k block][slot 8][8] bf16; a strip of the helper's is a super-block), a norm-weight object over 16 of
them, a row tile's outputs [g 2][slot 8][r 8]. L1 (the main core's own regions are its 4-slot design's at twice the
slots, but the residual): the main core's workspace holds the residual of slots 0 .. 3, the helper's memory that of 4
.. 7 (``hres``: the main core's south neighbour), the helper's scratch (its GEMV scratch and a K group's 16-k sums)
lies in the core south of it (``HS8_ROW``: the model's core there leaves it room). No moe, post, packed, paged designs.

Paged formats (``paged``: a model whose formats' kernels do not fit the cores together): their kernels are code pages
(``paging``, the memtile transport): per column the main and helper cores request a format's page for their half
before a tile of it (``gemv.render(paged=)``) from a store in their memtile (``PG_*``: its channel, BDs, bytes; the
pager's state words and wake lock on both cores), which ``page_store`` links and the device loads before a dispatch
whenever it is stale (``artlite.overlay.OverlayDevice.set_prologue``: first, then after another hardware context ran).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from aie.ir import Module
from aie.iron import Buffer, Kernel, Lock, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from .. import txn
from ..formats import FORMATS, iq3
from ..moe import route as moe_k
from ..operators import paging
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS, linear_tap
from ..operators.kernel_source import kernel_prefix
from ..operators.l1_layout import pin_l1_buffers
from ..operators.util import Attr, regions
from . import gemv
from .kernels import SRC as _SRC

SLOTS = gemv.SLOTS  # token slots of a design (4; ``DecodeOverlayOperator(slots=8)``: 8)
META = 64  # int32 metadata words at the start of an X object
XB = 4352  # bytes of an X object
XPAY = XB - 4 * META  # payload bytes of an X object
OBJ_K = XPAY // (2 * SLOTS)  # k of an input object (512: 64 k blocks of the slots' 8 values; 8 slots: 256, ``obj_k``)
NX_PER_NORM = 8  # input objects per norm-weight object (8 x 512 bf16 weights; 8 slots: 16 x 256, ``nx_per_norm``)
NORM_K = NX_PER_NORM * OBJ_K  # weights of a norm-weight object
C_LEN = 64  # bf16 elements of a C object
NBLK = 4  # blocks of row tiles a phase may have
MAX_K = 5120  # k of a phase's K group: its x panels (46 KB) in the helper's memory
CMD_WORDS = 16

# metadata words
M_FLAGS, M_NS, M_NX, M_K, M_NT, M_G, M_NRT, M_NB, M_B = range(9)  # M_B + 3 i: block i (end rt, fmt, valid rows)
M_SKIP = M_B + 3 * NBLK  # (20) X objects after this one that belong to another core
M_NASKIP, M_NREL, M_SLOT = 21, 22, 23
M_MODEL = 24  # the first word of the model cores' own
M_SP = 39  # (packed designs of 8 slots) a phase input's slots per k block (``packed_slots``; the host's per phase)
# (``post`` designs) an F_POST phase's layer output scale (bf16 bits); (``tok`` designs) a token object's index
# (M_TOK_KIND, < N_TOK) and per column its cache objects (M_TOK_NG + c); an object of another core with M_NASKIP = -1 -
# index: that token's count; (``post``) a phase's residual slice per column (M_H8) and 1 / the length of its norms
# (M_IH, fp32 bits): the model's hidden size or a drafter's (``hidden_meta``)
M_SCALE = 40
M_TOK_KIND, M_TOK_NG = 41, 48
M_H8, M_IH = 42, 43
B_F32 = 256  # in a block's format word: its rows go out as fp32 (two bf16 slots each)
# M_FLAGS
F_NORM = 1  # RMSNorm of the input (norm-weight objects); the outputs scaled by 1 / rms
F_LOADR = 2  # the payload is a slot's residual slice (fp32)
F_RESID = 4  # the rows add into the residual
F_EMITR = 8  # after the tiles, the residual goes out (bf16)
F_GU = 16  # tiles of [8 gate | 8 up] rows -> act(gate) up
F_MAIN = 32  # an object of the main core
F_MSG = 64  # the outputs go to the model's core on M instead of C
# (``post`` designs: post-norm residual updates, Gemma)
F_POST = 1 << 12  # a header: [partial sums | v, r] input, res = (res + v / rms(v) w_post) scale, x = r w1
F_VOUT = 1 << 13  # the rows add into pend (the sub-layer's output v); with F_EMITR v and its partial sums go out
F_PLI = 1 << 14  # the rows' outputs gelu(bf16(y)) pli, pli = bf16(p / rms(p) w + e): the meta object's payload
# [p: slot 4][256] | [e: slot 4][256] bf16 (p: the per-layer projection's rows, ``gemma4.overlay``; rms over the 256),
# w (bf16 hi | lo) after [w1 | w_post] in the phase's first norm-weight object
F_PZERO = 1 << 15  # pend = 0 first (an F_VOUT phase's first K group)
F_TOK = 1 << 16  # (another core's object) a token object: the main core keeps M_TOK_NG + col (its cache A objects)
F_KPAIR = 1 << 18  # (a header) its tiles hold row pairs of a matrix of K = POST_K (a row's super-blocks: two rows), run
# as two slots of one token: slot 0 = [x | 0], slot 1 = [0 | x] (the helper mirrors strip 0 into slot 1 for strip 1)
# (``moe`` designs: an MoE layer, module docstring "MoE")
B_GU = 512  # in a block's format word: its rows are [8 gate | 8 up] groups (F_GU for this block)
B_PAIR = 1024  # its rows are pairs [hi, lo] of one value (an F32 router's planes): the pairs' sums go out
B_KEEP = 2048  # (B_PAIR) the block's first value stays in the main core's state (S_SG + slot): the shared expert's gate
B_G2, B_G4 = 4096, gemv.B_G4  # its tiles hold row groups of 2 / 4 rows (``decode.image.Block.G``)
F_WS = 1 << 19  # the rows times the slots' scales (wsc: the shared expert's gate, an expert's routing weights)
F_ROUTE = 1 << 20  # the route object: the router's logits as its payload -> the routing (control packets, expert object)
F_EXPERT = 1 << 21  # the expert object: the column's items, M_NT their A objects (the main core's sub-phases)
F_COMB = 1 << 22  # the combine: its input objects (the columns' partials of a slot) add into the residual; F_EMITR
F_HOUT = 1 << 23  # (an expert's gate / up sub-phase) the outputs into the hold buffer: its down's input
F_ACCUM = 1 << 24  # (a down sub-phase) the rows add into acc ([slot][hidden] bf16; F_ACC0: stored), F_EMITA: then out
F_ACC0 = 1 << 25
F_EMITA = 1 << 26
XID = 3  # (moe) the X stream's packet id (its channel also carries the routing's control packets: ``moe.route.CID``)
# (moe) metadata words: a phase's first chunk of x / sum panels; the route object's (the layer's expert halves at
# M_RBASE (device address lo, hi) + k M_RSTR, M_RTGU + M_RTDN A objects each); the expert object's items (M_NITEM,
# then per item: e | h0 << 8 | h1 << 9, its weights per slot, fp32 bits)
M_NITEM, M_ITEM = 24, 25
ITEM_WORDS = 1 + SLOTS  # (e | h0 << 8 | h1 << 9, its weight in each slot)
M_RBASE, M_RSTR, M_RTGU, M_RTDN = 50, 52, 53, 54
M_XC0 = 55
assert M_ITEM + ITEM_WORDS * moe_k.MAX_ITEMS <= M_RBASE

# state words of the main core's workspace (the first ones are trip counts of the core program)
(S_NNO, S_NXN, S_NXP, S_NIN, S_NT, S_NE, S_NSKIP, S_NASKIP, S_NM, S_NR, S_FLAGS, S_NS, S_K8, S_G, S_NRT, S_NB, S_KC,
 S_RT, S_XD, S_NX, S_STG, S_ESLOT, S_EJ, S_ESRC, S_FMT, S_TL) = range(26)
# (``packed`` designs) the slots of an input k block, the strips (helper input commands) of an input object, less one
S_SP, S_NXS, S_NXS1 = 26, 27, 28
S_GN, S_GK = 29, 30  # (``gu_win`` designs) the rows in the gate / up windows, the windows of 16 so far in a SiLU group
S_B = 32  # the block table, 3 words per block
# (``post`` designs) the partial sums object (a trip count), the scale (bf16 bits), the column, the phase's residual
# slice and 1 / norm length (M_H8, M_IH); (``tok``) the cache objects per token object (S_NAS + M_TOK_KIND: N_TOK words)
S_NPS, S_SCALE, S_COL, S_H8, S_IH, S_NAS = 44, 45, 48, 49, 50, 52
N_TOK = 8
# (moe) the sub-phases of an expert object (a trip count), their local input strips (a trip count), the phase's first
# panel chunk, the next sub-phase, the shared gate's logit per slot (fp32 bits)
S_NSUB, S_NLS, S_XC0, S_SUB, S_SG = 44, 45, 46, 47, 48
S_WORDS = 64
assert S_TL < S_B and S_B + 3 * NBLK <= S_NPS and S_NAS + N_TOK <= S_WORDS and S_SG + SLOTS <= S_WORDS
# paged formats (module docstring): the memtile's page channel, its first page BD (none of aiecc's: ``Store.check``), the
# store's bytes (of up to 4 paged formats; of more: ``pg_store``); the pager's state words (data memory of the main and
# helper cores) and wake lock; the stream ids of the pages' and the requests' routes (main, helper)
PG_CH, PG_BD, PG_STORE, PG_STORE_BYTES = 5, 40, 0x78000, 0x8000
PG_STATE, PG_WAKE = 0xFFE0, 15  # (32-byte aligned: aiecc)
WS8_AT = 0x4800  # (8 slots) the main core's workspace (region 1 past its bank-B tables; then the sum panels)
PG_PAGE_ID, PG_REQ_ID = (1, 2), (3, 4)
# the event windows of a stats build (DecodeOverlayOperator(win=)): main's half tile, the helper's, main's row
# epilogue (collect), the helper's input objects
WINS = {"tile": 1, "htile": 2, "collect": 3, "hstrip": 4, "pre": 5, "prehdr": 6, "preplain": 7, "gap": 8, "gapm": 9}  # (gap: a phase's last tile to the next X object; gapm: to the next phase's)  # (pre: a phase's begin to its first tile)
# the helper's commands: CMD words 0 .. 3 (by kind), the kind at C_KIND
C_KIND = 4
K_TILE, K_STRIP, K_INPUT, K_PSTRIP, K_NOP, K_BEGIN, K_SUB, K_ACC, K_HELP = range(9)  # (K_SUB, K_ACC: moe; K_HELP:
# ``help_flag``)
C_SP = 5  # (K_STRIP, K_PSTRIP) the input's slots per k block (K_PSTRIP: + C_MIRROR: strip 0 into slot 1, F_KPAIR)
C_MIRROR = 8
POST_K = 256  # (post designs) k of a header phase's input object: [k block 32][v | r][slot 4][8] bf16


def hidden_meta(m: np.ndarray, hidden: int) -> None:
    """(``post`` designs) A phase's M_H8 / M_IH words for a residual of ``hidden`` values (hidden / 8 per column)."""
    m[M_H8] = hidden // 8
    m[M_IH] = np.float32(1.0 / hidden).view(np.int32)


def packed_slots(S: int) -> int:
    """(``packed`` designs) The slots of an input k block in a step of S slots: 1, 2 or 4 (S = 3 pads a slot; a design
    of 8: 8 from S 5 on). An input object then holds the k blocks of slots / SP strips (the helper's input commands:
    OBJ_K or POST_K k each)."""
    return 1 if S == 1 else (2 if S == 2 else (4 if S <= 4 else 8))




class DecodeOverlayOperator(ArtcOperator):
    """The shared decode overlay design (module docstring): the main and helper cores of every column. ``hidden``: the
    model's hidden size (the norm length, the residual slices); ``rows``: {tile format: rows per tile} of the formats
    the phases use; ``tile_bytes``: the A object size; ``nsb``: super-blocks per tile (its K chunk). The runtime
    sequence (W, ACT, OUT) is a probe that issues one task per stream (``check_endpoints``; never dispatched): the
    host runs the design with transactions.

    A model's subclass sets ``PREFIX`` (kernel and object names), ``msg`` / ``relay`` (the main core's M / R streams)
    and adds its cores: ``_extra_src`` (their kernel source), ``_extra_workers`` (their workers and D streams)."""

    PREFIX = "ovd"
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

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int = 9280, eps: float = 1e-6, nsb: int = 4,
                 act: str = "silu", post: bool = False, pli: bool = False, packed: bool = False, tok: bool = False,
                 gu_win: bool = False, paged: tuple[str, ...] = (), moe: tuple[int, int] | None = None,
                 max_k: int = MAX_K, win: str = "", slots: int = SLOTS):
        """act: the gate/up activation ("silu", "gelu"); post: post-norm residual updates (F_POST, F_VOUT); pli: the
        F_PLI epilogue (gelu(y) times per-layer inputs); packed: the phases' inputs hold the step's slots only
        (``packed_slots``), not 4; tok: token objects (F_TOK: the main cores skip an attention's cache objects by the
        count of its token, M_NASKIP < 0; ``post`` designs have them); gu_win: the gate/up phases' tiles hold fewer
        than 16 rows (their [8 gate | 8 up] groups cross tiles: a window per slot); paged: formats whose kernels are
        code pages (module docstring); moe: (experts, experts used per token) of the MoE layers' phases (module
        docstring "MoE": the X stream packet-switched, the router's blocks, row-group tiles, the expert's sub-phases, the
        combine); max_k: the K of a phase's group at most (its x panels in the helper's memory; moe: one more chunk,
        the side chunk); slots: the token slots of a step (``gemv``'s x panels; 8: module docstring "Designs of 8
        slots")."""
        assert slots in (4, 8) and (slots == 4 or not (post or pli or tok or gu_win or paged or moe)), slots
        self.slots = slots
        self.obj_k = XPAY // (2 * slots)  # (k of an input object)
        self.nx_per_norm = NORM_K // self.obj_k  # (input objects per norm-weight object)
        self.chunk = gemv.Chunk(nsb, slots)
        self.moe = moe is not None
        self.experts, self.top = moe or (0, 0)
        assert max_k % self.chunk.k == 0 and max_k <= MAX_K
        self.max_k = max_k
        self.max_g = max_k // self.chunk.k + int(self.moe)  # K chunks of a phase's group (moe: and the side chunk)
        assert hidden % (COLS * C_LEN) == 0 and hidden % self.chunk.k == 0 and hidden <= max_k
        assert not moe or not (post or pli or packed or tok) and act == "silu" and nsb == 4
        assert all(gemv.tile_rows_of(f, tile_bytes, nsb) == r for f, r in rows.items()), rows
        assert all(gemv.supported(f, nsb, slots) for f in rows), (rows, slots)
        assert act in ("silu", "gelu") and (not pli or act == "gelu")
        self.hidden, self.rows, self.tile_bytes, self.eps = hidden, dict(sorted(rows.items())), tile_bytes, eps
        self.act, self.post, self.pli, self.packed, self.tok, self.gu_win = act, post, pli, packed, tok or post, gu_win
        self.paged = tuple(sorted(paged))
        assert set(self.paged) <= set(rows), (paged, rows)
        assert not gu_win or act == "silu"
        self.rowsplit = sorted(f for f in rows if gemv.row_split(f, nsb))
        # (the x panels as [core half][chunk]: the two cores' panels of a tile in different L1 regions of the helper's
        # memory, not one chunk's 128 panels in a row: their reads at once conflict; a 512-k input strip is a half)
        self.xh = nsb == 4 and not self.rowsplit and not post
        self.wide = any(r > 16 for r in rows.values())  # (tiles of more than 16 rows: 4-slot designs)
        assert slots == 4 or not self.wide, rows
        self.yt = (128 if self.wide else 64) * slots // 4  # (floats of a row tile's outputs)
        # (floats of the helper's partial rows: its row groups of a tile in the super-block split, or half a row-split's)
        self.yn = 128 if any(r > 16 and f not in self.rowsplit for f, r in rows.items()) else gemv.y_floats(slots)
        self.yb = self.yt  # (floats of a tile's row block: 16 rows [g 2][slot][r], 32 rows [g 4][slot][r])
        if self.moe:  # (row groups: a tile's 4 blocks, the helper's two; tiles of more than 16 rows: blocks of 32 rows,
            # their gate / up rows through the windows, ``gu_win``)
            self.yb = 128 if any(r > 16 for r in rows.values()) else 64
            self.yt, self.yn = 4 * self.yb, 2 * self.yb
            self.gu_win = self.gu_win or self.yb == 128
            self.PACKET, self.BD_COUNT = {"X": XID}, {"A": 3}  # (shim BD 3: the experts' tasks, ``moe``)
        self.win = win  # (a stats build: the event window of a region, WINS)
        self.ablate: tuple[str, ...] = ()  # (a timing build: work compiled out, results wrong; "strip": the input strips)
        self.h8 = hidden // COLS
        self.res_pitch = -(-(self.h8 + 32) // 16) * 16  # (fp32 per slot: the window adds reach 32 past a row)
        self.res_slots = min(slots, 4)  # (the slots of the residual in the main core's workspace; the others' in hres)
        super().__init__()

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
            sizes += [("pend", 4 * SLOTS * self.res_pitch), ("ivv", 4 * 32)]
        if self.pli:  # ([g 4][slot][r 8] bf16: p, then pli; e; [slot][8] fp32 sums, then the slots' 1 / rms(p) [slot][r 8]
            # bf16 hi | lo)
            sizes += [("pli", 2 * 2 * 4 * SLOTS * 8 + 4 * 32)]
        if self.gu_win:
            sizes += [("gw", 4 * SLOTS * (64 if self.yb == 128 else 32))]
        if self.moe:  # (the slots' scales of F_WS, the hold buffer ([h | h] of a half: [k block][slot][8] bf16), the
            # routing's C objects (control packets, the expert object's metadata), the expert object's plan)
            sizes += [("wsc", 4 * 32), ("hold", 2 * SLOTS * OBJ_K), ("rbuf", 2 * C_LEN * moe_k.RBUF_OBJECTS),
                      ("plan", 4 * moe_k.PLAN_WORDS)]
        out = regions(sizes)
        assert self.slots == 8 or MAX_K + 2048 <= self.hs_bytes  # (the 16-k sums of a K group, fp32, and hpstrip's nb
        # in the helper's scratch)
        return out

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (COLS * self.tile_bytes // 2,), torch.bfloat16),
                ArgSpec("in", (COLS * XB // 2,), torch.bfloat16),
                ArgSpec("out", (2 * COLS * C_LEN,), torch.bfloat16)]

    def _consts(self) -> dict[str, int]:
        """The metadata, flag and state words the kernel sources take."""
        return {k: v for k, v in globals().items() if k[:2] in ("M_", "F_", "S_", "B_", "K_") or k in ("C_KIND", "C_MIRROR")}

    def _extra_src(self, p: str, consts: dict[str, int]) -> str:
        """The kernel source of the model's cores."""
        return ""

    @property
    def pfx(self) -> str:
        """The placeholder prefix of the kernels (``kernel_source``), under ``PREFIX``."""
        return kernel_prefix(self.PREFIX)

    def get_kernel_src(self) -> str:
        p = self.pfx
        consts = self._consts()
        fmts = list(self.rows)
        assert not self.xh or self.obj_k * 2 * self.slots // 4 == self.chunk.k
        src = gemv.render(f"{p}m", fmts, tile_bytes=self.tile_bytes, nsb=self.chunk.nsb, paged=self.paged,
                          pg=self._page_spec() if self.paged else None, rg=self.moe, yb=self.yb, slots=self.slots)
        src += self._extra_src(p, consts)
        gv = {"NSB": self.chunk.nsb, "XP_CHUNK": self.chunk.xp, "XS_CHUNK": self.chunk.xs,
              "XS_SLOT": gemv.xs_slot(self.slots),
              "XPH": self.max_g * self.chunk.xp // 2, "XPHC": self.chunk.xp // 2,
              "PANEL": gemv.PANEL,
              "Y_FLOATS": gemv.Y_FLOATS}
        return src + _SRC.render(p=p, mp=f"{p}m", w_=Attr(self.workspace()), H=self.hidden, H8=self.h8, SHM=self.SHM,
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
                                 packed=self.packed, C_SP=C_SP, MAX_K=MAX_K, ablate=self.ablate,
                                 moe=self.moe, gate=self.gate, help_flag=self.help_flag, help_word=self.help_word,
                                 **self._moe_vars(p, consts), **consts)

    def _moe_vars(self, p: str, consts: dict[str, int]) -> dict[str, object]:
        """(moe) The template variables of the MoE paths: the routing's source (``moe.route``), the side chunk, the
        plan's layout."""
        if not self.moe:
            return {}
        route = moe_k.render(p, consts | {"META": META, "SLOTS": SLOTS, "ITEM_WORDS": ITEM_WORDS,
                                          "w_": Attr(self.workspace())}, experts=self.experts, top=self.top, cols=COLS)
        return {"route": route, "SIDE": self.max_g - 1, "PLAN_SUBS": moe_k.PLAN_SUBS, "PLAN_META": moe_k.PLAN_META,
                "ITEM_WORDS": ITEM_WORDS}

    @property
    def pg_store(self) -> tuple[int, int, int]:
        """(first BD, store address, bytes) of the page store: PG_BD, PG_STORE, PG_STORE_BYTES for up to 4 paged
        formats; of more, the BDs from 48 - 2 n (the odd channels' last) and 48 KB (6 formats' pages of ~3-4 KB per
        core half)."""
        n = len(self.paged)
        return (PG_BD, PG_STORE, PG_STORE_BYTES) if n <= 4 else (48 - 2 * n, 0x74000, 0xC000)

    def _page_spec(self) -> gemv.PageSpec:
        """The pager of the paged formats (main core: half 0, helper: half 1)."""
        (h0, ctl), (h1, _) = (paging.request(stream_id=i, channel=PG_CH) for i in PG_REQ_ID)
        bd0 = self.pg_store[0]
        return gemv.PageSpec(state=0x70000 + PG_STATE, wake=PG_WAKE, hdr=(h0, h1),
                             bd0=(bd0, bd0 + len(self.paged)), ctl=ctl)

    def page_store(self) -> paging.Store | None:
        """The compiled design's page store (the paged formats' pages of the main cores, then of the helpers), checked
        against the design's configuration; None without paged formats."""
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
        bd0, base, size = self.pg_store
        store = paging.Store(base=base, size=size, channel=PG_CH, bd0=bd0,
                             pagers=(pager(0, self.MAIN_ROW, self.STACK), pager(1, self.HELPER_ROW, self.HELPER_STACK)))
        store.check(wd, range(COLS))
        return store

    def page_prologue(self, store: paging.Store) -> np.ndarray:
        """The transaction that makes the pages loadable: the store and its BDs in every memtile, every paging core's
        state words cleared (a configuration rewrites the slot's start with the startup code)."""
        t = txn.Txn()
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
        if self.moe:  # (acc: the experts' partials of the column, in the helper's memory)
            acc_ty = np.ndarray[(SLOTS * self.hidden,), bf]
            k_collect = Kernel(f"{p}_collect", "kernel.o", [r_ty, ws_ty, acc_ty])
            k_emit = Kernel(f"{p}_emit", "kernel.o", [ws_ty, c_ty, acc_ty])
            k_sub = Kernel(f"{p}_sub", "kernel.o", [ws_ty, cmd_ty])
            k_lstrip = Kernel(f"{p}_lstrip", "kernel.o", [ws_ty, cmd_ty])

        mr, hr = self.MAIN_ROW, self.HELPER_ROW
        a_l3 = [ObjectFifo(a_ty, name=f"{P}_A{c}", depth=self.A_DEPTH) for c in range(COLS)]
        a_l1 = [a_l3[c].cons(depth=self.A_DEPTH).forward(tile=Tile(c, 1), depth=self.A_FWD, name=f"{P}_a{c}") for c in range(COLS)]
        # (moe: the X stream packet-switched, its channel also carries the control packets into the shim's TileControl)
        x_l3 = [ObjectFifo(x_ty, name=f"{P}_X{c}", depth=self.X_DEPTH, **({"packet": True, "packet_id": XID} if self.moe
                                                                            else {})) for c in range(COLS)]
        x_l1 = [x_l3[c].cons(depth=self.X_DEPTH).forward(tile=Tile(c, 1), depth=1, name=f"{P}_x{c}") for c in range(COLS)]
        c_fifos = [ObjectFifo(c_ty, name=f"{P}_C{c}", depth=2) for c in range(COLS)]
        cmds = [ObjectFifo(cmd_ty, name=f"{P}_cmd{c}", depth=2) for c in range(COLS)]
        m_fifos = [ObjectFifo(c_ty, name=f"{P}_M{c}", depth=2) for c in range(COLS)] if self.msg else []
        r_fifos = [ObjectFifo(rr_ty, name=f"{P}_R{c}", depth=2) for c in range(COLS)] if self.relay else []
        ress = [ObjectFifo(r_ty, name=f"{P}_res{c}", depth=2) for c in range(COLS)]

        def main_fn(a_f, x_f, c_f, cmd_f, res_f, *rest):
            rest = list(rest)
            if "IQ3_XXS" in self.rows:  # (the grid tables: placed, read through their addresses)
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
            ea = (acc,) if self.moe or s8 else ()  # (collect's and emit's last argument: moe's acc, 8 slots' hres)

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
        xps = []
        m_ws, m_xs = [], []
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
                pg_flows += [paging.page_flow(mem, t, channel=PG_CH, stream_id=PG_PAGE_ID[h]),
                             paging.request_flow(t, mem, stream_id=PG_REQ_ID[h])]
        grids = {}  # (IQ3_XXS: per core the grid table's two copies at iq3.TAB, used through their addresses only)
        if "IQ3_XXS" in self.rows:
            t_ty = np.ndarray[(iq3.TAB_BYTES // 4,), np.dtype[np.uint32]]
            for c in range(COLS):
                for h, r in enumerate((mr, hr)):
                    grids[c, h] = [Buffer(t_ty, tile=Tile(c, r), name=f"{P}_grid{c}_{r}{i}",
                                          address=iq3.TAB[h][i] - 0x70000, initial_value=iq3.grid_words())
                                   for i in range(2)]
        # (paged: the main cores take COLS + their column, not a constant 0 in column 0, which can give its control code
        # another size: every core's resident kernels must lie where the pages were linked, ``paging.link``)
        for c in range(COLS):
            ws = Buffer(ws_ty, name=f"{P}_ws{c}", **({"tile": Tile(c, mr)} if self.LENDS_WS else {}))
            m_ws.append(ws)
            moe_args = [k_sub, k_lstrip, Buffer(acc_ty, name=f"{P}_acc{c}", tile=Tile(c, hr))] if self.moe else []
            hres = [Buffer(hres_ty, name=f"{P}_hres{c}", tile=Tile(c, hr))] if s8 else []
            xp = Buffer(xp_ty, name=f"{P}_xp{c}", tile=Tile(c, hr))  # (the x panels in the helper's memory)
            xps.append(xp)
            xs = Buffer(xs_ty, name=f"{P}_xs{c}", tile=Tile(c, mr))
            m_xs.append(xs)
            hs = Buffer(hs_ty, name=f"{P}_hs{c}", tile=Tile(c, self.HS8_ROW if s8 else hr))
            sq = Buffer(sq_ty, name=f"{P}_sq{c}", tile=Tile(c, hr))
            workers.append(Worker(main_fn, [a_l1[c].cons(depth=self.A_L1), x_l1[c].cons(depth=1), c_fifos[c].prod(), cmds[c].prod(),
                                            ress[c].cons(), *([m_fifos[c].prod()] if self.msg else []),
                                            *([r_fifos[c].cons()] if self.relay else []), *moe_args, *hres,
                                            *([k_xgate] if self.gate else []), k_begin, k_nbegin,
                                            k_strip_n,
                                            k_strip, k_input, k_inv, k_cmd, k_tile, k_collect, k_emit, k_relay,
                                            *([k_psums] if self.post else []), ws, xp, xs,
                                            (COLS if self.paged else 0) + c, *pg_main.get(c, []),
                                            *grids.get((c, 0), [])],
                                  tile=Tile(c, mr), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                  while_true=False))
            help_args = []
            if self.help_flag:
                hk, hb = self._help(c)
                help_args = [Buffer(hst_ty, name=f"{P}_hst{c}", tile=Tile(c, hr), initial_value=np.zeros(4, np.int32)),
                             hk, hb]
            workers.append(Worker(helper_fn, [cmds[c].cons(), ress[c].prod(), k_htask, hs, sq, xp, *hres, *help_args,
                                              *pg_help.get(c, []), *grids.get((c, 1), [])],
                                  tile=Tile(c, hr),
                                  stack_size=self.HELPER_STACK, dynamic_objfifo_lowering=True, while_true=False))
        s = _Streams(p=p, a_ty=a_ty, x_ty=x_ty, c_ty=c_ty, r_ty=rr_ty, a_l1=a_l1, x_l1=x_l1, m_fifos=m_fifos, r_fifos=r_fifos,
                     ws=m_ws, xs=m_xs, xp=xps)
        extra, d_fifos = self._extra_workers(s)
        workers += extra
        a_prod = [f.prod(tile=Tile(c, 0), channel=self.ENDPOINTS["A"][1]) for c, f in enumerate(a_l3)]
        x_prod = [f.prod(tile=Tile(c, 0), channel=self.ENDPOINTS["X"][1]) for c, f in enumerate(x_l3)]
        c_cons = [f.cons(tile=Tile(c, 0), channel=self.ENDPOINTS["C"][1]) for c, f in enumerate(c_fifos)]
        d_cons = [f.cons(tile=Tile(c, 0), channel=self.ENDPOINTS["D"][1]) for c, f in d_fifos]
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
        for f in pg_flows:
            rt.add_flow(f)
        for lk in pg_locks + self._locks:
            rt.add_lock(lk)
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        if s8:  # (the workspace and the sum panels in the main core's regions 1, 2: the A buffers in regions 0, 3)
            at = {f"{P}_ws{c}": WS8_AT for c in range(COLS)} | {f"{P}_xs{c}": WS8_AT + W["size"] for c in range(COLS)}
            assert WS8_AT + W["size"] + self.xs_bytes + self.tile_bytes + XB + 4 * 2 * C_LEN <= 0x10000
            pin_l1_buffers(module, fifo=lambda name, depth: None, buffer=at.get)
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


# ----------------------------------------------------------------------
# Host side: the X objects of a phase's input
# ----------------------------------------------------------------------


def n_inputs(K: int, slots: int = SLOTS) -> int:
    """Input objects of a phase of K elements per slot (after its metadata object) of a design of ``slots``."""
    k = XPAY // (2 * slots)
    assert K % k == 0, K
    return K // k


def region_bytes(K: int, slots: int = SLOTS) -> int:
    """Bytes of a phase's input in DDR: [metadata object | x: [k block K / 8][slot ``slots``][8] bf16]."""
    return XB + 2 * slots * K


def input_fill(K: int, sp: int = SLOTS, slots: int = SLOTS) -> tuple[int, tuple[int, int, int, int]]:
    """(bytes, shim BD dims) of the X fill of a phase's input from the start of its region: the metadata object, then
    object i = the XB bytes that end with x's i-th XPAY bytes (strides of XPAY, extents of XB); x of sp slots per k
    block (``packed_slots``) of a design of ``slots``."""
    return (1 + -(-n_inputs(K, slots) * sp // slots)) * XB, (XB // 8, 2, XB // 8, XPAY // 4)


def post_region_bytes(K: int) -> int:
    """Bytes of a header phase's input in DDR: [metadata object | the partial sums object's payload (the 8 columns'
    C objects at 128 c) | x: [k block K / 8][v | r][slot 4][8] bf16]."""
    return XB + XPAY + 4 * SLOTS * K


def post_input_fill(K: int, sp: int = SLOTS) -> tuple[int, tuple[int, int, int, int]]:
    """(bytes, shim BD dims) of the X fill of a header phase's input (as ``input_fill``: the metadata object, the
    partial sums object, then the input objects: K / POST_K strips, SLOTS / sp per object)."""
    return (2 + -(-(K // POST_K) * sp // SLOTS)) * XB, (XB // 8, 2, XB // 8, XPAY // 4)


def interleave(x: np.ndarray, sp: int = SLOTS) -> np.ndarray:
    """S <= sp slots' x (S, K) -> the input layout [k block][slot sp][8] (K / 8, sp, 8) bf16 bits (uint16)."""
    S, K = x.shape
    out = np.zeros((K // 8, sp, 8), np.uint16)
    xb = torch.from_numpy(np.asarray(x, np.float32)).bfloat16().view(torch.int16).numpy().view(np.uint16)
    out[:, :S] = xb.reshape(S, K // 8, 8).transpose(1, 0, 2)
    return out


def norm_objects(w: np.ndarray, tile_bytes: int) -> np.ndarray:
    """The norm-weight A objects of a weight vector (bf16 values): object n holds the weights of input objects
    ``NX_PER_NORM`` n .. at its start."""
    per = NORM_K
    n = -(-len(w) // per)
    out = np.zeros((n, tile_bytes // 2), np.uint16)
    wb = torch.from_numpy(np.array(w, np.float32)).bfloat16().view(torch.int16).numpy().view(np.uint16)
    for i in range(n):
        part = wb[i * per:(i + 1) * per]
        out[i, :len(part)] = part
    return out
