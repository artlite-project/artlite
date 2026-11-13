"""The wire protocol between the host and the cores of the decode overlay (``decode.design``): the layout of an X
object, its metadata words and flags, the state words of the main core, the helper's commands, the packet and page ids,
and the host's layout of a phase's input in DDR. The words are declared with ``operators.words`` (their checks run at
import); the kernels receive them as ``consts()``, never by prefix.

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

Metadata words (``XMETA``): the words from 0 to ``M_MODEL`` are every object's header. The kinds of object own the words
past it, each its own (``objects``: one kind's words may reuse another's offsets): a phase's metadata object (``PHASE``),
a token object (``TOKEN``), the MoE routing's objects (``MOE``); the attention cores' (``decode.attn``) and the models'
kinds extend or sit beside them (a model's words in a phase's object: its kind extends ``PHASE``). The flag bits of the
cores' objects in M_FLAGS are one space: a design's kinds (``XMETA.consts``) take each bit once.

State words (``STATE``): those of the main core's workspace; the first ones are trip counts of the core program, and
the words of the post-norm designs, the token objects and the MoE layers are features of one array (``POST_STATE``,
``TOK_STATE``, ``MOE_STATE``: the MoE words reuse the others', ``STATE.together`` refuses both in one design). The
helper's command (``CMD``): its kind at ``C_KIND``, the input's slots per k block at ``C_SP``.

Where the words are used: the main and helper cores' kernels (``design.cc.jinja``, ``gemv``), the attention cores
(``decode.attn``), the models' cores (``models.*.decode``: their words extend these), the host (``decode.phases``,
``decode.model``, the models' overlays).
"""

import numpy as np
from ml_dtypes import bfloat16

from ..formats.common import MMV_SLOTS
from ..moe.route import MAX_ITEMS
from ..operators.design import COLS
from ..operators.words import WordMap

SLOTS = MMV_SLOTS  # token slots of a design (4; ``DecodeOverlayOperator(slots=8)``: 8)
META = 64  # int32 metadata words at the start of an X object
XB = 4352  # bytes of an X object
XPAY = XB - 4 * META  # payload bytes of an X object
OBJ_K = XPAY // (2 * SLOTS)  # k of an input object (512: 64 k blocks of the slots' 8 values; 8 slots: 256, ``obj_k``)
NX_PER_NORM = 8  # input objects per norm-weight object (8 x 512 bf16 weights; 8 slots: 16 x 256, ``nx_per_norm``)
NORM_K = NX_PER_NORM * OBJ_K  # weights of a norm-weight object
C_LEN = 64  # bf16 elements of a C object
NBLK = 4  # blocks of row tiles a phase may have
MAX_K = 5120  # k of a phase's K group: its x panels (46 KB) in the helper's memory
CMD_WORDS = 16  # words of the helper's command
S_WORDS = 64  # state words of the main core's workspace
N_TOK = 8  # token objects (``tok`` designs: the cache objects per token object, TOK_STATE's S_NAS)
POST_K = 256  # (post designs) k of a header phase's input object: [k block 32][v | r][slot 4][8] bf16
ITEM_WORDS = 1 + SLOTS  # (MoE designs, of 4 slots) the expert object's item: e | h0 << 8 | h1 << 9, its weight in each slot

# ----------------------------------------------------------------------
# X metadata
# ----------------------------------------------------------------------

XMETA = WordMap("X metadata", META, objects=True)
_HDR = XMETA.common
M_FLAGS, M_NS, M_NX, M_K, M_NT, M_G, M_NRT, M_NB = _HDR.words("M_FLAGS M_NS M_NX M_K M_NT M_G M_NRT M_NB")
M_B = _HDR.block("M_B", 3 * NBLK)  # M_B + 3 i: block i (end row tile, format word, valid rows)
M_SKIP = _HDR.word("M_SKIP")  # X objects after this one that belong to another core
M_NASKIP, M_NREL, M_SLOT = _HDR.words("M_NASKIP M_NREL M_SLOT")
M_MODEL = _HDR.next_free()  # the first word of a kind's own

PHASE = XMETA.kind("phase")
# (packed designs of 8 slots) a phase input's slots per k block (``packed_slots``; the host's per phase)
M_SP = PHASE.word("M_SP", at=39)
# (``post`` designs) an F_POST phase's layer output scale (bf16 bits)
M_SCALE = PHASE.word("M_SCALE")
# (``post`` designs) a phase's residual slice per column (M_H8) and 1 / the length of its norms (M_IH, fp32 bits): the
# model's hidden size or a drafter's (``hidden_meta``)
M_H8, M_IH = PHASE.words("M_H8 M_IH", at=42)
M_XC0 = PHASE.word("M_XC0", at=55)  # (moe) a phase's first chunk of x / sum panels
F_NORM = PHASE.flag("F_NORM", 0)  # RMSNorm of the input (norm-weight objects); the outputs scaled by 1 / rms
F_LOADR = PHASE.flag("F_LOADR", 1)  # the payload is a slot's residual slice (fp32)
F_RESID = PHASE.flag("F_RESID", 2)  # the rows add into the residual
F_EMITR = PHASE.flag("F_EMITR", 3)  # after the tiles, the residual goes out (bf16)
F_GU = PHASE.flag("F_GU", 4)  # tiles of [8 gate | 8 up] rows -> act(gate) up
F_MAIN = PHASE.flag("F_MAIN", 5)  # an object of the main core
F_MSG = PHASE.flag("F_MSG", 6)  # the outputs go to the model's core on M instead of C
# (``post`` designs: post-norm residual updates, Gemma)
F_POST = PHASE.flag("F_POST", 12)  # a header: [partial sums | v, r] input, res = (res + v / rms(v) w_post) scale, x = r w1
F_VOUT = PHASE.flag("F_VOUT", 13)  # the rows add into pend (the sub-layer's output v); with F_EMITR v and its partial sums go out
F_PLI = PHASE.flag("F_PLI", 14)  # the rows' outputs gelu(bf16(y)) pli, pli = bf16(p / rms(p) w + e): the meta object's payload
# [p: slot 4][256] | [e: slot 4][256] bf16 (p: the per-layer projection's rows, ``gemma4.overlay``; rms over the 256),
# w (bf16 hi | lo) after [w1 | w_post] in the phase's first norm-weight object
F_PZERO = PHASE.flag("F_PZERO", 15)  # pend = 0 first (an F_VOUT phase's first K group)
F_KPAIR = PHASE.flag("F_KPAIR", 18)  # (a header) its tiles hold row pairs of a matrix of K = POST_K (a row's super-blocks: two
# rows), run as two slots of one token: slot 0 = [x | 0], slot 1 = [0 | x] (the helper mirrors strip 0 into slot 1 for strip 1)
# (``moe`` designs: an MoE layer, ``decode.design``'s "MoE")
F_WS = PHASE.flag("F_WS", 19)  # the rows times the slots' scales (wsc: the shared expert's gate, an expert's routing weights)
F_COMB = PHASE.flag("F_COMB", 22)  # the combine: its input objects (the columns' partials of a slot) add into the residual; F_EMITR
F_HOUT = PHASE.flag("F_HOUT", 23)  # (an expert's gate / up sub-phase) the outputs into the hold buffer: its down's input
F_ACCUM = PHASE.flag("F_ACCUM", 24)  # (a down sub-phase) the rows add into acc ([slot][hidden] bf16; F_ACC0: stored), F_EMITA: then out
F_ACC0 = PHASE.flag("F_ACC0", 25)
F_EMITA = PHASE.flag("F_EMITA", 26)
# a block's format word: the format at the low 8 bits (``formats``' fid), then
B_F32 = PHASE.flag("B_F32", 8, of="format")  # its rows go out as fp32 (two bf16 slots each)
B_GU = PHASE.flag("B_GU", 9, of="format")  # (moe) its rows are [8 gate | 8 up] groups (F_GU for this block)
B_PAIR = PHASE.flag("B_PAIR", 10, of="format")  # its rows are pairs [hi, lo] of one value (an F32 router's planes): the pairs' sums go out
B_KEEP = PHASE.flag("B_KEEP", 11, of="format")  # (B_PAIR) the block's first value stays in the main core's state (S_SG + slot): the shared expert's gate
B_G2 = PHASE.flag("B_G2", 12, of="format")  # its tiles hold row groups of 2 / 4 rows (``decode.image.Block.G``)
B_G4 = PHASE.flag("B_G4", 13, of="format")

TOKEN = XMETA.kind("token")
# (``tok`` designs) a token object's index (< N_TOK) and per column its cache objects; an object of another core with
# M_NASKIP = -1 - index: that token's count
M_TOK_KIND = TOKEN.word("M_TOK_KIND", at=41)
M_TOK_NG = TOKEN.block("M_TOK_NG", COLS, at=48)
F_TOK = TOKEN.flag("F_TOK", 16)  # (another core's object) a token object: the main core keeps M_TOK_NG + col (its cache A objects)

# (moe) the route object's words (the layer's expert halves at M_RBASE (device address lo, hi) + k M_RSTR, M_RTGU + M_RTDN A
# objects each) and the expert object's (its items: M_NITEM, then per item e | h0 << 8 | h1 << 9, its weight per slot,
# fp32 bits): the routing writes both (``moe.route``)
MOE = XMETA.kind("moe")
M_NITEM = MOE.word("M_NITEM", at=M_MODEL)
M_ITEM = MOE.block("M_ITEM", ITEM_WORDS * MAX_ITEMS)
M_RBASE = MOE.block("M_RBASE", 2, at=50)
M_RSTR, M_RTGU, M_RTDN = MOE.words("M_RSTR M_RTGU M_RTDN")
F_ROUTE = MOE.flag("F_ROUTE", 20)  # the route object: the router's logits as its payload -> the routing (control packets, expert object)
F_EXPERT = MOE.flag("F_EXPERT", 21)  # the expert object: the column's items, M_NT their A objects (the main core's sub-phases)

# ----------------------------------------------------------------------
# State words of the main core
# ----------------------------------------------------------------------

STATE = WordMap("main core state", S_WORDS)
_S = STATE.common
(S_NNO, S_NXN, S_NXP, S_NIN, S_NT, S_NE, S_NSKIP, S_NASKIP, S_NM, S_NR, S_FLAGS, S_NS, S_K8, S_G, S_NRT, S_NB, S_KC,
 S_RT, S_XD, S_NX, S_STG, S_ESLOT, S_EJ, S_ESRC, S_FMT, S_TL) = _S.words(
    "S_NNO S_NXN S_NXP S_NIN S_NT S_NE S_NSKIP S_NASKIP S_NM S_NR S_FLAGS S_NS S_K8 S_G S_NRT S_NB S_KC S_RT S_XD S_NX "
    "S_STG S_ESLOT S_EJ S_ESRC S_FMT S_TL", at=0)
# (``packed`` designs) the slots of an input k block, the strips (helper input commands) of an input object, less one
S_SP, S_NXS, S_NXS1 = _S.words("S_SP S_NXS S_NXS1")
S_GN, S_GK = _S.words("S_GN S_GK")  # (``gu_win`` designs) the rows in the gate / up windows, the windows of 16 so far in a SiLU group
S_B = _S.block("S_B", 3 * NBLK, at=32)  # the block table, 3 words per block
S_TEND = _S.word("S_TEND", at=S_WORDS - 1)  # (tile) the window's end
# (``post`` designs) the partial sums object (a trip count), the scale (bf16 bits), the column, the phase's residual
# slice and 1 / norm length (M_H8, M_IH)
POST_STATE = STATE.kind("post")
S_NPS, S_SCALE = POST_STATE.words("S_NPS S_SCALE", at=44)
S_COL, S_H8, S_IH = POST_STATE.words("S_COL S_H8 S_IH", at=48)
TOK_STATE = STATE.kind("tok")
S_NAS = TOK_STATE.block("S_NAS", N_TOK, at=52)  # (``tok``) the cache objects per token object (S_NAS + M_TOK_KIND)
# (moe) the sub-phases of an expert object (a trip count), their local input strips (a trip count), the phase's first
# panel chunk, the next sub-phase, the shared gate's logit per slot (fp32 bits)
MOE_STATE = STATE.kind("moe", exclusive_with=(POST_STATE, TOK_STATE))
S_NSUB, S_NLS, S_XC0, S_SUB = MOE_STATE.words("S_NSUB S_NLS S_XC0 S_SUB", at=44)
S_SG = MOE_STATE.block("S_SG", SLOTS, at=48)

# ----------------------------------------------------------------------
# The helper's command
# ----------------------------------------------------------------------

CMD = WordMap("helper command", CMD_WORDS)
C_KIND = CMD.common.word("C_KIND", at=4)  # words 0 .. 3 by kind
K_TILE, K_STRIP, K_INPUT, K_PSTRIP, K_NOP, K_BEGIN, K_SUB, K_ACC, K_HELP = CMD.common.values(
    "K_TILE K_STRIP K_INPUT K_PSTRIP K_NOP K_BEGIN K_SUB K_ACC K_HELP", of="kind")  # (K_SUB, K_ACC: moe; K_HELP: ``help_flag``)
C_SP = CMD.common.word("C_SP")  # (K_STRIP, K_PSTRIP) the input's slots per k block
C_MIRROR = CMD.common.flag("C_MIRROR", 3, of="sp")  # (K_PSTRIP: C_SP + C_MIRROR: strip 0 into slot 1, F_KPAIR)

# ----------------------------------------------------------------------
# Streams, packets and pages
# ----------------------------------------------------------------------

XID = 3  # (moe, bcast) the X stream's packet id (moe: its channel also carries the routing's control packets: ``moe.route.CID``)
XC_ID, BC_ID, BC_COL, X_CH = 4, 5, 3, 1  # (bcast) the X objects' packet id past the memtile, the broadcast's, its shim, the cores' X channel
AID, AB_ID, A_MT = 6, 10, 0  # (a_bcast) the A stream's packet id, its first broadcast's, the memtiles' A channel
# paged formats (``decode.design``'s "Paged formats"): the memtile's page channel, its first page BD (none of aiecc's:
# ``Store.check``), the store's bytes (of up to 4 paged formats; of more: ``pg_store``); the pager's state words (data
# memory of the main and helper cores) and wake lock; the stream ids of the pages' and the requests' routes (main, helper)
PG_CH, PG_BD, PG_STORE, PG_STORE_BYTES = 5, 40, 0x78000, 0x8000
PG_STATE, PG_WAKE = 0xFFE0, 15  # (32-byte aligned: aiecc)
PG_PAGE_ID, PG_REQ_ID = (1, 2), (3, 4)
# the event windows of a stats build (DecodeOverlayOperator(win=)): main's half tile, the helper's, main's row
# epilogue (collect), the helper's input objects; gap: a phase's last tile to the next X object, gapm: to the next phase's;
# pre: a phase's begin to its first tile
WINS = {"tile": 1, "htile": 2, "collect": 3, "hstrip": 4, "pre": 5, "prehdr": 6, "preplain": 7, "gap": 8, "gapm": 9}


def consts() -> dict[str, int]:
    """The words, flags and values the decode design's kernels take (``DecodeOverlayOperator.get_kernel_src``)."""
    return XMETA.consts(PHASE, TOKEN, MOE) | STATE.consts(POST_STATE, TOK_STATE, MOE_STATE) | CMD.consts()


# ----------------------------------------------------------------------
# Host side: the X objects of a phase's input
# ----------------------------------------------------------------------


def hidden_meta(m: np.ndarray, hidden: int) -> None:
    """(``post`` designs) A phase's M_H8 / M_IH words for a residual of ``hidden`` values (hidden / 8 per column)."""
    m[M_H8] = hidden // 8
    m[M_IH] = np.float32(1.0 / hidden).view(np.int32)


def packed_slots(S: int) -> int:
    """(``packed`` designs) The slots of an input k block in a step of S slots: 1, 2 or 4 (S = 3 pads a slot; a design
    of 8: 8 from S 5 on). An input object then holds the k blocks of slots / SP strips (the helper's input commands:
    OBJ_K or POST_K k each)."""
    return 1 if S == 1 else (2 if S == 2 else (4 if S <= 4 else 8))


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
    xb = np.asarray(x, np.float32).astype(bfloat16).view(np.uint16)
    out[:, :S] = xb.reshape(S, K // 8, 8).transpose(1, 0, 2)
    return out


def norm_objects(w: np.ndarray, tile_bytes: int) -> np.ndarray:
    """The norm-weight A objects of a weight vector (bf16 values): object n holds the weights of input objects
    ``NX_PER_NORM`` n .. at its start."""
    per = NORM_K
    n = -(-len(w) // per)
    out = np.zeros((n, tile_bytes // 2), np.uint16)
    wb = np.asarray(w, np.float32).astype(bfloat16).view(np.uint16)
    for i in range(n):
        part = wb[i * per:(i + 1) * per]
        out[i, :len(part)] = part
    return out
