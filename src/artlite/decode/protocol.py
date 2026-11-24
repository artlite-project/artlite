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
  chunks, a row tile's chunks in a row; a ``B_PT`` block's pairs of row tiles share the tile of their last chunk, the
  second one a chunk less: ``S_PT``; blocks of row tiles ``M_B``: end row tile, format (| ``B_F32``), valid
  rows). Per tile the main core posts the helper's command, runs its half, adds the helper's partial rows; a
  complete row tile's rows per slot: ``F_RESID`` into the residual (fp32), ``F_GU`` (rows 0 .. 7 gate, 8 .. 15 up) ->
  bf16(act(bf16(gate))) up (``kmath.act``'s SiLU table), else the rows (bf16; ``B_F32``: fp32 as two bf16); ``F_EMITR``:
  after the tiles the residual goes out (bf16; ``F_ROTE``: as rotated copies, ``decode.design``'s "Rotated inputs").

Metadata words (``XMETA``): the words from 0 to ``M_MODEL`` are every object's header. The kinds of object own the words
past it, each its own (``objects``: one kind's words may reuse another's offsets): a phase's metadata object (``PHASE``),
a token object (``TOKEN``), the MoE routing's objects (``MOE``); the attention cores' (``decode.attn``) and the models'
kinds extend or sit beside them (a model's words in a phase's object: its kind extends ``PHASE``). The flag bits of the
cores' objects in M_FLAGS are one space: a design's kinds (``XMETA.consts``) take each bit once.

State words (``STATE``): those of the main core's workspace; the first ones are trip counts of the core program, and
the words of the post-norm designs, the token objects and the MoE layers are features of one array (``POST_STATE``,
``TOK_STATE``, ``MOE_STATE``: the MoE words reuse the others', ``STATE.together`` refuses both in one design); the rotated
inputs' (``ROT_STATE``) and the last MoE word lie past the S_WORDS of the others, in a rotated or MoE design's longer state.
The
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
DIMS_OBJ = (XB // 8, 2, XB // 8, XPAY // 4)  # shim BD dims of X objects of XB bytes, XPAY apart (the overlapping regions)
STAGE_PAD = 2 * (XB - XPAY)  # bytes before a staged x (``staged_region_bytes``): an input object's header as strips of cells
STAGED_X = XB + STAGE_PAD  # bytes from a staged region's start to its x
OBJ_K = XPAY // (2 * SLOTS)  # k of an input object (512: 64 k blocks of the slots' 8 values; 8 slots: 256, ``obj_k``)
NX_PER_NORM = 8  # input objects per norm-weight object (8 x 512 bf16 weights; 8 slots: 16 x 256, ``nx_per_norm``)
NORM_K = NX_PER_NORM * OBJ_K  # weights of a norm-weight object
C_LEN = 64  # bf16 elements of a C object
NBLK = 4  # blocks of row tiles a phase may have
MAX_K = 5120  # k of a phase's K group: its x panels (46 KB) in the helper's memory
CMD_WORDS = 16  # words of the helper's command
S_WORDS = 64  # state words of the main core's workspace (a ``rot``, ``moe`` or ``states`` design's: ``S_WORDS_ROT``, words past them)
S_WORDS_ROT = 80
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
# (rot) the rotation objects (A objects of 2 groups' ``paro.TanForm.objects``: counted in M_NT, after the tiles), the
# copies of an F_ROTE emission (each of the column's residual slice), F_ROTX's groups (of the column's slice)
M_NRA, M_NRC, M_RGN = PHASE.words("M_NRA M_NRC M_RGN", at=28)
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
# (``rot`` designs: PARO linears, ``decode.design``'s "Rotated inputs") the residual goes out (F_EMITR) as M_NRC rotated
# copies of M_RGN groups each, each after its partial sums object; an input of rotated x after a partial sums object (the
# outputs scaled by the slots' 1 / rms); a rotation phase: per column its slice of the input (X objects) rotated and out
# (F_RACT: the slice is [gate | up] per k block, rotated is act(bf16(gate)) up)
F_ROTE = PHASE.flag("F_ROTE", 7)
F_PSQ = PHASE.flag("F_PSQ", 27)
F_ROTX = PHASE.flag("F_ROTX", 28)
F_RACT = PHASE.flag("F_RACT", 29)
F_KPAIR = PHASE.flag("F_KPAIR", 18)  # (a header) its tiles hold row pairs of a matrix of K = POST_K (a row's super-blocks: two
# rows), run as two slots of one token: slot 0 = [x | 0], slot 1 = [0 | x] (the helper mirrors strip 0 into slot 1 for strip 1)
# (``moe`` designs: an MoE layer, ``decode.design``'s "MoE")
F_WS = PHASE.flag("F_WS", 19)  # the rows times the slots' scales (wsc: the shared expert's gate, an expert's routing weights)
F_COMB = PHASE.flag("F_COMB", 22)  # the combine: its input objects (the columns' partials of a slot) add into the residual; F_EMITR
F_HOUT = PHASE.flag("F_HOUT", 23)  # (an expert's gate / up sub-phase) the outputs into the hold buffer: its down's input
F_ACCUM = PHASE.flag("F_ACCUM", 24)  # (a down sub-phase) the rows add into acc ([slot][hidden] bf16; F_ACC0: stored), F_EMITA: then out
F_ACC0 = PHASE.flag("F_ACC0", 25)
F_EMITA = PHASE.flag("F_EMITA", 26)
# (``rot`` and ``post`` designs, a PARO Gemma's R phases, ``gemma4.paro_steps``: a kind of its own, whose bits other
# designs' kinds take) an F_ROTX phase's slice is [v | r] per k block of a post-norm update's region: rotated is n = (r +
# v w_post / rms(v)) scale (w_post of its groups in their rotation objects' spare bytes, ``gemma4.image``); F_RPS its
# first sub-phase: before the first group the column's partial sums of res^2 go out; F_RUPD (with F_RPS: an update's
# first R phase): the partial sums object of v follows the metadata object (1 / rms(v), kept for the update's other R
# phases) and res's exact update goes out before the sums; F_RSLOT: the slice is a 64-slot input's first 32 channels of
# each slot (proj's, ``gemma4.image``)
RPOST = XMETA.kind("rpost", extends=(PHASE,))
F_RPOST, F_RUPD, F_RPS, F_RSLOT = (RPOST.flag(n, b) for n, b in (("F_RPOST", 8), ("F_RUPD", 9), ("F_RPS", 10),
                                                                 ("F_RSLOT", 11)))
# a block's format word: the format at the low 8 bits (``formats``' fid), then
B_F32 = PHASE.flag("B_F32", 8, of="format")  # its rows go out as fp32 (two bf16 slots each)
B_GU = PHASE.flag("B_GU", 9, of="format")  # (moe) its rows are [8 gate | 8 up] groups (F_GU for this block)
B_PAIR = PHASE.flag("B_PAIR", 10, of="format")  # its rows are pairs [hi, lo] of one value (an F32 router's planes): the pairs' sums go out
B_KEEP = PHASE.flag("B_KEEP", 11, of="format")  # (B_PAIR) the block's first value stays in the main core's state (S_SG + slot): the shared expert's gate
B_G2 = PHASE.flag("B_G2", 12, of="format")  # its tiles hold row groups of 2 / 4 rows (``decode.image.Block.G``)
B_G4 = PHASE.flag("B_G4", 13, of="format")
# a split gate / up block (``decode.image.Block.fmt2``: a SiLU group's gate rows in tiles of the format at bits 0 .. 7, then its
# up rows in tiles of the format at bits 16 .. 23, B_NA and B_NB tiles: F_GU of its own, ``gusplit``)
B_SPLIT = PHASE.flag("B_SPLIT", 14, of="format")
# its row tiles in pairs sharing the tile of their last chunk (``decode.image.Block.pt``): a pair's tiles are the first
# row tile's full chunks, the tail tile (its first half: that row tile's, the second: the next one's), the next one's full
# chunks; an odd block's last row tile as a plain one
B_PT = PHASE.flag("B_PT", 15, of="format")
B_FMT2, B_NA, B_NB = 16, 24, 28  # (a split block's second format, its gate and up tiles per group: their bits' shifts)
# (``qtails`` designs) its row tiles in tail-last groups (``decode.image.Region.qt``): a group of up to 4 row tiles (one of
# the phase's M_NRT: its rows 4 R) is their full chunks (M_G - 1 each), then their tail tile (row groups: B_G4, super-block j
# row tile j's tail); the block's last group's row tiles less one at B_QM (2 bits)
B_QT = PHASE.flag("B_QT", 16, of="format")
B_QM = 24
# (``qtails`` designs) its tiles hold row groups of 8 rows of half a super-block each (``decode.image.Block.G`` 8: an
# expert unit's down of K 128)
B_G8 = PHASE.flag("B_G8", 17, of="format")

# (``states`` designs: Inkling's, ``models.inkling2``) a phase whose input is fp32 states: per k range of 1024 / sp k
# (sp: the step's packed slots) M_NR objects, one per state b < M_NR, its range [k block][slot sp][8] fp32 (the state
# array's, ``models.inkling2.overlay``), mixed by the slots' coefficients of set M_CSET (-1: one state, coefficient 1) into
# the x panels (the helper's ``hsacc`` / ``hspan``)
INK = XMETA.kind("ink", extends=(PHASE,))
F_STATES = INK.flag("F_STATES", 8)
# its first X object after the metadata is a partial sums object (the 8 columns' [slot 4][32] fp32 at 512 c: lanes b < 8
# the states' sums of squares, 8 + 8 j + b their dots with query j < 3 of the mix): the helper's coefficient sets M_CSET ..
# M_CSET + M_NQ - 1 of its M_NMIX states (``hcoef``) before its objects
F_COEF = INK.flag("F_COEF", 9)
M_NR, M_CSET, M_NQ, M_NMIX = INK.words("M_NR M_CSET M_NQ M_NMIX", at=44)
# the phase's rows (into a zeroed res: F_RESID, F_COMB) are a sub-layer's output v, which the residual stream's step takes
# after the tiles (the helper's ``hink``, ``hips``): the post conv out = (v + sum_t w_t xp_t) / sqrt(5) over the visit's
# history (an X object: its 3 inputs before the step's, fp32 [3][hidden / 8]) and the step's rows, with the conv's taps (an A
# object: bf16 [4][hidden / 8]); cur (state n of the column's states, fp32 in the helper's south neighbour) += out, or with
# ``block`` state n + 1 = out (n stays: a completed block state); then the partial sums of the next mix (as F_COEF reads
# them) of states 0 .. n + block with the nqb queries of the next A object. Out (C): the history window (the 3 inputs, the
# step's S: [3 + S][hidden / 8] fp32), cur's slice ([k block][slot sp][8] fp32) and the partial sums ([slot 4][32] fp32).
# M_INKN: n | block << 4 | nqb << 12 | bank << 16 | nm
# << 20 | bank_m << 24. bank (bits 16 .. 18: enable, bank w, its state i): cur also into bank w's state i (the column's
# copy, and its slice out a second time, after the partial sums); nm (with nqb): the message mixes' partial sums of bank
# bank_m's 2 states with the nm queries after the nqb in the queries object, as two more objects (queries 0 .. 2, 3)
F_INK = INK.flag("F_INK", 10)
M_INKN = INK.word("M_INKN", at=48)
# a blend phase (a loop visit's controller, Inkling's passes after the first): its input object (M_NX 1) holds g's halves
# (the controller's down: [half 2][column 8][slot 4][8] fp32), its tiles (M_NT) are A objects of the column's forget rows,
# then its gain rows (bf16 [72][64] each): out = cur - tanh(F g) cur + tanh(G g) msg (cur: the column's states 0 .. n
# mixed by coefficient set j, msg: bank w's 2 states by set m), its slice out ([k block][slot sp][8] fp32) after the last.
# M_BLEND: j | m << 4 | n << 8 | w << 12
F_BLEND = INK.flag("F_BLEND", 11)
M_BLEND = INK.word("M_BLEND", at=49)
INK_STATES = 5  # the column's states the helper keeps (fp32 [state][slot 4][hidden / 8]; F_LOADR: slot M_SLOT of state M_INKN)
INK_BANKS = 2  # the banks of 2 states each after them (the loop's passes: ``M_INKN``'s bank, ``M_BLEND``'s w)

# (``states`` designs) another core's object whose X and A objects interleave after it (Inkling's attention: per cache group
# with a bias, its E object, then its A objects): the main core skips M_EGRP times one X object and M_EGA A objects, then
# M_ETAIL A objects (after M_SKIP / M_NASKIP's): both cores take the broadcast streams' objects in one order
EGRP = XMETA.kind("ink groups")
M_EGRP, M_EGA, M_ETAIL = EGRP.words("M_EGRP M_EGA M_ETAIL", at=45)

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
M_IGS = MOE.block("M_IGS", MAX_ITEMS, at=56)  # (``units`` routings) the expert object's items' gate / up scales (fp32 bits)
F_ROUTE = MOE.flag("F_ROUTE", 20)  # the route object: the router's logits as its payload -> the routing (control packets, expert object)
F_EXPERT = MOE.flag("F_EXPERT", 21)  # the expert object: the column's items, M_NT their A objects (the main core's sub-phases)

# ----------------------------------------------------------------------
# State words of the main core
# ----------------------------------------------------------------------

STATE = WordMap("main core state", S_WORDS_ROT)
_S = STATE.common
(S_NNO, S_NXN, S_NXP, S_NIN, S_NT, S_NE, S_NSKIP, S_NASKIP, S_NM, S_NR, S_FLAGS, S_NS, S_K8, S_G, S_NRT, S_NB, S_KC,
 S_RT, S_XD, S_NX, S_STG, S_ESLOT, S_EJ, S_ESRC, S_FMT, S_TL) = _S.words(
    "S_NNO S_NXN S_NXP S_NIN S_NT S_NE S_NSKIP S_NASKIP S_NM S_NR S_FLAGS S_NS S_K8 S_G S_NRT S_NB S_KC S_RT S_XD S_NX "
    "S_STG S_ESLOT S_EJ S_ESRC S_FMT S_TL", at=0)
# (``packed`` designs) the slots of an input k block, the strips (helper input commands) of an input object, less one
S_SP, S_NXS, S_NXS1 = _S.words("S_SP S_NXS S_NXS1")
# (``gu_win`` designs) the rows in the gate / up windows (a split block's: its next tile's place in its group, ``gusplit``),
# the windows of 16 so far in a SiLU group
S_GN, S_GK = _S.words("S_GN S_GK")
# (``tails`` designs) the row tile's place in a B_PT block's pairs: 1 its first (its tail tile's second half: the next
# row tile's), 2 its second (its tail came with the first's: one chunk less), else 0
S_PT = _S.word("S_PT")
S_B = _S.block("S_B", 3 * NBLK, at=32)  # the block table, 3 words per block
S_TEND = _S.word("S_TEND", at=S_WORDS - 1)  # (tile) the window's end
# (``post`` designs) the partial sums object (a trip count), the scale (bf16 bits), the column, the phase's residual
# slice and 1 / norm length (M_H8, M_IH)
POST_STATE = STATE.kind("post")
S_NPS, S_SCALE = POST_STATE.words("S_NPS S_SCALE", at=44)
S_COL, S_H8, S_IH = POST_STATE.words("S_COL S_H8 S_IH", at=48)
TOK_STATE = STATE.kind("tok")
S_NAS = TOK_STATE.block("S_NAS", N_TOK, at=52)  # (``tok``) the cache objects per token object (S_NAS + M_TOK_KIND)
# (moe) the passes of an object (a trip count: a phase's 1, an expert object's sub-phases), a sub-phase's local input
# strips (a trip count), the phase's first panel chunk, the next sub-phase, the shared gate's logit per slot (fp32 bits),
# whether a pass sets up a sub-phase (a trip count: the helper's ``hsub``)
MOE_STATE = STATE.kind("moe", exclusive_with=(POST_STATE,))
S_NSUB, S_NLS, S_XC0, S_SUB = MOE_STATE.words("S_NSUB S_NLS S_XC0 S_SUB", at=44)
S_SG = MOE_STATE.block("S_SG", SLOTS, at=48)
S_NSUBC = MOE_STATE.word("S_NSUBC", at=S_WORDS + 8)
# (rot) the partial sums object (a trip count), F_ROTE's rotation objects, the groups of the next rotation object, F_ROTX's
# X objects, the rotation objects of the next one (trip counts); the copy (F_ROTX: the group in its X object), the group
# and the groups of the phase's rotations; (rot and post) an R phase's residual update before its first group (a trip
# count: ``wpupd``)
ROT_STATE = STATE.kind("rot")
S_NPQ, S_NRA, S_NGA, S_NRX, S_NRP, S_RC, S_RG, S_RN = ROT_STATE.words("S_NPQ S_NRA S_NGA S_NRX S_NRP S_RC S_RG S_RN",
                                                                      at=S_WORDS)
S_NWP = ROT_STATE.word("S_NWP", at=S_WORDS + 9)
# (``states``) an F_STATES phase's states per k range (log2) and coefficient set (M_NR, M_CSET): past S_WORDS, in a states
# design's longer state
STATES_STATE = STATE.kind("states", exclusive_with=(ROT_STATE,))
# (S_NCF: a trip count, F_COEF; S_CQN: M_NQ | M_NMIX << 8)
S_NST, S_CSET, S_NCF, S_CQN = STATES_STATE.words("S_NST S_CSET S_NCF S_CQN", at=S_WORDS + 10)
S_NEGRP, S_NEGA, S_NETL = STATES_STATE.words("S_NEGRP S_NEGA S_NETL", at=S_WORDS)  # (trip counts: ``EGRP``)
# (F_INK) the residual stream's step, its queries object (trip counts), M_INKN, the history's X object (its address; then
# the message mixes' partial sums in the helper's scratch, ``emit``)
S_NINK = STATES_STATE.word("S_NINK", at=S_WORDS + 3)
S_NIQB, S_INKN, S_IX = STATES_STATE.words("S_NIQB S_INKN S_IX", at=S_WORDS + 5)
S_BLEND = STATES_STATE.word("S_BLEND", at=S_WORDS + 14)  # (F_BLEND) M_BLEND
# (``qtails`` designs, of the longer state: B_QT blocks) the row tile of its group (its rows at yt + S_QY: block S_QS of
# the group's), the group's row tiles (0: a block of plain row tiles)
QT_STATE = STATE.kind("qt", exclusive_with=(ROT_STATE,))
S_QS = QT_STATE.word("S_QS", at=S_WORDS + 4)
S_QM = QT_STATE.word("S_QM", at=S_WORDS + 9)
S_QY = QT_STATE.word("S_QY", at=S_WORDS + 15)

# ----------------------------------------------------------------------
# The helper's command
# ----------------------------------------------------------------------

CMD = WordMap("helper command", CMD_WORDS)
C_KIND = CMD.common.word("C_KIND", at=4)  # words 0 .. 3 by kind
K_TILE, K_STRIP, K_INPUT, K_PSTRIP, K_NOP, K_BEGIN, K_SUB, K_ACC, K_HELP = CMD.common.values(
    "K_TILE K_STRIP K_INPUT K_PSTRIP K_NOP K_BEGIN K_SUB K_ACC K_HELP", of="kind")  # (K_SUB, K_ACC: moe; K_HELP: ``help_flag``)
# (``rot`` designs) the main core's rotations on the helper: the partial sums object (``psq``), the residual's copies
# (``rote``), a slice's (``rotx``), an expert half's hold buffer (``hrot``); words 0 .. 2: the A object, the X object, the
# main core's workspace (as the helper sees them)
K_PSQ, K_ROTE, K_ROTX, K_HROT = CMD.common.values("K_PSQ K_ROTE K_ROTX K_HROT", of="kind", start=9)
C_SP = CMD.common.word("C_SP")  # (K_STRIP, K_PSTRIP) the input's slots per k block
C_MIRROR = CMD.common.flag("C_MIRROR", 3, of="sp")  # (K_PSTRIP: C_SP + C_MIRROR: strip 0 into slot 1, F_KPAIR)
# (``states``) an fp32 state object: words 0 .. 2 its payload, the norm weights of its range (0: none), its index d in the
# phase's objects; C_NST log2 of the states per range, C_CSET the coefficient set (-1: unit), C_XC0 the phase's first panel chunk, C_ACC the
# main core's res region (the range's sum)
K_SSTRIP, K_COEF = CMD.common.values("K_SSTRIP K_COEF", of="kind", start=13)
C_NST, C_CSET, C_XC0, C_ACC = CMD.common.words("C_NST C_CSET C_XC0 C_ACC", at=6)
# (``states``) the residual stream's step (F_INK): words 0 .. 2 the history's payload, the A object, res (as the helper sees
# them), C_INKN the phase's M_INKN, C_NS its slots; K_ILOAD (F_LOADR): word 0 the payload, C_INKN state << 2 | slot
K_INK, K_IPS, K_ILOAD = CMD.common.values("K_INK K_IPS K_ILOAD", of="kind", start=15)
C_INKN, C_NS = CMD.common.words("C_INKN C_NS")
# (``states``) a blend's input object (word 0 its payload, 3 -1) or A object i (word 1, 3 i; ``hblend``: 2 res, C_BLEND
# M_BLEND)
K_BLEND = CMD.common.values("K_BLEND", of="kind", start=18)[0]
C_BLEND = CMD.common.word("C_BLEND")
C_NG = CMD.common.word("C_NG")  # (``qtails``, K_SSTRIP) the phase's 128-k groups (its last range: what is left of them)

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


def consts(rpost: bool = False, states: bool = False) -> dict[str, int]:
    """The words, flags and values the decode design's kernels take (``DecodeOverlayOperator.get_kernel_src``; ``rpost``:
    a rotated post design's, RPOST's flags too; ``states``: INK's)."""
    return XMETA.consts(PHASE, TOKEN, MOE, *([RPOST] if rpost else []), *([INK, EGRP] if states else [])) | \
        STATE.consts(POST_STATE, TOK_STATE, MOE_STATE, ROT_STATE, STATES_STATE, QT_STATE) | CMD.consts() | \
        {"B_FMT2": B_FMT2, "B_NA": B_NA, "B_NB": B_NB, "B_QM": B_QM}


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
    return (1 + -(-n_inputs(K, slots) * sp // slots)) * XB, DIMS_OBJ


def staged_cell(slots: int = SLOTS) -> int:
    """Bytes of a staged x's cell: twice a k block's [slot ``slots``][8] bf16 (``staged_region_bytes``)."""
    return 2 * 2 * slots * 8


def staged_region_bytes(K: int, slots: int = SLOTS) -> int:
    """Bytes of a phase's input in DDR whose x is staged: [metadata object | STAGE_PAD bytes | x: [k block K / 8]
    [cell]], a cell [slot ``slots``][8] bf16 and as many bytes past it. Several producers each write whole k blocks'
    [slot][8] into the cells from a slot of their own on (the attention's passes of a step of several sequences,
    ``decode.attn256``): what they write past their own slots lands on the later producers' slots, which those write
    after it, or past the slots. The X fill reads the cells' first halves (``staged_fill``); x's first cell is
    ``STAGED_X`` bytes into the region."""
    return STAGED_X + staged_cell(slots) * (K // 8)


def staged_fill(K: int, slots: int = SLOTS) -> tuple[int, tuple[int, int, int, int]]:
    """(bytes, shim BD dims) of the X fill of a staged x's input objects, from XB bytes into its region on (after the
    metadata object, a fill of its own): object i the XB bytes that end with x's i-th XPAY bytes, read as strips of a k
    block's bytes a cell apart (its first strips, the header: whatever the cells before hold, STAGE_PAD bytes of them
    before x), the objects XPAY of x apart."""
    kb = staged_cell(slots) // 2
    assert XB % kb == 0 and XPAY % kb == 0, (slots, kb)
    return n_inputs(K, slots) * XB, (kb // 4, XB // kb, 2 * kb // 4, XPAY // kb * 2 * kb // 4)


def post_region_bytes(K: int) -> int:
    """Bytes of a header phase's input in DDR: [metadata object | the partial sums object's payload (the 8 columns'
    C objects at 128 c) | x: [k block K / 8][v | r][slot 4][8] bf16]."""
    return XB + XPAY + 4 * SLOTS * K


def post_input_fill(K: int, sp: int = SLOTS) -> tuple[int, tuple[int, int, int, int]]:
    """(bytes, shim BD dims) of the X fill of a header phase's input (as ``input_fill``: the metadata object, the
    partial sums object, then the input objects: K / POST_K strips, SLOTS / sp per object)."""
    return (2 + -(-(K // POST_K) * sp // SLOTS)) * XB, (XB // 8, 2, XB // 8, XPAY // 4)


def rot_region_bytes(K: int, sp: int = SLOTS) -> int:
    """(rot) Bytes of an F_PSQ phase's input in DDR: [metadata object | the partial sums object's payload (the 8 columns'
    C objects at 128 c) | x: [k block K / 8][slot sp][8] bf16] (as ``post_region_bytes``)."""
    return XB + XPAY + 2 * sp * K


def rot_input_fill(K: int, sp: int = SLOTS, slots: int = SLOTS) -> tuple[int, tuple[int, int, int, int]]:
    """(bytes, shim BD dims) of the X fill of an F_PSQ phase's input (``input_fill`` with the partial sums object after
    the metadata object)."""
    return (2 + -(-n_inputs(K, slots) * sp // slots)) * XB, DIMS_OBJ


def rot_objects(objs: np.ndarray, tile_bytes: int) -> np.ndarray:
    """(rot) Rotation objects (G, 3328) uint8 (``paro.TanForm.objects``, a column's groups in order) -> the A objects
    of two groups each (ceil(G / 2), tile_bytes)."""
    G, n = objs.shape
    assert 2 * n <= tile_bytes, (n, tile_bytes)
    out = np.zeros((-(-G // 2), tile_bytes), np.uint8)
    for i in range(G):
        out[i // 2, n * (i % 2):n * (i % 2 + 1)] = objs[i]
    return out


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
