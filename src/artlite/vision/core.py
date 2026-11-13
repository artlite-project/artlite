"""The cores' program of the vision overlay (``design``): a C++ main reading commands from the row's B stream. The main
loop (``sd15_run``), GEMMs and convs are resident code; the other kinds are a role's code (``ROLES``: "a" attention of
d 80, "s" of d 160, "c" of d 40, "v" the VAE decoder's single-head attention of d 512, "gn" the GroupNorm applies,
"ln" the LayerNorm / RMSNorm apply and the GEGLU / GELU epilogues of the feed-forward GEMMs that follow it, "w" MiniCPM-V's
window attention, "r" the RoPE epilogue of the vision encoders' Q and K GEMMs (with LFM3-VL's per-head RMSNorm,
LFM3-Audio's two-pass heads) and a SiLU epilogue at GELU's place, "d" LFM3-VL's depthwise 3 x 3 conv, the strip convs and
the SwiGLU / gated-product epilogues at GEGLU's and GELU's places, "e" the epilogues of LFM3-Audio's speech generator at
RoPE's place (C_EPI: heads of 40, fp32 tiles, the attention gate's factors), "g" the d-160 attention with per-query
masks and LFM3's output gate (its codec's window attention); together they do not fit 16 KB of program memory), each a page (``paging``) whose entry (``sd15_ext_x``) the main loop calls through the
pager's state words (the ext buffer: page id, entry; "ln": the GEMM's GEGLU second, its GELU third; "d": SwiGLU second,
the product third: a GEMM's command picks the code by the page its op loads). A page arrives in the B stream
(``device.VisionDevice``: two B slots of control packets, the first in slot 0, its first word ``paging.header()``): the
core holds both slots, its MM2S 1 streams them into its own program memory while it sleeps in ``sd15_pg_self``, then it
gives them back.

A command is one B slot: the header words ``C_*`` and, from word ``C_BIAS``, the op's biases (bf16 [pass][80]); the
core copies it into its workspace and releases the slot. Kinds:
- ``K_NOP``: nothing (the host's parity filler: an op's command in B slot 0, ``seq``);
- ``K_TABLE``: the next B slot's first ``TAB`` bytes -> the activation table (``kmath``'s: SiLU, or GELU fed 2 |x|);
- ``K_GEMM``: per token tile (C_NTILE) and pass (C_NPASS: 80 output channels each): C := the pass's bias; per k-tile
  (C_NKT) an A k-tile (``A_OBJ`` objects hold 128 / m of them, in two halves: ``design.R23``) and a B slot -> the MAC
  (m = C_M tokens, 64 or 16: an fp32 C tile, rounded to bf16 once after the residual); a conv (C_CONV, the tile = the
  column's tokens: C_R rows of groups, x = g + groups lane): per chunk of 64 input channels (C_NCH) an A object of its
  input band rows (the halo rows included) and per tap a B slot, in the host's tap order: the taps' A are the object's
  groups or lane-shifted copies (``sd15_shift``) of its edge groups, their MAC calls, walks and shift phases host-built
  (``ops.Gemm.conv_words``, ``sd15_conv``); then C_NRES A objects of a residual tile (bf16, the C-tile layout) added to
  C; statistics of the result (C_STATS: GroupNorm group sums of the
  op (C_STF: carried over from / to the next op), LayerNorm per-token sums of a tile, each emitted as one output
  chunk after the tile / the op); the output (C_OUT): the bf16 C tile (m x 160 B; O_DS: its groups 0 and 2, a
  stride-2 conv's x, in 2 x 2 chunks), its bfp16 panels (m x 90 B,
  [rb][cb]), their transposes (V^T: [rb][cb] of [8 channels][8 tokens], the exponent per channel), GEGLU (the
  pass's channels [hid 40 | gate 40] -> hid gelu(gate): bfp16 panels [rb][5]), GELU (gelu of the 80: bfp16 panels
  [rb][10]) or RoPE (role r's entry: the pass is a head, its channels [first dims 40 | second dims 40] of 40 pairs
  (k, 40 + k); the residual objects its tokens' cos | sin, a C tile [rb][cb][64] of cos (cb < 5) and sin; each pair
  turned, then bfp16 panels [rb][cb]; C_ROPE2 (m 16): a head of two passes, normed over both, both turned at its
  second pass) or, with role r's page, SiLU (bfp16 panels [rb][10]) in ``OCH`` chunks (the last one padded);
- ``K_DW`` (role d): a depthwise 3 x 3 conv of one 80-channel block of the column's tile (``sd15_ext_d``);
- ``K_DW1`` (role d): a depthwise conv of strips (``sd15_dw1``): per row block its inputs' strip in an A object half,
  the taps' offsets in the command (C_TOFF), 8 tokens 80 elements apart; mode 1 times a C row block (a causal 1-D
  conv), mode 2 + a bias, SiLU of the strip first where C_DWS (a stride-2 3 x 3 conv);
- ``K_APPLY`` (roles gn / ln): GroupNorm (+ SiLU) / LayerNorm / RMSNorm of C_NBLK 80-channel blocks of a bf16 C tile (the A
  stream: 3 row blocks per A half, ``ops.Apply``) -> bfp16 panels [rb][cb] per block: the next B slot holds gamma,
  beta (GroupNorm [c16][gamma 16 | beta 16], LayerNorm [gamma | beta], fp32); GroupNorm (C_JOIN: the images are one
  image's halves, their sums added): the B slots after it C_NREC
  group records (``REC_SLOT`` per slot, summed); LayerNorm: the first A object the tile's per-token records of the
  column's 4 cores;
- ``K_ATTN`` (roles a, s): per unit (C_NUNIT: an image and head of d 160) its queries (an A object: C_NQP query
  panels (1, 2 or 4) as pairs [pair][d-block 20][2], a pair per half; a single panel's pair partner is junk), per key
  group (C_NG of 32
  keys, C_NK of them valid in the last one) a K slot [key pair 2][d-block 20][2] and a V^T slot [d-block pair 10][key
  block 4][2] (8 dims x 8 keys): per query pair S = Q K^T (the GEMM MAC loop, m 16) and the softmax, then P V^T of the
  pairs (m 16 npair); K is given back before V is taken (the next K arrives during P V^T). The softmax: the reference
  max of each query from the first group (with C_RAISE raised, O and l rescaled, where a later group's P saturates
  at 2^63 above it: an attention sink, the group's S computed again and the reference raised to its maximum; else P
  saturates; q pre-scaled by log2 e / sqrt d), P = 2^(s - m) to 0.59 bf16 ULP, l = the row sums of the bfp16 P;
  out: O / l as panels [query panel][d-block]. A group whose keys are all masked (C_KMASK) is streamed and skipped
  (the first group computed sets the reference max). C_DENSE (any number of groups): the first C_NLIVE groups live,
  their last one's masks C_KMASK's words 0, 1 (words 2, 3: -1, every key; 4, 5: 0, the dead groups'). C_QMASK (d
  160): per query its own masks, from the unit's query object's second half ([group][key panel][2] words: bit 8 q + k
  of word h, query 4 h + q, key 8 kp + k), a group without a key kept skipped.

Statistics records (``REC`` bytes, an output chunk): GroupNorm [image 2][S1, S2][16 groups] fp32 of groups of C_GSIZE
channels counted from C_G0 (the core's first output channel C_CH0 + its passes); LayerNorm [S1 m | S2 m] fp32.
Token panels: (image row, group, image): row block rb of a tile is image rb % 2.
"""

import functools

import numpy as np

from ..gemm.mac import render_mac
from ..operators import kmath, paging, templating
from ..operators.kernel_source import kernel_prefix

# command header words
(C_KIND, C_M, C_NTILE, C_NPASS, C_NKT, C_NRES, C_OUT, C_STATS, C_GSIZE, C_CH0, C_G0, C_NBLK, C_MODE, C_NREC, C_INVN,
 C_EPS, C_NUNIT, C_NQP, C_NG, C_NK, C_CONV, C_NCH, C_R, C_ROT, C_STF, C_ND, C_NKP, C_TPAR, C_JOIN) = range(29)
C_RAISE = 29  # (attention: later key groups may raise the reference max: ``sd15_attn_raise``)
C_QKN = 32  # (a RoPE GEMM: each head RMS-normed first, C_INVN, C_EPS its 1 / n and eps, its weights at C_QKW)
C_DENSE, C_NLIVE = 30, 31  # (attention: the dense key masks, the live groups: the module docstring)
C_QMASK = 33  # (attention of d 160: per-query key masks from each unit's query object: the module docstring)
C_ROPE2 = 34  # (a RoPE GEMM of m 16: heads of two passes, ``sd15_ext_r``; the word toggles 1 / 2 per pass)
C_EPI = 35  # (role e's epilogue at O_ROPE: E_ROPE40 two heads of 40 a pass, E_F32 fp32 out, E_GATE 1 + 2^y as bf16)
E_ROPE40, E_F32, E_GATE = 1, 2, 3
C_KMASK = (576, 704)  # (attention of d 80 / 160: per image its key groups' valid bits, [group][2] words, at most 32
# groups; -1: all; around the command's walks)
C_GINV = 36  # (GroupNorm: ``group_inv(C_GSIZE)``, the group of channel c being c C_GINV >> 20)
C_MASKS = 256  # (conv: the lane shifts' byte masks, 1152 B: after the biases of at most 4 passes)
C_CALLS = 544  # (conv: the taps' MAC calls, CALL words each: ``ops.Gemm.conv_words``)
C_QKW = C_CALLS  # (a RoPE GEMM with C_QKN, not a conv: its heads' norm weights, 80 bf16)
CALL, NCALL = 10, 11  # (words of a call, the most calls)
CROSS = (1, 2, 5, 6)  # (the calls of the 4-band convs' corner taps: their walk crosses between cbuf and the object)
C_PATS = C_CALLS + CALL * NCALL  # (the MAC's walks of B and C, a dims_3d_t each: a conv's, a GEMM's, attention's P V^T)
C_WALKS = C_PATS - 7  # (a GEMM's and attention's P V^T: the MAC's walk of A)
C_TAPW = C_PATS + 14  # (conv: per tap of the order its calls and shift phase)
C_PHASE = C_TAPW + 9  # (conv: the shift phases, per output row a sd15_shift's offsets)
SH_OBJ = 1 << 31  # (a shift phase's destination offset in the A object, else in cbuf)
ST_KEEP, ST_HOLD = 1, 2  # (C_STF: a statistics record continued from / kept for the next op: one record over several
# ops; LayerNorm's: ops of one tile)
C_BIAS = 64  # (word of the biases: bf16 [pass][80])
C_GTAB = 64  # (GroupNorm apply: ``group_words``, two per 16 channels)
C_ZERO = 48  # (16 zero words: a residual of zeros)
K_NOP, K_TABLE, K_GEMM, K_APPLY, K_ATTN, K_WATTN, K_DW, K_DW1 = 1, 2, 3, 4, 5, 6, 8, 9  # (not paging.header(), 7: a
# page's first word)
DW_STRIP, DW_C, DW_MASK = 1600, 3200, 4480  # (a depthwise conv slot: its strips, C, mask words: ``ops.DwConv``)
C_NOBJ, C_DWM, C_NTAP, C_DWS = 37, 38, 39, 40  # (K_DW1: A objects, mode, taps, SiLU of the strips: ``ops.StripConv``)
C_NPAIR = 41  # (attention of d 40: query pairs a unit, 0: 8)
C_GATE = 42  # (attention of d 40: the byte of the output gate's factors 1 / sigmoid(g) (bf16) in a unit's query object
# (0: none); role g's at WIN_GATE)
WIN_GATE = 4608 + 128  # (role g: the gate's factors in a unit's query object, after its masks: ``ops.WinAttention``)
C_DWC = 43  # (K_DW1 mode 1: a half's C row block, in elements after its strip)
C_TOFF = 64  # (K_DW1: per tap its element offset in a half's strip)
DW1_STRIP = 2048  # (K_DW1 mode 2: a half's strip, elements)
O_BF16, O_BFP16, O_GEGLU, O_VT, O_DS, O_GELU, O_ROPE = 0, 1, 2, 3, 4, 5, 6
S_NONE, S_GN, S_LN = 0, 1, 2
A_GN, A_GNSILU, A_LN, A_RMS = 1, 2, 3, 4
REC_GN = 256  # (bytes of a GroupNorm record)
REC_SLOT = 22  # (GroupNorm records per B slot)
P = kernel_prefix("sd15")  # (the kernels' names: ``kernel_source``)
ROLES = ("a", "c", "v", "gn", "ln", "w", "r", "d", "s", "e", "g")
WINS = ("gnc", "mac", "out", "acqa", "acqb", "apply", "aqk", "apv", "afin", "aexp", "aexp2", "nomac", "a40qk")  # (nomac: no GEMM / conv MAC, an ablation)

_SRC = templating.load("artlite/vision/core.cc.jinja")


def group_words(ch0: int, nch: int, gsize: int, g0: int, nreal: int = 0) -> np.ndarray:
    """C_GTAB of a GroupNorm apply core (channels [ch0, ch0 + nch) in groups of gsize >= 4, its record window from group
    g0; nreal: only the first that many are real, the rest pads in the last real channel's group): per 16 channels from
    ch0 the window indices of their groups in lane order (up to four: bits 0..4, 5..9, 10..14, 15..19; a third or
    fourth repeats the last) and the fourth's first lane (bits 20..24, 16: none), then the lanes of the first group and
    of the third (masks, bits 0..15, 16..31: the second group's lanes are the rest)."""
    assert gsize >= 4 and nch % 16 == 0
    nreal = nreal or nch
    out = []
    for c in range(ch0, ch0 + nch, 16):
        grp = [min(c + j, ch0 + nreal - 1) // gsize for j in range(16)]
        gs = sorted(set(grp))
        assert len(gs) <= 4 and all(0 <= g - g0 < 16 for g in gs), (c, gs, g0)
        ga, gb, gc, gd = (gs + [gs[-1]] * 3)[:4] if len(gs) > 2 else (gs[0], gs[0], gs[-1], gs[-1])
        ma = sum(1 << j for j in range(16) if grp[j] == ga)
        mc = sum(1 << j for j in range(16) if grp[j] == gc and gc != ga)
        e3 = grp.index(gd) if len(gs) == 4 else 16
        out += [(ga - g0) | (gb - g0) << 5 | (gc - g0) << 10 | (gd - g0) << 15 | e3 << 20, ma | mc << 16]
    return np.array(out, np.uint32).view(np.int32)


@functools.cache
def group_inv(gsize: int) -> int:
    """C_GINV: ceil(2^20 / gsize), exact as c // gsize for every channel index c < 4096."""
    gi = -(-(1 << 20) // gsize)
    assert all(c * gi >> 20 == c // gsize for c in range(4096)), gsize
    return gi


@functools.cache
def shift_masks(joined: bool = False) -> bytes:
    """``sd15_shift``'s byte masks (``C_MASKS``): per byte of a 576-byte cycle (8 panels [kb][image 2]) 0xFF where kept
    after a shift down (lane 0 of each panel cleared), then up (lane 7 cleared). joined: the two images are one image's
    halves side by side, a cut lane of image 1 down / image 0 up the pixel across the seam: only image 0's lane 0 (down)
    and image 1's lane 7 (up) cleared."""
    def cut(d: int, i: int) -> bool:
        img = i // 72 % 2
        return (i % 72 < 9 and not (joined and img)) if d == 0 else (i % 72 >= 63 and not (joined and not img))
    return bytes(0 if cut(d, i) else 0xFF for d in range(2) for i in range(576))


def source(*, win: str = "") -> str:
    from . import design as d
    assert win in ("",) + WINS, win
    assert paging.header() not in (K_NOP, K_TABLE, K_GEMM, K_APPLY, K_ATTN, K_WATTN, K_DW, K_DW1), "a command kind reads as a page"
    return _SRC.render(
        p=P, pager=paging.source(P, wake=d.L_WAKE, send=d.L_SEND), header=paging.header(), lock_base=paging.LOCK_BASE,
        L_DONE=d.L_DONE, macf=render_mac(name=f"{P}_macf", k=d.KT, f32=True, nbr_arg=True, pats=True, c2_bank="b"),
        pvm=render_mac(name=f"{P}_pvm", k=32, f32=True, nbr_arg=True), Q=d.Q,
        gelu_tmax=kmath.GELU.t_max, win=win, CALL=CALL, CROSS=CROSS, A_OBJ=d.A_OBJ, WinGate=WIN_GATE, R23=d.R23, A_HI=d.A_HI, QS=(d.A_OBJ // 2 + d.A_HI) // 4, OCH=d.OCH, OSL=d.OSL, NO=d.NO, WS=d.WS, TAB=d.TAB,
        **{k: v for k, v in globals().items() if k[:2] in ("C_", "K_", "O_", "S_", "A_", "E_") or k[:3] in ("REC", "ST_", "DW_", "DW1")})
