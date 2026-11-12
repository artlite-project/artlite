"""LFM3-Audio's speech encoder (projector ``lfm3a``: LFM3-Audio-2B's mmproj) on the vision overlay (``artlite.vision``:
one design for every network, its ops built by host-built transactions).

The network (llama.cpp's ``clip_graph_lfm3a``, transformers' ``Lfm3AudioEncoder`` + ``Lfm3AudioProjector``): a clip's
log-mel spectrogram (128 bins, 100 frames a second) through a downsampler of three 3 x 3 stride-2 convs (8x in time
and frequency: conv_in 1 -> 256 channels, then two depthwise convs each followed by a 1 x 1 conv 256 -> 256, SiLU
after each stage, the 16 frequencies x 256 channels flattened into lin_out -> 768), a row per 8 frames, then 17 LFM2
layers [RMSNorm -> operator | RMSNorm -> SwiGLU 768 -> 1792 -> 768] (operators: a short conv, in_proj -> [B; C; x],
y = C conv3(B x) causal, out_proj; or attention, 8 query heads on 4 KV heads of 96, q / k RMSNorm per head, NEOX RoPE
on all 96 dims at the row's position, a causal window of 32 rows), the final RMSNorm and the projector (LayerNorm ->
768 -> 2048 + b -> GELU (erf) -> 2048 -> 2048 + b).

Widths: the residual's 768 channels in C = 960 (3 passes of 80 a core row; the pads zero), the feed-forward's 1792 in
1920, a head's 96 dims in 160 (two passes of the RoPE epilogue ``rope2``: pairs (i, 48 + i) at (i, 40 + i) for i < 40 in
its first pass, i = 40 .. 47 at (i - 40, i) in its second; V's 96 then 64 zeros; O's columns likewise). GEMMs of a
norm's panels take K 768 (the pads skipped).

Layout: a band of 512 rows, column c's token panel j holding rows 64 j + 8 c .. + 8 (``ops.WinAttention``: the
attention's K / V reach every column on the B streams, so the columns share windows: window pair j is panel j of each
column). The encoder runs a clip band by band (``Encoder``): one band program (a dispatch) for every band, the
residual resident in the memtile from the downsampler through the layers to the projector; what a band leaves the next
one (the state: each conv layer's last 2 rows of B x, each attention layer's last key group, the downsampler stages'
last rows) is carried by a dispatch of its own between bands (``carry``). A stream's position is the band's first row
(the RoPE tables, written per band; the first band's masks keep no key before the stream).

The downsampler on a band (its 4096 frames): stage 1 as a GEMM of the host's im2col panels (per pixel its 9 taps in a
block, the tenth input a constant 1 where the pixel is inside the clip: the bias in the weights, so pixels past the
clip's end are 0) -> bf16 pre-activations in DDR (S1 [1 + 2048 times][1 + 64 freqs][320]: index 0 of each the zero
padding, or the previous band's last time); stage 2 strip convs (``ops.StripConv`` mode 2: SiLU of the strips first,
outputs 8 times at one frequency) -> the 1 x 1 conv (GEMM, bf16 out) -> S2 [1 + 1024][1 + 32][320], its first time past
the clip's end zeroed (a drain at an address set per band); stage 3 strip convs of the row's own panels (8 rows at one
frequency) -> the 1 x 1 conv + SiLU (panels) -> lin_out's panels in DDR ([kb 32 f + c / 8][panel], its columns permuted
on the host from HF's c 16 + f) -> lin_out + b -> the residual x0 and its records.
"""

import math
from collections.abc import Callable

import numpy as np

from artlite.operators import kmath
from artlite.operators.design import COLS, ROWS
from artlite.overlay import Transaction
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gelu_table, gemm
from artlite.vision.ops import ROPE2_TAB, Gemm, StripConv, WinAttention
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import ln, pad, vit_level
from artlite.vision.weights import Wt, cat, computed, eye, take, values

H, C, FFN_REAL, FFN, HEADS, KVH, HD, HP = 768, 960, 1792, 1920, 8, 4, 96, 160
EPS = 1e-5
THETA = 1e6
WINDOW = 32
QSCALE = 1.4426950408889634 / math.sqrt(HD)  # (the queries' scale: the softmax in log2 units)
ISOLATED, KR = 9, 832  # (the residual's channels in bfp16 blocks of their own; the norm panels' K: 104 blocks)
ROWS_BAND = 512  # (rows of a band)
FRAMES_BAND = 8 * ROWS_BAND
DCH, DPAD = 256, 320  # (the downsampler's channels, padded)
T1, F1, T2, F2, F3 = 4 * ROWS_BAND, 64, 2 * ROWS_BAND, 32, 16  # (a band's times and the frequencies of stages 1, 2, 3)
LO_KB = F3 * DCH // 8  # (lin_out's K in kb: 512)
SILU_TABLE = np.tile(kmath.SILU.table_words(), 2).view(np.uint8)
PROJ_UP, PROJ_DN = 640, 640  # (the projector's GEMMs in output chunks)


# -- the layouts of a head's 96 dims in 160 channels


def rope2_dims() -> np.ndarray:
    """Per channel of a head's 160 its dim (-1: a pad): the rope2 pairs (module docstring)."""
    out = -np.ones(HP, np.int64)
    for i in range(40):
        out[i], out[40 + i] = i, HD // 2 + i
    for i in range(HD // 2 - 40):
        out[80 + i], out[120 + i] = 40 + i, HD // 2 + 40 + i
    return out


def head_rows(heads: int, rope: bool) -> np.ndarray:
    """Rows (Q / K: rope2's, V's: plain) or O's columns of ``heads`` heads of 160 from HF's of 96."""
    j = rope2_dims() if rope else np.where(np.arange(HP) < HD, np.arange(HP), -1)
    return np.concatenate([np.where(j >= 0, h * HD + j, -1) for h in range(heads)])


def pairs_angles(pos: np.ndarray) -> np.ndarray:
    """The rope2 tables' angles of positions pos (n,): (n, 48) float64, pair i at pos theta^(-2 i / 96)."""
    return np.asarray(pos, np.float64)[:, None] * THETA ** (-np.arange(HD // 2, dtype=np.float64) * 2 / HD)


def rope2_tables(pos: np.ndarray) -> np.ndarray:
    """The rope2 epilogue's tables of 16 tokens at positions pos: [A | B] C-tile layouts of 16 x 80 bf16 (ROPE2_TAB
    bytes: A's pairs 0 .. 39, B's 40 .. 47 then cos 1, sin 0)."""
    ang = pairs_angles(pos)
    a, b = np.ones((16, 80), np.float32), np.ones((16, 80), np.float32)
    a[:, :40], a[:, 40:] = np.cos(ang[:, :40]), np.sin(ang[:, :40])
    b[:, 40:] = 0
    b[:, :8], b[:, 40:48] = np.cos(ang[:, 40:]), np.sin(ang[:, 40:])
    out = np.zeros(ROPE2_TAB, np.uint8)
    out[:2560], out[2560:] = L.ctiles(a, 16)[:2560], L.ctiles(b, 16)[:2560]
    return out


# -- weights


def vec(get: Callable[[str], Wt | np.ndarray], name: str, n: int = C) -> np.ndarray:
    """A vector of the file (a norm's weights, a bias), zero-padded to n."""
    return pad(values(get(name)).reshape(-1), n)


def residual_order(get: Callable[[str], Wt | np.ndarray], kinds: tuple[str, ...]) -> np.ndarray:
    """The residual's channel order (C entries: HF's channel, -1 a pad): the ``ISOLATED`` largest channels each in a bfp16
    block of its own after the others (the RMSNorms' panels: a block's 8 channels share an exponent, and the residual
    has a few channels 10 - 30x the rest, ~20 after the final norm: in a shared block they leave the others ~5 bits).
    The largest by the weights, the same for every input: alternately those the layers write most (the residual writers'
    row norms, summed) and those the final norm weights most."""
    wr = np.zeros(H)
    for il, kd in enumerate(kinds):
        for t in ("ffn_down", "conv_pw2" if kd == "conv" else "attn_out"):
            wr += np.linalg.norm(values(get(f"a.blk.{il}.{t}.weight")).reshape(H, -1), axis=1)
    post = np.abs(values(get("a.post_ln.weight")).reshape(-1))
    iso: list[int] = []
    for a, b in zip(np.argsort(-wr, kind="stable"), np.argsort(-post, kind="stable"), strict=True):
        for c in (int(a), int(b)):
            if c not in iso:
                iso.append(c)
    iso = iso[:ISOLATED]
    rest = [c for c in range(H) if c not in iso]
    order = rest + [-1] * (-len(rest) % 8) + [x for c in iso for x in [c] + [-1] * 7]
    assert len(order) == KR
    return np.array(order + [-1] * (C - KR), np.int64)


def rvec(get: Callable[[str], Wt | np.ndarray], name: str, order: np.ndarray) -> np.ndarray:
    """A residual-ordered vector of the file (``residual_order``)."""
    return take(values(get(name)).reshape(-1), order)


def layer_weights(get: Callable[[str], Wt | np.ndarray], il: int, kind: str, order: np.ndarray) -> dict:
    """Layer il's padded weights by role, the residual's channels in ``order`` (``residual_order``: the norms' weights,
    the residual writers' rows, the norm panels' readers' columns, K 832): "ln1", "ln2" (RMSNorm weights), "gu" ([up |
    gate] rows), "down" (K 1792); a conv layer's "bx" ([B | x] rows, each padded to 960), "c", "taps" (960, 3), "out" (K
    768: the product's channels); an attention layer's "q", "k" (rope2 rows), "v", "o" (columns by head: K 1280), "qn" /
    "kn" (the q / k norms' 160 weights of a head in rope2's layout, q's times log2 e / sqrt(96))."""
    p = f"a.blk.{il}."
    ko = order[:KR]
    cols = lambda m: take(m, ko, 1)
    w: dict = {"kind": kind, "ln1": rvec(get, p + "ln1.weight", order), "ln2": rvec(get, p + "ln2.weight", order),
               "gu": cols(cat([pad(get(p + "ffn_up.weight"), FFN), pad(get(p + "ffn_gate.weight"), FFN)])),
               "down": take(get(p + "ffn_down.weight"), order)}
    if kind == "conv":
        ip = get(p + "conv_pw1.weight")
        rows = lambda k: take(ip, np.arange(k * H, (k + 1) * H))
        w["bx"] = cols(cat([pad(rows(0), C), pad(rows(2), C)]))
        w["c"] = cols(pad(rows(1), C))
        taps = values(get(p + "conv_dw.weight")).astype(np.float32).reshape(H, 3)
        w["taps"] = np.concatenate([taps, np.zeros((C - H, 3), np.float32)])
        w["out"] = take(get(p + "conv_pw2.weight"), order)
        return w
    w["q"] = cols(take(get(p + "attn_q.weight"), head_rows(HEADS, True)))
    w["k"] = cols(take(get(p + "attn_k.weight"), head_rows(KVH, True)))
    w["v"] = cols(take(get(p + "attn_v.weight"), head_rows(KVH, False)))
    w["o"] = take(take(get(p + "attn_out.weight"), head_rows(HEADS, False), 1), order)
    idx = rope2_dims()
    for n, s in (("qn", QSCALE), ("kn", 1.0)):
        nw = values(get(p + f"attn_{n[0]}_norm.weight")).astype(np.float32).reshape(-1)
        w[n] = np.where(idx >= 0, nw[np.maximum(idx, 0)], 0).astype(np.float32) * np.float32(s)
    return w


def kinds_of(names: object, layers: int) -> tuple[str, ...]:
    """The layers' kinds ("conv", "attn") of an mmproj's tensor names."""
    return tuple("conv" if f"a.blk.{il}.conv_dw.weight" in names else "attn" for il in range(layers))


def down_weights(get: Callable[[str], Wt | np.ndarray], order: np.ndarray) -> dict:
    """The downsampler's weights: "in" (stage 1: 320 rows of K 16: taps 3 dy + dx (dy the frequency's), then the bias at
    input 9), "mid_taps" / "end_taps" (256, 9) and their biases, "mid" / "end" (the 1 x 1 convs, 320 x 256) and their
    biases, "out" (lin_out: K 4096 in order f 256 + c, rows 960) and its bias."""
    v = lambda n: values(get(n)).astype(np.float32)
    w0 = v("a.conv1d.0.weight").reshape(DCH, 9)
    stage1 = np.zeros((DPAD, 16), np.float32)
    stage1[:DCH, :9], stage1[:DCH, 9] = w0, v("a.conv1d.0.bias").reshape(-1)
    out: dict = {"in": computed("lfm3a.conv_in", stage1)}
    for nm, k in (("mid", 2), ("end", 5)):
        out[f"{nm}_taps"] = v(f"a.conv1d.{k}.weight").reshape(DCH, 9)
        out[f"{nm}_tb"] = v(f"a.conv1d.{k}.bias").reshape(-1)
        out[nm] = pad(computed(f"lfm3a.{nm}", v(f"a.conv1d.{k + 1}.weight").reshape(DCH, DCH)), DPAD)  # (1 x 1: F32)
        out[f"{nm}_b"] = vec(get, f"a.conv1d.{k + 1}.bias", DPAD)
    cols = np.array([c * F3 + f for f in range(F3) for c in range(DCH)])
    out["out"] = take(take(get("a.pre_encode.out.weight"), cols, 1), order)
    out["out_b"] = rvec(get, "a.pre_encode.out.bias", order)
    return out


def weights(get: Callable[[str], Wt | np.ndarray], kinds: tuple[str, ...], proj: int) -> dict:
    """Every padded weight of the encoder (``layer_weights`` per layer, ``down_weights`` as "down", the final norm
    "post", the projector's LayerNorm "pg" / "pb", its GEMMs "up" / "dn" and biases), as ``get`` gives them (views of the
    file), never materialized."""
    order = residual_order(get, kinds)
    w: dict = {"down": down_weights(get, order), "layers": len(kinds), "kinds": kinds, "proj": proj, "order": order,
               **{il: layer_weights(get, il, k, order) for il, k in enumerate(kinds)}}
    w["post"] = rvec(get, "a.post_ln.weight", order)
    w["pg"], w["pb"] = rvec(get, "mm.a.mlp.0.weight", order), rvec(get, "mm.a.mlp.0.bias", order)
    up = -(-proj // 320) * 320
    w["up"] = take(pad(get("mm.a.mlp.1.weight"), up), order[:KR], 1)
    w["up_b"] = vec(get, "mm.a.mlp.1.bias", up)
    w["dn"] = pad(get("mm.a.mlp.3.weight"), up)
    w["dn_b"] = vec(get, "mm.a.mlp.3.bias", up)
    w["eye"] = pad(eye(KR), C)
    return w


# -- the band program


S1_POS, S2_POS = DPAD * 2, DPAD * 2  # (bytes of a stage's position: 320 bf16)
S1_ROW, S2_ROW = (1 + F1) * S1_POS, (1 + F2) * S2_POS  # (bytes of a time: its frequencies, the padding first)
S1_BYTES = (1 + T1) * S1_ROW + 8192  # (and a margin: block 3's strips read 64 channels past a position)
S2_BYTES = (1 + T2) * S2_ROW + 8192
G_ROW = C * 2
G_BYTES = (2 + ROWS_BAND) * G_ROW  # (a conv layer's B x: rows -2, -1 (the state), the band's)
MEL_PX = 256  # (stage 1's pixels of a GEMM's column: 4 tiles; its drain's 32 row blocks: a lock counts 63)
MEL_OPS = T1 * F1 // (COLS * MEL_PX)
MEL_BYTES = 8 * MEL_PX // 8 * L.PANEL  # (an op's column: [kb 8][32 panels], the k-tile's 64 inputs: those past the 16
# zero, as the MAC multiplies them too and a stale block's exponent overflows its products to NaN)
LO_BYTES = LO_KB * 8 * L.PANEL  # (a column's lin_out panels)
ROPE_BYTES = 4 * ROPE2_TAB  # (a column's tables: 4 tiles of 16)
MASK_BYTES = 8 * 128  # (a column's masks: 8 units)
OUT_W = 2240  # (the projector's output: 2048 in passes)
TAG_ZROW = "zrow"


def tokens(c: int) -> np.ndarray:
    """Column c's rows of a band (its 64 tokens: panel j's 8 rows 64 j + 8 c ..)."""
    return (64 * np.arange(8)[:, None] + 8 * c + np.arange(8)[None]).reshape(-1)


class Net:
    """The encoder's ops on a builder (module docstring): ``band`` (one band: the downsampler, the layers, the projector)
    and ``carry`` (the state into the next band's place), over the DDR images named by ``p``."""

    def __init__(self, w: dict, p: str = "a"):
        self.w, self.p = w, p
        self.lv = vit_level(1)

    def img(self, bd: Builder, name: str, nbytes: int) -> int:
        return bd.ddr(f"{self.p}.{name}", nbytes)

    def fill(self, bd: Builder, name: str, per_col: int, addr: int, nbytes: int, off: int = 0) -> dict[int, ColOps]:
        """Fills of DDR image ``name`` ([col][per_col]) at ``off`` into the memtile at addr."""
        base = self.img(bd, name, COLS * per_col)
        bd.reading(f"{self.p}.{name}")
        return {c: ColOps(fills=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(base + c * per_col + off, nbytes, None))])
                for c in range(COLS)}

    def rms(self, bd: Builder, x: Tensor, n: Tensor, g: np.ndarray, name: str, rec: int) -> None:
        ln(bd, x, n, g, None, name=name, rec=rec, C=C, H=H, eps=EPS, mode="rms")

    # -- the downsampler

    def down(self, bd: Builder) -> tuple[Tensor, int]:
        w, lv = self.w["down"], self.lv
        s1, s2 = self.img(bd, "s1", S1_BYTES), self.img(bd, "s2", S2_BYTES)
        bd.table("silu", SILU_TABLE)
        na = 16 * MEL_PX // 8 * L.PANEL  # (the A panels: the GEMM's walk reads 16 kb)
        for k in range(MEL_OPS):  # (stage 1: column c's 256 pixels of op k at frequency 8 (k % 8) + c, times 256 (k // 8) ..)
            a = bd.mem.alloc(na)
            bd.op(self.fill(bd, "mel", MEL_OPS * MEL_BYTES, a, MEL_BYTES, k * MEL_BYTES), {})
            g = Gemm(m=64, ntile=MEL_PX // 64, K=16, N=DPAD, x=a, y=0, out="bf16", x_np=MEL_PX // 8, packed=True)
            y = bd.mem.alloc(ROWS * g.packed_block)
            g.y = y
            bd.gemm(g, w["in"], None, name=f"{self.p}.in")
            drains = {}
            for c in range(COLS):
                f1, t0 = 8 * (k % 8) + c, MEL_PX * (k // 8)
                drains[c] = ColOps(drains=[self._to_grid(y, g.packed_block, MEL_PX // 8,
                                                         s1 + (1 + t0) * S1_ROW + (1 + f1) * S1_POS, S1_ROW)])
            bd.written(f"{self.p}.s1")
            bd.defer(drains, release=[(y, ROWS * g.packed_block)])
            bd.mem.release(a, na)
        bd.flush()
        for g2 in range(F2 // 2):  # (stage 2: 4 strip convs (2 frequencies x 2 halves of their times), the 1 x 1 conv)
            bd.flush()  # (the last group's drains: their memory)
            yp = alloc(bd, lv, DPAD, "pn", pnp=32)
            st = bd.mem.alloc(StripConv.STAGE2)
            for i in range(4):
                f2, half = 2 * g2 + i // 2, i % 2
                op = StripConv(mode=2, y=yp.addr, off=StripConv.taps2(0), nobj=8, tp0=8 * i, y_np=32, silu=True)
                cols = {}
                for c in range(COLS):  # (column c: panels p = 16 c + 8 half + o, times 8 p ..: S1 times 16 p - 1 ..)
                    p0 = 16 * c + 8 * half
                    src = s1 + 16 * p0 * S1_ROW + 2 * f2 * S1_POS
                    cols[c] = ColOps(fills=[op.fill2(st, src, r, pos=S1_ROW // 4, line=S1_POS // 4, step=16 * S1_ROW // 4)
                                            for r in range(ROWS)])
                bd.reading(f"{self.p}.s1")
                bd.op(cols, {})
                img = bd.image(f"{self.p}.mid.dw", op.image(w["mid_taps"], w["mid_tb"]))
                bd.op({c: ColOps(sends=op.sends2(st), recvs=op.recvs()) for c in range(COLS)}, op.row_ops(img))
            bd.mem.release(st, StripConv.STAGE2)
            g = Gemm(m=64, ntile=4, K=DCH, N=DPAD, x=yp.addr, y=0, out="bf16", x_np=32, packed=True)
            y = bd.mem.alloc(ROWS * g.packed_block)
            g.y = y
            bd.gemm(g, w["mid"], w["mid_b"], name=f"{self.p}.mid")
            free(bd, yp)
            drains = {c: ColOps(drains=[self._to_grid(y + 2 * h * 10240, g.packed_block, 16,
                                                      s2 + (1 + 128 * c) * S2_ROW + (1 + 2 * g2 + h) * S2_POS, S2_ROW)
                                        for h in range(2)]) for c in range(COLS)}
            bd.written(f"{self.p}.s2")
            bd.defer(drains, release=[(y, ROWS * g.packed_block)])
        bd.flush()
        bd.ddr_sync()  # (the stage's drains written: the zeroed time is one of them)
        # (S2's first time past the clip's end zeroed: 6 x 4 KB of the memtile's zeros at an address set per band)
        bd.op({0: ColOps(drains=[MemTask([{"addr": bd.zero, "nwords": 1024}], repeat=6,
                                         ddr=(s2, 4096, None, (1024, 6), TAG_ZROW))])}, {})
        lo = self.img(bd, "lo", COLS * LO_BYTES)
        bd.table("silu", SILU_TABLE)
        for f3 in range(F3):  # (stage 3: column c's panels j (rows 64 j + 8 c ..) at frequency f3, the 1 x 1 conv + SiLU)
            st = bd.mem.alloc(StripConv.STAGE2)
            d3 = alloc(bd, lv, DPAD, "pn")
            op = StripConv(mode=2, y=d3.addr, off=StripConv.taps2(0), nobj=8, y_np=8, silu=True)
            cols = {}
            for c in range(COLS):  # (panel j = 8 j + c of the band's rows: S2 times 16 (8 j + c) - 1 ..)
                src = s2 + 16 * c * S2_ROW + 2 * f3 * S2_POS
                cols[c] = ColOps(fills=[op.fill2(st, src, r, pos=S2_ROW // 4, line=S2_POS // 4, step=128 * S2_ROW // 4)
                                        for r in range(ROWS)])
            bd.reading(f"{self.p}.s2")
            bd.op(cols, {})
            img = bd.image(f"{self.p}.end.dw", op.image(w["end_taps"], w["end_tb"]))
            bd.op({c: ColOps(sends=op.sends2(st), recvs=op.recvs()) for c in range(COLS)}, op.row_ops(img))
            bd.mem.release(st, StripConv.STAGE2)
            e3 = alloc(bd, lv, DPAD, "pn")
            bd.gemm(Gemm(m=64, ntile=1, K=DCH, N=DPAD, x=d3.addr, y=e3.addr, out="gelu", act="silu"), w["end"],
                    w["end_b"], name=f"{self.p}.end")
            free(bd, d3)
            nb = 32 * 8 * L.PANEL  # (its first 32 kb: the 256 channels)
            bd.written(f"{self.p}.lo")
            bd.defer({c: ColOps(drains=[MemTask([{"addr": e3.addr, "nwords": nb // 4}],
                                                ddr=(lo + c * LO_BYTES + f3 * nb, nb, None))]) for c in range(COLS)},
                     release=[(e3.addr, e3.size)])
        bd.flush()
        a = alloc(bd, lv, F3 * DCH, "pn")
        bd.op(self.fill(bd, "lo", LO_BYTES, a.addr, LO_BYTES), {})
        sp = bd.mem.alloc(8192, top=True)  # (a residual's last objects read 6912 B past it)
        x = alloc(bd, lv, C, top=True)  # (the residual of every layer: at the top, the layers' tensors below it)
        bd.mem.release(sp, 8192)
        rec = gemm(bd, a, w["out"], w["out_b"], name=f"{self.p}.lin_out", out=x, ln=True)
        free(bd, a)
        return x, rec

    @staticmethod
    def _to_grid(y: int, block: int, n: int, ddr: int, row: int) -> MemTask:
        """A drain of n row blocks of packed bf16 C tiles (``Gemm(packed=True)``: 4 blocks ``block`` bytes apart, 8
        tokens a row block) into a grid of positions of 320 bf16: row block k's 8 tokens at ``ddr`` + (8 k + i) ``row``
        (the times of one frequency)."""
        return MemTask([{"addr": y, "nwords": 8 * 160, "dims": _mdims(block), "iteration": (320, n)}], repeat=n,
                       ddr=(ddr, 8 * 640, (160, 8, row // 4, 1), (8 * row // 4, n)))

    # -- the layers

    def layer(self, bd: Builder, il: int, x: Tensor, rec: int) -> int:
        return (self.conv if self.w[il]["kind"] == "conv" else self.attn)(bd, il, x, rec)

    def ffn(self, bd: Builder, il: int, x: Tensor, rec: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["ln2"], name + ".ln2", rec)
        bd.table("silu", SILU_TABLE)
        h = alloc(bd, lv, FFN, "pn")
        bd.gemm(Gemm(m=64, ntile=1, K=KR, N=2 * FFN, x=n.addr, y=h.addr, out="geglu", act="silu", x_np=n.np, y_np=h.np),
                w["gu"], None, name=name + ".gu")
        free(bd, n)
        rec = gemm(bd, Tensor(lv, FFN_REAL, h.addr, "pn", pnp=h.np), w["down"], None, name=name + ".down", out=x, res=x,
                   ln=True)
        free(bd, h)
        return rec

    def conv(self, bd: Builder, il: int, x: Tensor, rec: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        gname = f"{self.p}.g{il}"
        ga = bd.ddr(gname, G_BYTES)
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["ln1"], name + ".ln1", rec)
        nk = Tensor(lv, KR, n.addr, "pn", pnp=n.np)
        c = alloc(bd, lv, C)
        gemm(bd, nk, w["c"], None, name=name + ".c", out=c)
        gb = 2 * C // 80 * 5760  # (the product's passes: 5760 B each)
        g = bd.mem.alloc(gb)
        bd.gemm(Gemm(m=64, ntile=1, K=KR, N=2 * C, x=n.addr, y=g, out="prod", x_np=n.np), w["bx"], None, name=name + ".bx")
        free(bd, n)
        bd.written(gname)
        # (row block j = panel j's 8 rows 64 j + 8 c ..: per token its 24 passes' 5 x 8 channels -> G row 2 + the row; an
        # op of its own: g's memory)
        bd.defer({col: ColOps(drains=[MemTask([{"addr": g, "nwords": 8 * 480, "dims": [(4, 1), (5, 32), (24, 1440), (0, 4)],
                                                "iteration": (160, 8)}], repeat=8,
                                              ddr=(ga + (2 + 8 * col) * G_ROW, 8 * G_ROW, (960, 4, 960, 1), (64 * G_ROW // 4, 8)))])
                  for col in range(COLS)}, release=[(g, gb)])
        bd.flush()
        y = alloc(bd, lv, C, "pn")
        st = bd.mem.alloc(2 * StripConv.STAGE1)
        bd.reading(gname)
        ops = [StripConv(mode=1, y=y.addr, off=(0, 80, 160), blocks=tuple(3 * r + k for r in range(ROWS))) for k in range(3)]

        def fills(k: int) -> dict[int, ColOps]:  # (panel j's strip: G rows 64 j + 8 c - 2 .. at index 64 j + 8 c)
            return {col: ColOps(fills=[ops[k].fill1(st + (k % 2) * StripConv.STAGE1, ga + 8 * col * G_ROW, C, 64)])
                    for col in range(COLS)}
        bd.op(fills(0), {})
        for k, op in enumerate(ops):
            img = bd.image(f"{name}.dw{k}", op.image(w["taps"], None))
            cols = {col: ColOps(sends=op.sends1(st + (k % 2) * StripConv.STAGE1, c.addr), recvs=op.recvs())
                    for col in range(COLS)}
            bd.op(bd.merge(cols, fills(k + 1)) if k < 2 else cols, op.row_ops(img))
        bd.mem.release(st, 2 * StripConv.STAGE1)
        free(bd, c)
        rec = gemm(bd, Tensor(lv, H, y.addr, "pn", pnp=y.np), w["out"], None, name=name + ".out", out=x, res=x, ln=True)
        free(bd, y)
        return self.ffn(bd, il, x, rec)

    def attn(self, bd: Builder, il: int, x: Tensor, rec: int) -> int:
        w, lv, name = self.w[il], self.lv, f"{self.p}.l{il}"
        kvname = f"{self.p}.kv{il}"
        kv = bd.ddr(kvname, WinAttention.kv_image(8))
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["ln1"], name + ".ln1", rec)
        rt = bd.mem.alloc(ROPE_BYTES)
        bd.op(self.fill(bd, "rope", ROPE_BYTES, rt, ROPE_BYTES), {})
        kt, vt = alloc(bd, lv, KVH * HP, "pn"), alloc(bd, lv, KVH * HP, "pn")
        for t in range(4):  # (an op per tile of 16: a head's two passes in one op)
            bd.gemm(Gemm(m=16, ntile=1, K=KR, N=KVH * HP, x=n.addr + 2 * t * L.PANEL, y=kt.addr + 2 * t * L.PANEL,
                         res=rt + t * ROPE2_TAB, out="rope2", qkn=w["kn"], qkn_n=HD, qkn_eps=EPS, x_np=n.np, y_np=kt.np),
                    w["k"], None, name=name + ".k")
        gemm(bd, Tensor(lv, KR, n.addr, "pn", pnp=n.np), w["v"], None, name=name + ".v", out=vt, kind="vt")
        at = WinAttention(q=0, o=0, mask=0, kv=kv, kv_head_bytes=WinAttention.kv_image(8) // ROWS)
        bd.written(kvname)
        bd.defer({col: at.kv_drains(kt.addr, vt.addr, col, 0, kv) for col in range(COLS)},
                 release=[(kt.addr, kt.size), (vt.addr, vt.size)])
        qt = alloc(bd, lv, HEADS * HP, "pn")
        for t in range(4):
            for p0 in (0, 2):
                bd.gemm(Gemm(m=16, ntile=1, K=KR, N=HEADS * HP, x=n.addr + 2 * t * L.PANEL, y=qt.addr + 2 * t * L.PANEL,
                             res=rt + t * ROPE2_TAB, out="rope2", qkn=w["qn"], qkn_n=HD, qkn_eps=EPS, x_np=n.np,
                             y_np=qt.np, p0=p0, npo=2), w["q"], None, name=f"{name}.q{p0}")
        free(bd, n)
        bd.mem.release(rt, ROPE_BYTES)
        mk = bd.mem.alloc(MASK_BYTES + D.A_OBJ // 2)  # (each unit's mask object reads A_OBJ / 2 bytes on)
        bd.op(self.fill(bd, "mask", MASK_BYTES, mk, MASK_BYTES), {})
        o = alloc(bd, lv, HEADS * HP, "pn")
        assert qt.addr + (HEADS * HP // 8 + 12) * 8 * L.PANEL <= 512 * 1024, "the query objects' walk past the memtile"
        at = WinAttention(q=qt.addr, o=o.addr, mask=mk, kv=kv, kv_head_bytes=WinAttention.kv_image(8) // ROWS)
        bd.flush()
        bd.reading(kvname)
        cmd = bd.image(f"{self.p}.win", at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd, 0))
        free(bd, qt)
        bd.mem.release(mk, MASK_BYTES + D.A_OBJ // 2)
        rec = gemm(bd, o, w["o"], None, name=name + ".o", out=x, res=x, ln=True)
        free(bd, o)
        return self.ffn(bd, il, x, rec)

    # -- the projector

    def final(self, bd: Builder, x: Tensor, rec: int) -> None:
        w, lv = self.w, self.lv
        n = alloc(bd, lv, C, "pn")
        self.rms(bd, x, n, w["post"], f"{self.p}.post", rec)
        free(bd, x)
        t = alloc(bd, lv, C)
        rec = gemm(bd, Tensor(lv, KR, n.addr, "pn", pnp=n.np), w["eye"], None, name=f"{self.p}.eye", out=t, ln=True)
        free(bd, n)
        n2 = alloc(bd, lv, C, "pn")
        ln(bd, t, n2, w["pg"], w["pb"], name=f"{self.p}.pln", rec=rec, C=C, H=H, eps=EPS)
        free(bd, t)
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        h = alloc(bd, lv, OUT_W, "pn")
        gemm(bd, Tensor(lv, KR, n2.addr, "pn", pnp=n2.np), w["up"], w["up_b"], name=f"{self.p}.up", out=h, kind="gelu",
             hilo=True)
        free(bd, n2)
        outb = lv.ct(OUT_W)
        out = self.img(bd, "emb", COLS * outb)
        y = alloc(bd, lv, PROJ_DN)
        hk = Tensor(lv, self.w["proj"], h.addr, "pn", pnp=h.np)
        for k in range(-(-OUT_W // PROJ_DN)):
            nk = min(PROJ_DN, OUT_W - PROJ_DN * k)
            yk = Tensor(lv, nk, y.addr)
            gemm(bd, hk, w["dn"][PROJ_DN * k:PROJ_DN * k + nk], w["dn_b"][PROJ_DN * k:PROJ_DN * k + nk],
                 name=f"{self.p}.dn{k}", out=yk, hilo=True)
            bd.written(f"{self.p}.emb")
            bd.op({c: ColOps(drains=[MemTask([{"addr": y.addr, "nwords": yk.nbytes // 4}],
                                             ddr=(out + c * outb + lv.ct(PROJ_DN * k), yk.nbytes, None))])
                   for c in range(COLS)}, {})
        free(bd, y)
        free(bd, h)

    def band(self, bd: Builder) -> None:
        x, rec = self.down(bd)
        for il in range(self.w["layers"]):
            rec = self.layer(bd, il, x, rec)
        self.final(bd, x, rec)

    # -- the state into the next band's place

    def carried(self) -> list[tuple[str, int, int, int, int]]:
        """What ``carry`` copies: (image, its bytes, from, to, bytes) - S1's / S2's last time into time -1, each conv
        layer's last 2 rows of B x into rows -2, -1, each attention layer's unit 8 slot 1 (the band's last group) into
        unit 0's (every KV head)."""
        jobs = [("s1", S1_BYTES, T1 * S1_ROW, 0, S1_ROW), ("s2", S2_BYTES, T2 * S2_ROW, 0, S2_ROW)]
        ub, kvb = WinAttention.unit_bytes(), WinAttention.kv_image(8)
        for il, kd in enumerate(self.w["kinds"]):
            if kd == "conv":
                jobs.append((f"g{il}", G_BYTES, ROWS_BAND * G_ROW, 0, 2 * G_ROW))
            else:
                jobs += [(f"kv{il}", kvb, h * kvb // ROWS + 8 * ub + 2 * D.Q, h * kvb // ROWS + 2 * D.Q, 2 * D.Q)
                         for h in range(ROWS)]
        return jobs

    def carry(self, bd: Builder) -> None:
        """The state into the next band's place (``carried``), through column 0's memtile."""
        jobs = self.carried()
        at = 0
        fills, drains = [], []
        for name, size, src, dst, nb in jobs:
            base = self.img(bd, name, size)
            bd.reading(f"{self.p}.{name}")
            fills.append(MemTask([{"addr": at, "nwords": nb // 4}], ddr=(base + src, nb, None)))
            drains.append(MemTask([{"addr": at, "nwords": nb // 4}], ddr=(base + dst, nb, None)))
            bd.written(f"{self.p}.{name}")
            at += -(-nb // 64) * 64
        assert at <= D.HEAP
        for k in range(0, len(fills), 3):
            bd.op({0: ColOps(fills=fills[k:k + 3])}, {})
        for k in range(0, len(drains), 3):
            bd.op({0: ColOps(drains=drains[k:k + 3])}, {})


def _mdims(block: int) -> list[tuple[int, int]]:
    """The memtile walk of a row block's 8 tokens of packed bf16 C tiles (``Net._to_grid``): per token its 4 blocks'
    10 x 8 channels."""
    return [(4, 1), (10, 32), (4, block // 4), (0, 4)]


# -- the host side of a band


GROUPS = T1 // MEL_PX  # (stage 1's time groups of a band: ops 8 g .. 8 g + 7 hold group g's times)


def im2col(melp: np.ndarray, band: int, valid_t1: int, groups: tuple[int, int] = (0, GROUPS)) -> np.ndarray:
    """Stage 1's panels of a band's time groups [g0, g1) (DDR image <p>.mel: [column][op][kb 8][panel 32][8][9] bytes,
    kb 2 .. 7 zero; here [column][the groups' ops]): column c of op k the pixels of frequency 8 (k % 8) + c, times
    256 (k // 8) + i of the band; a pixel's taps 3 dy + dx (mel bin 2 f + dy - 1, frame 2 t + dx - 1) and the constant 1
    at input 9, all zero for times from valid_t1 on (the clip's end: only the ops that hold earlier times are
    converted). melp: the band's frames from 4096 band - 1 (130 bins: the first and last zero, the bins from 1; frames
    past the clip zero)."""
    g0, g1 = groups
    out = np.zeros((COLS, 8 * (g1 - g0), 8, MEL_PX // 8, 8, 9), np.uint8)
    nk = 8 * max(0, min(-(-(valid_t1 - T1 * band) // MEL_PX), g1) - g0)  # (the ops with a time inside the clip)
    if nk > 0:
        k, c, px = 8 * g0 + np.arange(nk), np.arange(COLS), np.arange(MEL_PX)
        f1 = (8 * (k % 8))[:, None, None] + c[None, :, None]
        t1 = (MEL_PX * (k // 8))[:, None, None] + px[None, None, :]
        x = np.zeros((nk, COLS, MEL_PX, 16), np.float32)
        for dy in range(3):
            for dx in range(3):  # (bin 2 f + dy - 1 at index + 1; frame 2 t + dx - 1 at + 1)
                x[..., 3 * dy + dx] = melp[2 * f1 + dy, 2 * t1 + dx]
        x[..., 9] = 1
        x[(T1 * band + t1 >= valid_t1).repeat(COLS, 1)] = 0
        b = L.blocks(x.reshape(nk, COLS, MEL_PX // 8, 8, 2, 8))  # (op, col, panel, lane, kb, 9)
        out[:, :nk, :2] = b.transpose(1, 0, 4, 2, 3, 5)
    return out


def band_mel(mel: np.ndarray, band: int, start: int = 0) -> np.ndarray:
    """``im2col``'s melp of a clip's mel (128, frames: from frame ``start`` on) for a band."""
    f0 = FRAMES_BAND * band - 1
    out = np.zeros((130, FRAMES_BAND + 3), np.float32)
    a, b = max(f0, start), min(f0 + FRAMES_BAND + 3, start + mel.shape[1])
    if b > a:
        out[1:129, a - f0:b - f0] = mel[:, a - start:b - start]
    return out


def rope_image(r0: int) -> np.ndarray:
    """The band's rope2 tables (DDR image <p>.rope: [column][tile 4][ROPE2_TAB]) of rows at positions r0 + their row."""
    return np.concatenate([rope2_tables(r0 + tokens(c)[16 * t:16 * t + 16]) for c in range(COLS) for t in range(4)])


def mask_image(start: bool) -> np.ndarray:
    """The columns' attention masks (DDR image <p>.mask: [column][unit 8][128 B]); start: a stream's first band."""
    return np.concatenate([WinAttention.masks(c, 8, start).view(np.uint8).reshape(-1) for c in range(COLS)])


class Encoder:
    """The encoder (module docstring) as dispatches on a VisionDevice: ``band`` and ``carry`` programs (built at
    construction, over the same weight images: ``Builder.fork``). A clip (``__call__``) or a stream's bands (``reset``,
    ``run_band``): per band the host writes its inputs (stage 1's panels, the RoPE tables, the masks, where S2's time past
    the clip goes), runs the band program and reads its rows; the carry program between bands.

    A stream's band in progress (``run_band(partial=True)``: live audio) runs again as its frames arrive: every layer is
    causal, so its rows before the frames' end are the band's, but for the rounding of V^T's bfp16 blocks (8 keys, the
    later ones of a partial run past the frames: the rows of such a block as far from float64 as the band's, ~2%
    from them), and the state carried on comes from its complete run.
    Stage 1's panels are converted for the time groups the new frames reach. Such a stream keeps its state on the host
    between bands (``saved``), as a clip encoded meanwhile (``owner``: who ran the encoder last) overwrites it."""

    def __init__(self, npu: VisionDevice, w: dict, *, name: str = "a", state: dict | None = None):
        """state: a ``state()`` on a device whose images were restored (``VisionDevice.restore``: the programs taken,
        their shim addresses resolved on this device's images)."""
        self.npu, self.w, self.p = npu, w, name
        self.net = Net(w, name)
        self.bd = Builder(npu, reuse=state is not None)
        self.pieces = dict(state["pieces"]) if state else {"band": self._piece(self.net.band),
                                                           "carry": self._piece(self.net.carry)}
        npu.alloc(f"{name}.junk", 8 * 4096)  # (S2's zeroed time of a band before the clip's end)
        self.band_tr, self.carry_tr = (self._transaction(self.pieces[k]) for k in ("band", "carry"))
        self.band_tags = self.pieces["band"]["tags"]
        self.proj = w["proj"]
        self.band = 0
        self._carry = False  # (the last band ran complete: the carry program before the next)
        self._t1 = None  # (stage 1's times of the band in progress in the mel image; None: none of its own yet)
        self.owner: object | None = None  # (the stream that ran the encoder last; None: a clip)

    def _piece(self, build: Callable[[Builder], None]) -> dict:
        """A program: its words, their shim addresses by image (``VisionDevice.refs``), its tags ([word, tag])."""
        b = self.bd.fork()
        build(b)
        words = b.finish()
        return {"words": words, "refs": self.npu.refs(b.s.t.addrs), "tags": list(b.s.tags)}

    def _transaction(self, piece: dict) -> Transaction:
        words = piece["words"].copy()
        self.npu.resolve(words, piece["refs"], lambda n: n)
        return Transaction(self.npu.dev, words)

    def state(self) -> dict:
        """What a restored device's encoder takes besides its images (``__init__``'s state): the programs."""
        return {"pieces": dict(self.pieces)}

    def reset(self) -> None:
        """A new stream: its state zero (the downsampler stages' time -1, the conv layers' rows -2, -1; the first band's
        masks keep no key before it)."""
        p, z = self.p, lambda n: np.zeros(n, np.uint8)
        self.npu.write(f"{p}.s1", z(S1_ROW))
        self.npu.write(f"{p}.s2", z(S2_ROW))
        for il, kd in enumerate(self.w["kinds"]):
            if kd == "conv":
                self.npu.write(f"{p}.g{il}", z(2 * G_ROW))
        self.band, self._carry, self._t1 = 0, False, None

    def carry(self) -> None:
        """The state of the band that ran complete into the next band's place now (else before the next band)."""
        if self._carry:
            self.npu.run(self.carry_tr)
            self._carry = False

    def saved(self) -> list[np.ndarray]:
        """The stream's state at the next band's start (after ``carry``): the bytes ``Net.carried`` copies."""
        assert not self._carry
        return [self.npu.read(f"{self.p}.{name}", nb, dst) for name, _, _, dst, nb in self.net.carried()]

    def restore(self, band: int, saved: list[np.ndarray] | None) -> None:
        """A stream's state at band ``band``'s start (``saved``; None: band 0's, zero)."""
        if saved is None:
            self.reset()
        else:
            for (name, _, _, dst, _), data in zip(self.net.carried(), saved, strict=True):
                self.npu.write(f"{self.p}.{name}", data, dst)
            self._carry, self._t1 = False, None
        self.band = band

    def run_band(self, mel: np.ndarray, frames: int, *, start: int = 0, partial: bool = False) -> np.ndarray:
        """The stream's band (its rows 512 band ..: (512, proj) float32) of the clip's log-mel spectrogram mel (128,
        frames ``start`` .. >= the band's; ``frames``: the clip's frames so far, the end's zero padding); ``partial``:
        the band runs again with more frames (its rows past the frames' end are not the stream's), else the next call
        runs the next band, the state carried on."""
        p, b = self.p, self.band
        self.carry()
        t1 = -(-frames // 2)
        if self._t1 is None:  # (the band's first run: its tables and every group)
            g0, g1 = 0, GROUPS
            self.npu.write(f"{p}.rope", rope_image(ROWS_BAND * b))
            self.npu.write(f"{p}.mask", mask_image(b == 0))
        else:  # (the groups the new frames reach; the later ones stay zero)
            g0, g1 = self._t1 // MEL_PX, min(-(-(t1 - T1 * b) // MEL_PX), GROUPS)
        if g1 > g0:
            img = im2col(band_mel(mel, b, start), b, t1, (g0, g1))
            for c in range(COLS):
                self.npu.write(f"{p}.mel", img[c], (c * MEL_OPS + 8 * g0) * MEL_BYTES)
        self._t1 = min(t1 - T1 * b, T1) if partial else None
        t2 = -(-frames // 4) - T2 * b  # (S2's first time past the clip's end, if in this band)
        zrow = self.npu.address(f"{p}.s2", (1 + t2) * S2_ROW) if 0 <= t2 < T2 else self.npu.address(f"{p}.junk")
        tr = self.band_tr
        for idx, tag in self.band_tags:
            assert tag == TAG_ZROW
            tr.words[idx], tr.words[idx + 1] = zrow & 0xFFFFFFFF, zrow >> 32
        tr.flush()
        self.npu.run(tr)
        if not partial:
            self.band += 1
            self._carry = True
        lv = self.net.lv
        raw = self.npu.read(f"{p}.emb", COLS * lv.ct(OUT_W)).reshape(COLS, -1)
        out = np.empty((ROWS_BAND, self.proj), np.float32)
        for c in range(COLS):
            out[tokens(c)] = L.from_ctiles(raw[c], 64, OUT_W, 64)[:, :self.proj]
        return out

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """A clip's log-mel spectrogram (128, frames) -> its rows' embeddings (rows, proj)."""
        frames = mel.shape[1]
        rows = -(-frames // 8)
        self.reset()
        self.owner = None
        out = [self.run_band(mel, frames) for _ in range(-(-rows // ROWS_BAND))]
        return np.concatenate(out)[:rows]
