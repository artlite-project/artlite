"""The MOSS encoder's layer as one program of the vision overlay: a ``WhisperEncoderLayer`` in one dispatch.

The layer is ``WhisperEncoderLayer``'s self-attention half: pre-LayerNorm (``self_attn_layer_norm``), q / k / v of 16 heads
of 64 dims, one bidirectional softmax over every key of the 1,500-row sequence (the 64-row grid's padding rows are zero
keys; their queries are dropped on the host), ``out_proj`` and the residual. The residual's 1,024 channels take the
overlay's width ``C = 1,280`` (the pads zero), a head of 64 dims its 80-channel slot, and the FFN half (the second
LayerNorm, fc1 4,096 -> 4,160 with the erf-GELU, fc2 1,280, the second residual): the whole layer in one dispatch.

The geometry is the overlay's: 3 bands of 512 rows (per column 64 tokens a tile, ``m = 64``), the weight matrices in
the ``Gemm`` B layout, the K / V^T slots of the whole 1,536-key sequence in one DDR image on the attention rows' B
streams (``ops.Attention`` d 80, the second image the first's groups joined: ``Keys.joint``). One dispatch, four phases,
passed twice over the bands (a band's attention reads every band's keys):

1. ``ln`` page: the pre-LN (``ops.Apply``, mode LN) of the input's C tiles into panels (the GEMMs' A operand), pass 1 from
   a copy op's statistics of the band's C tiles;
2. resident GEMM, pass 1: k and the V^T panels of a band, their slot drains into the keys' image (K = the residual
   width's real 1,024 channels);
3. pass 2 (``ln`` page again, the band's statistics from DDR): q, then the ``a`` page's attention over all 1,536 keys;
4. resident GEMM: ``out_proj`` (K = the padded q width 1,280) with the residual added;
5. ``ln`` page: the post attention pre-LN, the residual parked in DDR meanwhile (its records are the out projection's);
6. resident GEMMs: fc1 (K 1,024, N 4,160) with the ``ln`` page's erf-GELU epilogue, the residual back, fc2 (K 4,160)
   with the second residual, the output's band.

Code pages are the program's budget: a core's program memory is 16 KB, the overlay's resident code is 9,136 B, and the
layer's phases run two of its pages - the LayerNorm apply (``ln``, 7,056 B) and the d-80 attention (``a``, 6,592 B). A page
and the resident code must fit together, so the ``a`` page's ceiling (8,600 B, ``A_PAGE_BUDGET``) is the judge: above it
the GELU epilogue would move into the resident code and the ``ln`` page yield. Both fit: 9,136 + 7,056 = 16,192 B.

The host side the program's ops read: the matrices and vectors in the program's layouts (``layer_weights``: head-major, the
residual width, q pre-scaled by log2 e / sqrt 64; the biases as the GEMMs' command words, not a folded K column), and
``layer_front`` - the arithmetic every conversion of the program does (the C tiles' bf16, the LayerNorm's fp32 statistics
over the 1,024 real channels, the GEMMs' bfp16 panels and weights with the command's bf16 biases, the d-80 attention's
per-query maximum, P rounded once to bfp16 with the row sums of that same P, and the output projection's bf16 residual) -
the gate that ``tests/operators/test_moss_layer.py`` holds against HF's layer in float64. The dispatch's ops (``front``,
on a ``Builder``) are one transaction (``Layer``: 72,708 words, 290,832 B, 48 DDR images, 4.4 MB of K / V^T slots),
verified on a fake device. The FFN half adds no code page (its GELU epilogue is an entry of the same ``ln`` page:
resident 9,136 B + ``ln`` 16,192 B of 16,384).

``Chain`` runs 24 of them: layer i's own weight images, its band the previous layer's output (the bands alternate x ->
y -> x), the K / V^T slots in two images.
"""

import math
import os
from dataclasses import dataclass
from math import erf

import numpy as np
from ml_dtypes import bfloat16

from artlite.operators import kmath
from artlite.operators.design import COLS
from artlite.operators.paging import PM_BYTES
from artlite.parallel_compile import compile_all
from artlite.vision import layout as L
from artlite.vision.builder import Builder
from artlite.vision.design import VisionDesign
from artlite.vision.device import VisionDevice, role_pages
from artlite.vision.layers import alloc, free, gelu_table, gemm, ln_rec
from artlite.vision.levels import TokenLevel
from artlite.vision.ops import Apply
from artlite.vision.program import Program
from artlite.vision.vit import Bands, Keys, attend, k_view, kv, ln, ln_records, vit_level
from artlite.vision.weights import Wt, computed, pad_index, take

A_PAGE_BUDGET = 8600  # (the d-80 attention's page: past this the GELU moves resident and the ln page yields)
PAGES = ("ln", "a")  # (the pages the layer's phases run)


def compiled(design: VisionDesign | None = None) -> VisionDesign:
    """The vision overlay's design, compiled (the layer program's design: the one every vision network runs)."""
    design = design or VisionDesign()
    compile_all([design])
    return design


def page_sizes(design: VisionDesign) -> dict[str, int]:
    """The compiled design's pages by role (``vision.device.role_pages``): the bytes each adds at the slot above the
    resident code, and the resident code itself under "_resident"."""
    pager = role_pages(os.path.dirname(design._program.insts_path))
    return {str(role): page.nbytes for role, page in pager.pages.items()} | {"_resident": pager.slot}


def program_memory(sizes: dict[str, int], pages: tuple[str, ...] = PAGES) -> dict[str, int]:
    """Per page the resident code and its page together (a core holds one page at a time: 16 KB)."""
    out = {page: sizes["_resident"] + sizes[page] for page in pages}
    if any(n > PM_BYTES for n in out.values()):
        raise AssertionError(f"the resident code and a page exceed the 16 KB program memory: {out}")
    return out


H, C = 1024, 1280  # (the residual's real channels and the overlay's width; a head of 64 dims takes 80)
HEADS, HD, HP = 16, 64, 80
FFN, FFN_REAL = 4160, 4096  # (the feed-forward's real units and the overlay's width)
EPS = 1e-5  # (WhisperConfig's / nn.LayerNorm's)
QSCALE = L.LOG2E / math.sqrt(HD)  # (the queries' scale: the softmax in log2 units, HF's 1 / sqrt(64) in log2 e)
FOLDED = ("q", "k", "v", "o")  # (the matrices: head-major rows, o's columns)


def head_rows(heads: int = HEADS, hd: int = HD, hp: int = HP) -> np.ndarray:
    """The rows (or columns) of ``heads`` heads of ``hp`` from the file's heads of ``hd``: each head's hd, then zeros
    (-1); ``vit.plain_index``'s indices, here without the vision host package's import path."""
    j = np.arange(hp)
    return np.concatenate([np.where(j < hd, h * hd + j, -1) for h in range(heads)])


def pad_heads(x: np.ndarray, dim: int = 0) -> np.ndarray:
    """Rows (dim 0) or columns (1) of the heads of HD -> heads of HP (zeros after each head's)."""
    return take(x, head_rows(), dim)


def pad(x: np.ndarray, rows: int, cols: int = 0) -> np.ndarray:
    """x's rows (and columns) zero-padded to rows (cols)."""
    out = take(x, pad_index(rows, x.shape[0]))
    return take(out, pad_index(cols, out.shape[1]), 1) if cols else out


def as_bf16(x: np.ndarray) -> np.ndarray:
    """A tensor rounded to bf16 (the C tiles' and the commands' biases' precision)."""
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def erf_gelu(x: np.ndarray) -> np.ndarray:
    """The exact erf GELU (HF's ``ACT2FN["gelu"]``), float64 in / float64 out."""
    v = np.asarray(x, np.float64)
    return 0.5 * v * (1.0 + np.vectorize(erf, otypes=[np.float64])(v / math.sqrt(2.0)))


def as_bfp16(x: np.ndarray) -> np.ndarray:
    """A tensor through the bfp16 panels' blocks (8 x 8: one exponent, 8-bit mantissas: the MAC's operand)."""
    return L.from_panels(L.panels(np.asarray(x, np.float32)), *x.shape)


def layer_weights(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The layer's matrices and vectors in the program's layouts from the checkpoint's HF tensors (fp32) as ``w``
    (q_w / q_b / k_w / v_w / v_b / o_w / o_b / ln_w / ln_b): q / k / v rows and o's columns by head (each head's 64
    dims at its 80-channel slot, zeros after), the out projection's rows at the residual width, q (and its bias) times
    the queries' scale, the K each matrix reads (q / k / v the 1,024 real channels, o the padded q width 1,280), the
    pre-LN's gamma / beta at the residual width."""
    A = HEADS * HP
    wq = np.asarray(w["q_w"], np.float32) * np.float32(QSCALE)
    return {"q": pad(pad_heads(wq, 0), A, H), "q_b": pad(pad_heads(np.asarray(w["q_b"], np.float32) * np.float32(QSCALE), 0), A),
            "k": pad(pad_heads(w["k_w"], 0), A, H), "k_b": np.zeros(A, np.float32),
            "v": pad(pad_heads(w["v_w"], 0), A, H), "v_b": pad(pad_heads(w["v_b"], 0), A),
            "o": pad(pad_heads(np.asarray(w["o_w"], np.float32), 1), C, A),
            "o_b": pad(np.asarray(w["o_b"], np.float32).reshape(-1), C),
            "fc1": pad(np.asarray(w["fc1_w"], np.float32), FFN, H),
            "fc1_b": pad(np.asarray(w["fc1_b"], np.float32).reshape(-1), FFN),
            "fc2": pad(np.asarray(w["fc2_w"], np.float32), C, FFN),
            "fc2_b": pad(np.asarray(w["fc2_b"], np.float32).reshape(-1), C),
            "ln_g": pad(np.asarray(w["ln_w"], np.float32).reshape(-1), C),
            "ln_b": pad(np.asarray(w["ln_b"], np.float32).reshape(-1), C),
            "ln_f_g": pad(np.asarray(w["f_ln_w"], np.float32).reshape(-1), C),
            "ln_f_b": pad(np.asarray(w["f_ln_b"], np.float32).reshape(-1), C)}


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, *, eps: float = EPS, real: int = H) -> np.ndarray:
    """The pre-LN of the C tiles x (rows, the residual width: its pads zero) as the ``Apply`` computes it: the statistics
    over the ``real`` channels (the pads contribute nothing), fp32, over the whole row; gamma / beta at the width."""
    v = np.asarray(x, np.float64)[:, :real]
    mean = v.mean(1, keepdims=True)
    var = v.var(1, keepdims=True)  # (the biased variance: E[x^2] - E[x]^2, HF's LayerNorm)
    inv = 1.0 / np.sqrt(var + eps)
    return (np.asarray(x, np.float64) - mean) * inv * np.asarray(gamma, np.float64) + np.asarray(beta, np.float64)


def attention(n: np.ndarray, w: dict[str, np.ndarray], *, keys: int) -> np.ndarray:
    """The d-80 attention of the pre-LN's panels n (rows, 1,024) as the ``a`` page computes it: q / k / v by the GEMMs
    (bfp16 panels and weights, the command's bf16 biases, the heads' 80-channel slots, each head's pads zero), the
    per-query reference maximum over the ``keys`` valid keys of the 64-key groups, P = 2^(s - m) rounded once to bfp16
    (the masked keys' exactly zero), O = P V in fp32 with l the row sums of that same bfp16 P, O / l and the output
    projection's A rounded to bfp16."""
    A = HEADS * HP
    rows = n.shape[0]
    a = as_bfp16(n)
    q, k, v = (as_bfp16(a @ as_bfp16(w[t]).T + as_bf16(w[t + "_b"])).reshape(rows, HEADS, HP) for t in ("q", "k", "v"))
    ctx = np.zeros((rows, HEADS, HP), np.float32)
    for h in range(HEADS):
        s = q[:, h, :HD] @ k[:, h, :HD].T
        m = s[:, :keys].max(1, keepdims=True)
        p = np.exp2(s - m)
        p[:, keys:] = 0.0  # (the padded keys of the groups: masked before the panels' rounding, as -2^14 is)
        p = as_bfp16(p)
        o = p @ v[:, h, :HD]
        ctx[:, h, :HD] = o / p.sum(1, keepdims=True)
    return as_bfp16(ctx.reshape(rows, A) @ as_bfp16(w["o"]).T + as_bf16(w["o_b"]))[:, :H]


def layer_front(x: np.ndarray, w: dict[str, np.ndarray], *, keys: int = 1500, real: int = H) -> np.ndarray:
    """The layer's attention half as the program computes it on the input's rows x (rows, 1,024): the C tiles' bf16 x,
    the pre-LN into panels, q / k / v, the attention over the ``keys`` valid rows, out_proj and the bf16 residual
    (the output's C tile)."""
    wide = np.zeros((x.shape[0], C), np.float32)  # (the C tiles: the residual's real channels, the pads zero)
    wide[:, :real] = as_bf16(x)[:, :real]
    n = as_bfp16(layer_norm(wide, w["ln_g"], w["ln_b"], real=real))  # (the panels at the residual width)
    return as_bf16(attention(n[:, :real], w, keys=keys) + wide[:, :real])


def block(x: np.ndarray, w: dict[str, np.ndarray], *, keys: int = 1500, real: int = H) -> np.ndarray:
    """The whole layer as the program computes it on x (rows, 1,024): the attention half (``layer_front``) then the FFN
    half - the post attention pre-LN's panels, fc1 4,096 -> 4,160 with the erf-GELU epilogue's bfp16 panels, fc2 (K 4,160)
    and the bf16 residual."""
    x1 = layer_front(x, w, keys=keys, real=real)
    wide = np.zeros((x.shape[0], C), np.float32)
    wide[:, :real] = x1
    n = as_bfp16(layer_norm(wide, w["ln_f_g"], w["ln_f_b"], real=real))
    h = np.asarray(erf_gelu(as_bfp16(as_bfp16(n[:, :real]) @ as_bfp16(w["fc1"]).T + as_bf16(w["fc1_b"]))), np.float32)
    out = as_bfp16(h) @ as_bfp16(w["fc2"]).T + as_bf16(w["fc2_b"])
    return as_bf16(out[:, :real] + x1)


def block_reference(x: np.ndarray, w: dict[str, np.ndarray], *, keys: int = 1500, eps: float = EPS) -> np.ndarray:
    """HF's layer 0 in float64 on the bf16 input: the attention half (``reference``) then the FFN
    (HF's layer in float64, the second half)."""
    x1 = reference(x, w, keys=keys, eps=eps)[:, :H]
    v = np.asarray(x1, np.float64)
    mean, var = v.mean(1, keepdims=True), v.var(1, keepdims=True)
    xn = (v - mean) / np.sqrt(var + eps) * np.asarray(w["f_ln_w"], np.float64) + np.asarray(w["f_ln_b"], np.float64)
    h = erf_gelu(xn @ np.asarray(w["fc1_w"], np.float64).T + np.asarray(w["fc1_b"], np.float64))
    out = h @ np.asarray(w["fc2_w"], np.float64).T + np.asarray(w["fc2_b"], np.float64)
    return np.asarray(x1 + out[:, :H], np.float32)


def reference(x: np.ndarray, w: dict[str, np.ndarray], *, keys: int = 1500, eps: float = EPS) -> np.ndarray:
    """HF's layer 0's attention half on x (the input's bf16 rounding included: the program's C tiles), float64: its
    pre-LN, q / k / v, the softmax over the ``keys`` rows, out_proj and the residual."""
    xv = np.asarray(as_bf16(x), np.float64)[:, :H]
    mean, var = xv.mean(1, keepdims=True), xv.var(1, keepdims=True)
    xn = (xv - mean) / np.sqrt(var + eps) * np.asarray(w["ln_w"], np.float64) + np.asarray(w["ln_b"], np.float64)
    q = (xn @ np.asarray(w["q_w"], np.float64).T + np.asarray(w["q_b"], np.float64)) / math.sqrt(HD)
    k = xn @ np.asarray(w["k_w"], np.float64).T
    v = xn @ np.asarray(w["v_w"], np.float64).T + np.asarray(w["v_b"], np.float64)
    qh, kh, vh = (t.reshape(-1, HEADS, HD) for t in (q, k, v))
    ctx = np.zeros((x.shape[0], HEADS, HD), np.float64)
    for h in range(HEADS):
        s = qh[:, h, :] @ kh[:keys, h, :].T
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        ctx[:, h, :] = (p @ vh[:keys, h, :]) / p.sum(1, keepdims=True)
    return (ctx.reshape(-1, HEADS * HD) @ np.asarray(w["o_w"], np.float64).T + np.asarray(w["o_b"], np.float64) +
            xv).astype(np.float32)


# -- the dispatch: the residual's 3 bands, the key sequence, the two passes


LV = vit_level(3)  # (the layer's level: 3 bands of one 64-token tile per column: 1,536 rows, the 1,500 padded)
NAME = "moss.l0"


def sequence_rows(lv: TokenLevel = LV, tokens: int = 1500) -> np.ndarray:
    """Per joint key position of the layer's attention (``ops.Attention``'s slot order, 2 x 768 for 3 bands) the tensor
    row (band, column, row) whose key it holds: image 0's keys first (the level's 12 groups), then image 1's (a joint
    sequence, ``Keys.joint``); a group of 64 keys is two columns' 4 token panels (kp0 0 / 4 of ``kv_drains``), a column's
    panels of one image the even / odd ones of a tile (token panel tp of image tp % 2), and a key panel's 8 tokens the
    cross-product''s lanes."""
    n_per_image = 8 * COLS * (lv.np // 2) * lv.nb  # (768: an image's keys)
    assert lv.m == 64 and lv.np == 8 and lv.nb == 3 and n_per_image == 768, (lv.m, lv.np, lv.nb)
    pos = np.arange(2 * n_per_image)
    img, in_image = pos // n_per_image, pos % n_per_image
    g, key = in_image // 64, in_image % 64
    band, cpair = g // 4, g % 4
    col = 2 * cpair + key // 32
    k = key % 32
    panel = 2 * (k // 8) + img
    return band * 512 + col * 64 + 8 * panel + k % 8


def valid_keys(lv: TokenLevel = LV, tokens: int = 1500) -> np.ndarray:
    """The layer's key masks over the joint sequence: the rows past the sequence's ``tokens`` are padding (the 64-row
    grid's tail), the rest are keys."""
    return sequence_rows(lv) < tokens


def row_cols(band: int, col: int) -> np.ndarray:
    """The tensor rows a band image's column holds: the 64 consecutive rows 512 band + 64 col .. (the host's layout)."""
    return 512 * band + 64 * col + np.arange(64)


KEYS = Keys(NAME + ".attn", joint=True, valid=(valid_keys(), valid_keys()), heads=HEADS)


def layer_wt(w: dict[str, np.ndarray], name: str = NAME) -> dict[str, Wt | np.ndarray]:
    """The layer's matrices as the program's weight images: ``computed`` (the host's fp32 values, packed to bfp16 by
    ``Slots`` at the build's end; a name per matrix, restorable), the biases and the pre-LN's vectors as they are."""
    images = layer_weights(w)
    return {k: (computed(f"{name}.{k}", v) if k in FOLDED else v) for k, v in images.items()}


def front(bd: Builder, w: dict[str, Wt | np.ndarray], *, name: str = NAME, lv: TokenLevel = LV, kv_ddr: int = 0,
          keys: Keys = KEYS, bands: tuple[str, str] | None = None, attention: bool = True) -> None:
    """The layer's front as one dispatch's ops: pass 1 per band its key (the pre-LN's copy with statistics, the K and V^T
    into the slots), pass 2 per band its queries (the pre-LN again, Q, the attention over every key, out_proj and the
    residual into the output's band)."""
    xin, xout = bands or (f"{name}.x", f"{name}.y")
    x, y = Bands(lv, xin, C), Bands(lv, xout, C)
    for b in range(lv.nb):  # (the keys of every band first: an op's attention reads them all)
        t = x.x.load(bd, b)
        rec = ln_records(bd, t, name=f"{name}.lnrec{b}")
        bd.defer(bd.spill(f"{xin}.rec.{b}", rec, ln_rec(lv)))
        n = alloc(bd, lv, C, "pn")
        bd.apply(Apply(m=lv.m, nblk=C // 320, mode="ln", x=t.addr, y=n.addr, ntile=lv.ntile, rec=rec, C=H, eps=EPS,
                       y_np=n.np), w["ln_g"], w["ln_b"], name=f"{name}.ln1.{b}")
        bd.mem.release(rec, ln_rec(lv))
        free(bd, t)
        kv(bd, w, name, n, kv_ddr, keys, b, q=False)
        free(bd, n)
    bd.flush()
    bd.ddr_sync()
    for b in range(lv.nb):  # (the queries: the same pre-LN, the keys' records from pass 1)
        t, rec = x.load(bd, b)
        n = alloc(bd, lv, C, "pn")
        ln(bd, t, n, w["ln_g"], w["ln_b"], name=f"{name}.ln2.{b}", rec=rec, C=C, H=H, eps=EPS)
        qt = alloc(bd, lv, keys.width, "pn")
        gemm(bd, k_view(n, w["q"]), w["q"], w["q_b"], name=f"{name}.q{b}", out=qt, kind="bfp16")
        free(bd, n)
        if attention:  # (off: the queries stand in for the context - the same ops minus the page's and the keys' work)
            o = attend(bd, qt, kv_ddr, keys)
            free(bd, qt)
        else:
            o = qt
        rec = gemm(bd, o, w["o"], w["o_b"], name=f"{name}.o{b}", out=t, res=t, ln=True)
        free(bd, o)
        # (the FFN half: the post attention pre-LN, x parked in DDR (its records drained with the LN op: the hidden panels
        # take the memtile), fc1 + the erf-GELU epilogue's page, x back, fc2 and the residual)
        park = f"{xin}.p{b}"  # (a scratch image of the layer's input parity: the residual parked across fc1)
        n2 = alloc(bd, lv, C, "pn", top=True)
        bd.defer(bd.spill(park, t.addr, t.nbytes), release=[(t.addr, t.size or t.nbytes)])
        ln(bd, t, n2, w["ln_f_g"], w["ln_f_b"], name=f"{name}.lnf{b}", rec=rec, C=C, H=H, eps=EPS)
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        h = alloc(bd, lv, FFN, "pn")
        gemm(bd, k_view(n2, w["fc1"]), w["fc1"], w["fc1_b"], name=f"{name}.fc1.{b}", out=h, kind="gelu")
        free(bd, n2)
        yt = alloc(bd, lv, C)
        bd.op(bd.load(park, yt.addr, yt.nbytes), {})
        rec = gemm(bd, k_view(h, w["fc2"]), w["fc2"], w["fc2_b"], name=f"{name}.fc2.{b}", out=yt, res=yt, ln=True)
        free(bd, h)
        y.store(bd, b, yt, rec)


@dataclass
class Layer:
    """The layer's front on a vision device: its DDR images (the input's and the output's bands, the keys'), the weights'
    images and the dispatch's program. ``write`` / ``read`` the rows: band b's column c holds the 64 consecutive rows
    ``row_cols`` (bf16 C tiles, the pads zero)."""

    npu: VisionDevice
    w: dict[str, Wt | np.ndarray]
    lv: TokenLevel = LV
    name: str = NAME
    keys: Keys = KEYS

    def __post_init__(self) -> None:
        self.npu.alloc(f"{self.name}.kv", self.keys.attention(self.lv).kv_bytes())
        bd = Builder(self.npu)
        front(bd, self.w, name=self.name, lv=self.lv, kv_ddr=self.npu.address(f"{self.name}.kv"),
              keys=self.keys)
        self.program = Program.of(self.npu, bd)
        self.transaction = self.program.transaction(self.npu)

    def write(self, x: np.ndarray) -> None:
        """The layer's input rows (the grid's 1,536) into the input's band images."""
        for b in range(self.lv.nb):
            data = np.zeros((COLS, self.lv.ct(C)), np.uint8)
            for c in range(COLS):
                rows = np.zeros((64, C), np.float32)
                rows[:, :H] = np.asarray(x, np.float32)[row_cols(b, c)]
                data[c] = L.ctiles(rows, self.lv.m)
            self.npu.write(f"{self.name}.x.{b}", data.reshape(-1))

    def read(self) -> np.ndarray:
        """The output's rows (the grid's 1,536; the input's token order)."""
        out = np.zeros((512 * self.lv.nb, C), np.float32)
        for b in range(self.lv.nb):
            data = self.npu.read(f"{self.name}.y.{b}").reshape(COLS, -1)
            for c in range(COLS):
                out[row_cols(b, c)] = L.from_ctiles(data[c], 64, C, self.lv.m)
        return out[:, :H]

    def run(self) -> None:
        self.npu.run(self.transaction)


@dataclass
class Chain:
    """``len(ws)`` layers of the same program on one vision device: layer i's weight images (``{name}.l{i}``), its band
    images the previous layer's output (the bands alternate ``x`` -> ``y`` -> ``x``), the K / V^T slots in two images
    (``{name}.kv0`` / ``kv1``: one dispatch's write of each layer's keys, the other layer's drain overlapping nothing).
    ``write`` the first layer's input rows, ``run`` every layer's dispatch (blocking), ``read`` the last layer's rows."""

    npu: VisionDevice
    ws: list[dict[str, np.ndarray]]
    lv: TokenLevel = LV
    name: str = "moss"
    keys: Keys = KEYS

    def __post_init__(self) -> None:
        kvb = self.keys.attention(self.lv).kv_bytes()
        self.kv = [self.npu.alloc(f"{self.name}.kv{n}", kvb) for n in range(2)]
        self.programs: list[Program] = []
        for i, w in enumerate(self.ws):
            bands = (f"{self.name}.x", f"{self.name}.y") if i % 2 == 0 else (f"{self.name}.y", f"{self.name}.x")
            bd = Builder(self.npu)
            front(bd, layer_wt(w, f"{self.name}.l{i}"), name=f"{self.name}.l{i}", lv=self.lv,
                  kv_ddr=self.npu.address(f"{self.name}.kv{i % 2}"), keys=self.keys, bands=bands)
            self.programs.append(Program.of(self.npu, bd))
        self.transactions = [p.transaction(self.npu) for p in self.programs]
        self.out = f"{self.name}.y" if (len(self.ws) - 1) % 2 == 0 else f"{self.name}.x"

    def write(self, x: np.ndarray) -> None:
        """The first layer's input rows into its band images."""
        for b in range(self.lv.nb):
            data = np.zeros((COLS, self.lv.ct(C)), np.uint8)
            for c in range(COLS):
                rows = np.zeros((64, C), np.float32)
                rows[:, :H] = np.asarray(x, np.float32)[row_cols(b, c)]
                data[c] = L.ctiles(rows, self.lv.m)
            self.npu.write(f"{self.name}.x.{b}", data.reshape(-1))

    def read(self) -> np.ndarray:
        """The last layer's output rows (the grid's; the first layer's token order)."""
        out = np.zeros((512 * self.lv.nb, C), np.float32)
        for b in range(self.lv.nb):
            data = self.npu.read(f"{self.out}.{b}").reshape(COLS, -1)
            for c in range(COLS):
                out[row_cols(b, c)] = L.from_ctiles(data[c], 64, C, self.lv.m)
        return out[:, :H]

    def warm(self) -> None:
        """Every layer's first dispatch, before the measured ones (a program's first pays its context's configuration)."""
        for tr in self.transactions:
            self.npu.run(tr)

    def run(self) -> list[float]:
        """Every layer's dispatch in order (blocking); the dispatches' seconds."""
        import time

        times = []
        for tr in self.transactions:
            t0 = time.perf_counter()
            self.npu.run(tr)
            times.append(time.perf_counter() - t0)
        return times
