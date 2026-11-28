"""Inkling 2's vision encoder (projector ``inkling2v``: Inkling 2 Mini's mmproj) on the vision overlay
(``artlite.vision``: one design for every image network, its ops built by host-built transactions; the ViT's layers
``artlite.vision.vit``'s, channels and heads padded there).

The network (llama.cpp's ``mtmd_graph_inkling2v``, transformers' ``Inkling2VisionModel``): an image of ph x pw patches
of 20 pixels (both even: the image padded, ``host``), the patch embedding (a conv, no bias), the learned 32 x 32
position embedding interpolated bilinearly to the grid (the corners not aligned), a class token before the patches
(``class_embd`` + ``class_position_embd``), pre_ln, 23 layers of attention over the image's patches and its class token
with a 2-D RoPE (in each head of 64 dims, pair i = dims (2 i, 2 i + 1) turned by (x + 1) s^i for i < 16 and by (y +
1) s^(i - 16) above, s = 10000^(-1 / 16); the class token at (0, 0)), the attention's output and the feed-forward's
(GELU erf) times the layer scales ``ls1`` / ``ls2`` before they join the residual; then, the class token dropped, each 2
x 2 window's patches (0, 0), (0, 1), (1, 0), (1, 1) merged (4 hidden, no norm) -> mm.1 -> GELU (erf) -> mm.2 (the
language model's width), no biases: one token per window, raster order.

Tower (``tower``): 23 layers of 1024, 16 heads of 64, FFN 4096 (C 1280, FFN 4160), LayerNorm eps 1e-5.

Layout: ``artlite.vision.windows``'s (bands of 512 patches, a window's 4 patches in one tile, the attention's keys one
dense joint sequence) with the class token at patch 0 of the first window past the image (``windows.Layout.cls``): an
image of n merged tokens takes ceil((n + 1) / 128) bands. The class token's slot has zero panels (its constant input
too) and its position row the class embedding.

Weights: the layer scales folded into the rows of attn_out and ffn_down and into their biases (``weights.scale_rows``:
the packer multiplies the rows as it packs them), the queries' scale log2 e / 8 into q; RoPE in the Q and K GEMMs'
epilogue (``ops.Gemm(out="rope")``: a head's 80 channels pairs (k, 40 + k)): a head's pair i at channels (i, 40 + i),
i < 32 (``rope_heads``), each slot's cos | sin row (``rope_rows``: x first, positions from 1, the class token's 0).

The patch embedding takes the pixels' exact 8-bit integers q = p - 128 (the normalization folded into its weights:
``vit.folded_patch``), hi + lo weights (``ops.Gemm``), the position rows as its residual (host rows, bf16). A padding
pixel (-1 on the 0 .. 255 scale) is 1 / 127.5 below black: its panels take black (q = -128) and its patch's position
row the rest, -1 / 127.5 times the patch weights summed over the padding pixels (``pad_rows``: per pattern of padding
columns and rows at the image's right and bottom edges, from ``patch_sums``). pre_ln: a LayerNorm apply into bf16 C
tiles (``ops.Apply(ctiles=True)``: the residual stream's start at bf16 precision; its bfp16 panels instead double the
embeddings' error), their LayerNorm records from a copy op (``vit.ln_records``). The merger's MLP reads the last
residual's bfp16 panels (``windows.Merger(norm=False)``), hi + lo weights.

Precision (``artlite.vision.vit``'s rounding points): per merged token ~4% from the float64 encoder (median; the CPU at
the overlay's rounding points the same), where llama.cpp's own embeddings are 1.3% off (its im2col rounds the pixels to
fp16, which the network amplifies ~17x: layer 0 fed llama.cpp's pre_ln gives its output to 2e-7); the language model's KL
with the overlay's embeddings stays near its KL with llama.cpp's (``tests/inkling2v``).

Dispatches (``Encoder``: ``windows.WindowEncoder``): a program per nb (built on first use): nb 1 runs as one dispatch;
larger images as the patch embedding, pieces of ``layers_per_piece(nb)`` layers (one dispatch each: well under the
driver's 2 s), the final merger.
"""

import math
from collections.abc import Callable

import numpy as np

from artlite.operators.design import COLS
from artlite.vision import windows as W
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gemm, ln_rec
from artlite.vision.ops import Apply
from artlite.vision.program import ProgramStore
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import (
    Bands,
    Keys,
    Vit,
    blocks4,
    folded_patch,
    kv_bytes,
    layer_bands,
    ln_records,
    pad,
    pad_heads,
)
from artlite.vision.vit import vit_level as level
from artlite.vision.weights import Wt, computed, f32, scale_rows, take, values
from artlite.vision.windows import Layout, Names, out_width

from .host import PATCH, TOKEN_PX

PIX, ONE, KP = 3 * PATCH * PATCH, 3 * PATCH * PATCH, 1216  # (a patch's inputs (c, ky, kx); the constant input 1 in a
# dedicated bfp16 block; the panels' K: whole k-tiles)
FOLD_A = 1 / 127.5  # (the pixels' scale in the folded patch weights: ``vit.folded_patch``)
UPC, DNC = 960, 640  # (the MLP's output chunks: mm.1's (3 passes a row), mm.2's (2))
PIECE_COST = (1.6, 0.15)  # (a layer ~1.6 nb + 0.15 nb^2 ms: ``windows.piece_layers``)
THETA = 10000.0


def tower(hidden: int, heads: int, ffn: int, eps: float) -> Vit:
    """The widths of an Inkling 2 tower (NotImplementedError: one the overlay does not run)."""
    return Vit.of(hidden, heads, ffn, eps, gelu="gelu_erf")


def merger(vt: Vit) -> W.Merger:
    """The final merger of a tower: mm.1's 4 x H outputs in chunks of UPC (and the rest: whole passes), no norm."""
    fup = -(-4 * vt.H // 320) * 320
    return W.Merger(H=vt.H, C=vt.C, eps=vt.EPS, up=(UPC,) * (fup // UPC) + ((fup % UPC,) if fup % UPC else ()), dn=DNC,
                    norm=False)


def rope_index(heads: range, hd: int, hp: int) -> np.ndarray:
    """Rows of ``heads`` heads of hp for the RoPE epilogue (pairs (k, hp / 2 + k)) from the file's heads of hd whose
    pairs are interleaved: dims 2 k, 2 k + 1 at k, hp / 2 + k (k < hd / 2), zeros (-1) elsewhere."""
    j = np.arange(hp)
    k = j % (hp // 2)
    return np.concatenate([np.where(k < hd // 2, h * hd + 2 * k + (j >= hp // 2), -1) for h in heads])


def rope_heads(w: Wt | np.ndarray, vt: Vit) -> Wt | np.ndarray:
    """Q or K rows (or a bias) in the RoPE epilogue's head layout (``rope_index``)."""
    return take(w, rope_index(range(vt.HEADS), vt.HD, vt.HP))


def rope_rows(ph: int, pw: int, vt: Vit) -> tuple[np.ndarray, np.ndarray]:
    """The patches' (raster order) and the class token's channels of the RoPE epilogue (hp each): cos of a head's
    pairs, then 1 to hp / 2, sin of the pairs, then 0; pair i's angle (x + 1) s^i for i < HD / 4, (y + 1) s^(i - HD /
    4) above, s = 10000^(-4 / HD); the class token's 0."""
    q = vt.HD // 4
    f = THETA ** (-np.arange(q, dtype=np.float64) / q)
    y, x = np.divmod(np.arange(ph * pw), pw)
    ang = np.concatenate([np.zeros((1, 2 * q)), np.concatenate([(x + 1)[:, None] * f, (y + 1)[:, None] * f], 1)])
    out = np.zeros((len(ang), vt.HP), np.float32)
    out[:, :vt.HP // 2] = 1
    out[:, :2 * q], out[:, vt.HP // 2:vt.HP // 2 + 2 * q] = np.cos(ang), np.sin(ang)
    return out[1:], out[0]


def pos_rows(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """The learned position embedding (n x n rows of the hidden, float32) at a ph x pw patch grid (raster order):
    bilinear, the corners not aligned, in float32 as llama.cpp's ``ggml_interpolate`` (GGML_SCALE_MODE_BILINEAR)."""
    n = round(math.sqrt(table.shape[0]))
    if (ph, pw) == (n, n):
        return np.array(table, np.float32)
    t = np.asarray(table, np.float32).reshape(n, n, -1)
    f32_ = np.float32

    def axis(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = (np.arange(m, dtype=f32_) + f32_(0.5)) / (f32_(m) / f32_(n)) - f32_(0.5)
        i0 = np.floor(s).astype(np.int64)
        i1 = np.clip(i0 + 1, 0, n - 1)
        i0 = np.clip(i0, 0, n - 1)
        return i0, i1, np.clip(s - i0.astype(f32_), 0, 1).astype(f32_)
    y0, y1, dy = axis(ph)
    x0, x1, dx = axis(pw)
    dy, dx = dy[:, None, None], dx[None, :, None]
    a, b, c, d = t[y0][:, x0], t[y0][:, x1], t[y1][:, x0], t[y1][:, x1]
    out = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy
    return out.reshape(ph * pw, -1).astype(np.float32)


def patch_sums(w: np.ndarray) -> np.ndarray:
    """The patch weights (rows, 3, P, P) summed over the channels and the bottom-right corners: (rows, P + 1, P + 1)
    float64, entry [a, b] the sum over the last a pixel rows and the last b pixel columns (``pad_rows``)."""
    s = np.asarray(w, np.float64).sum(1)[:, ::-1, ::-1].cumsum(1).cumsum(2)
    out = np.zeros((s.shape[0], PATCH + 1, PATCH + 1))
    out[:, 1:, 1:] = s
    return out


def pad_rows(sums: np.ndarray, height: int, width: int) -> np.ndarray:
    """Per patch of an image of height x width pixels padded to multiples of ``TOKEN_PX`` (raster order) its position
    row's correction for its padding pixels: -1 / 127.5 times the patch weights summed over them (``patch_sums``:
    the pixels past the right edge or the bottom one), float64 (patches, rows)."""
    ph, pw = -(-height // TOKEN_PX) * 2, -(-width // TOKEN_PX) * 2
    out = np.zeros((ph, pw, sums.shape[0]))
    ry = np.clip((np.arange(ph) + 1) * PATCH - height, 0, PATCH)  # (padding rows of each patch row)
    rx = np.clip((np.arange(pw) + 1) * PATCH - width, 0, PATCH)
    for i in np.flatnonzero(ry):
        out[i] -= sums[:, ry[i], PATCH]
    for j in np.flatnonzero(rx):
        out[:, j] -= sums[:, PATCH, rx[j]]
    for i in np.flatnonzero(ry):
        for j in np.flatnonzero(rx):
            out[i, j] += sums[:, ry[i], rx[j]]  # (the corner, counted twice)
    return out.reshape(ph * pw, -1) * FOLD_A


def layer_weights(get: Callable[[str], Wt | np.ndarray], il: int, vt: Vit) -> dict[str, Wt | np.ndarray]:
    """Layer il's padded weights by role: q (x log2 e / sqrt HD) and k (rows by head for the RoPE epilogue:
    ``rope_heads``), v (rows by head), o (columns by head, rows x ls1), up, down (rows x ls2) and their biases
    (``*_b``), ln1 / ln2 gamma and beta (``*_g``, ``*_b``); q, k, v and up of K = the hidden; ``get(name)``: a tensor of
    the mmproj, a matrix (rows, cols: ``weights.Wt`` or a tensor) or a vector (1, cols) float32."""
    p = f"v.blk.{il}."
    C, A, F = vt.C, vt.HEADS * vt.HP, vt.FFN
    vec = lambda n: values(get(p + n)).reshape(-1)
    ls1, ls2, qs = vec("ls1.weight"), vec("ls2.weight"), np.float32(vt.qscale)
    w = {"q": pad(rope_heads(f32(get(p + "attn_q.weight")) * float(qs), vt), A),
         "q_b": rope_heads(vec("attn_q.bias") * qs, vt),
         "k": pad(rope_heads(f32(get(p + "attn_k.weight")), vt), A), "k_b": rope_heads(vec("attn_k.bias"), vt),
         "v": pad(pad_heads(f32(get(p + "attn_v.weight")), vt=vt), A), "v_b": pad_heads(vec("attn_v.bias"), vt=vt)}
    w["o"] = pad(pad_heads(scale_rows(f32(get(p + "attn_out.weight")), ls1), 1, vt), C, A)
    w["o_b"] = pad(vec("attn_out.bias") * ls1, C)
    w["up"] = pad(f32(get(p + "ffn_up.weight")), F)
    w["up_b"] = pad(vec("ffn_up.bias"), F)
    w["down"] = pad(scale_rows(f32(get(p + "ffn_down.weight")), ls2), C, F)
    w["down_b"] = pad(vec("ffn_down.bias") * ls2, C)
    for n in ("ln1", "ln2"):
        w[n + "_g"] = pad(vec(n + ".weight"), C)
        w[n + "_b"] = pad(vec(n + ".bias"), C)
    return w


def weights(get: Callable[[str], Wt | np.ndarray], layers: int, proj: int, vt: Vit) -> dict:
    """Every padded weight of the encoder: "patch" (``patch_weights``), "pre_g" / "pre_b" (pre_ln), per layer il its
    ``layer_weights``, "up" / "down" (mm.1 on the merged 4 x C channels -> FUP, mm.2 FUP -> ``out_width``) and their
    (zero) biases; "pos" (the 32 x 32 position embedding, float32), "cls" (the class token's row, C), "sums"
    (``patch_sums``); "layers", "proj", "vit" (the widths). Matrices as ``get`` gives them (``weights.Wt``: views of
    the file), never materialized."""
    md, C = merger(vt), vt.C
    vec = lambda n: values(get(n)).reshape(-1)
    wp = values(get("v.patch_embd.weight")).astype(np.float64).reshape(vt.H, 3, PATCH, PATCH)
    w: dict = {"patch": computed("inkling2v.patch", folded_patch([wp.reshape(vt.H, PIX)], c=C, kp=KP, one=ONE)),
               "pre_g": pad(vec("v.pre_ln.weight"), C), "pre_b": pad(vec("v.pre_ln.bias"), C),
               "pos": vec("v.position_embd.weight").reshape(-1, vt.H), "sums": patch_sums(wp),
               "cls": pad(vec("v.class_embd") + vec("v.class_position_embd.weight"), C),
               "layers": layers, "proj": proj, "vit": vt, **{il: layer_weights(get, il, vt) for il in range(layers)}}
    w["up"] = pad(blocks4(get("mm.1.weight"), 1, vt.H, C), md.FUP, md.merged)
    w["up_b"] = np.zeros(md.FUP, np.float32)
    w["down"] = pad(f32(get("mm.2.weight")), out_width(proj), md.FUP)
    w["down_b"] = np.zeros(out_width(proj), np.float32)
    return w


def layers_per_piece(nb: int, layers: int = 23) -> int:
    """Layers of a dispatch at nb bands (``windows.piece_layers``)."""
    return W.piece_layers(nb, layers, PIECE_COST)


def keys_of(name: str, vt: Vit, **kw: object) -> Keys:
    """A level's attention keys (one dense joint sequence) of the tower's heads."""
    return Keys(name, joint=True, dense=True, heads=vt.HEADS, **kw)


def patch_embed(bd: Builder, w: dict, nm: Names, nb: int, vt: Vit) -> None:
    """The patch embedding of nb bands and pre_ln: per band a GEMM of the host's patch panels (DDR image <p>.pix.b)
    with the position rows (<p>.pos.b) as its residual and its LayerNorm records, pre_ln into bf16 C tiles -> the
    residual's bands before layer 0 and their records (a copy op's)."""
    lv = level(nb)
    C = vt.C
    out = Bands(lv, nm.x(0), C)
    pn, cn = lv.pn(KP), lv.ct(C)
    for b in range(nb):
        pi, ri = bd.ddr(f"{nm.p}.pix.{b}", COLS * pn), bd.ddr(f"{nm.p}.pos.{b}", COLS * cn)
        pt = alloc(bd, lv, KP, "pn")
        e = alloc(bd, lv, C)
        bd.op({c: ColOps(fills=[MemTask([{"addr": pt.addr, "nwords": pn // 4}], ddr=(pi + c * pn, pn, None)),
                               MemTask([{"addr": e.addr, "nwords": cn // 4}], ddr=(ri + c * cn, cn, None))])
               for c in range(COLS)}, {})
        rec = gemm(bd, pt, w["patch"], None, name=f"{nm.p}.patch", out=e, res=e, ln=True, hilo=True)
        free(bd, pt)
        x = alloc(bd, lv, C)
        bd.apply(Apply(m=lv.m, nblk=C // 320, mode="ln", x=e.addr, y=x.addr, ntile=lv.ntile, rec=rec, C=vt.H,
                       eps=vt.EPS, ctiles=True), w["pre_g"], w["pre_b"], name=f"{nm.p}.pre")
        bd.mem.release(rec, ln_rec(lv))
        free(bd, e)
        out.store(bd, b, x, ln_records(bd, x, name=f"{nm.p}.x0", C=C))


def layers(bd: Builder, w: dict, nm: Names, nb: int, il: range, kv: int, vt: Vit) -> None:
    """Layers ``il`` on nb bands (the residual from image x(il[0]) to x(il[-1] + 1))."""
    lv = level(nb)
    keys = keys_of(nm.attn(nb), vt)
    rt = lv.ct(vt.HP)

    def rope(b_: Builder, band: int) -> Tensor:
        t = alloc(b_, lv, vt.HP)
        b_.ddr(f"{nm.p}.rope.{band}", COLS * rt)
        b_.op(b_.load(f"{nm.p}.rope.{band}", t.addr, rt), {})
        return t
    for i in il:
        layer_bands(bd, w[i], nm.layer(i), Bands(lv, nm.x(i), vt.C), Bands(lv, nm.x(i + 1), vt.C), kv_ddr=kv, keys=keys,
                    park=lambda b: f"{nm.p}.park", rope=rope, vt=vt)


class Encoder(W.WindowEncoder):
    """The encoder (module docstring) as dispatches on a VisionDevice (``windows.WindowEncoder``)."""

    PATCH = PATCH

    def __init__(self, npu: VisionDevice, w: dict, *, max_bands: int, name: str = "i", state: dict | None = None,
                 store: ProgramStore | None = None):
        """w: ``weights`` (its "vit": the tower's widths); max_bands: the largest level (the K / V^T slots' DDR,
        allocated first); state: a ``state()`` on a device whose images were restored (``VisionDevice.restore``: its
        programs taken, new ones built over the images as they are); store: where those are kept
        (``WindowEncoder``)."""
        self.vt = w["vit"]
        super().__init__(npu, w, max_bands=max_bands, name=name, state=state, store=store,
                         kv_bytes=kv_bytes(level(max_bands), keys_of("", self.vt)))

    def layout(self, px: np.ndarray, two: bool = False) -> Layout:
        """(an image's pixels of any size: padded to whole merged tokens, a class token)"""
        assert px.ndim == 3 and not two, px.shape
        return Layout(-(-px.shape[0] // TOKEN_PX), -(-px.shape[1] // TOKEN_PX), cls=True)

    def patch_embed(self, bd: Builder, nb: int) -> None:
        patch_embed(bd, self.w, self.nm, nb, self.vt)

    def layers(self, bd: Builder, nb: int, il: range) -> None:
        layers(bd, self.w, self.nm, nb, il, self.kv, self.vt)

    def final(self, bd: Builder, nb: int) -> None:
        W.merger(bd, self.w, self.nm, nb, self.nl, merger(self.vt))

    def layers_per_piece(self, nb: int) -> int:
        return layers_per_piece(nb, self.nl)

    def pixel_panels(self, lay: Layout, px: np.ndarray) -> dict[str, np.ndarray]:
        """(the padding pixels black: their difference is the position rows', ``pad_rows``)"""
        full = np.zeros((lay.gh * TOKEN_PX, lay.gw * TOKEN_PX, 3), np.uint8)
        full[:px.shape[0], :px.shape[1]] = px
        return W.patch_panels(lay, full, self.nm.p, kp=KP, one=ONE, patch=PATCH)

    def size_inputs(self, lay: Layout, size: tuple[int, int]) -> dict[str, np.ndarray]:
        """The size's rows: positions (the padding pixels' corrections in them, the class token's row), RoPE tables,
        key masks (the level's attention command)."""
        nm, nb, vt = self.nm, lay.nb, self.vt
        ph, pw = 2 * lay.gh, 2 * lay.gw
        pr = np.zeros((ph * pw, vt.C), np.float32)
        pr[:, :vt.H] = pos_rows(self.w["pos"], ph, pw) + pad_rows(self.w["sums"], *size)
        ok = lay.keys()
        att = keys_of(nm.attn(nb), vt, valid=(ok, ok)).attention(level(nb)).command()
        rope, rope_cls = rope_rows(ph, pw, vt)
        return ({nm.attn(nb): att} | W.band_rows(lay, pr, f"{nm.p}.pos", self.w["cls"]) |
                W.band_rows(lay, rope, f"{nm.p}.rope", rope_cls))
