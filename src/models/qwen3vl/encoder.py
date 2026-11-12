"""Qwen3-VL's vision encoder (projector ``qwen3vl_merger``: Qwen3.5 / 3.6 / 3.8's mmproj, vllm-sr's d3) on the vision
overlay (``artlite.vision``: one design for every image network, its ops built by host-built transactions; the ViT's
layers ``artlite.vision.vit``'s, channels and heads padded there).

The network (llama.cpp's ``clip_graph_qwen3vl``, transformers' ``Qwen3VLVisionModel``): an image of ph x pw patches of
16 pixels (both even: ``host.target_size``), the patch embedding (a Conv3d of two temporal kernels on a still image:
their sum), the learned 48 x 48 position embedding interpolated bilinearly (align corners) to the grid, the layers of
attention over all of the image's patches with a 2-D RoPE (vision mode: in each head of d dims, dims i and i + d / 2 a
pair, turned by y s^i for i < d / 4 and by x s^(i - d / 4) above, s = 10000^(-4 / d)), post_ln per patch, then each 2 x
2 window's patches (0, 0), (0, 1), (1, 0), (1, 1) merged (4 hidden) -> mm.0 -> GELU (erf: transformers' ``nn.GELU``) ->
mm.2 (the language model's width): one token per window, raster order.

Towers (``tower``: ``vit.Vit`` widths from the mmproj): 27 layers of 1152, 16 heads of 72, FFN 4304 (Qwen3.5 9B+, 3.6,
3.8: C 1280, FFN 4480); 24 of 1024, 16 heads of 64, FFN 4096 (Qwen3.5-2B's, d3-nano: C 1280, FFN 4160); 12 of 768, 12
heads of 64, FFN 3072 (Qwen3.5-0.8B's, d3-lite: C 960, FFN 3200). The GEMMs of a norm's panels take K = the hidden (its
pads skipped: the same sums).

Layout: ``artlite.vision.windows``'s (bands of 512 patches: 8 columns x 64 tokens, a window's 4 patches in one tile;
the attention's keys one dense joint sequence, so the transactions depend on the band count only; post_ln's panels go
to DDR in the merged layout for the merger's MLP: ``windows.merger``).

RoPE: the Q and K GEMMs turn each head's pairs in their epilogue (``ops.Gemm(out="rope")``): a head's 80 channels are
its first d / 2 dims, zeros to 40, its last d / 2, zeros (``rope_heads``: the q, k rows; the dot products do not
change), the residual stream brings each band's cos | sin tile (``inputs``). The patch embedding takes the pixels' exact 8-bit
integers (q = p - 128, the normalization folded into its weights), the patch embedding and the MLP hi + lo weights
(``ops.Gemm``); positions are the patch embedding's residual (host rows, bf16).

Precision (``artlite.vision.vit``'s): per merged token ~4% from the float64 encoder (median; 512 x 512: p90 ~12%), as
the CPU at the overlay's rounding points; a few tokens 0.3-2.8x their norm off (the last layer's massive activation:
at 512 x 512 a total of 0.07-0.10, whichever tokens the last bits pick), where llama.cpp's own embeddings are up to
0.22x off (total 0.014). The language model's KL against llama.cpp's run stays at llama.cpp's level (``tests/qwen3vl``).

Dispatches (``Encoder``: ``windows.WindowEncoder``): a program per nb (built on first use): nb 1 runs as one dispatch;
larger images as the patch embedding, pieces of ``layers_per_piece(nb)`` layers (one dispatch each: well under the
driver's 2 s), the final merger.
"""

import math
from collections.abc import Callable

import numpy as np

from artlite.operators.design import COLS
from artlite.vision import windows as W
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gemm
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import VIT, Bands, Keys, Vit, kv_bytes, layer_bands, pad, pad_heads
from artlite.vision.vit import vit_level as level
from artlite.vision.weights import Wt, computed, f32, take, values
from artlite.vision.windows import Layout, Names, out_width

PATCH = 16
PIX, ONE, KP = 768, 768, 832  # (a patch's inputs (c, ky, kx) of 3 x 16 x 16; the constant input 1, a bfp16 block of
# its own; the panels' K: whole k-tiles)
UPC, DNC = 960, 640  # (the MLP's output chunks: mm.0's (3 passes a row), mm.2's (2))
PIECE_BAND_LAYERS = 16  # (a piece's band-layers at most: its build ~0.4 s)
PIECE_MS = 500.0  # (a dispatch's estimated time at most)


def tower(hidden: int, heads: int, ffn: int, eps: float) -> Vit:
    """The widths of a Qwen3-VL tower (module docstring); NotImplementedError: one the overlay does not run."""
    return VIT if (hidden, heads, ffn) == (VIT.H, VIT.HEADS, VIT.FFN_REAL) and eps == VIT.EPS else \
        Vit.of(hidden, heads, ffn, eps)


def merger(vt: Vit) -> W.Merger:
    """The final merger of a tower: mm.0's 4 x H outputs in chunks of UPC (and the rest: whole passes)."""
    fup = -(-4 * vt.H // 320) * 320
    return W.Merger(H=vt.H, C=vt.C, eps=vt.EPS, up=(UPC,) * (fup // UPC) + ((fup % UPC,) if fup % UPC else ()), dn=DNC)


def rope_rows(vt: Vit) -> np.ndarray:
    """``rope_heads``' rows: per head of HD dims its 80 channels, dims 0 .. HD / 2 - 1 at 0 .., HD / 2 .. at 40 ..
    (pair k: channels k, 40 + k), -1 elsewhere."""
    HD, HP = vt.HD, vt.HP
    j = np.arange(HP)
    d = np.where(j < HP // 2, j, j - HP // 2 + HD // 2)
    ok = (j % (HP // 2)) < HD // 2
    return np.concatenate([np.where(ok, h * HD + d, -1) for h in range(vt.HEADS)])


def rope_heads(w: Wt | np.ndarray, vt: Vit = VIT) -> Wt | np.ndarray:
    """Rows of the heads of HD -> heads of 80 for the RoPE epilogue (``rope_rows``; zeros elsewhere)."""
    return take(w, rope_rows(vt))


def layer_weights(get: Callable[[str], Wt | np.ndarray], il: int, vt: Vit = VIT) -> dict[str, Wt | np.ndarray]:
    """Layer il's padded weights by role: q (x log2 e / sqrt HD), k (rows by head for the RoPE epilogue:
    ``rope_heads``), v (rows by head), o (columns by head), up, down and their biases (``*_b``), ln1 / ln2 gamma and
    beta (``*_g``, ``*_b``); q, k, v and up of K = the hidden; ``get(name)``: a tensor of the mmproj, a matrix (rows,
    cols: ``weights.Wt`` or a tensor) or a vector (1, cols) float32."""
    p = f"v.blk.{il}."
    H, C, A, F = vt.H, vt.C, vt.HEADS * vt.HP, vt.FFN
    qkv, qkv_b = f32(get(p + "attn_qkv.weight")), f32(get(p + "attn_qkv.bias")).reshape(-1)
    part = lambda t, i: t[i * H:(i + 1) * H]
    rope, qs = lambda t: rope_heads(t, vt), vt.qscale
    w = {"q": pad(rope(part(qkv, 0) * qs), A), "q_b": rope(part(qkv_b, 0) * qs),
         "k": pad(rope(part(qkv, 1)), A), "k_b": rope(part(qkv_b, 1)),
         "v": pad(pad_heads(part(qkv, 2), vt=vt), A), "v_b": pad_heads(part(qkv_b, 2), vt=vt)}
    w["o"] = pad(pad_heads(f32(get(p + "attn_out.weight")), 1, vt), C, A)
    w["o_b"] = pad(f32(get(p + "attn_out.bias")).reshape(-1), C)
    w["up"] = pad(f32(get(p + "ffn_up.weight")), F)
    w["up_b"] = pad(f32(get(p + "ffn_up.bias")).reshape(-1), F)
    w["down"] = pad(f32(get(p + "ffn_down.weight")), C, F)
    w["down_b"] = pad(f32(get(p + "ffn_down.bias")).reshape(-1), C)
    for n in ("ln1", "ln2"):
        w[n + "_g"] = pad(f32(get(p + n + ".weight")).reshape(-1), C)
        w[n + "_b"] = pad(f32(get(p + n + ".bias")).reshape(-1), C)
    return w


def _blocks4(v: Wt | np.ndarray, axis: int = 0, vt: Vit = VIT) -> Wt | np.ndarray:
    """A merged token's 4 x H channels (along axis: 0, a vector's or a matrix's rows; 1, a matrix's columns) -> 4 x C
    (each patch's padded)."""
    return take(v, np.array([k * vt.H + i if i < vt.H else -1 for k in range(4) for i in range(vt.C)]), axis)


def weights(get: Callable[[str], Wt | np.ndarray], layers: int, proj: int, vt: Vit = VIT) -> dict:
    """Every padded weight of the encoder: "patch" (``patch_weights``), per layer il its ``layer_weights``, "post"
    (post_ln's gamma, beta), "up" / "down" (mm.0 on the merged 4 x C channels -> FUP, mm.2 FUP -> ``out_width``) and
    their biases; "layers", "proj", "vit" (the widths). Matrices as ``get`` gives them (``weights.Wt``: views of the
    file), never materialized."""
    md = merger(vt)
    w: dict = {"patch": patch_weights(get, vt), "layers": layers, "proj": proj, "vit": vt,
               **{il: layer_weights(get, il, vt) for il in range(layers)}}
    w["post_g"], w["post_b"] = (pad(f32(get(f"v.post_ln.{n}")).reshape(-1), vt.C) for n in ("weight", "bias"))
    w["up"] = pad(_blocks4(get("mm.0.weight"), 1, vt), md.FUP, md.merged)
    w["up_b"] = pad(f32(get("mm.0.bias")).reshape(-1), md.FUP)
    w["down"] = pad(f32(get("mm.2.weight")), out_width(proj), md.FUP)
    w["down_b"] = pad(f32(get("mm.2.bias")).reshape(-1), out_width(proj))
    return w


def patch_weights(get: Callable[[str], Wt | np.ndarray], vt: Vit = VIT) -> dict[str, Wt | np.ndarray]:
    """The patch embedding as a GEMM of the integer panels of ``Layout.patches`` (KP inputs -> C): the two temporal
    kernels summed, the normalization x = (p / 255 - 0.5) / 0.5 = a q + o of q = p - 128 (a = 1 / 127.5, o = 1 / 255)
    folded in: input k < 768 the weights' column k times a, input 768 (the constant 1) their offset sum_k w_k o
    (computed on the host at every load, ``weights.computed``); the bias."""
    H, C = vt.H, vt.C
    w = (values(get("v.patch_embd.weight")).astype(np.float64) +
         values(get("v.patch_embd.weight.1")).astype(np.float64)).reshape(H, PIX)
    a = 1 / 127.5
    o = 128 * a - 1
    wt = np.zeros((C, KP), np.float32)
    wt[:H, :PIX], wt[:H, ONE] = w * a, w.sum(1) * o
    return {"w": computed("qwen3vl.patch", wt), "b": pad(f32(get("v.patch_embd.bias")).reshape(-1), C)}


def pos_rows(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """The learned position embedding (n x n rows of 1152, float32) at a ph x pw patch grid (raster order): bilinear
    with the corners aligned (llama.cpp's GGML_SCALE_FLAG_ALIGN_CORNERS, transformers' linspace grid), float32."""
    n = round(math.sqrt(table.shape[0]))
    t = table.reshape(n, n, -1)

    def axis(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.arange(m, dtype=np.float64) * ((n - 1) / (m - 1)) if m > 1 else np.zeros(1)
        i0 = np.minimum(np.floor(s).astype(np.int64), n - 1)
        return i0, np.minimum(i0 + 1, n - 1), (s - i0).astype(np.float32)
    y0, y1, fy = axis(ph)
    x0, x1, fx = axis(pw)
    top = t[y0][:, x0] * (1 - fx)[None, :, None] + t[y0][:, x1] * fx[None, :, None]
    bot = t[y1][:, x0] * (1 - fx)[None, :, None] + t[y1][:, x1] * fx[None, :, None]
    return (top * (1 - fy)[:, None, None] + bot * fy[:, None, None]).reshape(ph * pw, -1).astype(np.float32)


def rope_table(ph: int, pw: int, vt: Vit = VIT) -> np.ndarray:
    """Per patch (raster order) its 80 channels of the RoPE epilogue: cos of pairs 0 .. HD / 2 - 1, 1 to 40, sin of
    the pairs, 0 to 80; pair i's angle y s^i (i < HD / 4), x s^(i - HD / 4) (above), s = 10000^(-4 / HD)."""
    return W.rope_table(ph, pw, vt.HD, 10000.0, vt.HP)


def layers_per_piece(nb: int, layers: int = 27) -> int:
    """Layers of a dispatch at nb bands: all at nb 1, else at most PIECE_BAND_LAYERS band-layers (its build time) and
    about PIECE_MS (a layer of the 1152 tower ~1.6 nb + 0.15 nb^2 ms: its GEMMs and its attention over every key)."""
    if nb == 1:
        return layers
    return max(1, min(layers, PIECE_BAND_LAYERS // nb, int(PIECE_MS / (1.6 * nb + 0.15 * nb * nb))))


def patch_embed(bd: Builder, w: dict[str, np.ndarray], nm: Names, nb: int, vt: Vit = VIT) -> None:
    """The patch embedding of nb bands: per band a GEMM of the host's patch panels (DDR image <p>.pix.b) with the
    position rows (<p>.pos.b) as its residual -> the residual's bands before layer 0 and their records."""
    lv = level(nb)
    C = vt.C
    out = Bands(lv, nm.x(0), C)
    pn, cn = lv.pn(KP), lv.ct(C)
    for b in range(nb):
        pi, ri = bd.ddr(f"{nm.p}.pix.{b}", COLS * pn), bd.ddr(f"{nm.p}.pos.{b}", COLS * cn)
        pt = alloc(bd, lv, KP, "pn")
        x = alloc(bd, lv, C)
        bd.op({c: ColOps(fills=[MemTask([{"addr": pt.addr, "nwords": pn // 4}], ddr=(pi + c * pn, pn, None)),
                               MemTask([{"addr": x.addr, "nwords": cn // 4}], ddr=(ri + c * cn, cn, None))])
               for c in range(COLS)}, {})
        rec = gemm(bd, pt, w["w"], w["b"], name=f"{nm.p}.patch", out=x, res=x, ln=True, hilo=True)
        free(bd, pt)
        out.store(bd, b, x, rec)


def keys_of(name: str, vt: Vit = VIT, **kw: object) -> Keys:
    """A level's attention keys (one dense joint sequence) of the tower's heads."""
    return Keys(name, joint=True, dense=True, heads=vt.HEADS, **kw)


def layers(bd: Builder, w: dict, nm: Names, nb: int, il: range, kv: int, vt: Vit = VIT) -> None:
    """Layers ``il`` on nb bands (the residual from image x(il[0]) to x(il[-1] + 1))."""
    lv = level(nb)
    keys = keys_of(nm.attn(nb), vt)
    rt = level(nb).ct(vt.HP)

    def rope(b_: Builder, band: int) -> Tensor:
        t = alloc(b_, lv, vt.HP)
        b_.ddr(f"{nm.p}.rope.{band}", COLS * rt)
        b_.op(b_.load(f"{nm.p}.rope.{band}", t.addr, rt), {})
        return t
    for i in il:
        layer_bands(bd, w[i], nm.layer(i), Bands(lv, nm.x(i), vt.C), Bands(lv, nm.x(i + 1), vt.C), kv_ddr=kv, keys=keys,
                    park=lambda b: f"{nm.p}.park", rope=rope, vt=vt)


def final(bd: Builder, w: dict, nm: Names, nb: int, nl: int, vt: Vit = VIT) -> None:
    """post_ln of the residual after nl layers and the merger's MLP (``windows.merger``)."""
    W.merger(bd, w, nm, nb, nl, merger(vt))


class Encoder(W.WindowEncoder):
    """The encoder (module docstring) as dispatches on a VisionDevice (``windows.WindowEncoder``)."""

    def __init__(self, npu: VisionDevice, w: dict, pos: np.ndarray, *, max_bands: int, name: str = "q",
                 state: dict | None = None):
        """w: ``weights`` (its "vit": the tower's widths); pos: the learned position embedding (48 x 48 rows of the
        hidden, float32); max_bands: the largest level (the K / V^T slots' DDR, allocated first); state: a ``state()``
        on a device whose images were restored (``VisionDevice.restore``: its programs taken, new ones built over the
        images as they are)."""
        self.pos, self.vt = pos, w["vit"]
        super().__init__(npu, w, max_bands=max_bands, name=name, state=state,
                         kv_bytes=kv_bytes(level(max_bands), keys_of("", self.vt)))

    def patch_embed(self, bd: Builder, nb: int) -> None:
        patch_embed(bd, self.w["patch"], self.nm, nb, self.vt)

    def layers(self, bd: Builder, nb: int, il: range) -> None:
        layers(bd, self.w, self.nm, nb, il, self.kv, self.vt)

    def final(self, bd: Builder, nb: int) -> None:
        final(bd, self.w, self.nm, nb, self.nl, self.vt)

    def layers_per_piece(self, nb: int) -> int:
        return layers_per_piece(nb, self.nl)

    def pixel_panels(self, lay: Layout, px: np.ndarray) -> dict[str, np.ndarray]:
        return W.patch_panels(lay, px, self.nm.p, kp=KP, one=ONE)

    def size_inputs(self, lay: Layout) -> dict[str, np.ndarray]:
        """The size's rows: positions, RoPE tables, key masks (the level's attention command)."""
        nm, nb, vt = self.nm, lay.nb, self.vt
        ph, pw = 2 * lay.gh, 2 * lay.gw
        pr = np.zeros((ph * pw, vt.C), np.float32)
        pr[:, :vt.H] = pos_rows(self.pos, ph, pw)
        ok = lay.keys()
        att = keys_of(nm.attn(nb), vt, valid=(ok, ok)).attention(level(nb)).command()
        return ({nm.attn(nb): att} | W.band_rows(lay, pr, f"{nm.p}.pos") |
                W.band_rows(lay, rope_table(ph, pw, vt), f"{nm.p}.rope"))
