"""Qwen3-VL's vision encoder (projector ``qwen3vl_merger``: Qwen3.5 / 3.6 / 3.8's mmproj) on the vision overlay
(``artlite.vision``: one design for every image network, its ops built by host-built transactions; the ViT's layers
``artlite.vision.vit``'s, channels and heads padded there).

The network (llama.cpp's ``clip_graph_qwen3vl``, transformers' ``Qwen3VLVisionModel``): an image of ph x pw patches of
16 pixels (both even: ``host.target_size``), the patch embedding (a Conv3d of two temporal kernels on a still image:
their sum), the learned 48 x 48 position embedding interpolated bilinearly (align corners) to the grid, 27 layers of
attention over all of the image's patches with a 2-D RoPE (vision mode: in each head dims i and i + 36 a pair, turned by
y s^i for i < 18 and by x s^(i - 18) above, s = 10000^(-1/18)), post_ln per patch, then each 2 x 2 window's patches
(0, 0), (0, 1), (1, 0), (1, 1) merged (4608) -> mm.0 -> GELU (erf: transformers' ``nn.GELU``) -> mm.2 (the language
model's width): one token per window, raster order.

Layout (``Layout``): a level of ``nb`` bands of 512 patches (8 columns x 64 tokens), nb = ceil(windows / 128). Window w
(raster order) is at (p, band, column, lane) = w's digits in (2, nb, 8, 8) (p the slowest); its patch s in token
panel 2 s + p of that band and column, at that lane. The attention's keys are the joint sequence [p][band][column][s]
[lane] (``ops.Attention``: the overlay's two images are p), whose valid keys are whole 64-key groups up to the one of the
last window (a lane pattern), so that ``ops.Attention(dense=True)`` takes any image size as data: the transactions
depend on nb only. Slots past the image take the patches of window w mod windows (typical values in the bfp16 blocks
over tokens, V^T's, that mix real and padding tokens; their keys masked). The merge: a band's 4 token-panel pairs are
its windows' 4 patches, so post_ln's panels go to DDR in the merged layout (a walk of their 72-byte panels): 4 bands'
merged tokens are one m-64 tile (token panel 2 (band % 4) + p) for the merger's MLP, its weights streamed once per 4
bands.

RoPE: the Q and K GEMMs turn each head's pairs in their epilogue (``ops.Gemm(out="rope")``): a head's 80 channels are
its first 36 dims, 4 zero, its last 36, 4 zero (``rope_heads``: the q, k rows; the dot products do not change), the
residual stream brings each band's cos | sin tile (``inputs``). The patch embedding takes the pixels' exact 8-bit
integers (q = p - 128, the normalization folded into its weights), the patch embedding and the MLP hi + lo weights
(``ops.Gemm``); positions are the patch embedding's residual (host rows, bf16).

Dispatches (``Encoder``): a program per nb (built on first use, ``programs``): nb 1 runs as one dispatch; larger
images as the patch embedding, pieces of ``layers_per_piece(nb)`` layers (one dispatch each: well under the driver's 2
s), the final merger. A piece is built once for layers 0 .. k - 1 and run at layer i0 by rewriting its shim addresses
for layer i0's images (``VisionDevice.refs`` / ``resolve``: the residual's ping-pong images swapped at odd i0).
"""

import math
import re
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from artlite.operators import kmath
from artlite.operators.design import COLS
from artlite.overlay import Transaction
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gelu_table, gemm
from artlite.vision.ops import _dims
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import (
    FFN,
    HD,
    HEADS,
    HP,
    QSCALE,
    Bands,
    C,
    H,
    Keys,
    kv_bytes,
    layer_bands,
    ln,
    pad,
    pad_heads,
    xfer,
)
from artlite.vision.vit import vit_level as level
from artlite.vision.weights import Wt, computed, take, values

PATCH = 16
PIX, ONE, KP = 768, 768, 832  # (a patch's inputs (c, ky, kx) of 3 x 16 x 16; the constant input 1, a bfp16 block of
# its own; the panels' K: whole k-tiles)
MERGED, FUP = 4 * C, 4800  # (a merged token's channels: 4 x 1280; mm.0's 4608 outputs padded to 15 passes a row)
UPC, DNC = 960, 640  # (the MLP's output chunks: mm.0's (3 passes a row), mm.2's (2))
WIN = 128  # (windows of a band: 8 columns x 2 images x 8 lanes)
PIECE_BAND_LAYERS = 16  # (a piece's band-layers at most: its build ~0.4 s)
PIECE_MS = 500.0  # (a dispatch's estimated time at most)


ROPE_ROWS = np.array([h * HD + (j if j < HD // 2 else j - HP // 2 + HD // 2) if j < HD // 2 or HP // 2 <= j < HP // 2 + HD // 2
                      else -1 for h in range(HEADS) for j in range(HP)])  # (``rope_heads``)
BLOCKS4 = np.array([k * H + i if i < H else -1 for k in range(4) for i in range(C)])  # (``_blocks4``)


def rope_heads(w: Wt | torch.Tensor) -> Wt | torch.Tensor:
    """Rows of 16 heads of 72 -> 16 heads of 80 for the RoPE epilogue: dims 0 .. 35 at 0 .. 35, 36 .. 71 at 40 .. 75
    (pair k: channels k, 40 + k), zeros elsewhere."""
    return take(w, ROPE_ROWS)


def layer_weights(get: Callable[[str], Wt | torch.Tensor], il: int) -> dict[str, Wt | torch.Tensor]:
    """Layer il's padded weights by role: q (x log2 e / sqrt 72), k (rows by head for the RoPE epilogue:
    ``rope_heads``), v (rows by head), o (columns by head), up, down and their biases (``*_b``), ln1 / ln2 gamma and
    beta (``*_g``, ``*_b``); ``get(name)``: a tensor of the mmproj, a matrix (rows, cols: ``weights.Wt`` or a tensor)
    or a vector (1, cols) float32."""
    p = f"v.blk.{il}."
    qkv, qkv_b = get(p + "attn_qkv.weight").float(), get(p + "attn_qkv.bias").float().reshape(-1)
    part = lambda t, i: t[i * H:(i + 1) * H]
    w = {"q": pad(rope_heads(part(qkv, 0) * QSCALE), C, C), "q_b": rope_heads(part(qkv_b, 0) * QSCALE),
         "k": pad(rope_heads(part(qkv, 1)), C, C), "k_b": rope_heads(part(qkv_b, 1)),
         "v": pad(pad_heads(part(qkv, 2)), C, C), "v_b": pad_heads(part(qkv_b, 2))}
    w["o"] = pad(pad_heads(get(p + "attn_out.weight").float(), 1), C, C)
    w["o_b"] = pad(get(p + "attn_out.bias").float().reshape(-1), C)
    w["up"] = pad(get(p + "ffn_up.weight").float(), FFN, C)
    w["up_b"] = pad(get(p + "ffn_up.bias").float().reshape(-1), FFN)
    w["down"] = pad(get(p + "ffn_down.weight").float(), C, FFN)
    w["down_b"] = pad(get(p + "ffn_down.bias").float().reshape(-1), C)
    for n in ("ln1", "ln2"):
        w[n + "_g"] = pad(get(p + n + ".weight").float().reshape(-1), C)
        w[n + "_b"] = pad(get(p + n + ".bias").float().reshape(-1), C)
    return w


def _blocks4(v: Wt | torch.Tensor, axis: int = 0) -> Wt | torch.Tensor:
    """A merged token's 4 x 1152 channels (along axis: 0, a vector's or a matrix's rows; 1, a matrix's columns) -> 4 x
    1280 (each patch's padded)."""
    return take(v, BLOCKS4, axis)


def out_width(proj: int) -> int:
    """The embeddings' channels on the NPU: the language model's width in whole passes (320 a row of 4)."""
    return -(-proj // 320) * 320


def weights(get: Callable[[str], Wt | torch.Tensor], layers: int, proj: int) -> dict:
    """Every padded weight of the encoder: "patch" (``patch_weights``), per layer il its ``layer_weights``, "post"
    (post_ln's gamma, beta), "up" / "down" (mm.0 on the merged 4 x 1280 channels -> FUP, mm.2 FUP -> ``out_width``) and
    their biases; "layers", "proj". Matrices as ``get`` gives them (``weights.Wt``: views of the file), never
    materialized."""
    w: dict = {"patch": patch_weights(get), "layers": layers, "proj": proj, **{il: layer_weights(get, il) for il in range(layers)}}
    w["post_g"], w["post_b"] = (pad(get(f"v.post_ln.{n}").float().reshape(-1), C) for n in ("weight", "bias"))
    w["up"] = pad(_blocks4(get("mm.0.weight"), 1), FUP, MERGED)
    w["up_b"] = pad(get("mm.0.bias").float().reshape(-1), FUP)
    w["down"] = pad(get("mm.2.weight").float(), out_width(proj), FUP)
    w["down_b"] = pad(get("mm.2.bias").float().reshape(-1), out_width(proj))
    return w


def patch_weights(get: Callable[[str], Wt | torch.Tensor]) -> dict[str, Wt | torch.Tensor]:
    """The patch embedding as a GEMM of the integer panels of ``Layout.patches`` (KP inputs -> C): the two temporal
    kernels summed, the normalization x = (p / 255 - 0.5) / 0.5 = a q + o of q = p - 128 (a = 1 / 127.5, o = 1 / 255)
    folded in: input k < 768 the weights' column k times a, input 768 (the constant 1) their offset sum_k w_k o
    (computed on the host at every load, ``weights.computed``); the bias."""
    w = (values(get("v.patch_embd.weight")).astype(np.float64) +
         values(get("v.patch_embd.weight.1")).astype(np.float64)).reshape(H, PIX)
    a = 1 / 127.5
    o = 128 * a - 1
    wt = np.zeros((C, KP), np.float32)
    wt[:H, :PIX], wt[:H, ONE] = w * a, w.sum(1) * o
    return {"w": computed("qwen3vl.patch", wt), "b": pad(get("v.patch_embd.bias").float().reshape(-1), C)}


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


def rope_table(ph: int, pw: int) -> np.ndarray:
    """Per patch (raster order) its 80 channels of the RoPE epilogue: cos of pairs 0 .. 35, 1 (4), sin of pairs 0 ..
    35, 0 (4); pair i's angle y s^i (i < 18), x s^(i - 18) (above), s = 10000^(-1/18)."""
    f = 10000.0 ** (-np.arange(HD // 4, dtype=np.float64) / (HD // 4))
    y, x = np.divmod(np.arange(ph * pw), pw)
    ang = np.concatenate([y[:, None] * f, x[:, None] * f], 1)
    out = np.zeros((ph * pw, HP), np.float32)
    out[:, :HP // 2] = 1
    out[:, :HD // 2], out[:, HP // 2:HP // 2 + HD // 2] = np.cos(ang), np.sin(ang)
    return out


@dataclass(frozen=True)
class Layout:
    """An image of gh x gw merged tokens (windows of 2 x 2 patches) on a level of ``nb`` bands (module docstring)."""

    gh: int
    gw: int

    @property
    def windows(self) -> int:
        return self.gh * self.gw

    @property
    def nb(self) -> int:
        return -(-self.windows // WIN)

    def slot_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Per slot ([band][column][token 64], token = 8 token panel + lane) its window (the real one: w mod windows)
        and patch s; plus the real windows' mask."""
        nb = self.nb
        b, c, t = np.meshgrid(np.arange(nb), np.arange(COLS), np.arange(64), indexing="ij")
        tp, lane = np.divmod(t, 8)
        s, p = np.divmod(tp, 2)
        w = ((p * nb + b) * COLS + c) * 8 + lane
        return w % self.windows, s, w < self.windows

    def slot_patches(self) -> np.ndarray:
        """Per slot ([band][column][64]) its patch's index in the image's raster order (2 gh x 2 gw patches)."""
        w, s, _ = self.slot_windows()
        r, c = np.divmod(w, self.gw)
        return (2 * r + s // 2) * (2 * self.gw) + 2 * c + s % 2

    def keys(self) -> np.ndarray:
        """The valid keys of the joint sequence [p][band][column][s][lane]."""
        nb = self.nb
        p, b, c, _, lane = np.meshgrid(*(np.arange(n) for n in (2, nb, COLS, 4, 8)), indexing="ij")
        return (((p * nb + b) * COLS + c) * 8 + lane < self.windows).reshape(-1)

    def merged(self, out: np.ndarray) -> np.ndarray:
        """The MLP's tiles (quads of 4 bands, [quad][column][token 64]: token panel 2 (band % 4) + p, lane; channels)
        -> the windows' embeddings in raster order."""
        nb = self.nb
        q, c, t = np.meshgrid(np.arange(-(-nb // 4)), np.arange(COLS), np.arange(64), indexing="ij")
        j, lane = np.divmod(t, 8)
        b, p = 4 * q + j // 2, j % 2
        w = ((p * nb + b) * COLS + c) * 8 + lane
        ok = (b < nb) & (w < self.windows)
        emb = np.empty((self.windows, out.shape[-1]), np.float32)
        emb[w[ok]] = out.reshape(-1, out.shape[-1])[ok.reshape(-1)]
        return emb


def layers_per_piece(nb: int, layers: int = 27) -> int:
    """Layers of a dispatch at nb bands: all at nb 1, else at most PIECE_BAND_LAYERS band-layers (its build time) and
    about PIECE_MS (a layer ~1.6 nb + 0.15 nb^2 ms: its GEMMs and its attention over every key)."""
    if nb == 1:
        return layers
    return max(1, min(layers, PIECE_BAND_LAYERS // nb, int(PIECE_MS / (1.6 * nb + 0.15 * nb * nb))))


class Names:
    """The DDR images of the encoder (``name`` their prefix)."""

    def __init__(self, name: str):
        self.p = name

    def x(self, i: int) -> str:  # (the residual before layer i: two images, alternating)
        return f"{self.p}.x{'ab'[i % 2]}"

    def layer(self, il: int) -> str:
        return f"{self.p}.l{il}"

    def attn(self, nb: int) -> str:
        return f"{self.p}.attn.{nb}"

    def rename(self, off: int) -> Callable[[str], str]:
        """The images of a piece built from layer 0 when run from layer ``off``: layer il's -> il + off's, the
        residual's two swapped at odd off."""
        lay = re.compile(rf"^{re.escape(self.p)}\.l(\d+)\.")
        xs = {f"{self.p}.xa.": f"{self.p}.xb.", f"{self.p}.xb.": f"{self.p}.xa."}

        def f(n: str) -> str:
            n = lay.sub(lambda m: f"{self.p}.l{int(m[1]) + off}.", n)
            k = n[:len(self.p) + 4]
            return xs[k] + n[len(k):] if off % 2 and k in xs else n
        return f


def patch_embed(bd: Builder, w: dict[str, torch.Tensor], nm: Names, nb: int) -> None:
    """The patch embedding of nb bands: per band a GEMM of the host's patch panels (DDR image <p>.pix.b) with the
    position rows (<p>.pos.b) as its residual -> the residual's bands before layer 0 and their records."""
    lv = level(nb)
    out = Bands(lv, nm.x(0))
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


def layers(bd: Builder, w: dict, nm: Names, nb: int, il: range, kv: int) -> None:
    """Layers ``il`` on nb bands (the residual from image x(il[0]) to x(il[-1] + 1))."""
    lv = level(nb)
    keys = Keys(nm.attn(nb), joint=True, dense=True)
    rt = level(nb).ct(HP)

    def rope(b_: Builder, band: int) -> Tensor:
        t = alloc(b_, lv, HP)
        b_.ddr(f"{nm.p}.rope.{band}", COLS * rt)
        b_.op(b_.load(f"{nm.p}.rope.{band}", t.addr, rt), {})
        return t
    for i in il:
        layer_bands(bd, w[i], nm.layer(i), Bands(lv, nm.x(i)), Bands(lv, nm.x(i + 1)), kv_ddr=kv, keys=keys,
                    park=lambda b: f"{nm.p}.park", rope=rope)


def final(bd: Builder, w: dict, nm: Names, nb: int, nl: int) -> None:
    """post_ln of the residual after nl layers and the merger's MLP: per band LayerNorm -> panels drained to DDR in
    the merged layout of its quad (<p>.mg.q: [column][kb 640][token panel 8][72 B]; band b's patch s, image p at kb
    160 s + kb, token panel 2 (b % 4) + p); per quad its tile of 64 merged tokens: mm.0 + GELU in chunks to DDR
    (<p>.hid.q), mm.2 in chunks -> the embeddings' C tiles (<p>.out.q: [column][out_width / 80][ct_bytes(64)])."""
    lv, l1 = level(nb), level(1)
    x = Bands(lv, nm.x(nl))
    pn, mgb, hid = lv.pn(C), l1.pn(MERGED), l1.pn(FUP)
    outw = w["down"].shape[0]
    walk = _dims((18, 1), (2, 18), (C // 8, 144), (4, 36))  # (stream order: s, kb, p, a panel's 18 words)
    for b in range(nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, lv, C, "pn")
        ln(bd, xb, n, w["post_g"], w["post_b"], name=f"{nm.p}.post", rec=rec)
        free(bd, xb)
        bd.op(xfer(bd, f"{nm.p}.mg.{b // 4}", mgb, fill=False, mem=n.addr, nbytes=pn, off=2 * (b % 4) * L.PANEL,
                   mdims=walk, sdims=(36, MERGED // 8, 144, 144 * MERGED // 8)), {})
        free(bd, n)
    for qd in range(-(-nb // 4)):
        mg = alloc(bd, l1, MERGED, "pn")
        spare = bd.mem.alloc(9216)  # (the GEMM's A walk reads past its panels' end)
        bd.op(xfer(bd, f"{nm.p}.mg.{qd}", mgb, fill=True, mem=mg.addr, nbytes=mgb), {})
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        g = alloc(bd, l1, UPC, "pn")
        for k in range(FUP // UPC):
            gemm(bd, mg, w["up"][UPC * k:UPC * (k + 1)], w["up_b"][UPC * k:UPC * (k + 1)], name=f"{nm.p}.up{k}", out=g,
                 kind="gelu", hilo=True)
            bd.op(xfer(bd, f"{nm.p}.hid.{qd}", hid, fill=False, mem=g.addr, nbytes=g.nbytes, off=k * g.nbytes), {})
        free(bd, g)
        bd.mem.release(spare, 9216)
        free(bd, mg)
        h = alloc(bd, l1, FUP, "pn")
        spare = bd.mem.alloc(9216)
        bd.op(xfer(bd, f"{nm.p}.hid.{qd}", hid, fill=True, mem=h.addr, nbytes=hid), {})
        y = alloc(bd, l1, DNC)
        for k in range(outw // DNC + (outw % DNC > 0)):
            n = min(DNC, outw - DNC * k)
            yk = Tensor(l1, n, y.addr)
            gemm(bd, h, w["down"][DNC * k:DNC * k + n], w["down_b"][DNC * k:DNC * k + n], name=f"{nm.p}.down{k}",
                 out=yk, hilo=True)
            bd.op(xfer(bd, f"{nm.p}.out.{qd}", l1.ct(outw), fill=False, mem=y.addr, nbytes=yk.nbytes,
                       off=k * l1.ct(DNC)), {})
        free(bd, y)
        bd.mem.release(spare, 9216)
        free(bd, h)


@dataclass
class Piece:
    """A dispatch of a program: its transaction words, their shim addresses by image (``VisionDevice.refs``) and the
    layers it runs from layer 0 (a piece of layers: run at another layer by ``Names.rename``; else None)."""

    words: np.ndarray
    refs: tuple
    layers: int | None = None


class Encoder:
    """The encoder (module docstring) as dispatches on a VisionDevice: per nb a program (``programs``: its pieces built
    on first use, over the same weight images: ``Builder.fork``). Per image the host writes its inputs (``inputs``:
    patch panels, position rows, RoPE tables; the level's key masks into its attention command), runs the pieces and
    reads the embeddings."""

    def __init__(self, npu: VisionDevice, w: dict, pos: np.ndarray, *, max_bands: int, name: str = "q",
                 state: dict | None = None):
        """w: ``weights``; pos: the learned position embedding (48 x 48 rows of 1152, float32); max_bands: the largest
        level (the K / V^T slots' DDR, allocated first); state: a ``state()`` on a device whose images were restored
        (``VisionDevice.restore``: its programs taken, new ones built over the images as they are)."""
        self.npu, self.w, self.pos, self.nm = npu, w, pos, Names(name)
        self.nl, self.proj, self.outw = w["layers"], w["proj"], w["down"].shape[0]
        self.bd = Builder(npu, reuse=state is not None)
        self.kv = self.bd.ddr(f"{name}.kv", kv_bytes(level(max_bands), Keys("", joint=True, dense=True)))
        self.max_bands = max_bands
        for nb in range(1, max_bands + 1):  # (every level's attention command, before any program)
            self.bd.ddr(self.nm.attn(nb), D.Q)
        self.programs: dict[int, list[Piece]] = {} if state is None else dict(state["programs"])
        self.built: Callable[[int, list[Piece], bool], None] | None = None  # (a new program's hook: a cache's; the
        # bool: it took only images that existed before it)
        self.tr: dict[tuple[int, int], Transaction] = {}
        self._inputs: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()
        self.program(1)  # (every layer's images: nb 1 runs them all in one dispatch)
        self.program(max_bands)  # (every band's DDR images: the layout that later programs take)

    def state(self) -> dict:
        """What a restored device's Encoder takes besides its images (``__init__``'s state): the programs built."""
        return {"programs": dict(self.programs)}

    def _piece(self, build: Callable[[Builder], None], nlayers: int | None = None) -> Piece:
        b = self.bd.fork()
        build(b)
        words = b.finish()
        return Piece(words, self.npu.refs(b.s.t.addrs), nlayers)

    def program(self, nb: int) -> list[Piece]:
        """The pieces of nb bands (built on first use): nb 1 one; else the patch embedding, a piece of k layers (and
        one of the rest), the final merger (``run`` runs the layer pieces at every layer)."""
        if nb in self.programs:
            return self.programs[nb]
        assert nb <= self.max_bands, (nb, self.max_bands)
        w, nm, nl = self.w, self.nm, self.nl
        had = set(self.npu._images)
        if nb == 1:
            def full(b: Builder) -> None:
                patch_embed(b, w["patch"], nm, 1)
                layers(b, w, nm, 1, range(nl), self.kv)
                final(b, w, nm, 1, nl)
            out = [self._piece(full)]
        else:
            k = layers_per_piece(nb, nl)
            out = [self._piece(lambda b: patch_embed(b, w["patch"], nm, nb)),
                   self._piece(lambda b: layers(b, w, nm, nb, range(k), self.kv), k)]
            if nl % k:
                out.append(self._piece(lambda b: layers(b, w, nm, nb, range(nl % k), self.kv), nl % k))
            out.append(self._piece(lambda b: final(b, w, nm, nb, nl)))
        self.programs[nb] = out
        if self.built:
            self.built(nb, out, set(self.npu._images) <= had)
        return out

    def _transaction(self, nb: int, i: int, at: int) -> Transaction:
        """Piece i of nb's program for a run from layer ``at``: its shim addresses resolved on this device's images
        (another process's program: its images restored at other addresses), a layer piece's per run for layer at's
        images, rewritten in place."""
        pc = self.program(nb)[i]
        key = (nb, i)
        if key not in self.tr:
            words = pc.words.copy()
            self.npu.resolve(words, pc.refs, lambda n: n)
            self.tr[key] = Transaction(self.npu.dev, words)
        tr = self.tr[key]
        if pc.layers is not None:
            self.npu.resolve(tr.words, pc.refs, self.nm.rename(at))
            tr.flush()
        return tr

    def schedule(self, nb: int) -> list[tuple[int, int]]:
        """The dispatches of an image of nb bands: (piece, its first layer)."""
        prog = self.program(nb)
        if nb == 1:
            return [(0, 0)]
        k = prog[1].layers
        out = [(0, 0)] + [(1, i0) for i0 in range(0, self.nl - self.nl % k, k)]
        if self.nl % k:
            out.append((2, self.nl - self.nl % k))
        return out + [(len(prog) - 1, 0)]

    def inputs(self, lay: Layout, px: np.ndarray) -> dict[str, np.ndarray]:
        """An image's DDR images: per band its patches' integer panels (<p>.pix.b), position rows (<p>.pos.b) and RoPE
        tables (<p>.rope.b); the level's attention command with the image's key masks. px: uint8 (H, W, 3)."""
        nm, nb, lv = self.nm, lay.nb, level(lay.nb)
        ph, pw = 2 * lay.gh, 2 * lay.gw
        q = px.reshape(ph, PATCH, pw, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(ph * pw, PIX).astype(np.int16) - 128
        sp = lay.slot_patches().reshape(nb, COLS, 64)
        out = {}
        pix = np.zeros((COLS, 64, KP), np.int8)
        pix[..., ONE] = 1
        for b in range(nb):
            pix[..., :PIX] = q[sp[b]]
            out[f"{nm.p}.pix.{b}"] = np.concatenate([L.int_panels(torch.from_numpy(pix[c])) for c in range(COLS)])
        key = (lay.gh, lay.gw)
        if key not in self._inputs:  # (the size's rows: positions, RoPE tables, key masks)
            pr = np.zeros((ph * pw, C), np.float32)
            pr[:, :H] = pos_rows(self.pos, ph, pw)
            rt = rope_table(ph, pw)
            ok = lay.keys()
            att = Keys(nm.attn(nb), valid=(ok, ok), joint=True, dense=True).attention(lv).command()
            sz = {f"{nm.attn(nb)}": att}
            for b in range(nb):
                sz[f"{nm.p}.pos.{b}"] = np.concatenate([L.ctiles(torch.from_numpy(pr[sp[b, c]]), 64) for c in range(COLS)])
                sz[f"{nm.p}.rope.{b}"] = np.concatenate([L.ctiles(torch.from_numpy(rt[sp[b, c]]), 64) for c in range(COLS)])
            self._inputs[key] = sz
            if len(self._inputs) > 4:
                self._inputs.popitem(last=False)
        self._inputs.move_to_end(key)
        return out | self._inputs[key]

    def embeddings(self, lay: Layout) -> np.ndarray:
        """The MLP's output C tiles -> the image's embeddings (windows, proj) in raster order."""
        l1, nq = level(1), -(-lay.nb // 4)
        tiles = np.empty((nq, COLS, 64, self.outw), np.float32)
        for qd in range(nq):
            raw = self.npu.read(f"{self.nm.p}.out.{qd}", COLS * l1.ct(self.outw)).reshape(COLS, -1)
            for c in range(COLS):
                tiles[qd, c] = L.from_ctiles(raw[c], 64, self.outw, 64).numpy()
        return lay.merged(tiles)[:, :self.proj]

    def __call__(self, px: np.ndarray) -> np.ndarray:
        """An image's pixels (uint8 (H, W, 3), multiples of 32: ``host.image_pixels``) -> its embeddings (H / 32 W / 32,
        proj) in raster order."""
        lay = Layout(px.shape[0] // 32, px.shape[1] // 32)
        assert lay.gh * 32 == px.shape[0] and lay.gw * 32 == px.shape[1], px.shape
        with ThreadPoolExecutor(1) as pool:  # (the inputs while the program builds, on first use)
            host = pool.submit(self.inputs, lay, px)
            plan = self.schedule(lay.nb)
            for n, data in host.result().items():
                self.npu.write(n, data)
        for i, at in plan:
            self.npu.run(self._transaction(lay.nb, i, at))
        return self.embeddings(lay)
