"""LFM3-VL's vision encoder (projector ``lfm3vl``: LFM3-VL-2B's and d1-2B's mmproj) on the vision overlay
(``artlite.vision``: one design for every image network, its ops built by host-built transactions; the layout, the
programs and the merger ``artlite.vision.windows``').

The network (llama.cpp's ``clip_graph_lfm3vl``, transformers' ``Lfm3VlVisionModel``): an image of ph x pw patches of
16 pixels (both even: ``host.target_size``), the patchifier (two temporal kernels on a still image: their sum, no bias),
RMSNorm, 14 layers [RMSNorm -> operator | RMSNorm -> SwiGLU 1024 -> 2560 -> 1024], each operator a gated depthwise 3 x 3
conv over the patch grid (in_proj -> [B; C; x], y = C conv(B x) with zero padding, out_proj) or bidirectional attention
(16 query heads, 8 KV heads of 64, q / k RMSNorm per head, the axial 2-D RoPE of theta 1e6: dims i, i + 32 a pair,
turned by y for i < 16, by x above), no final norm; the projector: LayerNorm per patch, the 2 x 2 merge, mm.0 -> GELU
(erf) -> mm.2. No biases but the projector's.

Widths: the residual's 1024 channels in C = 1280 (passes of 80 on 4 core rows; the pads zero: zero weight rows, the
norms over the 1024), a head's 64 dims in 80 (the RoPE epilogue's pairs (k, 40 + k): dims 0..31 at 0..31, 32..63 at
40..71; V's 64 then 16 zeros), the feed-forward's 2560 exact; GEMMs of a norm's panels take K 1024 (the pads skipped).
The query heads in the order h = r 8 + j of HF's head 2 j + r (``q_order``: Q's rows and O's columns), so that
query head h reads KV head h mod 8 (``ops.Attention`` kv_heads: each KV head's slots written for its two query heads).

A layer on nb bands (``layer``), two passes over the bands (its operator needs every band's keys / neighbours first):
- conv: per band LN1, in_proj as C (bf16 C tiles, spilled per band) and the gated product g = B x (``ops.Gemm(
  out="prod")`` of the [B | x] rows), drained to DDR as [window][class][1280] (``windows``' conv input); then per band
  4 depthwise conv ops (``ops.DwConv``: step k a block of 80 channels per core row; their staging double-buffered:
  step k + 1's fills with step k's op; the strips one window row up / down by per-size shim addresses), y = C conv(g)
  as panels, out_proj + x, the feed-forward.
- attention: per band LN1, K (the RoPE epilogue with the k norm) and V into the K / V^T slots; then per band LN1, Q
  (the q norm, x log2 e / 8), the attention over every key, O + x, the feed-forward.
- the feed-forward: LN2 (x parked in DDR meanwhile), gate / up as SwiGLU (``Gemm(act="silu")``: panels), x back, down + x.
The patch embedding: the pixels' exact 8-bit integers (q = p - 128, the normalization folded into its weights; hi +
lo weights), RMSNorm into panels, an identity GEMM back to C tiles (x0, with its records). A video's pair of frames
(``host.VideoPlan``): the two temporal kernels on their own frames, the pair's 2 x 768 inputs and the constant (KP 1600:
``patch_weights(pair=True)``); its other ops the image's.

Precision: each layer alone is within 1.2-1.9% of its own contribution (the bfp16 operands of its GEMMs; every token
within ~3.5%); the embeddings per merged token ~3.5% from the float64 encoder (median; p90 ~9%), a few tokens up to ~0.8
of their norm: patches whose final residual cancels to a fraction of their neighbours' (the projector's LayerNorm divides
by it). The language model's KL against llama.cpp's run is 0.0007 (with llama.cpp's own embeddings 0.0005:
``tests/lfm3vl``).
"""

import math
from collections.abc import Callable

import numpy as np

from artlite.operators import kmath
from artlite.operators.design import COLS, ROWS
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision import windows as W
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor, alloc, free, gemm
from artlite.vision.ops import DwConv, Gemm
from artlite.vision.seq import ColOps, MemTask
from artlite.vision.vit import Bands, Keys, VitLevel, attend, kv_bytes, ln, pad, vit_level
from artlite.vision.weights import Wt, cat, computed, eye, f32, take, values

H, C, HD, HP, HEADS, KVH, FFN = 1024, 1280, 64, 80, 16, 8, 2560
EPS = 1e-5
THETA = 1e6
QSCALE = 1.4426950408889634 / math.sqrt(HD)  # (the queries' scale: the softmax in log2 units)
PATCH = 16
PIX, ONE, KP = 768, 768, 832  # (a patch's inputs (c, ky, kx); the constant input 1; the panels' K: whole k-tiles)
ONE2, KP2 = 1536, 1600  # (a video pair's: its two frames' inputs, then the constant)
MERGER = W.Merger(H=H, C=C, eps=1e-6, up=(1280, 1280, 1280, 320), dn=640)  # (mm.0's 4096 outputs in 4160)
PIECE_BAND_LAYERS = 16  # (a piece's band-layers at most: its build time)
PIECE_MS = 500.0  # (a dispatch's estimated time at most)
SILU_TABLE = np.tile(kmath.SILU.table_words(), 2).view(np.uint8)


def q_order() -> np.ndarray:
    """Per query head h of the NPU's order its HF head (2 (h mod 8) + h // 8: query head h's KV head is h mod 8)."""
    h = np.arange(HEADS)
    return 2 * (h % KVH) + h // KVH


def rope_index(heads: np.ndarray) -> np.ndarray:
    """Rows of the RoPE epilogue's heads (80 each: dims 0..31 at 0..31, 32..63 at 40..71) from HF's heads ``heads``
    (64 rows each); -1: zero."""
    j = np.arange(HP)
    d = np.where(j < HP // 2, j, j - HP // 2 + HD // 2)
    ok = (j % (HP // 2)) < HD // 2
    return np.concatenate([np.where(ok, h * HD + d, -1) for h in heads])


def plain_index(heads: np.ndarray) -> np.ndarray:
    """Rows (V's) or columns (O's) of 80 per head: HF head h's 64, then 16 zeros."""
    j = np.arange(HP)
    return np.concatenate([np.where(j < HD, h * HD + j, -1) for h in heads])


def vec(get: Callable[[str], Wt | np.ndarray], name: str, n: int = C) -> np.ndarray:
    """A norm's weights (or any vector) of the file, zero-padded to n."""
    return pad(f32(get(name)).reshape(-1), n)


def layer_weights(get: Callable[[str], Wt | np.ndarray], il: int, kind: str) -> dict:
    """Layer il's padded weights by role: "ln1", "ln2" (RMSNorm weights), "gu" (the feed-forward's rows [up | gate],
    K 1024), "down" (K 2560); a conv layer's "bx" ([B | x] rows, each padded to 1280), "c" (C's rows), "taps" (1280, 3,
    3), "out" (K 1024); an attention layer's "q" (the NPU's head order, RoPE rows), "k" (RoPE rows), "v", "o" (columns by
    head: K 1280), "qn" / "kn" (the q / k norms' 80 weights in the RoPE rows, q's times log2 e / 8)."""
    p = f"v.blk.{il}."
    w: dict = {"kind": kind, "ln1": vec(get, p + "ln1.weight"), "ln2": vec(get, p + "ln2.weight"),
               "gu": cat([get(p + "ffn_up.weight"), get(p + "ffn_gate.weight")]), "down": pad(get(p + "ffn_down.weight"), C)}
    if kind == "conv":
        ip = get(p + "conv2d_in_proj.weight")
        rows = lambda k: take(ip, np.arange(k * H, (k + 1) * H))
        w["bx"] = cat([pad(rows(0), C), pad(rows(2), C)])
        w["c"] = pad(rows(1), C)
        taps = values(get(p + "conv2d_spatial.weight")).astype(np.float32).reshape(H, 3, 3)
        w["taps"] = np.concatenate([taps, np.zeros((C - H, 3, 3), np.float32)])
        w["out"] = pad(get(p + "conv2d_out_proj.weight"), C)
        return w
    qo = q_order()
    w["q"] = take(get(p + "attn_q.weight"), rope_index(qo))
    w["k"] = take(get(p + "attn_k.weight"), rope_index(np.arange(KVH)))
    w["v"] = take(get(p + "attn_v.weight"), plain_index(np.arange(KVH)))
    w["o"] = pad(take(get(p + "attn_out.weight"), plain_index(qo), 1), C)
    idx = rope_index(np.zeros(1, np.int64))
    for n, s in (("qn", QSCALE), ("kn", 1.0)):
        nw = f32(get(p + f"attn_{n[0]}_norm.weight")).reshape(-1)
        w[n] = np.where(idx >= 0, nw[np.maximum(idx, 0)], 0).astype(np.float32) * np.float32(s)
    return w


def kinds_of(names: object, layers: int) -> tuple[str, ...]:
    """The layers' kinds ("conv", "attn") of an mmproj's tensor names."""
    return tuple("conv" if f"v.blk.{il}.conv2d_in_proj.weight" in names else "attn" for il in range(layers))


def patch_weights(get: Callable[[str], Wt | np.ndarray], pair: bool = False) -> dict[str, Wt | np.ndarray]:
    """The patch embedding as a GEMM of the integer panels (KP inputs -> C): the two temporal kernels summed (``pair``:
    a video's pair of frames, KP2 inputs: kernel t on frame t's 768), the normalization x = (p / 255 - 0.5) / 0.5 =
    a q + o of q = p - 128 (a = 1 / 127.5, o = 1 / 255) folded in: input k < 768 the weights' column k times a, input
    768 (1536; the constant 1) their offset sum_k w_k o (computed at every load); no bias. "eg": the embedding norm's
    weights; "eye": the identity (C rows of K 1024)."""
    w0, w1 = (values(get(n)).astype(np.float64).reshape(H, PIX) for n in ("v.patch_embd.weight", "v.patch_embd.weight.1"))
    a = 1 / 127.5
    o = 128 * a - 1
    if pair:
        wt = np.zeros((C, KP2), np.float32)
        wt[:H, :PIX], wt[:H, PIX:2 * PIX], wt[:H, ONE2] = w0 * a, w1 * a, (w0 + w1).sum(1) * o
    else:
        wt = np.zeros((C, KP), np.float32)
        wt[:H, :PIX], wt[:H, ONE] = (w0 + w1) * a, (w0 + w1).sum(1) * o
    return {"w": computed("lfm3vl.patch2" if pair else "lfm3vl.patch", wt), "eg": vec(get, "v.embedding_norm.weight"),
            "eye": pad(eye(H), C)}


def weights(get: Callable[[str], Wt | np.ndarray], kinds: tuple[str, ...], proj: int) -> dict:
    """Every padded weight of the encoder: "patch", "patch2" (``patch_weights``: an image's, a video pair's), per layer
    il its ``layer_weights``, "kinds",
    "post_g" / "post_b" (the projector's LayerNorm), "up" / "down" (mm.0 on the merged 4 x 1280 channels -> 4160, mm.2
    -> the language model's width in passes) and their biases; "layers", "proj". Matrices as ``get`` gives them (views
    of the file), never materialized."""
    w: dict = {"patch": patch_weights(get), "patch2": patch_weights(get, pair=True), "layers": len(kinds), "proj": proj,
               "kinds": kinds,
               **{il: layer_weights(get, il, k) for il, k in enumerate(kinds)}}
    w["post_g"], w["post_b"] = vec(get, "mm.input_norm.weight"), vec(get, "mm.input_norm.bias")
    blocks = np.array([k * H + i if i < H else -1 for k in range(4) for i in range(C)])
    w["up"] = pad(take(get("mm.0.weight"), blocks, 1), MERGER.FUP)
    w["up_b"] = pad(f32(get("mm.0.bias")).reshape(-1), MERGER.FUP)
    w["down"] = pad(get("mm.2.weight"), W.out_width(proj), MERGER.FUP)
    w["down_b"] = pad(f32(get("mm.2.bias")).reshape(-1), W.out_width(proj))
    return w


def level(nb: int) -> VitLevel:
    return vit_level(nb)


def view(t: Tensor, n: int) -> Tensor:
    """The first n channels of the panel tensor t (a GEMM's K)."""
    return Tensor(t.lv, n, t.addr, "pn", pnp=t.np)


def rms(bd: Builder, x: Tensor, y: Tensor, g: np.ndarray, *, name: str, rec: int) -> None:
    """RMSNorm of the C tiles x (records at rec, freed) over their 1024 channels into the panels y."""
    ln(bd, x, y, g, None, name=name, rec=rec, C=C, H=H, eps=EPS, mode="rms")


def load_x(bd: Builder, x: Bands, b: int) -> Tensor:
    """Band b's C tiles of the residual x (without its records)."""
    t = alloc(bd, x.lv, C)
    bd.op(bd.load(x.x.band(b), t.addr, t.nbytes), {})
    return t


class Net:
    """The encoder's ops on a builder: ``patch``, ``layer`` (a layer on all bands), the DDR images by ``nm``'s names
    (module docstring)."""

    def __init__(self, w: dict, nm: W.Names, kv: int):
        self.w, self.nm, self.kv = w, nm, kv

    def keys(self, nb: int) -> Keys:
        return Keys(self.nm.attn(nb), joint=True, dense=True, kv_heads=KVH)

    def rope(self, bd: Builder, lv: VitLevel, band: int) -> Tensor:
        """Band b's cos | sin tile (``windows.rope_table``) in the memtile."""
        t = alloc(bd, lv, HP)
        rt = lv.ct(HP)
        bd.ddr(f"{self.nm.p}.rope.{band}", COLS * rt)
        bd.op(bd.load(f"{self.nm.p}.rope.{band}", t.addr, rt), {})
        return t

    def patch(self, bd: Builder, nb: int, pair: bool = False) -> None:
        """Per band the patch GEMM of the host's integer panels (<p>.pix.b; ``pair``: a video pair's, <p>.pixp.b), RMSNorm
        into panels, the identity GEMM back to C tiles: the residual's bands before layer 0 and their records."""
        w, nm, lv = self.w["patch2" if pair else "patch"], self.nm, level(nb)
        out = Bands(lv, nm.x(0))
        kp = KP2 if pair else KP
        pn = lv.pn(kp)
        for b in range(nb):
            pi = bd.ddr(f"{nm.p}.{'pixp' if pair else 'pix'}.{b}", COLS * pn)
            pt = alloc(bd, lv, kp, "pn")
            bd.op({c: ColOps(fills=[MemTask([{"addr": pt.addr, "nwords": pn // 4}], ddr=(pi + c * pn, pn, None))])
                   for c in range(COLS)}, {})
            p = alloc(bd, lv, C)
            rec = gemm(bd, pt, w["w"], None, name=f"{nm.p}.patch{'2' * pair}", out=p, ln=True, hilo=True)
            free(bd, pt)
            n = alloc(bd, lv, C, "pn")
            rms(bd, p, n, w["eg"], name=f"{nm.p}.en", rec=rec)
            free(bd, p)
            x0 = alloc(bd, lv, C)
            rec = gemm(bd, view(n, H), w["eye"], None, name=f"{nm.p}.eye", out=x0, ln=True)
            free(bd, n)
            out.store(bd, b, x0, rec)

    def ffn(self, bd: Builder, w: dict, name: str, x: Tensor, rec: int) -> tuple[Tensor, int]:
        """LN2, x parked in DDR, gate / up as SwiGLU panels, x back, down + x: the output's C tiles (x's memory freed)
        and records."""
        park, lv = f"{self.nm.p}.park", x.lv
        low = x.addr < D.HEAP // 2  # (the hidden at the other end of the memtile from x: ``vit.ffn``'s)
        n = alloc(bd, lv, C, "pn", top=low)
        bd.defer(bd.spill(park, x.addr, x.nbytes), release=[(x.addr, x.size or x.nbytes)])
        rms(bd, x, n, w["ln2"], name=name + ".ln2", rec=rec)
        bd.table("silu", SILU_TABLE)
        g = alloc(bd, lv, FFN, "pn", top=not low)
        for k in range(2):  # (16 passes a row: two ops, a command's biases take 14 passes at most)
            bd.gemm(Gemm(m=64, ntile=1, K=H, N=2 * FFN, x=n.addr, y=g.addr, out="geglu", act="silu", x_np=n.np, y_np=g.np,
                         p0=8 * k, npo=8), w["gu"], None, name=f"{name}.gu.{k}")
        free(bd, n)
        y = alloc(bd, lv, C)
        bd.op(bd.load(park, y.addr, y.nbytes), {})
        rec = gemm(bd, g, w["down"], None, name=name + ".down", out=y, res=y, ln=True)
        free(bd, g)
        return y, rec

    def layer(self, bd: Builder, i: int, nb: int, gmax: int) -> None:
        """Layer i on nb bands (the residual from image x(i) to x(i + 1)); gmax: the conv input's DDR bytes."""
        (self.conv if self.w[i]["kind"] == "conv" else self.attn)(bd, i, nb, gmax)

    def conv(self, bd: Builder, i: int, nb: int, gmax: int) -> None:
        w, nm, lv = self.w[i], self.nm, level(nb)
        name = nm.layer(i)
        x, out = Bands(lv, nm.x(i)), Bands(lv, nm.x(i + 1))
        gname = f"{nm.p}.g"
        ga = bd.ddr(gname, gmax)
        gbytes = 2 * C // 80 * 5760  # (the product's passes: 5760 B each)
        for b in range(nb):  # (LN1, C spilled, g = B x drained into the conv input)
            xb, rec = x.load(bd, b)
            n = alloc(bd, lv, C, "pn")
            rms(bd, xb, n, w["ln1"], name=name + ".ln1", rec=rec)
            free(bd, xb)
            c = alloc(bd, lv, C)
            gemm(bd, view(n, H), w["c"], None, name=name + ".c", out=c)
            bd.defer(bd.spill(f"{nm.p}.cg.{b}", c.addr, c.nbytes), release=[(c.addr, c.size or c.nbytes)])
            g = bd.mem.alloc(gbytes)
            bd.gemm(Gemm(m=64, ntile=1, K=H, N=2 * C, x=n.addr, y=g, out="prod", x_np=n.np), w["bx"], None,
                    name=name + ".bx")
            bd.written(gname)
            bd.defer({col: ColOps(drains=[MemTask([{"addr": g + 640 * p, "nwords": 5120,
                                                   "dims": [(4, 1), (5, 32), (32, 1440), (0, 4)], "iteration": (320, 4)}],
                                                 repeat=4, ddr=(ga + W.conv_window(int(W.windows_of(nb, b, col, p)[0])),
                                                                20480, (640, 8, 2560, 1), (640, 4)))
                                         for p in range(2)]) for col in range(COLS)}, release=[(g, gbytes)])
            free(bd, n)
        bd.op({}, {})  # (the last band's drains)
        S = DwConv.STAGE_BYTES
        for b in range(nb):  # (the depthwise conv in 4 steps, out_proj + x, the feed-forward)
            bd.flush()  # (the previous band's output stored: the staging takes 288 KB in one piece)
            y = alloc(bd, lv, C, "pn")
            st = bd.mem.alloc(2 * S)
            cname, mname = f"{nm.p}.cg.{b}", f"{nm.p}.cm.{b}"
            bd.reading(gname)
            bd.reading(cname)
            ca, ma = bd.npu.address(cname), bd.ddr(mname, COLS * DwConv.MASK_BYTES)
            ops = [DwConv(stage=st + (k % 2) * S, y=y.addr, k=k) for k in range(4)]

            def fills(k: int, b: int = b, ca: int = ca, ma: int = ma, ops: list[DwConv] = ops) -> dict[int, ColOps]:
                op, cols = ops[k], {}
                for col in range(COLS):
                    fl = [op.strip_fill(p, part, ga + W.conv_strip(nb, b, col, p, part, k), None if part == "mid" else part)
                          for p in range(2) for part in ("mid", "up", "down")]
                    fl.append(MemTask([op.c_fill()], ddr=op.c_ddr(ca + col * 16 * L.ct_bytes(64), k)))
                    fl.append(MemTask([op.mask_fill()], ddr=(ma + col * DwConv.MASK_BYTES, DwConv.MASK_BYTES, None)))
                    cols[col] = ColOps(fills=fl)
                return cols
            bd.op(fills(0), {})
            for k, op in enumerate(ops):
                img = bd.image(f"{name}.dw{k}", np.concatenate([np.concatenate([op.command(), DwConv.taps(w["taps"], 4 * r + k)])
                                                                for r in range(ROWS)]))
                cols = {col: op.col_ops() for col in range(COLS)}
                bd.op(bd.merge(cols, fills(k + 1)) if k < 3 else cols, op.row_ops(img))
            bd.mem.release(st, 2 * S)
            xb = load_x(bd, x, b)
            rec = gemm(bd, view(y, H), w["out"], None, name=name + ".out", out=xb, res=xb, ln=True)
            free(bd, y)
            yy, rec = self.ffn(bd, w, name, xb, rec)
            out.store(bd, b, yy, rec)

    def attn(self, bd: Builder, i: int, nb: int, gmax: int) -> None:
        w, nm, lv = self.w[i], self.nm, level(nb)
        name, keys = nm.layer(i), self.keys(nb)
        x, out = Bands(lv, nm.x(i)), Bands(lv, nm.x(i + 1))
        at = keys.attention(lv, kv=self.kv)
        for b in range(nb):  # (LN1, K and V into the slots)
            xb, rec = x.load(bd, b)
            n = alloc(bd, lv, C, "pn")
            rms(bd, xb, n, w["ln1"], name=name + ".ln1", rec=rec)
            free(bd, xb)
            rt = self.rope(bd, lv, b)
            kt, vt = alloc(bd, lv, KVH * HP, "pn"), alloc(bd, lv, KVH * HP, "pn")
            bd.gemm(Gemm(m=64, ntile=1, K=H, N=KVH * HP, x=n.addr, y=kt.addr, out="rope", res=rt.addr, qkn=w["kn"], qkn_n=HD,
                         qkn_eps=EPS, x_np=n.np, y_np=kt.np), w["k"], None, name=name + ".k")
            gemm(bd, view(n, H), w["v"], None, name=name + ".v", out=vt, kind="vt",
                 extra={c: at.kv_drains(kt.addr, vt.addr, c, part="k", band=b) for c in range(COLS)})
            free(bd, kt)
            bd.defer({c: at.kv_drains(kt.addr, vt.addr, c, part="v", band=b) for c in range(COLS)},
                     release=[(vt.addr, vt.size)])
            free(bd, n)
            free(bd, rt)
        bd.op({}, {})  # (the last band's V^T drains)
        bd.ddr_sync()
        for b in range(nb):  # (LN1, Q, the attention, O + x, the feed-forward)
            xb, rec = x.load(bd, b)
            n = alloc(bd, lv, C, "pn")
            rms(bd, xb, n, w["ln1"], name=name + ".ln1", rec=rec)
            qt = alloc(bd, lv, C, "pn")
            rt = self.rope(bd, lv, b)
            bd.gemm(Gemm(m=64, ntile=1, K=H, N=C, x=n.addr, y=qt.addr, out="rope", res=rt.addr, qkn=w["qn"], qkn_n=HD,
                         qkn_eps=EPS, x_np=n.np, y_np=qt.np), w["q"], None, name=name + ".q")
            free(bd, n)
            free(bd, rt)
            o = attend(bd, qt, self.kv, keys)
            free(bd, qt)
            rec = gemm(bd, o, w["o"], None, name=name + ".o", out=xb, res=xb, ln=True)
            free(bd, o)
            y, rec = self.ffn(bd, w, name, xb, rec)
            out.store(bd, b, y, rec)


class Encoder(W.WindowEncoder):
    """The encoder (module docstring) as dispatches on a VisionDevice (``windows.WindowEncoder``)."""

    def __init__(self, npu: VisionDevice, w: dict, *, max_bands: int, name: str = "l", state: dict | None = None):
        """w: ``weights``; max_bands: the largest level; state: a restored device's (``WindowEncoder``)."""
        self.kinds = w["kinds"]
        self.gmax = W.conv_ddr_bytes(max_bands)
        npu.alloc(f"{name}.g", self.gmax)  # (the conv input of the largest level, before any program)
        kvb = kv_bytes(level(max_bands), Keys("", joint=True, dense=True, kv_heads=KVH))
        self.net = Net(w, W.Names(name), 0)
        super().__init__(npu, w, max_bands=max_bands, name=name, state=state, kv_bytes=kvb)

    def patch_embed(self, bd: Builder, nb: int, pair: bool = False) -> None:
        self.net.kv = self.kv
        self.net.patch(bd, nb, pair)

    def layers(self, bd: Builder, nb: int, il: range) -> None:
        self.net.kv = self.kv
        for i in il:
            self.net.layer(bd, i, nb, self.gmax)

    def final(self, bd: Builder, nb: int) -> None:
        W.merger(bd, self.w, self.nm, nb, self.nl, MERGER)

    def layers_per_piece(self, nb: int) -> int:
        """Layers of a dispatch at nb bands: all at nb 1, else at most PIECE_BAND_LAYERS band-layers (its build time)
        and about PIECE_MS (a layer ~0.7 nb + 0.05 nb^2 ms: its GEMMs, its attention over every key)."""
        if nb == 1:
            return self.nl
        return max(1, min(self.nl, PIECE_BAND_LAYERS // nb, int(PIECE_MS / (0.7 * nb + 0.05 * nb * nb))))

    def _runs(self, nb: int) -> list[tuple[int, int]]:
        """(the leading conv layers, then runs of whole [attention, conv] pairs: pieces of one kind pattern)"""
        k = self.layers_per_piece(nb)
        if k >= self.nl:
            return [(0, self.nl)]
        lead = next((i for i, kd in enumerate(self.kinds) if kd == "attn"), self.nl)
        out, i = [], 0
        while i < lead:
            out.append((i, min(k, lead - i)))
            i += out[-1][1]
        step = max(1, k // 2 * 2)
        while i < self.nl:
            out.append((i, min(step, self.nl - i)))
            i += out[-1][1]
        return out

    def pixel_panels(self, lay: W.Layout, px: np.ndarray) -> dict[str, np.ndarray]:
        if px.ndim == 4:  # (a video's pair)
            return W.patch_panels(lay, px, self.nm.p, kp=KP2, one=ONE2, tag="pixp")
        return W.patch_panels(lay, px, self.nm.p, kp=KP, one=ONE)

    def size_inputs(self, lay: W.Layout) -> dict[str, np.ndarray]:
        """The size's rows: the attention command's key masks, the RoPE tables, the conv's edge masks."""
        nm, nb = self.nm, lay.nb
        ph, pw = 2 * lay.gh, 2 * lay.gw
        ok = lay.keys()
        att = Keys(nm.attn(nb), valid=(ok, ok), joint=True, dense=True, kv_heads=KVH).attention(level(nb)).command()
        out = {nm.attn(nb): att} | W.band_rows(lay, W.rope_table(ph, pw, HD, THETA, HP), f"{nm.p}.rope")
        for b in range(nb):
            out[f"{nm.p}.cm.{b}"] = W.dw_masks(lay.gh, lay.gw, nb, b).view(np.uint8).reshape(-1)
        return out

    def shift(self, tag: object, lay: W.Layout) -> int:
        return W.conv_shift(str(tag), lay.gw)
