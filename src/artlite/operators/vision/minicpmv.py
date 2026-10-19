"""MiniCPM-V 4.7's vision encoder (projector ``minicpmv4_7``: a SigLIP2 ViT of 27 layers, the ViT merger after layer 6,
the final 2 x 2 merger) on sd15's static design (``operators.sd15``: one design for both models, its ops built by
host-built transactions).

Channels: the hidden 1152 padded to ``C`` = 1280 (passes of 80 on 4 core rows): the residual's 1152 real channels,
then 128 that zero weight rows, biases, gamma and beta keep zero (LayerNorm's 1 / n over the 1152); a head's 72 dims
padded to 80 (Q / K / V rows, O's columns: 16 heads of d 80); the feed-forward's 4304 to ``FFN`` = 4480.

A layer (``layer``) on a level of two entries (sd15's two images): LN1 -> K, V (their slots drained to DDR), Q
(x log2 e / sqrt 72) -> attention (16 heads, the softmax's max raised past attention sinks) -> O + x (LayerNorm
records) -> LN2 (x parked in DDR meanwhile) -> up + GELU (panels) -> x back -> down + x (records). Its memtile
layout alternates between two (x low / high), the same every other layer.
"""

import math
from collections.abc import Callable

import torch

from ..design import COLS
from ..sd15 import design as D
from ..sd15.build import Builder, Level
from ..sd15.ops import Apply, Attention, Gemm
from ..sd15.unet import Tensor, alloc, free, gelu_table, gemm, ln_rec

H, C, HD, HP, HEADS = 1152, 1280, 72, 80, 16
FFN_REAL, FFN = 4304, 4480
EPS = 1e-6
LOG2E = 1.4426950408889634


def _heads(w: torch.Tensor) -> torch.Tensor:
    """Rows of 16 heads of 72 -> 16 heads of 80 (zero rows after each head's)."""
    out = torch.zeros((HEADS * HP,) + w.shape[1:], dtype=w.dtype)
    out.view(HEADS, HP, *w.shape[1:])[:, :HD] = w.reshape(HEADS, HD, *w.shape[1:])
    return out


def _pad(w: torch.Tensor, rows: int, cols: int = 0) -> torch.Tensor:
    out = torch.zeros((rows, cols) if cols else (rows,), dtype=w.dtype)
    out[tuple(slice(0, n) for n in w.shape)] = w
    return out


def layer_weights(get: Callable[[str], torch.Tensor], il: int) -> dict[str, torch.Tensor]:
    """Layer il's padded weights (float32) by role: q, k, v (rows by head), o (columns by head), up, down and their
    biases (``*_b``), ln1 / ln2 gamma and beta (``*_g``, ``*_b``); ``get(name)``: a tensor of the mmproj as
    (rows, cols) or (cols,)."""
    p = f"v.blk.{il}."
    s = LOG2E / math.sqrt(HD)
    w = {}
    for r in ("q", "k", "v"):
        wt, b = get(p + f"attn_{r}.weight").float(), get(p + f"attn_{r}.bias").float().reshape(-1)
        if r == "q":
            wt, b = wt * s, b * s
        w[r] = _pad(_heads(wt), C, C)
        w[r + "_b"] = _heads(b)
    o = get(p + "attn_out.weight").float()
    w["o"] = _pad(_heads(o.T).T, C, C)
    w["o_b"] = _pad(get(p + "attn_out.bias").float().reshape(-1), C)
    w["up"] = _pad(get(p + "ffn_up.weight").float(), FFN, C)
    w["up_b"] = _pad(get(p + "ffn_up.bias").float().reshape(-1), FFN)
    w["down"] = _pad(get(p + "ffn_down.weight").float(), C, FFN)
    w["down_b"] = _pad(get(p + "ffn_down.bias").float().reshape(-1), C)
    for n in ("ln1", "ln2"):
        w[n + "_g"] = _pad(get(p + n + ".weight").float().reshape(-1), C)
        w[n + "_b"] = _pad(get(p + n + ".bias").float().reshape(-1), C)
    return w


def ln_records(bd: Builder, x: Tensor, *, name: str) -> int:
    """The LayerNorm records of the C tiles x (a copy op with statistics: x from the host, no producer GEMM); returns
    their memtile address."""
    lv = x.lv
    rec = bd.mem.alloc(ln_rec(lv))
    y = bd.mem.alloc(lv.pn(C))
    bd.gemm(Gemm(m=lv.m, ntile=lv.ntile, K=0, N=C, x=0, y=y, out="bfp16", res=x.addr, stats="ln", rec=rec), None, None,
            name=name)
    bd.mem.release(y, lv.pn(C))
    return rec


def ln(bd: Builder, x: Tensor, y: Tensor, gamma: torch.Tensor, beta: torch.Tensor, *, name: str, rec: int) -> None:
    """LayerNorm of the C tiles x (its producer's records at rec, freed) into the panels y on every core row."""
    a = Apply(m=x.lv.m, nblk=C // 320, mode="ln", x=x.addr, y=y.addr, ntile=x.lv.ntile, rec=rec, C=H, eps=EPS,
              y_np=y.np)
    bd.apply(a, gamma, beta, name=name)
    bd.mem.release(rec, ln_rec(x.lv))


def attend(bd: Builder, q: Tensor, kv: int, *, name: str) -> Tensor:
    """The attention of the scaled queries' panels q on the K / V^T slots at DDR kv (every key of their entry): the
    output panels."""
    o = alloc(bd, q.lv, C, "pn")
    for t in range(max(q.lv.np // 8, 1)):  # (an op per tile)
        at = Attention(np=q.lv.np, q=q.addr, o=o.addr, kv=kv, d=HP, nbands=q.lv.nb, tile=t, heads=HEADS, raise_max=True)
        cmd = bd.image(name, at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd))
    return o


def layer(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int, *, kv: int) -> tuple[Tensor, int]:
    """A ViT layer (module docstring) on the residual's C tiles x with its LayerNorm records at rec (its producer's):
    the output's C tiles (x's memory freed) and their records. kv: a DDR buffer for the attention's K / V^T slots
    (``kv_bytes``)."""
    lv = x.lv
    n = alloc(bd, lv, C, "pn")
    ln(bd, x, n, w["ln1_g"], w["ln1_b"], name=name + ".ln1", rec=rec)
    kt, vt = alloc(bd, lv, C, "pn"), alloc(bd, lv, C, "pn")
    at = Attention(np=lv.np, q=0, o=0, kv=kv, d=HP, nbands=lv.nb, heads=HEADS)
    gemm(bd, n, w["k"], w["k_b"], name=name + ".k", out=kt, kind="bfp16")
    gemm(bd, n, w["v"], w["v_b"], name=name + ".v", out=vt, kind="vt",
         extra={c: at.kv_drains(kt.addr, vt.addr, c, part="k") for c in range(COLS)})
    free(bd, kt)
    qt = alloc(bd, lv, C, "pn")
    gemm(bd, n, w["q"], w["q_b"], name=name + ".q", out=qt, kind="bfp16",
         extra={c: at.kv_drains(kt.addr, vt.addr, c, part="v") for c in range(COLS)})
    bd.ddr_sync()
    free(bd, vt)
    free(bd, n)
    o = attend(bd, qt, kv, name=name + ".attn")
    free(bd, qt)
    rec = gemm(bd, o, w["o"], w["o_b"], name=name + ".o", out=x, res=x, ln=True)
    free(bd, o)
    # (the down GEMM reads the hidden panels and writes y: 506 of the memtile's 510 KB; layers alternate between x low,
    # the LN2 panels at the top and the hidden from the bottom (y then high), and x high, the LN2 panels at the bottom
    # and the hidden at the top (y then low))
    low = x.addr < D.HEAP // 2
    n = alloc(bd, lv, C, "pn", top=low)
    bd.defer(bd.spill(name + ".x", x.addr, x.nbytes), release=[(x.addr, x.size or x.nbytes)])  # (with LN2: the hidden
    # panels take the memtile)
    ln(bd, x, n, w["ln2_g"], w["ln2_b"], name=name + ".ln2", rec=rec)
    bd.table("gelu", gelu_table())
    g = alloc(bd, lv, FFN, "pn", top=not low)
    gemm(bd, n, w["up"], w["up_b"], name=name + ".up", out=g, kind="gelu")
    free(bd, n)
    y = alloc(bd, lv, C)
    bd.op(bd.load(name + ".x", y.addr, y.nbytes), {})
    rec = gemm(bd, g, w["down"], w["down_b"], name=name + ".down", out=y, res=y, ln=True)
    free(bd, g)
    return y, rec


def kv_bytes(lv: Level) -> int:
    """Bytes of a level's K / V^T slots (``layer``'s kv)."""
    return Attention(np=lv.np, q=0, o=0, kv=0, d=HP, nbands=lv.nb, heads=HEADS).kv_bytes()
