"""sd15's UNet blocks on a ``build.Builder`` (resnets, transformers, down / upsampling of the 32x32 .. 8x8 levels;
the banded transformer the 64x64 level's too: ``level64``).

A block takes its input with the GroupNorm records its first norm needs and returns its output the same way (records in
the layout its consumer asks for: ``stats`` = (group size, channel offset) of the consumer's GroupNorm, a
concatenation's second part at its offset): at 16x16 / 8x8 C-tile tensors in the memtile (a level fits it), at 32x32
DDR tensors per band (``DTensor``: the ``*_bands`` blocks run band by band, a band's working set the memtile's). Weights
by diffusers name (``w[p + ...]``).
"""

import functools
from dataclasses import dataclass

import numpy as np
import torch

from .. import act_lut
from ..design import COLS, ROWS
from . import design as D
from . import layout as L
from .build import Builder, Level, Records
from .ops import Apply, Attention, Gemm
from .seq import ColOps, MemTask


@dataclass
class Tensor:
    """A tensor of a level in the memtile: C tiles (kind "ct") or panels ("pn") of C channels at ``addr``; its
    GroupNorm records for the next norm (``rec``)."""

    lv: Level
    C: int
    addr: int
    kind: str = "ct"
    rec: list[Records] | None = None
    pnp: int = 0  # (panels: token panels per kb, if not the level's: a GEGLU output's padding)
    size: int = 0  # (the bytes allocated: ``alloc``)

    @property
    def np(self) -> int:
        return self.pnp or self.lv.np

    @property
    def nbytes(self) -> int:
        return self.lv.ct(self.C) if self.kind == "ct" else self.C // 8 * self.np * 72


def alloc(bd: Builder, lv: Level, C: int, kind: str = "ct", pnp: int = 0, top: bool = False, exact: bool = False) -> Tensor:
    """A memtile tensor (panels of tiles of 64: room for whole A objects of 2 k-tiles, the GEMMs' walk reads them;
    exact: none, the 64x64 convs' input rows)."""
    t = Tensor(lv, C, 0, kind, pnp=pnp)
    t.size = t.nbytes if kind == "ct" or lv.m != 64 or exact else -(-C // 128) * 16 * t.np * 72
    t.addr = bd.mem.alloc(t.size, top=top)
    return t


def free(bd: Builder, t: Tensor) -> None:
    bd.mem.release(t.addr, t.size or t.nbytes)


@functools.cache
def _silu() -> np.ndarray:
    return np.asarray(act_lut.SILU.table_buffer()).view(np.uint8)


def silu_table(bd: Builder) -> None:
    bd.table("silu", _silu())


@functools.cache
def gelu_table(table: act_lut.ActTable = act_lut.GELU) -> np.ndarray:
    """A GELU's table (act_lut.GELU, or GELU_ERF) for the GEGLU / GELU epilogues' table pass (the SiLU table's form, fed
    t' = 2 t: its segment index and remainder exactly 2x GELU's, so c1 / 2, c2 / 4; each copy padded to SiLU's words)."""
    import copy
    g = copy.copy(table)
    c = table.coefficients.copy()
    c[:, 1] /= 2
    c[:, 2] /= 4
    g.__dict__["coefficients"] = c
    w = g.table_words()
    w = np.concatenate([w, np.zeros(len(act_lut.SILU.table_words()) - len(w), np.uint32)])
    return np.concatenate([w, w]).view(np.uint8)


def gn_ops(lo: int, hi: int, win: int) -> list[tuple[int, int, tuple[int, ...]]]:
    """The apply ops of GroupNorm channels [lo, hi) (record windows of ``win`` channels from 0): (first channel, blocks
    of 80 per core, core rows). Each core's channels lie in one window: the range in equal parts, one per row, where
    they do; else per window on as many rows as divide it."""
    def parts(a: int, b: int, n: int) -> int:
        q, r = divmod(b - a, n)
        ok = not r and not q % 80 and q <= 640 and all((a + i * q) // win == (a + (i + 1) * q - 1) // win for i in range(n))
        return q // 80 if ok else 0
    if nblk := parts(lo, hi, ROWS):
        return [(lo, nblk, tuple(range(ROWS)))]
    out = []
    for a in range(lo // win * win, hi, win):
        a0, a1 = max(a, lo), min(a + win, hi)
        n = max(n for n in range(1, ROWS + 1) if parts(a0, a1, n))
        out.append((a0, parts(a0, a1, n), tuple(range(n))))
    return out


def gn_apply(bd: Builder, x: Tensor, y: Tensor, gamma: torch.Tensor, beta: torch.Tensor, *, name: str, eps: float,
             silu: bool, y_kb0: int = 0, ch0: int = 0, ccat: int = 0, extra: dict | None = None) -> None:
    """GroupNorm (32 groups, + SiLU) of x into the panels y (from kb y_kb0). x may be the part [ch0, ch0 + x.C) of a
    concatenation of ccat channels (its records in that space; gamma, beta: the part's). Ops of every core row, each
    core within one record window (``gn_ops``; extra: col ops merged into the first)."""
    C = ccat or x.C
    gs = C // 32
    if silu:
        silu_table(bd)
    for k, (a, nblk, rows) in enumerate(gn_ops(ch0, ch0 + x.C, 16 * gs)):
        ap = Apply(m=x.lv.m, nblk=nblk, mode="gnsilu" if silu else "gn", x=x.addr, y=y.addr, ntile=x.lv.ntile,
                   blk0=(a - ch0) // 80, y_kb0=y_kb0 + (a - ch0) // 8, y_np=y.np, gsize=gs, ch_off=a,
                   tokens=x.lv.H ** 2, eps=eps, rows=rows)
        bd.apply(ap, gamma, beta, name=f"{name}.{k}", records=x.rec, extra=extra if k == 0 else None)


def conv(bd: Builder, x: Tensor, w: torch.Tensor, b: torch.Tensor, *, name: str, out: Tensor, stride: int = 1,
         res: Tensor | None = None, stats: tuple[int, int] | None = None, tbias: str | None = None,
         band: int | None = None) -> None:
    """A 3x3 conv of the panels x into the C tiles out (+ res), out's records for ``stats``; tbias: the resnet whose
    time bias b includes (``Builder.gemm``: rewritten per step). band: of a banded level, x the whole column's panels,
    out band ``band``'s (row tiles: ``ops.Gemm`` rc)."""
    lv = out.lv
    N = w.shape[0]
    kw = {"m": lv.m, "ntile": 1, "K": x.C, "N": N, "x": x.addr, "y": out.addr, "out": "bf16", "conv": stride, "R": lv.R, "Lg": lv.Lg,
              "zero": bd.zero, "x_np": x.np}
    if band is not None:
        kw.update(ntile=lv.ntile, R=1, rc=lv.R, r0=band * lv.R // lv.nb)
    rec = bd.mem.alloc(4 * D.OCH, top=True) if stats else None
    if stats:
        kw.update(stats="gn", rec=rec, gsize=stats[0], ch_off=stats[1])
    bd.gemm(Gemm(**kw, res=res.addr if res is not None else None), w, b, name=name, tbias=tbias)
    if stats:
        out.rec = [bd.records(rec, name=f"{name}.rec", ch_off=stats[1], nr=N // 4, gsize=stats[0], band=band or 0,
                              nbands=lv.nb)]


def gemm(bd: Builder, x: Tensor, w: torch.Tensor, b: torch.Tensor | None, *, name: str, out: Tensor,
         res: Tensor | None = None, stats: tuple[int, int] | None = None, kind: str = "bf16", ln: bool = False,
         split: int = 1, extra: dict | None = None, band: int = 0, hilo: bool = False) -> int:
    """A GEMM (1x1 conv / linear) of the panels x into out (C tiles; panels: kind "bfp16" / "vt" / "geglu"), + res;
    out's GroupNorm records for ``stats``; ln: LayerNorm records, returns their memtile address (the caller frees
    ``ln_rec`` bytes after the apply). split: ops of npass / split passes (a lock counts a GEMM's sends to 63).
    extra: col ops (fills / drains) merged into the first op. hilo: hi + lo weights (``ops.Gemm``)."""
    lv = out.lv
    kw = {"m": lv.m, "ntile": lv.ntile, "K": x.C, "N": w.shape[0], "x": x.addr, "y": out.addr, "out": kind,
          "res": res.addr if res else None, "x_np": x.np, "y_np": out.np if kind != "bf16" else None, "hilo": hilo}
    rec = bd.mem.alloc(ln_rec(lv) if ln else 4 * D.OCH, top=not ln) if stats or ln else 0  # (LN: its apply's walk reads on)
    if stats:
        kw.update(stats="gn", rec=rec, gsize=stats[0], ch_off=stats[1])
    elif ln:
        kw.update(stats="ln", rec=rec)
    assert split == 1 or not (stats or ln)
    npass = w.shape[0] // 320
    for k in range(split):
        part = {} if split == 1 else {"p0": k * npass // split, "npo": npass // split}
        bd.gemm(Gemm(**kw, **part), w.flatten(1), b, name=name if split == 1 else f"{name}.{k}", extra=extra if k == 0 else None)
    if stats:
        out.rec = [bd.records(rec, name=f"{name}.rec", ch_off=stats[1], nr=w.shape[0] // 4, gsize=stats[0], band=band,
                              nbands=lv.nb)]
    return rec


def ln_rec(lv: Level) -> int:
    """Bytes of a band's LayerNorm records ([tile][core][OCH])."""
    return lv.ntile * 4 * D.OCH


def copy_stats(bd: Builder, x: Tensor, *, name: str, st: tuple[int, int], panels: Tensor | None = None,
               extra: dict | None = None, band: int = 0) -> list[Records]:
    """GroupNorm records of the C tiles x for ``st`` (a copy op: K 0, x as the residual), optionally its panels (extra:
    col ops merged into it)."""
    lv = x.lv
    rec = bd.mem.alloc(4 * D.OCH, top=True)
    y = panels.addr if panels is not None else bd.mem.alloc(lv.pn(x.C))
    g = Gemm(m=lv.m, ntile=lv.ntile, K=0, N=x.C, x=0, y=y, out="bfp16", res=x.addr, stats="gn", rec=rec, gsize=st[0],
             ch_off=st[1])
    bd.gemm(g, None, None, name=name, extra=extra)
    if panels is None:
        bd.mem.release(y, lv.pn(x.C))
    r = bd.records(rec, name=f"{name}.rec", ch_off=st[1], nr=x.C // 4, gsize=st[0], band=band, nbands=lv.nb)
    return [r]


def plain(bd: Builder, x: Tensor, y: Tensor, *, name: str) -> None:
    """The C tiles x as panels y: a copy (K 0, x as the residual) on every core row."""
    bd.gemm(Gemm(m=x.lv.m, ntile=x.lv.ntile, K=0, N=x.C, x=0, y=y.addr, out="bfp16", res=x.addr, y_np=y.np), None, None,
            name=name)


def resnet(bd: Builder, w: dict, p: str, x: Tensor, *, cout: int, time_bias: torch.Tensor, eps: float,
           stats: tuple[int, int] | None) -> Tensor:
    """diffusers' ResnetBlock2D: GN1 + SiLU, conv1 (+ the time bias), GN2 + SiLU, conv2 + x (or its 1x1 shortcut).
    x: C tiles with records for GN1 (freed); returns the output C tiles with records for ``stats``."""
    lv = x.lv
    p1 = alloc(bd, lv, x.C, "pn")
    gn_apply(bd, x, p1, w[p + "norm1.weight"], w[p + "norm1.bias"], name=p + "norm1", eps=eps, silu=True)
    h = alloc(bd, lv, cout)
    conv(bd, p1, w[p + "conv1.weight"], w[p + "conv1.bias"] + time_bias, name=p + "conv1", out=h,
         stats=(cout // 32, 0), tbias=p)
    free(bd, p1)
    sc = x
    if p + "conv_shortcut.weight" in w:
        px = alloc(bd, lv, x.C, "pn")
        plain(bd, x, px, name=p + "xpn")
        sc = alloc(bd, lv, cout)
        gemm(bd, px, w[p + "conv_shortcut.weight"], w[p + "conv_shortcut.bias"], name=p + "conv_shortcut", out=sc)
        free(bd, px)
        free(bd, x)
    p2 = alloc(bd, lv, cout, "pn")
    gn_apply(bd, h, p2, w[p + "norm2.weight"], w[p + "norm2.bias"], name=p + "norm2", eps=eps, silu=True)
    free(bd, h)
    conv(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", out=sc, res=sc, stats=stats)
    free(bd, p2)
    return sc



LOG2E = 1.4426950408889634


def ln_apply(bd: Builder, h: Tensor, y: Tensor, gamma: torch.Tensor, beta: torch.Tensor, *, name: str, rec: int) -> None:
    """LayerNorm of the C tiles h (records at rec: its producer's) into the panels y on every core row; frees rec."""
    a = Apply(m=h.lv.m, nblk=h.C // (80 * ROWS), mode="ln", x=h.addr, y=y.addr, ntile=h.lv.ntile, rec=rec, C=h.C,
              y_np=y.np)
    bd.apply(a, gamma, beta, name=name)
    bd.mem.release(rec, ln_rec(h.lv))


def attend(bd: Builder, q: Tensor, kv: int, *, name: str, nkeys: int = 0) -> Tensor:
    """Attention of the panels q (scaled queries, 8 heads) on the K / V^T slots at DDR kv (self-attention: the keys of
    all of the level's bands) on every core row: the output panels."""
    o = alloc(bd, q.lv, q.C, "pn")
    d = q.C // 8
    for t in range(1 if d == 40 else max(q.lv.np // 8, 1)):  # (an op per tile; d 40: per band)
        at = Attention(np=q.lv.np, q=q.addr, o=o.addr, kv=kv, nkeys=nkeys, d=d, nbands=q.lv.nb, tile=t)
        cmd = bd.image(name, at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd))
    return o


def _block_in(bd: Builder, w: dict, p: str, x: Tensor, *, kv: int, band: int = 0, extra: dict | None = None,
              free_x: bool = False, park: str = "") -> tuple[Tensor | None, Tensor]:
    """A transformer block's first part on (a band of) x: GN -> proj_in -> h, LN1 -> K, V^T (drained into the
    self-attention's slots at DDR kv: K's with V's GEMM, V^T's with Q's) and Q. Returns (h, the scaled queries'
    panels); free_x: x freed after the GroupNorm (extra: col ops merged into it); park: h drained into that DDR image
    with the K GEMM and freed (returned as None)."""
    lv, C = x.lv, x.C
    q, a = p + "transformer_blocks.0.", p + "transformer_blocks.0.attn1"
    pg = alloc(bd, lv, C, "pn")
    gn_apply(bd, x, pg, w[p + "norm.weight"], w[p + "norm.bias"], name=p + "norm", eps=1e-6, silu=False, extra=extra)
    if free_x:
        free(bd, x)
    h = alloc(bd, lv, C)
    rec = gemm(bd, pg, w[p + "proj_in.weight"], w[p + "proj_in.bias"], name=p + "proj_in", out=h, ln=True)
    free(bd, pg)
    n = alloc(bd, lv, C, "pn")
    ln_apply(bd, h, n, w[q + "norm1.weight"], w[q + "norm1.bias"], name=q + "norm1", rec=rec)
    if park:
        bd.defer(bd.spill(park, h.addr, h.nbytes), release=[(h.addr, h.nbytes)])
        h = None
    kt, vt = alloc(bd, lv, C, "pn"), alloc(bd, lv, C, "pn")
    at = Attention(np=lv.np, q=0, o=0, kv=kv, d=C // 8, nbands=lv.nb)
    gemm(bd, n, w[a + ".to_k.weight"], None, name=a + ".to_k", out=kt, kind="bfp16",
         extra=None)
    gemm(bd, n, w[a + ".to_v.weight"], None, name=a + ".to_v", out=vt, kind="vt",
         extra={c: at.kv_drains(kt.addr, vt.addr, c, part="k", band=band) for c in range(COLS)})
    free(bd, kt)
    qt = alloc(bd, lv, C, "pn")
    gemm(bd, n, w[a + ".to_q.weight"] * (LOG2E / (C // 8) ** 0.5), None, name=a + ".to_q", out=qt, kind="bfp16",
         extra={c: at.kv_drains(kt.addr, vt.addr, c, part="v", band=band) for c in range(COLS)})
    bd.ddr_sync()
    free(bd, vt)
    free(bd, n)
    return h, qt


def _block_out(bd: Builder, w: dict, p: str, h: Tensor, qt: Tensor, *, kv: int, ctx_kv: int, nctx: int) -> Tensor:
    """A transformer block's second part on (a band of) h with its queries qt (freed): + attn1 (the keys' slots at DDR
    kv) -> + attn2 (LN2; the context's K / V^T slots at DDR ctx_kv, nctx keys) -> + GEGLU feed-forward (LN3); h freed.
    Returns the proj_out input (panels)."""
    lv, C = h.lv, h.C
    q = p + "transformer_blocks.0."
    rec = 0
    for k, a in enumerate(("attn1", "attn2")):
        if a == "attn2":
            n = alloc(bd, lv, C, "pn")
            ln_apply(bd, h, n, w[q + "norm2.weight"], w[q + "norm2.bias"], name=q + "norm2", rec=rec)
            qt = alloc(bd, lv, C, "pn")
            gemm(bd, n, w[q + a + ".to_q.weight"] * (LOG2E / (C // 8) ** 0.5), None, name=q + a + ".to_q", out=qt,
                 kind="bfp16")
            free(bd, n)
        o = attend(bd, qt, kv if a == "attn1" else ctx_kv, name=q + a + ".attn", nkeys=0 if a == "attn1" else nctx)
        free(bd, qt)
        rec = gemm(bd, o, w[q + a + ".to_out.0.weight"], w[q + a + ".to_out.0.bias"], name=q + a + ".to_out", out=h,
                   res=h, ln=True)
        free(bd, o)
    n = alloc(bd, lv, C, "pn")
    ln_apply(bd, h, n, w[q + "norm3.weight"], w[q + "norm3.bias"], name=q + "norm3", rec=rec)
    bd.table("gelu", gelu_table())
    w1, b1, w2, b2 = (w[q + "ff.net.0.proj.weight"], w[q + "ff.net.0.proj.bias"], w[q + "ff.net.2.weight"],
                      w[q + "ff.net.2.bias"])
    hid = w1.shape[0] // 2
    half = hid // 2
    junk = 2 if lv.m == 16 else 0
    h4 = None
    for k in range(2):  # (the hidden channels in halves: the GEGLU output of all of them does not fit the memtile)
        g = alloc(bd, lv, half, "pn", pnp=lv.np + junk)
        rows = torch.cat([w1[k * half:(k + 1) * half], w1[hid + k * half:hid + (k + 1) * half]])
        bias = torch.cat([b1[k * half:(k + 1) * half], b1[hid + k * half:hid + (k + 1) * half]])
        gemm(bd, n, rows, bias, name=q + f"ff1.{k}", out=g, kind="geglu", split=2)
        if k == 1:
            free(bd, n)
            h4 = alloc(bd, lv, C, "pn")
        gemm(bd, g, w2[:, k * half:(k + 1) * half], b2 if k == 0 else None, name=q + f"ff2.{k}",
             out=h if k == 0 else h4, res=h, kind="bf16" if k == 0 else "bfp16")
        free(bd, g)
    free(bd, h)
    return h4


def transformer(bd: Builder, w: dict, p: str, x: Tensor, *, kv: int, ctx_kv: int, nctx: int,
                stats: tuple[int, int] | None, spill: bool) -> Tensor:
    """diffusers' Transformer2DModel of SD 1.x (one BasicTransformerBlock, 8 heads) on a level of one band: GN ->
    proj_in -> + attn1 (LN1) -> + attn2 (LN2; the context's K / V^T slots at DDR ctx_kv, nctx keys) -> + GEGLU
    feed-forward (LN3) -> proj_out + x. kv: a DDR buffer for the self-attention's slots; spill: x in DDR during the block
    (16x16: the memtile)."""
    lv, C = x.lv, x.C
    h, qt = _block_in(bd, w, p, x, kv=kv, extra=bd.spill(p + "x", x.addr, x.nbytes) if spill else None, free_x=spill)
    h4 = _block_out(bd, w, p, h, qt, kv=kv, ctx_kv=ctx_kv, nctx=nctx)
    if spill:
        x = alloc(bd, lv, C)
        bd.op(bd.load(p + "x", x.addr, x.nbytes), {})
    gemm(bd, h4, w[p + "proj_out.weight"], w[p + "proj_out.bias"], name=p + "proj_out", out=x, res=x, stats=stats)
    free(bd, h4)
    return x


@dataclass
class DTensor:
    """A tensor of a banded level in DDR: per band b its columns' memtile images ([col][band tensor bytes], DDR image
    ``band(b)``: ``Builder.spill`` / ``load``) of C tiles (kind "ct") or panels ("pn"); its GroupNorm records for the
    next norm (every band's)."""

    lv: Level
    C: int
    name: str
    kind: str = "ct"
    rec: list[Records] | None = None

    def band(self, b: int) -> str:
        return f"{self.name}.{b}"

    @property
    def nbytes(self) -> int:
        return self.lv.ct(self.C) if self.kind == "ct" else self.lv.pn(self.C)

    def load(self, bd: Builder, b: int, c0: int = 0, n: int | None = None) -> Tensor:
        """Band b in the memtile (an op of its fills); C tiles: its channels [c0, c0 + n) (whole 80-channel blocks)."""
        n = n or self.C - c0
        t = alloc(bd, self.lv, n, self.kind)
        off = self.lv.ct(c0) if c0 else 0
        assert self.kind == "ct" or not c0
        bd.reading(self.band(b))
        base = bd.npu.address(self.band(b))
        bd.op({c: ColOps(fills=[MemTask([{"addr": t.addr, "nwords": t.nbytes // 4}],
                                        ddr=(base + c * self.nbytes + off, t.nbytes, None))]) for c in range(COLS)}, {})
        t.rec = self.rec
        return t

    def store(self, bd: Builder, b: int, t: Tensor, *, defer: bool = True, c0: int = 0) -> None:
        """Band b (C tiles: its channels from c0) from the memtile tensor t (drained with the next op, t's bytes released
        after it; defer False: an op of its own, t freed); its records."""
        base = bd.ddr(self.band(b), COLS * self.nbytes)
        bd.written(self.band(b))
        off = self.lv.ct(c0) if c0 else 0
        cols = {c: ColOps(drains=[MemTask([{"addr": t.addr, "nwords": t.nbytes // 4}],
                                          ddr=(base + c * self.nbytes + off, t.nbytes, None))]) for c in range(COLS)}
        if defer:
            bd.defer(cols, release=[(t.addr, t.size or t.nbytes)])
        else:
            bd.op(cols, {})
            free(bd, t)
        self.rec = (self.rec or []) + (t.rec or [])


def transformer_bands(bd: Builder, w: dict, p: str, x: DTensor, *, kv: int, ctx_kv: int, nctx: int,
                      stats: tuple[int, int] | None, name: str) -> DTensor:
    """``transformer`` on a banded level (32x32 / 64x64: x and the output in DDR): every band's first part (its keys
    into the slots; h and Q parked in DDR, h with the K GEMM, Q with the next op; the last band's kept), then the
    bands' second parts (the last band first) and proj_out + x -> the output band (records for ``stats``)."""
    lv, C, nb = x.lv, x.C, x.lv.nb
    parked = DTensor(lv, C, name + ".h"), DTensor(lv, C, name + ".q", "pn")
    for b in range(nb):
        h, qt = _block_in(bd, w, p, x.load(bd, b), kv=kv, band=b, free_x=True,
                          park=parked[0].band(b) if b < nb - 1 else "")
        if b < nb - 1:
            parked[1].store(bd, b, qt)
    out = DTensor(lv, C, name)
    for b in reversed(range(nb)):
        if b < nb - 1:
            h, qt = alloc(bd, lv, C), alloc(bd, lv, C, "pn")
            bd.op(bd.merge(bd.load(parked[0].band(b), h.addr, h.nbytes), bd.load(parked[1].band(b), qt.addr, qt.nbytes)), {})
        h4 = _block_out(bd, w, p, h, qt, kv=kv, ctx_kv=ctx_kv, nctx=nctx)
        xb = x.load(bd, b)
        gemm(bd, h4, w[p + "proj_out.weight"], w[p + "proj_out.bias"], name=p + "proj_out", out=xb, res=xb, stats=stats,
             band=b)
        free(bd, h4)
        out.store(bd, b, xb, defer=False)
    return out


def band_view(t: Tensor, b: int) -> Tensor:
    """Band b of a whole column's panel tensor t (its token panels from b np; the kb stride t's)."""
    return Tensor(t.lv, t.C, t.addr + b * t.lv.np * L.PANEL, "pn", pnp=t.np)


def resnet_bands(bd: Builder, w: dict, p: str, x: DTensor, *, cout: int, time_bias: torch.Tensor, eps: float,
                 stats: tuple[int, int] | None, name: str) -> DTensor:
    """``resnet`` on a banded level (x and the output in DDR): GN1 + SiLU of each band into the whole column's panels
    (the convs' halo rows), conv1 (+ the time bias) per band with GN2's records (parked in DDR, the last band's kept),
    GN2 + SiLU of each band (the last first), conv2 + x (or its 1x1 shortcut) per band -> the output's band (records
    for ``stats``). A band's C tiles are a third of the memtile: an earlier band's output drained in an op of its own,
    before the next band's convs (barriers: those reuse its bytes)."""
    lv, nb = x.lv, x.lv.nb
    npw = lv.np * nb  # (a whole column's token panels)

    def norm(t: DTensor, wp: str, C: int, last: Tensor | None = None) -> Tensor:
        pw = alloc(bd, lv, C, "pn", pnp=npw)
        for b in reversed(range(nb)):
            tb = last if b == nb - 1 and last is not None else t.load(bd, b)
            tb.rec = t.rec  # (every band's)
            gn_apply(bd, tb, band_view(pw, b), w[wp + "weight"], w[wp + "bias"], name=wp[:-1], eps=eps, silu=True)
            free(bd, tb)
        return pw
    p1 = norm(x, p + "norm1.", x.C)
    h = DTensor(lv, cout, name + ".h")
    for b in range(nb):
        hb = alloc(bd, lv, cout)
        conv(bd, p1, w[p + "conv1.weight"], w[p + "conv1.bias"] + time_bias, name=p + "conv1", out=hb,
             stats=(cout // 32, 0), tbias=p, band=b)
        if b < nb - 1:
            h.store(bd, b, hb, defer=False)
    free(bd, p1)
    h.rec += hb.rec
    p2 = norm(h, p + "norm2.", cout, last=hb)
    out = DTensor(lv, cout, name)
    for b in range(nb):
        if p + "conv_shortcut.weight" in w:
            xb = x.load(bd, b)
            px = alloc(bd, lv, x.C, "pn", top=True)
            plain(bd, xb, px, name=p + "xpn")
            free(bd, xb)
            sc = alloc(bd, lv, cout)
            gemm(bd, px, w[p + "conv_shortcut.weight"], w[p + "conv_shortcut.bias"], name=p + "conv_shortcut", out=sc)
            free(bd, px)
        else:
            sc = x.load(bd, b)
        conv(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", out=sc, res=sc, stats=stats, band=b)
        out.store(bd, b, sc, defer=False)
    free(bd, p2)
    return out


def downsample_bands(bd: Builder, w: dict, p: str, x: DTensor) -> Tensor:
    """The stride-2 conv of a banded level's x (32x32) into the next level's C tiles (16x16, one band): x's panels of
    the whole column, the conv's even output rows (row tiles 0 and 2: a column's 16x16 rows) of every x, of which the
    even ones (groups 0 and 2: the next level's groups 0 and 1) are output (``ops.Gemm`` "ds")."""
    lv, npw = x.lv, x.lv.np * x.lv.nb
    N = w[p + "conv.weight"].shape[0]
    y = alloc(bd, Level(lv.H // 2), N)  # (first: the next level's tensors start at the memtile's bottom)
    px = alloc(bd, lv, x.C, "pn", pnp=npw)
    for b in range(lv.nb):
        xb = x.load(bd, b)
        plain(bd, xb, band_view(px, b), name=p + "xpn")
        free(bd, xb)
    g = Gemm(m=64, ntile=lv.R // 2, K=x.C, N=N, x=px.addr, y=y.addr, out="ds", conv=1, R=1, Lg=lv.Lg, zero=bd.zero,
             x_np=npw, rc=lv.R, rstep=2)
    bd.gemm(g, w[p + "conv.weight"], w[p + "conv.bias"], name=p + "conv")
    free(bd, px)
    return y


def upsample_bands(bd: Builder, w: dict, p: str, x: Tensor, *, stats: tuple[int, int] | None, name: str) -> DTensor:
    """Nearest 2x of the C tiles x (16x16, freed) into a banded level (32x32), then the conv: x's panels via DDR into the
    whole column's panels of the level (a 16x16 column's 2 rows are the 32x32 column's 4: row 2 r + f, group 2 g + e
    of panel (r, g): 4 fills), in K parts of 640 channels; per band the conv's halves of 640 outputs (+ the earlier
    part's partial sums, parked in DDR) -> the output's band (records for ``stats`` in the last part)."""
    lv = Level(2 * x.lv.H, 2)
    npw = lv.np * lv.nb
    px = alloc(bd, x.lv, x.C, "pn")
    plain(bd, x, px, name=p + "xpn")
    free(bd, x)
    nb = px.nbytes
    bd.op(bd.spill(name + ".up", px.addr, nb), {})
    bd.ddr_sync()
    free(bd, px)
    src = bd.npu.address(name + ".up")
    wc, bc = w[p + "conv.weight"], w[p + "conv.bias"]
    N = wc.shape[0]
    out = DTensor(lv, N, name)
    for k, c0 in enumerate(range(0, x.C, 640)):
        n = min(640, x.C - c0)
        pw = alloc(bd, lv, n, "pn", pnp=npw)
        pb = n // 8 * x.lv.np * L.PANEL  # (the part's panels per 16x16 column)
        bd.op({c: ColOps(fills=[MemTask([{"addr": pw.addr + (8 * f + 2 * e) * L.PANEL, "nwords": pb // 4,
                                          "dims": [(36, 1), (2, 72), (2, 288), (0, npw * 18)]}],
                                        ddr=(src + c * nb + c0 // 8 * x.lv.np * L.PANEL, pb, None))
                                for f in range(2) for e in range(2)]) for c in range(COLS)}, {})
        last = c0 + n == x.C
        for b in range(lv.nb):
            for o0 in range(0, N, 640):
                cb = out.load(bd, b, o0, 640) if k else alloc(bd, lv, 640)
                conv(bd, pw, wc[o0:o0 + 640, c0:c0 + n], bc[o0:o0 + 640] if k == 0 else None, name=f"{p}conv.{k}.{o0}",
                     out=cb, res=cb if k else None, stats=(stats[0], stats[1] + o0) if stats and last else None, band=b)
                out.store(bd, b, cb, defer=False, c0=o0)
        free(bd, pw)
    return out


def skip_bands(bd: Builder, x: DTensor, *, name: str, st: tuple[int, int]) -> DTensor:
    """A banded level's tensor as a skip: x with GroupNorm records in its concatenation's group space (st = (group size,
    channel offset)): a copy op per band."""
    out = DTensor(x.lv, x.C, x.name, rec=[])
    for c0 in range(0, x.C, 640):  # (in parts of 640 channels: a band's C tiles and their panels in the memtile)
        n = min(640, x.C - c0)
        for b in range(x.lv.nb):
            t = x.load(bd, b, c0, n)
            out.rec += copy_stats(bd, t, name=f"{name}.st{c0}", st=(st[0], st[1] + c0), band=b)
            free(bd, t)
    return out


def resnet_cat_bands(bd: Builder, w: dict, p: str, h: DTensor, skip: DTensor, *, cout: int, time_bias: torch.Tensor,
                     eps: float, stats: tuple[int, int] | None, name: str) -> DTensor:
    """``resnet_cat`` on a banded level (h, the skip and the output in DDR; their records in the concatenation's group
    space): conv1 and the 1x1 shortcut over K parts of at most 640 channels (h's, the skip's: a part's whole-column
    panels fit beside a band's C tiles), their partial sums (bf16 C tiles) parked in DDR per band; GN1 + SiLU of each
    part's channels, conv1's parts (+ the time bias, GN2's records in the last), the shortcut's parts per band, GN2 +
    SiLU, conv2 + the shortcut per band -> the output's band (records for ``stats``)."""
    lv, nb, ch, cs = h.lv, h.lv.nb, h.C, skip.C
    C, npw = ch + cs, h.lv.np * h.lv.nb
    recs = (h.rec or []) + (skip.rec or [])
    g1, b1, w1, wsc = w[p + "norm1.weight"], w[p + "norm1.bias"], w[p + "conv1.weight"], w[p + "conv_shortcut.weight"]
    parts = [(h, c0, min(640, ch - c0), c0) for c0 in range(0, ch, 640)] + \
        [(skip, c0, min(640, cs - c0), ch + c0) for c0 in range(0, cs, 640)]  # (source, its channel, channels, in C)
    c1 = DTensor(lv, cout, name + ".c1")
    for k, (src, c0, n, cc) in enumerate(parts):
        pw = alloc(bd, lv, n, "pn", pnp=npw)
        for b in reversed(range(nb)):
            xb = src.load(bd, b, c0, n)
            xb.rec = recs
            gn_apply(bd, xb, band_view(pw, b), g1[cc:cc + n], b1[cc:cc + n], name=f"{p}norm1.{k}", eps=eps, silu=True,
                     ch0=cc, ccat=C)
            free(bd, xb)
        for b in range(nb):
            cb = c1.load(bd, b) if k else alloc(bd, lv, cout)
            first, last = k == 0, k == len(parts) - 1
            conv(bd, pw, w1[:, cc:cc + n], w[p + "conv1.bias"] + time_bias if first else None, name=f"{p}conv1.{k}", out=cb,
                 res=None if first else cb, stats=(cout // 32, 0) if last else None, tbias=p if first else None, band=b)
            c1.store(bd, b, cb, defer=False)
        free(bd, pw)
    sc = DTensor(lv, cout, name + ".sc")
    for b in range(nb):
        sb = alloc(bd, lv, cout)
        for k, (src, c0, n, cc) in enumerate(parts):
            xb = src.load(bd, b, c0, n)
            px = alloc(bd, lv, n, "pn", top=True)
            plain(bd, xb, px, name=f"{p}xpn.{k}")
            free(bd, xb)
            gemm(bd, px, wsc[:, cc:cc + n].flatten(1), w[p + "conv_shortcut.bias"] if k == 0 else None,
                 name=f"{p}conv_shortcut.{k}", out=sb, res=sb if k else None)
            free(bd, px)
        sc.store(bd, b, sb, defer=False)
    p2 = alloc(bd, lv, cout, "pn", pnp=npw)
    for b in reversed(range(nb)):
        cb = c1.load(bd, b)
        gn_apply(bd, cb, band_view(p2, b), w[p + "norm2.weight"], w[p + "norm2.bias"], name=p + "norm2", eps=eps, silu=True)
        free(bd, cb)
    out = DTensor(lv, cout, name)
    for b in range(nb):
        sb = sc.load(bd, b)
        conv(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", out=sb, res=sb, stats=stats, band=b)
        out.store(bd, b, sb, defer=False)
    free(bd, p2)
    return out


@dataclass
class Skip:
    """A skip connection for the up path: parked in DDR (its C tiles: image ``ct``, its plain panels: ``pn``, the up
    path's shortcut input, and records in the up path's concatenation's group space), or resident (``ps``: the plain
    panels, ``pb``: the concatenation's GroupNorm + SiLU of the skip's channels, applied when the skip is made: no group
    straddles the concatenation, so its records suffice)."""

    lv: Level
    C: int
    ct: str = ""
    pn: str = ""
    rec: list[Records] | None = None
    ps: Tensor | None = None
    pb: Tensor | None = None


def make_skip(bd: Builder, x: Tensor, *, name: str, st: tuple[int, int], norm: tuple | None = None,
              defer: bool = True) -> Skip:
    """The C tiles x (kept in the memtile) as a skip: st = (group size, channel offset) of its concatenation. norm =
    (gamma, beta, eps) of the concatenation's channels of the skip: a resident skip (``Skip.pb``). Else parked in DDR:
    x's C tiles drained with the copy, its panels with the next op (defer: at the heap's top, so that their release
    leaves no hole between the next ops' tensors; else an op of their own: the next op's outputs need their bytes)."""
    pn = alloc(bd, x.lv, x.C, "pn", top=norm is None and defer)
    rec = copy_stats(bd, x, name=name + ".st", st=st, panels=pn,
                     extra=bd.spill(name + ".ct", x.addr, x.nbytes) if norm is None else None)
    if norm is not None:
        assert st[1] % st[0] == 0, "a group straddles the concatenation"
        g, b, eps = norm
        pb = alloc(bd, x.lv, x.C, "pn")
        gn_apply(bd, Tensor(x.lv, x.C, x.addr, rec=rec), pb, g, b, name=name + ".norm", eps=eps, silu=True, ch0=st[1],
                 ccat=st[1] + x.C)
        return Skip(x.lv, x.C, ps=pn, pb=pb)
    if defer:
        bd.defer(bd.spill(name + ".pn", pn.addr, pn.nbytes), release=[(pn.addr, pn.nbytes)])
    else:
        bd.op(bd.spill(name + ".pn", pn.addr, pn.nbytes), {})
        free(bd, pn)
    return Skip(x.lv, x.C, name + ".ct", name + ".pn", rec)


def resnet_cat(bd: Builder, w: dict, p: str, h: Tensor, skip: Skip, *, cout: int, time_bias: torch.Tensor,
               eps: float, stats: tuple[int, int] | None) -> Tensor:
    """A resnet of the up path on the concatenation [h | skip] (h: C tiles with records in the concatenation's group
    space): conv1 and the 1x1 shortcut in two K-chunks (h's, the skip's), the partials in the residual stream; ordered
    so that 16x16 fits the memtile (conv1's partial parked in DDR, drained with the next op; the skip's panels loaded
    with the shortcut's first GEMM; at 8x8 conv1's partial and a resident skip stay)."""
    lv, ch, cs = h.lv, h.C, skip.C
    C = ch + cs
    park = lv.H > 8
    w1, wsc = w[p + "conv1.weight"], w[p + "conv_shortcut.weight"]
    g1, b1 = w[p + "norm1.weight"], w[p + "norm1.bias"]
    recs = (h.rec or []) + (skip.rec or [])
    h.rec = recs
    pa = alloc(bd, lv, ch, "pn")
    gn_apply(bd, h, pa, g1[:ch], b1[:ch], name=p + "norm1.h", eps=eps, silu=True, ccat=C)
    c1 = alloc(bd, lv, cout)
    conv(bd, pa, w1[:, :ch], w[p + "conv1.bias"] + time_bias, name=p + "conv1.h", out=c1, tbias=p)
    free(bd, pa)
    if park:
        bd.defer(bd.spill(p + "c1", c1.addr, c1.nbytes), release=[(c1.addr, c1.nbytes)])
    ph = alloc(bd, lv, ch, "pn")
    plain(bd, h, ph, name=p + "hpn")
    free(bd, h)
    sc = alloc(bd, lv, cout)
    ps, fill = skip.ps, None
    if ps is None:
        ps = alloc(bd, lv, cs, "pn")
        fill = bd.load(skip.pn, ps.addr, ps.nbytes)
    gemm(bd, ph, wsc[:, :ch], w[p + "conv_shortcut.bias"], name=p + "conv_shortcut.h", out=sc, extra=fill)
    free(bd, ph)
    gemm(bd, ps, wsc[:, ch:], None, name=p + "conv_shortcut.s", out=sc, res=sc)
    free(bd, ps)
    pb = skip.pb
    if pb is None:
        sk = alloc(bd, lv, cs)
        sk.rec = recs
        bd.op(bd.load(skip.ct, sk.addr, sk.nbytes), {})
        pb = alloc(bd, lv, cs, "pn")
        gn_apply(bd, sk, pb, g1[ch:], b1[ch:], name=p + "norm1.s", eps=eps, silu=True, ch0=ch, ccat=C)
        free(bd, sk)
    if park:
        c1 = alloc(bd, lv, cout)
        bd.op(bd.load(p + "c1", c1.addr, c1.nbytes), {})
    conv(bd, pb, w1[:, ch:], None, name=p + "conv1.s", out=c1, res=c1, stats=(cout // 32, 0))
    free(bd, pb)
    p2 = alloc(bd, lv, cout, "pn")
    gn_apply(bd, c1, p2, w[p + "norm2.weight"], w[p + "norm2.bias"], name=p + "norm2", eps=eps, silu=True)
    free(bd, c1)
    conv(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", out=sc, res=sc, stats=stats)
    free(bd, p2)
    return sc


def downsample(bd: Builder, w: dict, p: str, x: Tensor, *, stats: tuple[int, int] | None) -> Tensor:
    """The stride-2 conv of the C tiles x (16x16) to the next level's C tiles (x kept)."""
    px = alloc(bd, x.lv, x.C, "pn")
    plain(bd, x, px, name=p + "xpn")
    y = alloc(bd, Level(x.lv.H // 2), w[p + "conv.weight"].shape[0])
    conv(bd, px, w[p + "conv.weight"], w[p + "conv.bias"], name=p + "conv", out=y, stride=2, stats=stats)
    free(bd, px)
    return y


def upsample(bd: Builder, w: dict, p: str, x: Tensor, *, stats: tuple[int, int] | None) -> Tensor:
    """Nearest 2x of the C tiles x, then the conv (x freed): the upsampled panels via DDR (16x16 column c's rows are
    8x8 row c: each panel (row, group, image) is the 8x8 panel (image), x = g + 2 lane), the conv on the next level."""
    lv2 = Level(2 * x.lv.H)
    px = alloc(bd, x.lv, x.C, "pn")
    plain(bd, x, px, name=p + "xpn")
    free(bd, x)
    nb = px.nbytes
    bd.op(bd.spill(p + "up", px.addr, nb), {})
    bd.ddr_sync()
    free(bd, px)
    u = alloc(bd, lv2, x.C, "pn")
    base = bd.npu.address(p + "up")
    cols = {c: ColOps(fills=[MemTask([{"addr": u.addr + (2 * r + g) * 2 * L.PANEL, "nwords": nb // 4,
                                       "dims": [(36, 1), (x.C // 8, lv2.np * 18)]}], ddr=(base + c * nb, nb, None))
                             for r in range(2) for g in range(2)]) for c in range(COLS)}
    bd.op(cols, {})
    y = alloc(bd, lv2, w[p + "conv.weight"].shape[0])
    conv(bd, u, w[p + "conv.weight"], w[p + "conv.bias"], name=p + "conv", out=y, stats=stats)
    free(bd, u)
    return y


CTX_KEYS = 77  # (the text context's tokens)


def section(bd: Builder, w: dict, x6: Tensor, *, biases: dict[str, torch.Tensor], ctx_kv: dict[str, int], kv: int,
            eps: float) -> Tensor:
    """The UNet's 16x16 / 8x8 levels: down_blocks.2 (2 resnets + transformers, the downsampler), down_blocks.3 (2
    resnets), the mid block, up_blocks.0 (3 resnets on concatenations, the upsampler), up_blocks.1 (3 resnets on
    concatenations + transformers), from x6 (16x16 C tiles, 640 channels: down_blocks.1's downsampler output) to
    up_blocks.1's output (16x16 C tiles, 1280). biases: the resnets' time biases; ctx_kv: per transformer prefix the
    DDR address of its context's K / V^T slots; kv: a DDR buffer for the self-attention slots (16x16's size)."""

    def res(p: str, h: Tensor, st: tuple[int, int] | None) -> Tensor:
        return resnet(bd, w, p, h, cout=1280, time_bias=biases[p], eps=eps, stats=st)

    def cat(p: str, h: Tensor, s: Skip, st: tuple[int, int] | None) -> Tensor:
        return resnet_cat(bd, w, p, h, s, cout=1280, time_bias=biases[p], eps=eps, stats=st)

    def tr(p: str, h: Tensor, st: tuple[int, int] | None) -> Tensor:
        return transformer(bd, w, p, h, kv=kv, ctx_kv=ctx_kv[p], nctx=CTX_KEYS, stats=st, spill=h.lv.H == 16)

    x6.rec = copy_stats(bd, x6, name="x6.st", st=(20, 0))
    s6 = make_skip(bd, x6, name="s6", st=(60, 1280), defer=False)
    d2 = "down_blocks.2."
    x = res(d2 + "resnets.0.", x6, (40, 0))
    x = tr(d2 + "attentions.0.", x, (40, 0))
    s7 = make_skip(bd, x, name="s7", st=(80, 1280))
    x = res(d2 + "resnets.1.", x, (40, 0))
    x = tr(d2 + "attentions.1.", x, None)
    s8 = make_skip(bd, x, name="s8", st=(80, 1280))
    y = downsample(bd, w, d2 + "downsamplers.0.", x, stats=(40, 0))
    free(bd, x)
    u0 = "up_blocks.0."

    def norm(j: int) -> tuple:  # (the skip channels' norm1 of up_blocks.0.resnets.j)
        q = f"{u0}resnets.{j}.norm1."
        return w[q + "weight"][1280:], w[q + "bias"][1280:], eps
    s9 = make_skip(bd, y, name="s9", st=(80, 1280), norm=norm(2))
    x = res("down_blocks.3.resnets.0.", y, (40, 0))
    s10 = make_skip(bd, x, name="s10", st=(80, 1280), norm=norm(1))
    x = res("down_blocks.3.resnets.1.", x, (40, 0))
    s11 = make_skip(bd, x, name="s11", st=(80, 1280), norm=norm(0))
    x = res("mid_block.resnets.0.", x, (40, 0))
    x = tr("mid_block.attentions.0.", x, (40, 0))
    x = res("mid_block.resnets.1.", x, (80, 0))
    x = cat(u0 + "resnets.0.", x, s11, (80, 0))
    x = cat(u0 + "resnets.1.", x, s10, (80, 0))
    x = cat(u0 + "resnets.2.", x, s9, None)
    x = upsample(bd, w, u0 + "upsamplers.0.", x, stats=(80, 0))
    u1 = "up_blocks.1."
    x = cat(u1 + "resnets.0.", x, s8, (40, 0))
    x = tr(u1 + "attentions.0.", x, (80, 0))
    x = cat(u1 + "resnets.1.", x, s7, (40, 0))
    x = tr(u1 + "attentions.1.", x, (60, 0))
    x = cat(u1 + "resnets.2.", x, s6, (40, 0))
    return tr(u1 + "attentions.2.", x, None)


def outer(bd: Builder, w: dict, x5: DTensor, s3: DTensor, *, biases: dict[str, torch.Tensor], ctx_kv: dict[str, int],
          kv: int, kv32: int, eps: float) -> DTensor:
    """The UNet from down_blocks.1 to up_blocks.2 (the 32x32 level banded, the 16x16 / 8x8 levels by ``section``): x5
    (32x32, 320 channels: down_blocks.0's downsampler output, records for down_blocks.1's first GroupNorm) and s3 (the
    same, records in up_blocks.2.resnets.2's concatenation) -> up_blocks.2's output (32x32, 640). kv32: a DDR buffer for
    the 32x32 self-attention's slots (kv: the 16x16 ones)."""
    d1, u2 = "down_blocks.1.", "up_blocks.2."

    def res(p: str, x: DTensor, st: tuple[int, int] | None) -> DTensor:
        return resnet_bands(bd, w, p, x, cout=640, time_bias=biases[p], eps=eps, stats=st, name=p[:-1])

    def tr(p: str, x: DTensor, st: tuple[int, int] | None) -> DTensor:
        return transformer_bands(bd, w, p, x, kv=kv32, ctx_kv=ctx_kv[p], nctx=CTX_KEYS, stats=st, name=p[:-1])

    def cat(p: str, h: DTensor, s: DTensor, st: tuple[int, int] | None) -> DTensor:
        return resnet_cat_bands(bd, w, p, h, s, cout=640, time_bias=biases[p], eps=eps, stats=st, name=p[:-1])
    x = tr(d1 + "attentions.0.", res(d1 + "resnets.0.", x5, (20, 0)), (20, 0))
    s4 = skip_bands(bd, x, name="s4", st=(40, 640))
    x = tr(d1 + "attentions.1.", res(d1 + "resnets.1.", x, (20, 0)), None)
    s5 = skip_bands(bd, x, name="s5", st=(60, 1280))
    y = section(bd, w, downsample_bands(bd, w, d1 + "downsamplers.0.", x), biases=biases, ctx_kv=ctx_kv, kv=kv, eps=eps)
    x = upsample_bands(bd, w, "up_blocks.1.upsamplers.0.", y, stats=(60, 0), name="up_blocks.1.up")
    x = tr(u2 + "attentions.0.", cat(u2 + "resnets.0.", x, s5, (20, 0)), (40, 0))
    x = tr(u2 + "attentions.1.", cat(u2 + "resnets.1.", x, s4, (20, 0)), (30, 0))
    return tr(u2 + "attentions.2.", cat(u2 + "resnets.2.", x, s3, (20, 0)), None)
