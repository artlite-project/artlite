"""The layers of Stable Diffusion's UNet and VAE decoder on a ``builder.Builder`` (the shared tensors and ``gemm``:
``artlite.vision.layers``): ``conv`` (3x3), ``plain`` (C tiles as panels), ``copy_stats`` (the GroupNorm records of C tiles),
the norms' applies into panels (``gn_apply``, ``ln_apply``) and ``attend`` (8 heads on K / V^T slots in DDR)."""

import numpy as np

from artlite.operators.design import COLS, ROWS
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.builder import Builder, Records
from artlite.vision.layers import Tensor, alloc, ln_rec, silu_table
from artlite.vision.ops import Apply, Attention, ConvSpec, Gemm


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


def gn_apply(bd: Builder, x: Tensor, y: Tensor, gamma: np.ndarray, beta: np.ndarray, *, name: str, eps: float,
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


def conv(bd: Builder, x: Tensor, w: np.ndarray, b: np.ndarray, *, name: str, out: Tensor, stride: int = 1,
         res: Tensor | None = None, stats: tuple[int, int] | None = None, tbias: str | None = None,
         band: int | None = None) -> None:
    """A 3x3 conv of the panels x into the C tiles out (+ res), out's records for ``stats``; tbias: the resnet whose
    time bias b includes (``Builder.gemm``: rewritten per step). band: of a banded level, x the whole column's panels,
    out band ``band``'s (row tiles: ``ops.Gemm`` rc)."""
    lv = out.lv
    N = w.shape[0]
    kw = {"m": lv.m, "ntile": 1, "K": x.C, "N": N, "x": x.addr, "y": out.addr, "out": "bf16", "x_np": x.np}
    cv = {"stride": stride, "R": lv.R, "Lg": lv.Lg, "zero": bd.zero}
    if band is not None:
        kw.update(ntile=lv.ntile)
        cv.update(R=1, rc=lv.R, r0=band * lv.R // lv.nb)
    rec = bd.mem.alloc(4 * D.OCH, top=True) if stats else None
    if stats:
        kw.update(stats="gn", rec=rec, gsize=stats[0], ch_off=stats[1])
    bd.gemm(Gemm(**kw, conv=ConvSpec(**cv), res=res.addr if res is not None else None), w, b, name=name, tbias=tbias)
    if stats:
        out.rec = [bd.records(rec, name=f"{name}.rec", ch_off=stats[1], nr=N // 4, gsize=stats[0], band=band or 0,
                              nbands=lv.nb)]


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


def ln_apply(bd: Builder, h: Tensor, y: Tensor, gamma: np.ndarray, beta: np.ndarray, *, name: str, rec: int) -> None:
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


def band_view(t: Tensor, b: int) -> Tensor:
    """Band b of a whole column's panel tensor t (its token panels from b np; the kb stride t's)."""
    return Tensor(t.lv, t.C, t.addr + b * t.lv.np * L.PANEL, "pn", pnp=t.np)
