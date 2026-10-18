"""sd15_v2's 64x64 level (``build.Level(64, 4)``): bands of 2 rows per column, a row's tokens as two tiles (its even
groups, its odd: ``ops.Gemm`` il). The residual stream and the blocks' C tiles in DDR per band (``unet.DTensor``); a
conv's input panels in DDR as the whole level's (``PImage``), from which each band's conv fills its rows and their halo
rows (the neighbour bands' and columns', zeros past the image)."""

from dataclasses import dataclass

import torch

from ..design import COLS
from . import design as D
from . import layout as L
from .build import Builder, Level
from .ops import Gemm
from .seq import ColOps, MemTask
from .unet import DTensor, Tensor, alloc, free, gn_apply

LV64 = Level(64, 4)
ROW = 16 * L.PANEL  # (a row's token panels per kb, bytes)


@dataclass
class PImage:
    """A 64x64 level's panels in DDR: per column [kb][row 8][16 token panels] (image ``name``)."""

    C: int
    name: str
    lv: Level = LV64

    @property
    def colbytes(self) -> int:
        return self.C // 8 * self.lv.R * ROW

    def store(self, bd: Builder, b: int, t: Tensor) -> None:
        """Band b's panels t ([kb][its 2 rows' token panels]) into its rows, drained with the next op (t's bytes
        released after it)."""
        base = bd.ddr(self.name, COLS * self.colbytes)
        bd.written(self.name)
        nkb, rb = self.C // 8, self.lv.R // self.lv.nb  # (rb: a band's rows)
        cols = {c: ColOps(drains=[MemTask([{"addr": t.addr, "nwords": t.nbytes // 4}],
                                          ddr=(base + c * self.colbytes + rb * b * ROW, t.nbytes,
                                               (rb * ROW // 4, nkb, self.lv.R * ROW // 4, rb * ROW // 4)))])
                for c in range(COLS)}
        bd.defer(cols, release=[(t.addr, t.size or t.nbytes)])

    def rows(self, bd: Builder, b: int, c0: int = 0, n: int | None = None) -> Tensor:
        """Band b's conv input: channels [c0, c0 + n) of rows rb b - 1 .. rb b + rb of every column (the row above
        from the band above or the west column's last, below from the band below or the east column's first, zeros
        past the image) as memtile panels [kb][rb + 2 rows' token panels] (an op of its fills)."""
        n = n or self.C - c0
        R, rb, nkb = self.lv.R, self.lv.R // self.lv.nb, n // 8
        x = alloc(bd, self.lv, n, "pn", pnp=16 * (rb + 2))
        bd.reading(self.name)
        base = bd.v2.address(self.name) + c0 // 8 * R * ROW
        zero = bd.zeros(nkb * ROW)
        kbx = (rb + 2) * ROW // 4  # (words: x's kb)

        def fill(at: int, nr: int, src: int, stride: int) -> MemTask:  # (nr rows to x's row at, per kb from src)
            r = ROW // 4
            return MemTask([{"addr": x.addr + at * ROW, "nwords": nkb * nr * r, "dims": [(r, 1), (nr, r), (nkb, kbx)]}],
                           ddr=(src, nkb * nr * ROW, (r, nr, r, stride)))
        kbs = R * ROW // 4  # (words: the image's kb)
        cols = {}
        for c in range(COLS):
            col = base + c * self.colbytes
            lo, hi = rb * b - 1, rb * b + rb + 1  # (the rows [lo, hi) of the column)
            fills = []
            if lo < 0:
                fills.append(fill(0, 1, zero, ROW // 4) if c == 0 else fill(0, 1, col - self.colbytes + (R - 1) * ROW, kbs))
            if hi > R:
                fills.append(fill(rb + 1, 1, zero, ROW // 4) if c == COLS - 1 else
                             fill(rb + 1, 1, col + self.colbytes, kbs))
            a, z = max(lo, 0), min(hi, R)
            fills.append(fill(a - lo, z - a, col + a * ROW, kbs))
            cols[c] = ColOps(fills=fills)
        bd.op(cols, {})
        return x


def conv64(bd: Builder, img: PImage, w: torch.Tensor, b: torch.Tensor | None, *, name: str, band: int, out: Tensor,
           c0: int = 0, res: Tensor | None = None, stats: tuple[int, int] | None = None, tbias: str | None = None) -> None:
    """A 3x3 conv (padding 1) of img's channels [c0, c0 + K) (w: (N, K, 3, 3), N <= 320: out the band's C tiles) for band
    ``band`` (+ res; out's GroupNorm records for ``stats``; tbias: ``unet.conv``'s)."""
    lv = img.lv
    x = img.rows(bd, band, c0, w.shape[1])
    kw = {"m": lv.m, "ntile": lv.ntile, "K": w.shape[1], "N": w.shape[0], "x": x.addr, "y": out.addr, "out": "bf16",
          "conv": 1, "R": 1, "Lg": lv.Lg, "zero": bd.zero, "x_np": x.np, "rc": lv.R // lv.nb + 2, "r0": 1, "il": True,
          "res": res.addr if res is not None else None}
    rec = bd.mem.alloc(4 * D.OCH, top=True) if stats else None
    if stats:
        kw.update(stats="gn", rec=rec, gsize=stats[0], ch_off=stats[1])
    bd.gemm(Gemm(**kw), w, b, name=name, tbias=tbias)
    free(bd, x)
    if stats:
        out.rec = [bd.records(rec, name=f"{name}.rec", ch_off=stats[1], nr=w.shape[0] // 4, gsize=stats[0], band=band,
                              nbands=lv.nb)]


def norm64(bd: Builder, x: DTensor, wp: str, w: dict, *, name: str, eps: float, c0: int = 0, n: int | None = None,
           ccat: int = 0, ch0: int = 0) -> PImage:
    """GroupNorm + SiLU of x's channels [c0, c0 + n) (records: x's, in a concatenation of ccat channels from ch0) into
    a PImage, band by band."""
    n = n or x.C - c0
    img = PImage(n, name)
    g, bt = w[wp + "weight"][ch0:ch0 + n], w[wp + "bias"][ch0:ch0 + n]
    for band in range(x.lv.nb):
        xb = x.load(bd, band, c0, n)
        xb.rec = x.rec
        t = alloc(bd, x.lv, n, "pn")
        gn_apply(bd, xb, t, g, bt, name=wp[:-1], eps=eps, silu=True, ch0=ch0, ccat=ccat)
        free(bd, xb)
        img.store(bd, band, t)
    return img


def resnet64(bd: Builder, w: dict, p: str, x: DTensor, *, cout: int, time_bias: torch.Tensor, eps: float,
             stats: tuple[int, int] | None, name: str) -> DTensor:
    """``unet.resnet`` on the 64x64 level (x and the output in DDR, 320 channels): GN1 + SiLU into a PImage, conv1 (+
    the time bias) per band with GN2's records, GN2 + SiLU into a PImage, conv2 + x per band -> the output's band
    (records for ``stats``)."""
    assert x.C == cout == 320
    p1 = norm64(bd, x, p + "norm1.", w, name=name + ".p1", eps=eps)
    h = DTensor(x.lv, cout, name + ".h", rec=[])
    for b in range(x.lv.nb):
        hb = alloc(bd, x.lv, cout)
        conv64(bd, p1, w[p + "conv1.weight"], w[p + "conv1.bias"] + time_bias, name=p + "conv1", band=b, out=hb,
               stats=(cout // 32, 0), tbias=p)
        h.store(bd, b, hb, defer=False)
    p2 = norm64(bd, h, p + "norm2.", w, name=name + ".p2", eps=eps)
    out = DTensor(x.lv, cout, name)
    for b in range(x.lv.nb):
        sc = x.load(bd, b)
        conv64(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", band=b, out=sc, res=sc, stats=stats)
        out.store(bd, b, sc, defer=False)
    return out
