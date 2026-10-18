"""sd15_v2's 64x64 level (``build.Level(64, 4)``): bands of 2 rows per column, a row's tokens as two tiles (its even
groups, its odd: ``ops.Gemm`` il). The residual stream and the blocks' C tiles in DDR per band (``unet.DTensor``); a
conv's input panels in DDR as the whole level's (``PImage``), from which each band's conv fills its rows and their halo
rows (the neighbour bands' and columns', zeros past the image)."""

from dataclasses import dataclass

import numpy as np
import torch

from ..design import COLS
from . import design as D
from . import layout as L
from .build import Builder, Level
from .ops import Gemm
from .seq import ColOps, MemTask
from .unet import DTensor, Tensor, alloc, free, gemm, gn_apply, plain

LV64 = Level(64, 4)
ROW = 16 * L.PANEL  # (a row's token panels per kb, bytes)
IL = (0, 2, 4, 6, 1, 3, 5, 7)  # (a row's groups in their order: even, then odd)
OUT = 4 * 8 * 128  # (conv_out's bytes per band and column: [tile][row block][64] of output channels 0..7)


def tokens(x: torch.Tensor) -> torch.Tensor:
    """(2 images, C, 64, 64) -> per column its tokens (COLS, 1024, C): column c's image rows 8 c .., token panel (row,
    group IL[k], image), lane j = pixel x = g + 8 j."""
    C = x.shape[1]
    v = x.reshape(2, C, COLS, 8, 8, 8)[..., list(IL)]  # i C c row j g
    return v.permute(2, 3, 5, 0, 4, 1).reshape(COLS, 1024, C)


def from_tokens(y: torch.Tensor) -> torch.Tensor:
    C = y.shape[-1]
    v = y.reshape(COLS, 8, 8, 2, 8, C).permute(3, 5, 0, 1, 4, 2)  # i C c row j k
    return v[..., [IL.index(g) for g in range(8)]].reshape(2, C, 64, 64)


def latent_panels(lat: torch.Tensor) -> list[np.ndarray]:
    """The latents (2, 4, 64, 64) as conv_in's PImage's first kb per column (its 8 channels: the 4 latent ones, zeros;
    the image's other kbs zeros)."""
    x = torch.zeros(2, 8, 64, 64)
    x[:, :lat.shape[1]] = lat
    t = tokens(x)
    return [L.panels(t[c]) for c in range(COLS)]


def noise(raw: np.ndarray) -> torch.Tensor:
    """conv_out's output image ([band][col][OUT]) -> (2, 4, 64, 64)."""
    ct = L.ct_bytes(64)
    r = raw.reshape(LV64.nb, COLS, 4, 8, 128)
    y = torch.zeros(COLS, 1024, 4)
    for b in range(LV64.nb):
        for c in range(COLS):
            buf = np.zeros((4, ct), np.uint8)
            buf[:, :8 * 1280].reshape(4, 8, 1280)[:, :, :128] = r[b, c]
            y[c, 256 * b:256 * (b + 1)] = L.from_ctiles(buf.reshape(-1), 256, 80, 64)[:, :4]
    return from_tokens(y)


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

    def store_up(self, bd: Builder, b: int, t: Tensor) -> None:
        """Nearest 2x of band b of a 32x32 level's panels t ([kb][its 2 rows of 8 token panels]): its row j as rows
        2 (2 b + j) + f, each its even and its odd groups (``ops.Gemm`` il: 32x32 group g is 64x64 groups 2 g, 2 g + 1),
        drained with the next op (t's bytes released after it)."""
        base = bd.ddr(self.name, COLS * self.colbytes)
        bd.written(self.name)
        nkb, half = self.C // 8, ROW // 2
        cols = {c: ColOps(drains=[MemTask([{"addr": t.addr + j * half, "nwords": nkb * half // 4,
                                            "dims": [(half // 4, 1), (nkb, ROW // 4)]}],
                                          ddr=(base + c * self.colbytes + (4 * b + 2 * j + f) * ROW + e * half, nkb * half,
                                               (half // 4, nkb, self.lv.R * ROW // 4, half // 4)))
                                  for j in range(2) for f in range(2) for e in range(2)]) for c in range(COLS)}
        bd.defer(cols, release=[(t.addr, t.size or t.nbytes)])

    def rows(self, bd: Builder, b: int, c0: int = 0, n: int | None = None) -> Tensor:
        """Band b's conv input: channels [c0, c0 + n) of rows rb b - 1 .. rb b + rb (``span``)."""
        rb = self.lv.R // self.lv.nb
        return self.span(bd, rb * b - 1, rb * b + rb + 1, c0, n)

    def span(self, bd: Builder, lo: int, hi: int, c0: int = 0, n: int | None = None) -> Tensor:
        """Channels [c0, c0 + n) of rows [lo, hi) of every column (rows past the column's: the west column's last, the
        east column's first; zeros past the image) as memtile panels [kb][hi - lo rows' token panels] (an op of its
        fills)."""
        n = n or self.C - c0
        R, nkb = self.lv.R, n // 8
        assert -1 <= lo and hi <= R + 1
        bd.reading(self.name)  # (first: the deferred drains pushed, their bytes released)
        x = alloc(bd, self.lv, n, "pn", pnp=16 * (hi - lo), exact=True)
        base = bd.v2.address(self.name) + c0 // 8 * R * ROW
        zero = bd.zeros(nkb * ROW)
        kbx = (hi - lo) * ROW // 4  # (words: x's kb)

        def fill(at: int, nr: int, src: int, stride: int) -> MemTask:  # (nr rows to x's row at, per kb from src)
            r = ROW // 4
            return MemTask([{"addr": x.addr + at * ROW, "nwords": nkb * nr * r, "dims": [(r, 1), (nr, r), (nkb, kbx)]}],
                           ddr=(src, nkb * nr * ROW, (r, nr, r, stride)))
        kbs = R * ROW // 4  # (words: the image's kb)
        cols = {}
        for c in range(COLS):
            col = base + c * self.colbytes
            fills = []
            if lo < 0:
                fills.append(fill(0, 1, zero, ROW // 4) if c == 0 else fill(0, 1, col - self.colbytes + (R - 1) * ROW, kbs))
            if hi > R:
                fills.append(fill(hi - lo - 1, 1, zero, ROW // 4) if c == COLS - 1 else
                             fill(hi - lo - 1, 1, col + self.colbytes, kbs))
            a, z = max(lo, 0), min(hi, R)
            fills.append(fill(a - lo, z - a, col + a * ROW, kbs))
            cols[c] = ColOps(fills=fills)
        bd.op(cols, {})
        return x


def conv64(bd: Builder, img: PImage, w: torch.Tensor, b: torch.Tensor | None, *, name: str, band: int,
           out: Tensor | None = None, c0: int = 0, res: Tensor | None = None, stats: tuple[int, int] | None = None,
           tbias: str | None = None) -> Tensor:
    """A 3x3 conv (padding 1) of img's channels [c0, c0 + K) (w: (N, K, 3, 3), N <= 320) for band ``band`` into out's C
    tiles (none: new ones, allocated after the input rows' fills, which take the drains pending: their bytes released
    first) (+ res; out's GroupNorm records for ``stats``; tbias: ``unet.conv``'s); returns out."""
    lv = img.lv
    x = img.rows(bd, band, c0, w.shape[1])
    out = out or alloc(bd, lv, w.shape[0])
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
    return out


def norm64(bd: Builder, x: DTensor, wp: str, w: dict, *, name: str, eps: float, c0: int = 0, n: int | None = None,
           ccat: int = 0, ch0: int = 0, rec: list | None = None) -> PImage:
    """GroupNorm + SiLU of x's channels [c0, c0 + n) (records: rec or x's; channels ch0 .. of a concatenation of ccat)
    into a PImage, band by band."""
    n = n or x.C - c0
    img = PImage(n, name)
    g, bt = w[wp + "weight"][ch0:ch0 + n], w[wp + "bias"][ch0:ch0 + n]
    for band in range(x.lv.nb):
        xb = x.load(bd, band, c0, n)
        xb.rec = rec if rec is not None else x.rec
        t = alloc(bd, x.lv, n, "pn")
        gn_apply(bd, xb, t, g, bt, name=name, eps=eps, silu=True, ch0=ch0, ccat=ccat)
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
        hb = conv64(bd, p1, w[p + "conv1.weight"], w[p + "conv1.bias"] + time_bias, name=p + "conv1", band=b,
                    stats=(cout // 32, 0), tbias=p)
        h.store(bd, b, hb)
    p2 = norm64(bd, h, p + "norm2.", w, name=name + ".p2", eps=eps)
    out = DTensor(x.lv, cout, name)
    for b in range(x.lv.nb):
        sc = x.load(bd, b)
        conv64(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", band=b, out=sc, res=sc, stats=stats)
        out.store(bd, b, sc)
    return out


def resnet_cat64(bd: Builder, w: dict, p: str, h: DTensor, skip: DTensor, *, cout: int, time_bias: torch.Tensor,
                 eps: float, stats: tuple[int, int] | None, name: str) -> DTensor:
    """``unet.resnet_cat`` on the 64x64 level (h, the skip and the output in DDR; their records in the concatenation's
    group space): GN1 + SiLU per part of 320 channels into PImages, conv1 over the parts per band (the partial sums in
    the band's C tiles; + the time bias, GN2's records in the last), the 1x1 shortcut over the parts per band, GN2 +
    SiLU, conv2 + the shortcut per band -> the output's band (records for ``stats``)."""
    lv, C = h.lv, h.C + skip.C
    recs = (h.rec or []) + (skip.rec or [])
    w1, wsc = w[p + "conv1.weight"], w[p + "conv_shortcut.weight"]
    parts = [(h, c0, c0) for c0 in range(0, h.C, 320)] + [(skip, 0, h.C)]  # (source, its channel, in C)
    imgs = [norm64(bd, src, p + "norm1.", w, name=f"{name}.p1.{k}", eps=eps, c0=c0, n=320, ccat=C, ch0=cc, rec=recs)
            for k, (src, c0, cc) in enumerate(parts)]
    c1 = DTensor(lv, cout, name + ".c1", rec=[])
    for b in range(lv.nb):
        cb = None
        for k, (_, _, cc) in enumerate(parts):
            first, last = k == 0, k == len(parts) - 1
            cb = conv64(bd, imgs[k], w1[:, cc:cc + 320], w[p + "conv1.bias"] + time_bias if first else None,
                        name=f"{p}conv1.{k}", band=b, out=cb, res=None if first else cb,
                        stats=(cout // 32, 0) if last else None, tbias=p if first else None)
        c1.store(bd, b, cb, defer=b < lv.nb - 1)  # (the last: the shortcut's tensors need its bytes)
    sc = DTensor(lv, cout, name + ".sc")
    for b in range(lv.nb):
        sb = alloc(bd, lv, cout)
        for k, (src, c0, cc) in enumerate(parts):
            xb = src.load(bd, b, c0, 320)
            px = alloc(bd, lv, 320, "pn", top=True)
            plain(bd, xb, px, name=f"{p}xpn.{k}")
            free(bd, xb)
            gemm(bd, px, wsc[:, cc:cc + 320].flatten(1), w[p + "conv_shortcut.bias"] if k == 0 else None,
                 name=f"{p}conv_shortcut.{k}", out=sb, res=sb if k else None)
            free(bd, px)
        sc.store(bd, b, sb, defer=False)  # (the next band's sum and parts need its bytes)
    p2 = norm64(bd, c1, p + "norm2.", w, name=name + ".p2", eps=eps)
    out = DTensor(lv, cout, name)
    for b in range(lv.nb):
        sb = sc.load(bd, b)
        conv64(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], name=p + "conv2", band=b, out=sb, res=sb, stats=stats)
        out.store(bd, b, sb)
    return out


def down64(bd: Builder, w: dict, p: str, x: DTensor, *, stats: tuple[int, int], name: str) -> DTensor:
    """The stride-2 conv of the 64x64 level's x into the 32x32 level (``Level(32, 2)``: column c's rows are its even
    rows; band b's the 64x64 rows 4 b, 4 b + 2): x's panels into a PImage, per 32x32 band the 64x64 rows 4 b - 1 ..
    4 b + 3 and the even tiles of its even rows (``ops.Gemm`` par0) -> the band's C tiles (records for ``stats``)."""
    img = PImage(x.C, name + ".pi")
    for b in range(x.lv.nb):
        xb = x.load(bd, b)
        t = alloc(bd, x.lv, x.C, "pn")
        plain(bd, xb, t, name=p + "xpn")
        free(bd, xb)
        img.store(bd, b, t)
    lv2 = Level(32, 2)
    wc, N = w[p + "conv.weight"], w[p + "conv.weight"].shape[0]
    out = DTensor(lv2, N, name, rec=[])
    for b in range(lv2.nb):
        xr = img.span(bd, 4 * b - 1, 4 * b + 4)
        y = alloc(bd, lv2, N)
        rec = bd.mem.alloc(4 * D.OCH, top=True)
        bd.gemm(Gemm(m=64, ntile=2, K=x.C, N=N, x=xr.addr, y=y.addr, out="bf16", conv=1, R=1, Lg=8, zero=bd.zero,
                     x_np=xr.np, rc=5, r0=1, rstep=2, il=True, par0=True, stats="gn", rec=rec, gsize=stats[0],
                     ch_off=stats[1]), wc, w[p + "conv.bias"], name=p + "conv")
        free(bd, xr)
        y.rec = [bd.records(rec, name=f"{p}conv.rec", ch_off=stats[1], nr=N // 4, gsize=stats[0], band=b, nbands=lv2.nb)]
        out.store(bd, b, y)
    return out


def up64(bd: Builder, w: dict, p: str, x: DTensor, *, stats: tuple[int, int], name: str) -> DTensor:
    """Nearest 2x of a 32x32 level's x (``Level(32, 2)``) into the 64x64 level, then the conv: x's panels drained into
    a PImage with the duplication (``PImage.store_up``), per band and output half of 320 channels the conv over K parts
    of 320 (the partial sums in the half's C tiles) -> the output's band (records for ``stats`` in the last part)."""
    img = PImage(x.C, name + ".up")
    for b in range(x.lv.nb):
        xb = x.load(bd, b)
        t = alloc(bd, x.lv, x.C, "pn")
        plain(bd, xb, t, name=p + "xpn")
        free(bd, xb)
        img.store_up(bd, b, t)
    wc, bc = w[p + "conv.weight"], w[p + "conv.bias"]
    N = wc.shape[0]
    out = DTensor(LV64, N, name, rec=[])
    for b in range(LV64.nb):
        for o0 in range(0, N, 320):
            cb = None
            for k, c0 in enumerate(range(0, x.C, 320)):
                last = c0 + 320 == x.C
                cb = conv64(bd, img, wc[o0:o0 + 320, c0:c0 + 320], bc[o0:o0 + 320] if k == 0 else None,
                            name=f"{p}conv.{k}.{o0}", band=b, out=cb, c0=c0, res=cb if k else None,
                            stats=(stats[0], stats[1] + o0) if last else None)
            out.store(bd, b, cb, c0=o0)
    return out


def conv_in64(bd: Builder, w: dict, lat: PImage, *, stats: tuple[int, int], name: str) -> DTensor:
    """conv_in on the latents' PImage (lat: 64 channels, the 4 latent ones then zeros) -> the 64x64 level's 320 channels
    per band (records for ``stats``)."""
    wi = w["conv_in.weight"]
    w64 = torch.zeros(wi.shape[0], lat.C, 3, 3)
    w64[:, :wi.shape[1]] = wi
    out = DTensor(LV64, wi.shape[0], name, rec=[])
    for b in range(LV64.nb):
        out.store(bd, b, conv64(bd, lat, w64, w["conv_in.bias"], name="conv_in", band=b, stats=stats))
    return out


def conv_out64(bd: Builder, w: dict, x: DTensor, *, eps: float, name: str) -> None:
    """conv_norm_out (GroupNorm + SiLU) and conv_out of the 64x64 level's x: its 4 output channels as the first of 320
    (zero weights past them) per band, their first 8 channels' blocks drained into DDR image ``name`` ([band][col][OUT]:
    ``noise``)."""
    img = norm64(bd, x, "conv_norm_out.", w, name=name + ".pn", eps=eps)
    wo = w["conv_out.weight"]
    w320, b320 = torch.zeros(320, x.C, 3, 3), torch.zeros(320)
    w320[:wo.shape[0]], b320[:wo.shape[0]] = wo, w["conv_out.bias"]
    base = bd.ddr(name, LV64.nb * COLS * OUT)
    for b in range(LV64.nb):
        cb = conv64(bd, img, w320, b320, name="conv_out", band=b)
        bd.op({c: ColOps(drains=[MemTask([{"addr": cb.addr, "nwords": OUT // 4,
                                           "dims": [(32, 1), (8, 320), (4, L.ct_bytes(64) // 4)]}],
                                         ddr=(base + (b * COLS + c) * OUT, OUT, None))]) for c in range(COLS)}, {})
        free(bd, cb)
    bd.ddr_sync()
