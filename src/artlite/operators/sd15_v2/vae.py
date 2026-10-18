"""sd15_v2's VAE decoder (SD 1.x AutoencoderKL's): one image on the UNet's design, its left and right halves as the
design's two images (``build.Level`` half; the convs' joined shift masks carry the pixels across the seam).

Levels (``LEVELS``): 64 px (a row = one tile of 4 groups: ``ops.Gemm`` rc; bands of 2 rows), 128 (two tiles a row,
its even and odd groups: il; bands of a row), 256 and 512 ("wide": tiles of 4 consecutive groups, bands of half a
row). A level's residual stream and blocks' C tiles in DDR per band (``unet.DTensor``); a conv's input as the whole
level's panels in DDR (``VImage``), a band's input rows filled from it (wide rows: the half row's window).

Channels (``Chans``): per producing core row a slice, padded to whole 80-channel passes in the C tiles (512: 4 x (128 +
32), 256: 4 x (64 + 16); the pads zero: zero weights, gamma and beta), at 16 GroupNorm groups apart in the statistics'
channel space (a slice in one record window); the panels of a conv's input hold the real channels only.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..design import COLS
from . import core as K
from . import design as D
from . import layout as L
from .build import Builder, Level
from .level64 import IL
from .ops import Apply, Gemm
from .seq import ColOps, MemTask
from .unet import DTensor, Tensor, _silu, alloc, free, plain

L64, L128, L256, L512 = Level(64, 4, True), Level(128, 16, True), Level(256, 64, True), Level(512, 128, True)
LEVELS = {lv.H: lv for lv in (L64, L128, L256, L512)}
APPLY_ROWS = (0, 1, 2, 3)


def kind(lv: Level) -> str:
    return "rc" if lv.Lg == 4 else "il" if lv.Lg == 8 else "wide"


def order(lv: Level) -> list[int]:
    """A row's groups in their storage order."""
    return list(IL) if lv.Lg == 8 else list(range(lv.Lg))


def tokens(x: torch.Tensor, lv: Level) -> torch.Tensor:
    """(1, C, H, H) -> per column its tokens (COLS, R 16 Lg, C): column c's rows c R .., token panel (row, group in the
    storage order, half), lane j = pixel half H / 2 + g + Lg j."""
    C = x.shape[1]
    v = x.reshape(C, COLS, lv.R, 2, 8, lv.Lg)[..., order(lv)]  # C c row half j g
    return v.permute(1, 2, 5, 3, 4, 0).reshape(COLS, lv.R * lv.Lg * 16, C)


def from_tokens(y: torch.Tensor, lv: Level) -> torch.Tensor:
    C = y.shape[-1]
    v = y.reshape(COLS, lv.R, lv.Lg, 2, 8, C).permute(5, 0, 1, 3, 4, 2)  # C c row half j g
    return v[..., [order(lv).index(g) for g in range(lv.Lg)]].reshape(1, C, lv.H, lv.H)


@dataclass(frozen=True)
class Chans:
    """C channels in ``slices`` slices (a producing core row's each), each padded to whole 80-channel passes (``per``
    real, ``pslice`` stored), at ``stride`` (16 groups) in the GroupNorm statistics' channel space."""

    C: int
    slices: int = 4

    @property
    def per(self) -> int:
        return self.C // self.slices

    @property
    def pslice(self) -> int:
        return -(-self.per // 80) * 80

    @property
    def P(self) -> int:  # (the stored channels)
        return self.slices * self.pslice

    @property
    def gsize(self) -> int:
        return self.C // 32

    @property
    def stride(self) -> int:
        return 16 * self.gsize

    def phys(self) -> torch.Tensor:
        """Each real channel's stored index."""
        c = torch.arange(self.C)
        return c // self.per * self.pslice + c % self.per

    def pad(self, v: torch.Tensor) -> torch.Tensor:
        """(C, ...) -> (P, ...): zeros at the pads."""
        out = torch.zeros(self.P, *v.shape[1:], dtype=v.dtype)
        out[self.phys()] = v
        return out

    def unpad(self, v: torch.Tensor) -> torch.Tensor:
        return v[..., self.phys()]

    def kb_runs(self) -> list[tuple[int, int, int]]:
        """Per slice (its first stored kb, its first real kb, its real kbs)."""
        return [(s * self.pslice // 8, s * self.per // 8, self.per // 8) for s in range(self.slices)]


def chans(C: int) -> Chans:
    """The decoder's layout of C channels: 128 in two slices (each on two core rows, on alternate tiles), else four."""
    return Chans(C, 2 if C == 128 else 4)


@dataclass
class VImage:
    """A level's conv input in DDR (image ``name``): per column [kb][row R][the row's token panels] of C real channels;
    wide rows as positions of 2 panels [its last group, groups 0 .. G - 1, its group 0] (the windows' edges)."""

    C: int
    name: str
    lv: Level

    @property
    def rowp(self) -> int:  # (a row's panels of a kb)
        return 2 * (self.lv.Lg + 2) if kind(self.lv) == "wide" else 2 * self.lv.Lg

    @property
    def colbytes(self) -> int:
        return self.C // 8 * self.lv.R * self.rowp * L.PANEL

    def base(self, bd: Builder) -> int:
        return bd.ddr(self.name, COLS * self.colbytes + 4096)  # (a wide window reads up to 2 positions past a row)

    def _drain(self, t: Tensor, ch: Chans | None, src_off: int, n_words: int, src_step: int, dst: int, dst_n0: int,
               dst_s1: int, nrun: int = 1) -> list[MemTask]:
        """A drain of t's real kbs (``ch``: t's slices; None: every kb is real): per real kb ``nrun`` strips of n_words
        from t's word src_off + src_step k, into DDR from dst (strips of dst_n0 words dst_s1 apart, the real kbs one
        image kb apart)."""
        kbw, ikb = t.np * 18, self.lv.R * self.rowp * 18  # (words: t's kb, the image's kb)
        ns, pk, nk = (ch.slices, ch.pslice // 8, ch.per // 8) if ch is not None else (1, 0, t.C // 8)
        dims = [(n_words, 1)] + ([(nrun, src_step)] if nrun > 1 else []) + [(nk, kbw)] + ([(0 if nrun > 1 else ns, pk * kbw)]
                                                                                           if ns > 1 else [])
        return [MemTask([{"addr": t.addr + 4 * src_off, "nwords": ns * nk * nrun * n_words, "dims": dims}],
                        ddr=(dst, 4 * ns * nk * nrun * n_words, (dst_n0, nrun * n_words // dst_n0, dst_s1, ikb)))]

    def store(self, bd: Builder, b: int, t: Tensor, ch: Chans | None = None) -> None:
        """Band b's panels t ([kb][band token panels], the stored channels ``ch``) into its rows (wide: its half row and
        the other end's edge position), drained with the next op (t's bytes released after it)."""
        base, lv = self.base(bd), self.lv
        bd.written(self.name)
        rp = self.rowp * 18  # (words: a row of a kb)
        cols = {}
        for c in range(COLS):
            col = base + c * self.colbytes
            if kind(lv) != "wide":
                rb = lv.R // lv.nb
                drains = self._drain(t, ch, 0, rb * rp, 0, col + 4 * rb * b * rp, rb * rp, rb * rp)
            else:
                r, h = divmod(b, 2)
                g2 = lv.Lg // 2  # (a half row's groups)
                row = col + 4 * r * rp
                drains = self._drain(t, ch, 0, g2 * 36, 0, row + 4 * 36 * (1 + h * g2), g2 * 36, g2 * 36)
                edge = (0, lv.Lg + 1) if h == 0 else ((g2 - 1) * 36, 0)  # (group 0 -> the last position; the last group
                # -> position 0)
                drains += self._drain(t, ch, edge[0], 36, 0, row + 4 * 36 * edge[1], 36, 36)
            cols[c] = ColOps(drains=drains)
        bd.defer(cols, release=[(t.addr, t.size or t.nbytes)])

    def store_up(self, bd: Builder, b: int, t: Tensor, src: Level, ch: Chans | None = None) -> None:
        """Nearest 2x of band b of level src's panels t into this level's rows: src row y -> rows 2 y, 2 y + 1, group g ->
        groups 2 g, 2 g + 1 (the storage orders: rc -> il, il -> wide, wide -> wide; wide: the edge positions too),
        drained with the next op (t's bytes released after it)."""
        base, lv = self.base(bd), self.lv
        bd.written(self.name)
        rp = self.rowp * 18
        cols = {}
        for c in range(COLS):
            col, drains = base + c * self.colbytes, []
            if kind(src) == "rc":  # (a band's 2 rows of 4 groups -> il rows [g0 .. g3 | g0 .. g3] of 8 panels each)
                assert kind(lv) == "il"
                for j in range(2):
                    for f in range(2):
                        y2 = 2 * (2 * b + j) + f
                        for e in range(2):
                            drains += self._drain(t, ch, j * 144, 144, 0, col + 4 * (y2 * rp + e * 144), 144, 144)
            elif kind(src) == "il":  # (a row of 8 groups in il order -> wide rows: groups 2 g + e at positions 1 + 2 g + e)
                assert kind(lv) == "wide" and lv.Lg == 16
                for f in range(2):
                    row = col + 4 * (2 * b + f) * rp
                    for e in range(2):  # (t's groups in natural order: il positions a + 4 b of g = 2 a + b)
                        for gb in range(2):
                            drains += self._drain(t, ch, 36 * 4 * gb, 36, 36, row + 4 * 36 * (1 + 2 * gb + e), 36, 144,
                                                  nrun=4)
                    drains += self._drain(t, ch, 36 * IL.index(7), 36, 0, row, 36, 36)  # (group 15 <- 7 at position 0)
                    drains += self._drain(t, ch, 0, 36, 0, row + 4 * 36 * (lv.Lg + 1), 36, 36)  # (group 0 at the end)
            else:  # (a half row of Lg_src / 2 groups -> 2 rows' groups 2 g + e)
                assert kind(lv) == "wide" and lv.Lg == 2 * src.Lg
                r, h = divmod(b, 2)
                g2 = src.Lg // 2
                for f in range(2):
                    row = col + 4 * (2 * r + f) * rp
                    for e in range(2):
                        drains += self._drain(t, ch, 0, 36, 36, row + 4 * 36 * (1 + 2 * h * g2 + e), 36, 72, nrun=g2)
                    if h == 0:  # (group 0 <- the band's first at the last position)
                        drains += self._drain(t, ch, 0, 36, 0, row + 4 * 36 * (lv.Lg + 1), 36, 36)
                    else:  # (the last group <- the band's last at position 0)
                        drains += self._drain(t, ch, (g2 - 1) * 36, 36, 0, row, 36, 36)
            cols[c] = ColOps(drains=drains)
        bd.defer(cols, release=[(t.addr, t.size or t.nbytes)])

    def fill(self, bd: Builder, lo: int, hi: int, p0: int, npos: int, c0: int, n: int) -> Tensor:
        """Channels [c0, c0 + n) of rows [lo, hi) of every column (rows past the column's: the west column's last, the
        east column's first; zeros past the image), positions [p0, p0 + npos) of each (wide; else the whole rows), as
        memtile panels [kb][row][position] (an op of its fills)."""
        R, nkb = self.lv.R, n // 8
        assert -1 <= lo and hi <= R + 1
        bd.reading(self.name)  # (first: the deferred drains pushed, their bytes released)
        nr = hi - lo
        x = alloc(bd, self.lv, n, "pn", pnp=2 * npos * nr, exact=True)
        rp = self.rowp * 18
        base = bd.v2.address(self.name) + c0 // 8 * R * rp * 4 + 4 * 36 * p0
        zero = bd.zeros(nkb * nr * npos * 144)
        kbx, kbs, pw = 2 * npos * nr * 18, R * rp, npos * 36  # (words: x's kb, the image's kb, a row's positions)

        def task(at: int, k: int, src: int, s1: int, s2: int) -> MemTask:  # (k rows to x's row at)
            return MemTask([{"addr": x.addr + 4 * at * pw, "nwords": nkb * k * pw, "dims": [(pw, 1), (k, pw), (nkb, kbx)]}],
                           ddr=(src, 4 * nkb * k * pw, (pw, k, s1, s2)))
        cols = {}
        for c in range(COLS):
            col, fills = base + c * self.colbytes, []
            if lo < 0:
                fills.append(task(0, 1, zero, pw, pw) if c == 0 else task(0, 1, col - self.colbytes + 4 * (R - 1) * rp, rp, kbs))
            if hi > R:
                fills.append(task(nr - 1, 1, zero, pw, pw) if c == COLS - 1 else task(nr - 1, 1, col + self.colbytes, rp, kbs))
            a, z = max(lo, 0), min(hi, R)
            fills.append(task(a - lo, z - a, col + 4 * a * rp, rp, kbs))
            cols[c] = ColOps(fills=fills)
        bd.op(cols, {})
        return x

    def fill_bytes(self, n: int) -> int:
        """Memtile bytes of a band's conv input of n channels (``rows``)."""
        lv = self.lv
        if kind(lv) == "wide":
            return n // 8 * 3 * 2 * (lv.Lg // 2 + 4) * L.PANEL
        return n // 8 * (lv.R // lv.nb + 2) * self.rowp * L.PANEL

    def rows(self, bd: Builder, b: int, c0: int = 0, n: int | None = None) -> tuple[Tensor, dict]:
        """Band b's conv input (channels [c0, c0 + n)): its rows and the halo rows (wide: the half row's window of
        Lg / 2 + 4 positions); returns it with its Gemm keywords."""
        n, lv = n or self.C - c0, self.lv
        if kind(lv) == "wide":
            r, h = divmod(b, 2)
            npos = lv.Lg // 2 + 4
            x = self.fill(bd, r - 1, r + 2, h * lv.Lg // 2, npos, c0, n)
            return x, {"Lg": 4, "rc": 3, "r0": 1, "wide": "fm" if h == 0 else "ml", "wpos": npos}
        rb = lv.R // lv.nb
        x = self.fill(bd, rb * b - 1, rb * b + rb + 1, 0, self.rowp // 2, c0, n)
        if kind(lv) == "il":
            return x, {"Lg": 8, "rc": rb + 2, "r0": 1, "il": True}
        return x, {"Lg": 4, "rc": rb + 2, "r0": 1}


def write(bd: Builder, img: VImage, x: torch.Tensor) -> None:
    """The host's image of x (1, C, H, H) into img's DDR image (a model's input; tests)."""
    lv = img.lv
    t = tokens(x, lv)
    if kind(lv) == "wide":  # (the rows' positions: [the last group, groups 0 .. G - 1, group 0])
        t = t.reshape(COLS, lv.R, lv.Lg, 16, -1)
        t = torch.cat([t[:, :, -1:], t, t[:, :, :1]], 2).reshape(COLS, -1, x.shape[1])
    base = img.base(bd)
    v = np.concatenate([L.panels(t[c]) for c in range(COLS)])
    bd.v2.write(img.name, v)
    assert v.size == COLS * img.colbytes and base


def stf(b: int, nb: int) -> int:
    """C_STF of band b of nb: one GroupNorm record per core over a layer's bands."""
    return (K.ST_KEEP if b else 0) | (K.ST_HOLD if b < nb - 1 else 0)


def conv(bd: Builder, img: VImage, w: torch.Tensor, b: torch.Tensor | None, *, ch: Chans, name: str, band: int,
         out: Tensor | None = None, res: DTensor | None = None, c0: int = 0, rec: int = 0) -> Tensor:
    """A 3x3 conv (padding 1) of img's channels [c0, c0 + K) (w: (P, K, 3, 3), its output rows stored as ``ch``: the
    same tensor for a layer's bands) for band ``band`` into out's C tiles, + out (partial sums); none: new ones, or
    res's band (+ res), allocated after the input's fills (which take the drains pending: their bytes released first);
    rec: GroupNorm records of the output there (one over the layer's bands: ``stf``); returns out."""
    lv = img.lv
    x, kw = img.rows(bd, band, c0, w.shape[1])
    acc = out is not None
    if res is not None and not acc:
        out = res.load(bd, band)
    out = out or alloc(bd, lv, ch.P)
    g = dict(m=64, ntile=lv.ntile, K=w.shape[1], N=ch.P, x=x.addr, y=out.addr, out="bf16", conv=1, R=1, zero=bd.zero,
             x_np=x.np, joined=True, res=out.addr if acc or res is not None else None, ch_step=ch.stride,
             ch_real=ch.per, tsplit=4 // ch.slices, **kw)
    if rec:
        g.update(stats="gn", rec=rec, gsize=ch.gsize, stf=stf(band, lv.nb))
    bd.gemm(Gemm(**g), w, b, name=f"{name}.{kw.get('wide', '')}{g.get('stf', '')}", wname=f"{name}.w")
    free(bd, x)
    return out


def norm(bd: Builder, x: DTensor, gamma: torch.Tensor, beta: torch.Tensor, *, ch: Chans, name: str, eps: float,
         silu: bool = True) -> VImage:
    """GroupNorm (+ SiLU) of x (its stored channels ``ch``, its records) into a VImage of the real channels, band by
    band: an apply on every core row (one slice each: a record window), the halves' statistics joined (gamma, beta:
    the real channels')."""
    lv = x.lv
    img = VImage(ch.C, name, lv)
    g, bt = ch.pad(gamma), ch.pad(beta)
    for b in range(lv.nb):
        bd.reading(x.band(b))  # (first: the pending drains pushed, their bytes released)
        xb = x.load(bd, b)
        if silu:
            bd.table("silu", _silu(), rows=APPLY_ROWS)
        t = alloc(bd, lv, ch.P, "pn")
        ap = Apply(m=64, nblk=ch.pslice // 80, mode="gnsilu" if silu else "gn", x=xb.addr, y=t.addr, ntile=lv.ntile,
                   y_np=t.np, gsize=ch.gsize, ch_step=ch.stride, tokens=lv.H ** 2, eps=eps, rows=APPLY_ROWS[:ch.slices],
                   join=True, nreal=ch.per)
        bd.apply(ap, g, bt, name=name + ".a", records=xb.rec)
        free(bd, xb)
        img.store(bd, b, t, ch)
    return img


def records(bd: Builder, rec: int, *, ch: Chans, name: str) -> list:
    """A layer's GroupNorm records (``conv``'s rec: one per core over its bands) drained to DDR."""
    r = bd.records(rec, name=name, ch_off=0, nr=ch.stride, gsize=ch.gsize)
    r.split = 4 // ch.slices
    return [r]


def conv_layer(bd: Builder, img: VImage, w: torch.Tensor, b: torch.Tensor | None, *, ch: Chans, name: str,
               res: DTensor | None = None, stats: bool = True) -> DTensor:
    """A 3x3 conv of every band of img (w, b: the real channels' (N, K, 3, 3), (N,); + res: its band in place) -> a
    DTensor of the stored channels ``ch`` with records for its norm. K in parts where a band's input and output do not
    fit the memtile together (the partial sums in the output's C tiles)."""
    lv = img.lv
    K = w.shape[1]
    kp = K
    while img.fill_bytes(kp) + lv.ct(ch.P) > D.HEAP - 32768:
        kp //= 2
    assert kp % 64 == 0 and K % kp == 0, (K, kp)
    wp, bp = ch.pad(w), ch.pad(b) if b is not None else None
    parts = [(c0, wp[:, c0:c0 + kp]) for c0 in range(0, K, kp)]
    out = DTensor(lv, ch.P, name, rec=[])
    rec = bd.mem.alloc(4 * D.OCH, top=True) if stats else 0
    for band in range(lv.nb):
        o = None
        for k, (c0, wk) in enumerate(parts):
            o = conv(bd, img, wk, bp if k == 0 else None, ch=ch, name=f"{name}.k{k}", band=band, out=o,
                     res=res if k == 0 else None, c0=c0, rec=rec if k == len(parts) - 1 else 0)
        out.store(bd, band, o)
    if stats:
        out.rec = records(bd, rec, ch=ch, name=name + ".rec")
    return out


def write_dtensor(bd: Builder, t: DTensor, x: torch.Tensor, ch: Chans) -> None:
    """The host's x (1, C, H, H) as t's bands (C tiles of the stored channels ``ch``)."""
    lv = t.lv
    xt = ch.pad(tokens(x, lv).permute(2, 0, 1)).permute(1, 2, 0)  # (COLS, tokens, P)
    for b in range(lv.nb):
        bd.image(t.band(b), np.concatenate([L.ctiles(xt[c, lv.T * b:lv.T * (b + 1)], 64) for c in range(COLS)]))


def with_stats(bd: Builder, x: DTensor, ch: Chans, *, name: str) -> None:
    """x's GroupNorm records (a copy op per band: K 0, x as the residual), for a DTensor written by the host."""
    lv = x.lv
    rec = bd.mem.alloc(4 * D.OCH, top=True)
    for b in range(lv.nb):
        xb = x.load(bd, b)
        y = alloc(bd, lv, ch.P)
        g = Gemm(m=64, ntile=lv.ntile, K=0, N=ch.P, x=0, y=y.addr, out="bf16", res=xb.addr, stats="gn", rec=rec,
                 gsize=ch.gsize, ch_step=ch.stride, ch_real=ch.per, tsplit=4 // ch.slices, stf=stf(b, lv.nb))
        bd.gemm(g, None, None, name=f"{name}.{g.stf}")
        free(bd, y)
        free(bd, xb)
    x.rec = records(bd, rec, ch=ch, name=name + ".rec")


def shortcut(bd: Builder, x: DTensor, w: torch.Tensor, b: torch.Tensor, *, cin: Chans, cout: Chans, name: str) -> DTensor:
    """The resnet's 1x1 conv_shortcut of x (stored ``cin``) per band -> a DTensor (stored ``cout``)."""
    lv = x.lv
    wp = cout.pad(cin.pad(w.flatten(1).T).T)  # ((P out, P in): zeros at both pads)
    bp = cout.pad(b)
    out = DTensor(lv, cout.P, name)
    for band in range(lv.nb):
        xb = x.load(bd, band)
        px = alloc(bd, lv, cin.P, "pn", top=True)
        plain(bd, xb, px, name=name + ".xpn")
        free(bd, xb)
        sb = alloc(bd, lv, cout.P)
        bd.gemm(Gemm(m=64, ntile=lv.ntile, K=cin.P, N=cout.P, x=px.addr, y=sb.addr, out="bf16", x_np=px.np,
                     tsplit=4 // cout.slices), wp, bp, name=name)
        free(bd, px)
        out.store(bd, band, sb, defer=False)
    return out


def resnet(bd: Builder, w: dict, p: str, x: DTensor, *, cin: Chans, cout: Chans, eps: float, name: str) -> DTensor:
    """diffusers' ResnetBlock2D (no time embedding): GN1 + SiLU, conv1, GN2 + SiLU, conv2 + x (or its 1x1 shortcut);
    x and the output in DDR with their records."""
    p1 = norm(bd, x, w[p + "norm1.weight"], w[p + "norm1.bias"], ch=cin, name=name + ".n1", eps=eps)
    h = conv_layer(bd, p1, w[p + "conv1.weight"], w[p + "conv1.bias"], ch=cout, name=p + "conv1")
    sc = x
    if p + "conv_shortcut.weight" in w:
        sc = shortcut(bd, x, w[p + "conv_shortcut.weight"], w[p + "conv_shortcut.bias"], cin=cin, cout=cout,
                      name=p + "conv_shortcut")
    p2 = norm(bd, h, w[p + "norm2.weight"], w[p + "norm2.bias"], ch=cout, name=name + ".n2", eps=eps)
    return conv_layer(bd, p2, w[p + "conv2.weight"], w[p + "conv2.bias"], ch=cout, name=p + "conv2", res=sc)


def upsample(bd: Builder, w: dict, p: str, x: DTensor, *, ch: Chans, name: str) -> DTensor:
    """Nearest 2x of x into the next level, then the upsampler's conv: x's panels drained into the next level's VImage
    with the duplication (``VImage.store_up``), the conv per band -> a DTensor with records for its norm."""
    lv2 = LEVELS[2 * x.lv.H]
    img = VImage(ch.C, name + ".up", lv2)
    for band in range(x.lv.nb):
        xb = x.load(bd, band)
        t = alloc(bd, x.lv, ch.P, "pn")
        plain(bd, xb, t, name=p + "xpn")
        free(bd, xb)
        img.store_up(bd, band, t, x.lv, ch)
    return conv_layer(bd, img, w[p + "conv.weight"], w[p + "conv.bias"], ch=ch, name=name)
