"""sd15's VAE decoder (SD 1.x AutoencoderKL's): one image on the UNet's design, its left and right halves as the
design's two images (``build.Level`` half; the convs' joined shift masks carry the pixels across the seam).

Levels (``LEVELS``): 64 px (a row = one tile of 4 groups: ``ops.Gemm`` rc; bands of 2 rows), 128 (two tiles a row,
its even and odd groups: il; bands of a row), 256 and 512 ("wide": tiles of 4 consecutive groups, bands of half a
row). A level's residual stream and blocks' C tiles in DDR per band (``unet.DTensor``); a conv's input as the whole
level's panels in DDR (``VImage``), a band's input rows filled from it (wide rows: the half row's window).

Channels (``Chans``): per producing core row a slice, padded to whole 80-channel passes in the C tiles (512: 4 x (128 +
32), 256: 4 x (64 + 16); the pads zero: zero weights, gamma and beta), at 16 GroupNorm groups apart in the statistics'
channel space (a slice in one record window); the panels of a conv's input hold the real channels only.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..design import COLS
from ..overlay.runtime import Transaction
from . import core as K
from . import design as D
from . import layout as L
from .build import Builder, Level
from .level64 import IL
from .ops import Apply, Gemm
from .runtime import Sd15Device
from .seq import ColOps, MemTask, RowOps
from .unet import LOG2E, DTensor, Tensor, _silu, alloc, free, plain

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

    def fill(self, bd: Builder, lo: int, hi: int, p0: int, npos: int, c0: int, n: int) -> tuple[Tensor, dict]:
        """Channels [c0, c0 + n) of rows [lo, hi) of every column (rows past the column's: the west column's last, the
        east column's first; zeros past the image), positions [p0, p0 + npos) of each (wide; else the whole rows), as
        memtile panels [kb][row][position]: the tensor and its fills (per column)."""
        R, nkb = self.lv.R, n // 8
        assert -1 <= lo and hi <= R + 1
        bd.reading(self.name)  # (first: the deferred drains pushed, their bytes released)
        nr = hi - lo
        x = alloc(bd, self.lv, n, "pn", pnp=2 * npos * nr, exact=True)
        rp = self.rowp * 18
        base = bd.npu.address(self.name) + c0 // 8 * R * rp * 4 + 4 * 36 * p0
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
        return x, cols

    def fill_bytes(self, n: int) -> int:
        """Memtile bytes of a band's conv input of n channels (``rows``)."""
        lv = self.lv
        if kind(lv) == "wide":
            return n // 8 * 3 * 2 * (lv.Lg // 2 + 4) * L.PANEL
        return n // 8 * (lv.R // lv.nb + 2) * self.rowp * L.PANEL

    def rows(self, bd: Builder, b: int, c0: int = 0, n: int | None = None) -> tuple[Tensor, dict, dict]:
        """Band b's conv input (channels [c0, c0 + n)): its rows and the halo rows (wide: the half row's window of
        Lg / 2 + 4 positions); returns it with its Gemm keywords and its fills."""
        n, lv = n or self.C - c0, self.lv
        if kind(lv) == "wide":
            r, h = divmod(b, 2)
            npos = lv.Lg // 2 + 4
            x, cols = self.fill(bd, r - 1, r + 2, h * lv.Lg // 2, npos, c0, n)
            return x, {"Lg": 4, "rc": 3, "r0": 1, "wide": "fm" if h == 0 else "ml", "wpos": npos}, cols
        rb = lv.R // lv.nb
        x, cols = self.fill(bd, rb * b - 1, rb * b + rb + 1, 0, self.rowp // 2, c0, n)
        if kind(lv) == "il":
            return x, {"Lg": 8, "rc": rb + 2, "r0": 1, "il": True}, cols
        return x, {"Lg": 4, "rc": rb + 2, "r0": 1}, cols


def image_bytes(img: VImage, x: torch.Tensor) -> np.ndarray:
    """The host's x (1, C, H, H) as img's DDR image (a model's input; tests)."""
    lv = img.lv
    t = tokens(x, lv)
    if kind(lv) == "wide":  # (the rows' positions: [the last group, groups 0 .. G - 1, group 0])
        t = t.reshape(COLS, lv.R, lv.Lg, 16, -1)
        t = torch.cat([t[:, :, -1:], t, t[:, :, :1]], 2).reshape(COLS, -1, x.shape[1])
    return np.concatenate([L.panels(t[c]) for c in range(COLS)])


def write(bd: Builder, img: VImage, x: torch.Tensor) -> None:
    img.base(bd)
    bd.npu.write(img.name, image_bytes(img, x))


def fetch(bd: Builder, x: DTensor, b: int) -> tuple[Tensor, dict]:
    """Band b of x in the memtile (allocated after the pending drains of an image written since the last sync are
    pushed): the tensor and its fills (per column), for an op of their own or merged into the op before its use."""
    bd.reading(x.band(b))
    t = alloc(bd, x.lv, x.C, x.kind)
    base = bd.npu.address(x.band(b))
    t.rec = x.rec
    return t, {c: ColOps(fills=[MemTask([{"addr": t.addr, "nwords": t.nbytes // 4}], ddr=(base + c * x.nbytes, t.nbytes,
                                                                                         None))]) for c in range(COLS)}


def load(bd: Builder, x: DTensor, b: int) -> Tensor:
    """Band b of x in the memtile (an op of its fills)."""
    t, cols = fetch(bd, x, b)
    bd.op(cols, {})
    return t


def stf(b: int, nb: int) -> int:
    """C_STF of band b of nb: one GroupNorm record per core over a layer's bands."""
    return (K.ST_KEEP if b else 0) | (K.ST_HOLD if b < nb - 1 else 0)


def prefetch(bd: Builder, get: Callable[[], tuple]) -> tuple | None:
    """get()'s tensors and fills where the memtile holds them beside the current op's (None: it does not)."""
    free_before = list(bd.mem.free)
    try:
        return get()
    except MemoryError:
        bd.mem.free = free_before
        return None


def conv(bd: Builder, x: Tensor, kw: dict, w: torch.Tensor, b: torch.Tensor | None, *, ch: Chans, name: str,
         band: int, nb: int, out: Tensor, acc: bool = False, rec: int = 0, extra: dict | None = None) -> None:
    """A 3x3 conv (padding 1) of the input rows x (``VImage.rows``: their Gemm keywords kw; w: (P, K, 3, 3), its output
    rows stored as ``ch``: the same tensor for a layer's bands) for band ``band`` of nb into out's C tiles (acc: + out,
    a residual or the partial sums); rec: GroupNorm records of the output there (one over the layer's bands:
    ``stf``); extra: fills merged into its op (the next band's inputs)."""
    g = dict(m=64, ntile=x.lv.ntile, K=w.shape[1], N=ch.P, x=x.addr, y=out.addr, out="bf16", conv=1, R=1, zero=bd.zero,
             x_np=x.np, joined=True, res=out.addr if acc else None, ch_step=ch.stride, ch_real=ch.per,
             tsplit=4 // ch.slices, **kw)
    if rec:
        g.update(stats="gn", rec=rec, gsize=ch.gsize, stf=stf(band, nb))
    bd.gemm(Gemm(**g), w, b, name=f"{name}.{kw.get('wide', '')}{g.get('stf', '')}", wname=f"{name}.w", extra=extra)


def norm(bd: Builder, x: DTensor, gamma: torch.Tensor, beta: torch.Tensor, *, ch: Chans, name: str, eps: float,
         silu: bool = True) -> VImage:
    """GroupNorm (+ SiLU) of x (its stored channels ``ch``, its records) into a VImage of the real channels, band by
    band: an apply on every core row (one slice each: a record window), the halves' statistics joined (gamma, beta:
    the real channels'); the next band's load merged into an apply where the memtile holds both."""
    lv = x.lv
    img = VImage(ch.C, name, lv)
    g, bt = ch.pad(gamma), ch.pad(beta)
    def inputs(b: int) -> tuple:  # (band b's input, its fills, its output)
        xb, cols = fetch(bd, x, b)
        return xb, cols, alloc(bd, lv, ch.P, "pn")
    nxt = None
    for b in range(lv.nb):
        xb, _, t = nxt if nxt else (load(bd, x, b), None, None)
        t = t or alloc(bd, lv, ch.P, "pn")
        if silu:
            bd.table("silu", _silu(), rows=APPLY_ROWS)
        nxt = prefetch(bd, lambda b=b: inputs(b + 1)) if b + 1 < lv.nb else None
        ap = Apply(m=64, nblk=ch.pslice // 80, mode="gnsilu" if silu else "gn", x=xb.addr, y=t.addr, ntile=lv.ntile,
                   y_np=t.np, gsize=ch.gsize, ch_step=ch.stride, tokens=lv.H ** 2, eps=eps, rows=APPLY_ROWS[:ch.slices],
                   join=True, nreal=ch.per)
        bd.apply(ap, g, bt, name=name + ".a", records=xb.rec, extra=nxt[1] if nxt else None)
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
    fit the memtile together (the partial sums in the output's C tiles); a conv's op takes the next one's input rows
    (and residual) where the memtile holds them too."""
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
    steps = [(band, k) for band in range(lv.nb) for k in range(len(parts))]

    def inputs(i: int, out: bool) -> tuple:  # (step i's input rows, their Gemm keywords, where ``out`` its output
        # (part 0: its residual band, or a new one), the fills)
        band, k = steps[i]
        x, kw, cols = img.rows(bd, band, parts[k][0], kp)
        if not out or k:
            return x, kw, None, cols
        if res is None:
            return x, kw, alloc(bd, lv, ch.P), cols
        r, rc = fetch(bd, res, band)
        return x, kw, r, bd.merge(cols, rc)
    nxt, o = None, None
    for i, (band, k) in enumerate(steps):
        if nxt is None:  # (its op takes the drains pending: their bytes released before the output's allocation)
            nxt = inputs(i, False)
            bd.op(nxt[3], {})
        x, kw, r, _ = nxt
        if k == 0:
            o = r or (load(bd, res, band) if res is not None else alloc(bd, lv, ch.P))
        nxt = prefetch(bd, lambda i=i: inputs(i + 1, True)) if i + 1 < len(steps) else None
        last = k == len(parts) - 1
        conv(bd, x, kw, parts[k][1], bp if k == 0 else None, ch=ch, name=f"{name}.k{k}", band=band, nb=lv.nb, out=o,
             acc=bool(k) or res is not None, rec=rec if last else 0, extra=nxt[3] if nxt else None)
        free(bd, x)
        if last:
            out.store(bd, band, o)
    if stats:
        out.rec = records(bd, rec, ch=ch, name=name + ".rec")
    return out


def dtensor_bytes(t: DTensor, x: torch.Tensor, ch: Chans) -> np.ndarray:
    """The host's x (1, C, H, H) as t's bands' DDR images ([band][col][80-channel block][tile][``layout.ct_bytes``]: C
    tiles of the stored channels ``ch``)."""
    lv, P, ct = t.lv, ch.P, L.ct_bytes(64)
    xt = ch.pad(tokens(x, lv).permute(2, 0, 1)).permute(1, 2, 0)  # (COLS, tokens, P)
    v = xt.reshape(COLS, lv.nb, lv.ntile, 8, 8, P // 80, 10, 8).permute(1, 0, 5, 2, 3, 6, 4, 7)  # band col cb80 t rb cb tok ch
    out = np.zeros((lv.nb, COLS, P // 80, lv.ntile, ct), np.uint8)
    out[..., :64 * 160] = v.bfloat16().reshape(lv.nb, COLS, P // 80, lv.ntile, -1).view(torch.uint8).numpy()
    return out.reshape(lv.nb, -1)


def write_dtensor(bd: Builder, t: DTensor, x: torch.Tensor, ch: Chans) -> None:
    for b, v in enumerate(dtensor_bytes(t, x, ch)):
        bd.image(t.band(b), v)


def read_dtensor(npu: Sd15Device, t: DTensor, ch: Chans) -> torch.Tensor:
    """t's real channels as (1, C, H, H)."""
    lv, P, ct = t.lv, ch.P, L.ct_bytes(64)
    raw = np.stack([npu.read(t.band(b), COLS * t.nbytes) for b in range(lv.nb)]).reshape(lv.nb, COLS, P // 80, lv.ntile, ct)
    v = torch.from_numpy(np.ascontiguousarray(raw[..., :64 * 160])).view(torch.bfloat16)
    y = v.reshape(lv.nb, COLS, P // 80, lv.ntile, 8, 10, 8, 8).permute(1, 0, 3, 4, 6, 2, 5, 7)  # col band t rb tok cb80 cb ch
    return from_tokens(ch.unpad(y.float().reshape(COLS, -1, P)), lv)


def with_stats(bd: Builder, x: DTensor, ch: Chans, *, name: str) -> None:
    """x's GroupNorm records (a copy op per band: K 0, x as the residual), for a DTensor written by the host."""
    lv = x.lv
    rec = bd.mem.alloc(4 * D.OCH, top=True)
    for b in range(lv.nb):
        xb = load(bd, x, b)
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
        xb = load(bd, x, band)
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
        xb = load(bd, x, band)
        t = alloc(bd, x.lv, ch.P, "pn")
        plain(bd, xb, t, name=p + "xpn")
        free(bd, xb)
        img.store_up(bd, band, t, x.lv, ch)
    return conv_layer(bd, img, w[p + "conv.weight"], w[p + "conv.bias"], ch=ch, name=name)


def conv_in(bd: Builder, w: dict, lat: VImage, *, scale: float, name: str) -> DTensor:
    """The latents' 1 / scaling_factor, post_quant_conv and conv_in as one 3x3 conv of the latents' image (``latents``:
    64 channels, the 4 latent ones, a ones channel (post_quant_conv's bias through conv_in's taps: zero past the image's
    edges, as the padding of post_quant_conv's output), zeros) -> 512 channels with records."""
    wi, bi = w["decoder.conv_in.weight"], w["decoder.conv_in.bias"]
    wq, bq = w["post_quant_conv.weight"][:, :, 0, 0], w["post_quant_conv.bias"]
    w64 = torch.zeros(wi.shape[0], 64, 3, 3)
    w64[:, :wq.shape[1]] = torch.einsum("omyx,mi->oiyx", wi, wq) / scale
    w64[:, wq.shape[1]] = torch.einsum("omyx,m->oyx", wi, bq)
    return conv_layer(bd, lat, w64, bi, ch=chans(wi.shape[0]), name=name)


def latents(lat: torch.Tensor) -> torch.Tensor:
    """The sampler's latents (1, 4, 64, 64) as ``conv_in``'s input channels (1, 64, 64, 64)."""
    x = torch.zeros(1, 64, *lat.shape[2:])
    x[:, :lat.shape[1]] = lat
    x[:, lat.shape[1]] = 1
    return x


OUTB = L512.ntile * 8 * 128  # (conv_out's bytes per band and column: [tile][row block][64] of output channels 0..7)


def conv_out(bd: Builder, w: dict, x: DTensor, *, C: int, eps: float, name: str) -> None:
    """conv_norm_out (GroupNorm + SiLU) and conv_out of the 512 level's x (C channels): its 3 output channels on every core row (one
    slice: Gemm tsplit 4), their first 8 channels' blocks drained into DDR image ``name`` ([band][col][OUTB]:
    ``image``)."""
    img = norm(bd, x, w["decoder.conv_norm_out.weight"], w["decoder.conv_norm_out.bias"], ch=chans(C),
               name=name + ".n", eps=eps)
    ch = Chans(w["decoder.conv_out.weight"].shape[0], 1)
    wp, bp = ch.pad(w["decoder.conv_out.weight"]), ch.pad(w["decoder.conv_out.bias"])
    lv, ct = x.lv, L.ct_bytes(64)
    base = bd.ddr(name, lv.nb * COLS * OUTB)
    bd.written(name)
    nxt = None
    for band in range(lv.nb):
        if nxt is None:
            nxt = img.rows(bd, band)
            bd.op(nxt[2], {})
        x, kw, _, cb = (*nxt, None) if len(nxt) == 3 else nxt
        cb = cb or alloc(bd, lv, ch.P)
        nxt = prefetch(bd, lambda band=band: (*img.rows(bd, band + 1), alloc(bd, lv, ch.P))) if band + 1 < lv.nb else None
        conv(bd, x, kw, wp, bp, ch=ch, name=name + ".c", band=band, nb=lv.nb, out=cb, extra=nxt[2] if nxt else None)
        free(bd, x)
        bd.defer({c: ColOps(drains=[MemTask([{"addr": cb.addr, "nwords": OUTB // 4, "dims": [(32, 1), (8, 320), (lv.ntile, ct // 4)]}],
                                            ddr=(base + (band * COLS + c) * OUTB, OUTB, None))]) for c in range(COLS)},
                 release=[(cb.addr, cb.size or cb.nbytes)])
    bd.op({}, {})
    bd.ddr_sync()


def image(raw: np.ndarray, n: int = 3) -> torch.Tensor:
    """conv_out's output image ([band][col][OUTB]) -> (1, n, 512, 512)."""
    lv = L512
    v = torch.from_numpy(np.ascontiguousarray(raw).reshape(lv.nb, COLS, lv.ntile, 8, 8, 16)).view(torch.bfloat16)[..., :n]
    t = v.float().permute(1, 0, 2, 3, 4, 5).reshape(COLS, lv.nb * lv.T, n)  # (col, band tile rb token, ch)
    return from_tokens(t, lv)


def decoder(bd: Builder, w: dict, lat: VImage, *, scale: float, eps: float, name: str) -> None:
    """The decoder from the latents' image: conv_in, the mid block (resnet, attention, resnet), the up blocks (their
    upsamplers), conv_norm_out and conv_out into DDR image ``name``."""
    h = conv_in(bd, w, lat, scale=scale, name="vae.conv_in")
    C = w["decoder.conv_in.weight"].shape[0]
    h = resnet(bd, w, "decoder.mid_block.resnets.0.", h, cin=chans(C), cout=chans(C), eps=eps, name="vae.mid0")
    h = mid_attention(bd, w, "decoder.mid_block.attentions.0.", h, eps=eps, name="vae.attn")
    h = resnet(bd, w, "decoder.mid_block.resnets.1.", h, cin=chans(C), cout=chans(C), eps=eps, name="vae.mid1")
    for i in range(4):
        for j in range(3):
            p = f"decoder.up_blocks.{i}.resnets.{j}."
            cout = w[p + "conv1.weight"].shape[0]
            h = resnet(bd, w, p, h, cin=chans(C), cout=chans(cout), eps=eps, name=f"vae.u{i}.{j}")
            C = cout
        if i < 3:
            h = upsample(bd, w, f"decoder.up_blocks.{i}.upsamplers.0.", h, ch=chans(C), name=f"vae.u{i}.up")
    conv_out(bd, w, h, C=C, eps=eps, name=name)


class Decoder:
    """The VAE decoder on a Sd15Device as one dispatch: ``self(latents)``, the sampler's latents (1, 4, 64, 64) -> the
    image (1, 3, 512, 512) in about [-1, 1]. Weights by diffusers name (``decoder.*``, ``post_quant_conv.*``)."""

    def __init__(self, npu: Sd15Device, w: dict, *, scale: float, eps: float = 1e-6):
        self.npu = npu
        bd = Builder(npu)
        self.lat = VImage(64, "vae.lat", L64)
        self.lat.base(bd)
        decoder(bd, w, self.lat, scale=scale, eps=eps, name="vae.out")
        self.words = bd.finish()
        self.tr = None
        self.addrs, self.images = bd.s.t.addrs, bd.images

    def state(self) -> dict:
        """What ``restored`` needs besides the device's images (the transaction's shim addresses as relocations)."""
        return {"lat": self.lat, "words": self.words, "rel": self.npu.relocations(self.addrs)}

    @classmethod
    def restored(cls, npu: Sd15Device, state: dict) -> "Decoder":
        """A Decoder of ``state()`` on a device whose images were restored (``Sd15Device.restore``)."""
        d = cls.__new__(cls)
        d.npu, d.tr, d.lat = npu, None, state["lat"]
        d.words = npu.relocate(state["words"], state["rel"])
        return d

    def __call__(self, lat: torch.Tensor) -> torch.Tensor:
        if self.tr is None:
            self.tr = Transaction(self.npu.dev, self.words)
        self.npu.write(self.lat.name, image_bytes(self.lat, latents(lat)))
        self.npu.run(self.tr)
        return image(self.npu.read("vae.out"))


GK = 16  # (keys of a d-512 attention's group: four B slots)


@dataclass
class Attention512:
    """Single-head attention of d 512 (the mid block's, role v: ``core``'s attn512) for the queries of a band: per core
    row ``units`` query panels of the column's (q: panels [d-block 64][np], scaled by log2 e / sqrt d), every core
    against all ``nkeys`` keys of the K / V^T image at DDR kv ([group][K d-blocks 0..31, 32..63, V^T likewise][Q];
    ``kv_drains``); the output panels at o ([d-block 80][np]: d-blocks 64..79 junk, the output chunks' padding)."""

    np: int
    q: int
    o: int
    kv: int
    nkeys: int
    padded: bool = False  # (q's kbs in Chans(512)'s stored layout: 4 slices of 20, 16 real)

    def __post_init__(self) -> None:
        assert self.nkeys % GK == 0 and self.np % 8 == 0
        self.units = self.np // 4  # (per core row)
        self.ng = self.nkeys // GK

    def command(self) -> np.ndarray:
        cmd = np.zeros(D.Q // 4, np.int32)
        cmd[[K.C_KIND, K.C_NUNIT, K.C_NG]] = K.K_ATTN, self.units, self.ng
        return cmd.view(np.uint8)

    def kv_bytes(self) -> int:
        return self.ng * 4 * D.Q

    def col_ops(self) -> ColOps:
        """Per row its query panels [units r ..) as A objects of two ([unit 2][d-block 64][panel]), their outputs."""
        co, kb = ColOps(), self.np * 18  # (words: a kb's panels)
        u = self.units
        for r in range(4):
            p0 = r * u
            dims = [(18, 1), (16, kb), (4, 20 * kb), (0, 18)] if self.padded else [(18, 1), (64, kb), (2, 18)]
            co.sends[r] = [MemTask([{"addr": self.q + 72 * p0, "nwords": D.A_OBJ // 4, "dims": dims,
                                     "iteration": (36, u // 2)}], repeat=u // 2)]
            co.recvs[r] = [MemTask([{"addr": self.o + 72 * p0, "nwords": u * 80 * 18, "dims": [(18, 1), (80, kb), (u, 18)]}])]
        return co

    def row_ops(self, cmd: int) -> dict[int, RowOps]:
        """Per row the command, then per unit the K / V^T image (every core takes it: the row's broadcast)."""
        return {r: RowOps(runs=[(cmd, D.Q)] + [(self.kv, self.kv_bytes())] * self.units,
                          b_objects=1 + self.units * 4 * self.ng, parity=False, role="v") for r in range(4)}


def kv512_image(k: torch.Tensor, v: torch.Tensor) -> np.ndarray:
    """K and V (keys, 512) as Attention512's image ([group][slot 4][Q]: K panels [d-block 32][key panel 2] of d-blocks
    0..31, of 32..63, then V^T panels likewise; a slot's last 1152 bytes unused) (tests)."""
    nkp = k.shape[0] // 8
    kp, vp = L.panels(k).reshape(64, nkp, 72), L.vt_panels(v).reshape(64, nkp, 72)
    out = np.zeros((nkp // 2, 4, D.Q), np.uint8)
    for sl, src in enumerate((kp[:32], kp[32:], vp[:32], vp[32:])):
        out[:, sl, :4608] = src.reshape(32, nkp // 2, 2, 72).transpose(1, 0, 2, 3).reshape(nkp // 2, -1)
    return out.reshape(-1)


def kv_drains(at: Attention512, t: Tensor, slot: int, band: int, col: int) -> ColOps:
    """Column ``col``'s drains of band ``band``'s K (slot 0) or V^T (slot 2) panels t (Chans(512)'s stored layout
    [kb 80][np]) into the image's slots slot, slot + 1 (d-blocks 0..31, 32..63) of its key groups ([d-block 32][key
    panel 2] each)."""
    kb, ng = t.np * 18, t.np // 2  # (words: a kb's panels; the column's groups in the band)
    g0 = (band * COLS + col) * ng
    return ColOps(drains=[MemTask([{"addr": t.addr + 4 * 40 * h * kb, "nwords": ng * 1152,
                                    "dims": [(36, 1), (16, kb), (2, 20 * kb), (0, 36)]}],
                                  ddr=(at.kv + (4 * g0 + slot + h) * D.Q, ng * 4608, (576, 2, 576, 4 * D.Q // 4)))
                          for h in range(2)])


def mid_attention(bd: Builder, w: dict, p: str, x: DTensor, *, eps: float, name: str) -> DTensor:
    """The mid block's attention (one head of 512 over the 4096 pixels) of x (the 64 level's, stored 512 channels):
    GroupNorm, then per band q (scaled by log2 e / sqrt 512: parked in DDR), k and v^T (into the K / V^T image), then
    per band the attention (``Attention512``, every core row in role v), proj_attn + x -> a DTensor with records for
    its norm."""
    lv, ch = x.lv, chans(512)
    ns = ("query.", "key.", "value.", "proj_attn.") if p + "query.weight" in w else ("to_q.", "to_k.", "to_v.", "to_out.0.")
    (wq, bq), (wk, bk), (wv, bv), (wo, bo) = ((w[p + n + "weight"].flatten(1), w[p + n + "bias"]) for n in ns)
    c = LOG2E / ch.C ** 0.5
    gn = norm(bd, x, w[p + "group_norm.weight"], w[p + "group_norm.bias"], ch=ch, name=name + ".n", eps=eps, silu=False)
    at0 = Attention512(np=lv.np, q=0, o=0, kv=0, nkeys=lv.H ** 2, padded=True)
    at0.kv = bd.ddr(name + ".kv", at0.kv_bytes())
    bd.written(name + ".kv")
    qd = DTensor(lv, ch.P, name + ".q", "pn")
    rb = lv.R // lv.nb

    pw = {n: (ch.pad(wt), ch.pad(bt)) for n, wt, bt in (("q", wq * c, bq * c), ("k", wk, bk), ("v", wv, bv), ("o", wo, bo))}

    def proj(t: Tensor, n: str, out: str, extra: dict | None = None) -> Tensor:
        y = alloc(bd, lv, ch.P, "pn")
        bd.gemm(Gemm(m=64, ntile=lv.ntile, K=ch.C, N=ch.P, x=t.addr, y=y.addr, out=out, x_np=t.np, y_np=y.np), *pw[n],
                name=f"{name}.{n}", extra=extra)
        return y
    for b in range(lv.nb):
        xg, cols = gn.fill(bd, rb * b, rb * b + rb, 0, gn.rowp // 2, 0, ch.C)
        bd.op(cols, {})
        qt = proj(xg, "q", "bfp16")
        kt = proj(xg, "k", "bfp16")
        vt = proj(xg, "v", "vt", extra={col: kv_drains(at0, kt, 0, b, col) for col in range(COLS)})
        free(bd, xg)
        free(bd, kt)
        bd.defer({col: kv_drains(at0, vt, 2, b, col) for col in range(COLS)}, release=[(vt.addr, vt.size)])
        qd.store(bd, b, qt)
    out = DTensor(lv, ch.P, name, rec=[])
    rec = bd.mem.alloc(4 * D.OCH, top=True)
    cmd = bd.image(name + ".cmd", at0.command())
    bd.reading(name + ".kv")
    for b in range(lv.nb):
        qb = load(bd, qd, b)
        o = alloc(bd, lv, ch.P, "pn")
        at = Attention512(np=lv.np, q=qb.addr, o=o.addr, kv=at0.kv, nkeys=at0.nkeys, padded=True)
        bd.op({col: at.col_ops() for col in range(COLS)}, at.row_ops(cmd))
        free(bd, qb)
        xb = load(bd, x, b)
        g = Gemm(m=64, ntile=lv.ntile, K=ch.C, N=ch.P, x=o.addr, y=xb.addr, out="bf16", res=xb.addr, x_np=o.np,
                 stats="gn", rec=rec, gsize=ch.gsize, ch_step=ch.stride, ch_real=ch.per, stf=stf(b, lv.nb))
        bd.gemm(g, *pw["o"], name=f"{name}.o.{g.stf}", wname=name + ".o.w")
        free(bd, o)
        out.store(bd, b, xb)
    out.rec = records(bd, rec, ch=ch, name=name + ".rec")
    return out
