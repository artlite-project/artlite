"""A network's tensors and layers on a ``build.Builder``: the vision overlay's ops (``ops``) as the networks call
them (``models.sd15``'s UNet and VAE decoder, ``models.minicpmv``'s encoder).

Tensors: ``Tensor`` (C tiles or panels of a level in the memtile, at an address of every column: ``alloc``, ``free``)
and ``DTensor`` (a banded level's tensor in DDR: per band its columns' memtile images, ``DTensor.load`` / ``store``).
Layers: ``gemm`` (a GEMM or 1x1 conv of panels, + a residual, with its output's GroupNorm or LayerNorm records),
``conv`` (3x3), ``plain`` (C tiles as panels), ``copy_stats`` (the GroupNorm records of C tiles), the norms' applies
into panels (``gn_apply``, ``ln_apply``), ``attend`` (8 heads on K / V^T slots in DDR), and the activation tables of the
epilogues (``silu_table``, ``gelu_table``).
"""

import functools
from dataclasses import dataclass

import numpy as np
import torch

from ..operators import kmath
from ..operators.design import COLS, ROWS
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
    w = kmath.SILU.table_words()
    return np.concatenate([w, w]).view(np.uint8)


def silu_table(bd: Builder) -> None:
    bd.table("silu", _silu())


@functools.cache
def gelu_table(table: kmath.ActTable = kmath.GELU) -> np.ndarray:
    """A GELU's table (kmath.GELU, or GELU_ERF) for the GEGLU / GELU epilogues' table pass (the SiLU table's form, fed
    t' = 2 t: its segment index and remainder exactly 2x GELU's, so c1 / 2, c2 / 4; each copy padded to SiLU's words)."""
    c = table.coefficients.copy()
    c[:, 1] /= 2
    c[:, 2] /= 4
    w = table.table_words(c)
    w = np.concatenate([w, np.zeros(len(kmath.SILU.table_words()) - len(w), np.uint32)])
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


def band_view(t: Tensor, b: int) -> Tensor:
    """Band b of a whole column's panel tensor t (its token panels from b np; the kb stride t's)."""
    return Tensor(t.lv, t.C, t.addr + b * t.lv.np * L.PANEL, "pn", pnp=t.np)
