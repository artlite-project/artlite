"""A network's tensors and layers on a ``builder.Builder``: the vision overlay's ops (``ops``) as the networks call
them (the encoders'; ``models.sd15``'s UNet and VAE decoder).

Tensors: ``Tensor`` (C tiles or panels of a level in the memtile, at an address of every column: ``alloc``, ``free``)
and ``DTensor`` (a banded level's tensor in DDR: per band its columns' memtile images, ``DTensor.load`` / ``store``).
Layers: ``gemm`` (a GEMM or 1x1 conv of panels, + a residual, with its output's GroupNorm or LayerNorm records) and the
activation tables of the epilogues (``silu_table``, ``gelu_table``). The convolutional networks' own layers are
``models.sd15.layers``.
"""

import functools
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ..operators import kmath
from . import design as D
from . import layout as L
from .builder import Builder, Records
from .levels import Level
from .ops import Gemm
from .weights import flat


class Kind(StrEnum):
    """What a tensor holds: bf16 C tiles or bfp16 panels."""

    CT = "ct"
    PN = "pn"


@dataclass
class Tensor:
    """A tensor of a level in the memtile: C tiles (kind "ct") or panels ("pn") of C channels at ``addr``; its
    GroupNorm records for the next norm (``rec``)."""

    lv: Level
    C: int
    addr: int
    kind: Kind = Kind.CT
    rec: list[Records] | None = None
    pnp: int = 0  # (panels: token panels per kb, if not the level's: a GEGLU output's padding)
    size: int = 0  # (the bytes allocated: ``alloc``)

    def __post_init__(self) -> None:
        self.kind = Kind(self.kind)

    @property
    def np(self) -> int:
        return self.pnp or self.lv.np

    @property
    def nbytes(self) -> int:
        return self.lv.ct(self.C) if self.kind == "ct" else self.C // 8 * self.np * L.PANEL


def alloc(bd: Builder, lv: Level, C: int, kind: Kind = Kind.CT, pnp: int = 0, top: bool = False, exact: bool = False) -> Tensor:
    """A memtile tensor (panels of tiles of 64: room for whole A objects of 2 k-tiles, the GEMMs' walk reads them;
    exact: none, the 64x64 convs' input rows)."""
    t = Tensor(lv, C, 0, Kind(kind), pnp=pnp)
    t.size = t.nbytes if t.kind == Kind.CT or lv.m != 64 or exact else -(-C // 128) * 16 * t.np * L.PANEL
    t.addr = bd.mem.alloc(t.size, top=top)
    return t


def free(bd: Builder, t: Tensor) -> None:
    bd.mem.release(t.addr, t.size or t.nbytes)


@functools.cache
def silu_words() -> np.ndarray:
    w = kmath.SILU.table_words()
    return np.concatenate([w, w]).view(np.uint8)


def silu_table(bd: Builder) -> None:
    bd.table("silu", silu_words())


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


def gemm(bd: Builder, x: Tensor, w: np.ndarray, b: np.ndarray | None, *, name: str, out: Tensor,
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
        bd.gemm(Gemm(**kw, **part), flat(w), b, name=name if split == 1 else f"{name}.{k}", extra=extra if k == 0 else None)
    if stats:
        out.rec = [bd.records(rec, name=f"{name}.rec", ch_off=stats[1], nr=w.shape[0] // 4, gsize=stats[0], band=band,
                              nbands=lv.nb)]
    return rec


def ln_rec(lv: Level) -> int:
    """Bytes of a band's LayerNorm records ([tile][core][OCH])."""
    return lv.ntile * 4 * D.OCH


@dataclass
class DTensor:
    """A tensor of a banded level in DDR: per band b its columns' memtile images ([col][band tensor bytes], DDR image
    ``band(b)``: ``Builder.spill`` / ``load``) of C tiles (kind "ct") or panels ("pn"); its GroupNorm records for the
    next norm (every band's)."""

    lv: Level
    C: int
    name: str
    kind: Kind = Kind.CT
    rec: list[Records] | None = None

    def __post_init__(self) -> None:
        self.kind = Kind(self.kind)

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
        bd.op(bd.fill(self.band(b), self.nbytes, mem=t.addr, nbytes=t.nbytes, off=off), {})
        t.rec = self.rec
        return t

    def store(self, bd: Builder, b: int, t: Tensor, *, defer: bool = True, c0: int = 0) -> None:
        """Band b (C tiles: its channels from c0) from the memtile tensor t (drained with the next op, t's bytes released
        after it; defer False: an op of its own, t freed); its records."""
        off = self.lv.ct(c0) if c0 else 0
        cols = bd.drain(self.band(b), self.nbytes, mem=t.addr, nbytes=t.nbytes, off=off)
        if defer:
            bd.defer(cols, release=[(t.addr, t.size or t.nbytes)])
        else:
            bd.op(cols, {})
            free(bd, t)
        self.rec = (self.rec or []) + (t.rec or [])


