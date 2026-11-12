"""The ViT layers of the vision encoders on the vision overlay (``models.minicpmv``'s SigLIP2 and ``models.qwen3vl``'s
Qwen3-VL ViTs; the default widths ``VIT``: 1152 wide, 16 heads of 72, a feed-forward of 4304, LayerNorm eps 1e-6; a
``Vit`` of other widths: Qwen3-VL's smaller towers, 1024 / 16 heads of 64 / 4096 and 768 / 12 heads of 64 / 3072).

Channels: the hidden H (1152) padded to ``C`` = 80 x heads (1280; passes of 80 on 4 core rows): the residual's H real
channels, then C - H that zero weight rows, biases, gamma and beta keep zero (LayerNorm's 1 / n over the H); a head's
dims (72) padded to 80 (Q / K / V rows, O's columns: heads of d 80, ``pad_heads``); the feed-forward's 4304 to ``FFN``
= 4480 (whole passes of 320).

The GEMMs' operands are bfp16 panels (C fp32 over the whole K: the overlay's). A layer (``layer``) on a level: LN1 ->
K, V (their slots drained to DDR), Q (x log2 e / sqrt HD) -> attention (its heads, the softmax's max raised past
attention sinks, key groups of padding skipped) -> O + x (LayerNorm records) -> LN2 (x parked in DDR meanwhile) -> up
+ GELU (panels) -> x back -> down + x (records). Its memtile layout alternates between two (x low / high), the same
every other layer. On a banded level (``layer_bands``) per band: its keys first, then its queries. With RoPE (Qwen3-VL)
the K and Q GEMMs turn each head's pairs by the band's cos | sin tile (``ops.Gemm(out="rope")``).

Precision: each layer alone is as close to float64 as the CPU at its rounding points (the tests' ``NpuEncoder``). Over
27 layers the roundings add up to ~4% per token (median; llama.cpp's bf16 activations: ~0.7%), in squares: the bfp16
weights (the down GEMM's most) and the bfp16 GEMM inputs (LN2's and GELU's most) ~35% each, the bf16 residual ~17%,
Q / K ~10% (Qwen3-VL at 512 x 512). A few tokens are much further off: both encoders' last layer writes a massive
activation into channel 514 (Qwen3-VL's: ~2000-12000, through ~30 FFN units that are on in most patches and off in
the rest); where those units are partly on, a small input error becomes a large error of x514, and post_ln divides
the patch by the sigma x514 sets. llama.cpp errs on the same tokens (Qwen3-VL: ~6x less), and which tokens depends on
the last bits of anything (float64 itself moves one by 67% between two JPEG decoders' pixels). Taking them to
llama.cpp's level needs bf16-level operands throughout (hi + lo inputs and weights, an fp32-level residual, Q / K: an
estimated ~2x the encoder's time); moving the largest units or channels into bfp16 blocks of their own (the pads) does
not help.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch  # noqa: TID251

from ..operators.design import COLS
from . import design as D
from .build import Builder, Level
from .layers import DTensor, Tensor, alloc, free, gelu_table, gemm, ln_rec
from .ops import Apply, Attention, Gemm
from .seq import ColOps, MemTask
from .weights import Wt, pad_index, take

H, C, HD, HP, HEADS = 1152, 1280, 72, 80, 16
FFN_REAL, FFN = 4304, 4480
EPS = 1e-6
LOG2E = 1.4426950408889634
QSCALE = LOG2E / math.sqrt(HD)  # (the queries' scale: the softmax in log2 units)


@dataclass(frozen=True)
class Vit:
    """A ViT's widths (module docstring): H real channels of the residual's C, HEADS heads of HD dims (each in HP),
    the feed-forward's FFN_REAL in FFN, LayerNorm's eps."""

    H: int = H
    C: int = C
    HD: int = HD
    HP: int = HP
    HEADS: int = HEADS
    FFN_REAL: int = FFN_REAL
    FFN: int = FFN
    EPS: float = EPS

    @classmethod
    def of(cls, hidden: int, heads: int, ffn: int, eps: float) -> "Vit":
        """The widths of a tower of ``hidden`` channels in ``heads`` heads (of at most 80 dims; 12 or 16 heads) and a
        feed-forward of ``ffn``."""
        hd = hidden // heads
        if hidden % heads or hd > HP or heads not in (12, 16):
            raise NotImplementedError(f"a ViT of {hidden} x {heads} heads (the overlay's: 12 or 16 heads of up to {HP})")
        return cls(hidden, HP * heads, hd, HP, heads, ffn, -(-ffn // 320) * 320, eps)

    @property
    def qscale(self) -> float:
        """(the queries' scale: the softmax in log2 units)"""
        return LOG2E / math.sqrt(self.HD)

    def head_rows(self) -> np.ndarray:
        """``pad_heads``' rows: per head its HD dims, then zeros to HP."""
        return np.array([h * self.HD + d if d < self.HD else -1 for h in range(self.HEADS) for d in range(self.HP)])


VIT = Vit()


@dataclass(frozen=True)
class VitLevel(Level):
    """A ViT level: ``nb`` bands of one tile per column of ``tm`` tokens (64; 16: the merges' tiles of merged tokens;
    both images: token panel tp of image tp % 2)."""

    tm: int = 64

    @property
    def T(self) -> int:
        return self.tm

    @property
    def m(self) -> int:
        return self.tm

    @property
    def ntile(self) -> int:
        return 1

    @property
    def np(self) -> int:
        return self.tm // 8


def vit_level(nb: int, tm: int = 64) -> VitLevel:
    return VitLevel(0, nb, False, tm)


@dataclass(frozen=True)
class Keys:
    """A level's attention keys: per image their validity (``valid``: None, all), joint (one sequence of both images'
    keys), dense (joint: any number of key groups, whole ones before the last valid one: ``ops.Attention``), kv_heads
    (grouped-query attention: ``ops.Attention``); ``name``:
    the DDR image of the level's attention command (shared by its layers: an encoder rewrites its key masks per
    dispatch)."""

    name: str
    valid: tuple | None = None
    joint: bool = False
    dense: bool = False
    kv_heads: int = 0
    heads: int = HEADS

    @property
    def width(self) -> int:
        """(the attention's channels: Q's, its output's)"""
        return self.heads * HP

    def attention(self, lv: Level, **kw: int) -> Attention:
        return Attention(np=lv.np, d=HP, nbands=lv.nb, heads=self.heads, raise_max=True, valid=self.valid,
                         joint=self.joint, dense=self.dense, kv_heads=self.kv_heads, **{"q": 0, "o": 0, "kv": 0} | kw)


def pad_heads(w: Wt | torch.Tensor, dim: int = 0, vt: Vit = VIT) -> Wt | torch.Tensor:
    """Rows (dim 0) or columns (1) of the heads of HD -> heads of HP (zeros after each head's)."""
    return take(w, vt.head_rows(), dim)


def pad(w: Wt | torch.Tensor, rows: int, cols: int = 0) -> Wt | torch.Tensor:
    """w's rows (and columns) zero-padded to rows (cols)."""
    out = take(w, pad_index(rows, w.shape[0]))
    return take(out, pad_index(cols, w.shape[1]), 1) if cols else out


def ln_records(bd: Builder, x: Tensor, *, name: str, C: int = C) -> int:
    """The LayerNorm records of the C tiles x (a copy op with statistics: x from the host, no producer GEMM); returns
    their memtile address."""
    lv = x.lv
    rec = bd.mem.alloc(ln_rec(lv))
    y = bd.mem.alloc(lv.pn(C))
    bd.gemm(Gemm(m=lv.m, ntile=lv.ntile, K=0, N=C, x=0, y=y, out="bfp16", res=x.addr, stats="ln", rec=rec), None, None,
            name=name)
    bd.mem.release(y, lv.pn(C))
    return rec


def ln(bd: Builder, x: Tensor, y: Tensor, gamma: torch.Tensor, beta: torch.Tensor | None, *, name: str, rec: int,
       C: int = C, H: int = H, eps: float = EPS, mode: str = "ln") -> None:
    """LayerNorm (mode "rms": RMSNorm, beta None) of the C tiles x (its producer's records at rec, freed) over their
    first H of C channels into the panels y on every core row."""
    a = Apply(m=x.lv.m, nblk=C // 320, mode=mode, x=x.addr, y=y.addr, ntile=x.lv.ntile, rec=rec, C=H, eps=eps,
              y_np=y.np)
    bd.apply(a, gamma, beta, name=name)
    bd.mem.release(rec, ln_rec(x.lv))


def attend(bd: Builder, q: Tensor, kv: int, keys: Keys) -> Tensor:
    """The attention of the scaled queries' panels q on the K / V^T slots at DDR kv (their level's ``keys``): the
    output panels."""
    o = alloc(bd, q.lv, keys.width, "pn")
    for t in range(max(q.lv.np // 8, 1)):  # (an op per tile)
        at = keys.attention(q.lv, q=q.addr, o=o.addr, kv=kv, tile=t)
        cmd = bd.image(keys.name, at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd))
    return o


def k_view(n: Tensor, w: Wt | torch.Tensor) -> Tensor:
    """The panels n as the A of a GEMM of the weights w: their K, the first of n's channels (the pads skipped)."""
    k = w.shape[1]
    return n if k == n.C else Tensor(n.lv, k, n.addr, "pn", pnp=n.np)


def _qk(bd: Builder, n: Tensor, w: dict[str, torch.Tensor], role: str, name: str, out: Tensor, rope: Tensor | None,
        extra: dict | None = None) -> None:
    """Q or K of the LN1 panels n into the panels out (rope: the band's cos | sin tile, each head's pairs turned)."""
    gemm(bd, k_view(n, w[role]), w[role], w[role + "_b"], name=f"{name}.{role}", out=out, kind="rope" if rope else "bfp16",
         res=rope, extra=extra)


def kv(bd: Builder, w: dict[str, torch.Tensor], name: str, n: Tensor, kvd: int, keys: Keys, band: int, *, q: bool,
       rope: Tensor | None = None) -> Tensor | None:
    """K and V of the LN1 panels n (their slots' drains at DDR kvd: band ``band``'s keys), and Q (q: its panels; else
    None, V^T's drains with the next op); rope: the band's cos | sin tile for K and Q."""
    lv = n.lv
    kt, vt = alloc(bd, lv, keys.width, "pn"), alloc(bd, lv, keys.width, "pn")
    at = keys.attention(lv, kv=kvd)
    _qk(bd, n, w, "k", name, kt, rope)
    gemm(bd, k_view(n, w["v"]), w["v"], w["v_b"], name=name + ".v", out=vt, kind="vt",
         extra={c: at.kv_drains(kt.addr, vt.addr, c, part="k", band=band) for c in range(COLS)})
    free(bd, kt)
    vd = {c: at.kv_drains(kt.addr, vt.addr, c, part="v", band=band) for c in range(COLS)}
    if not q:
        bd.defer(vd, release=[(vt.addr, vt.size)])
        return None
    qt = alloc(bd, lv, keys.width, "pn")
    _qk(bd, n, w, "q", name, qt, rope, extra=vd)
    free(bd, vt)
    return qt


def attn_out(bd: Builder, w: dict[str, torch.Tensor], name: str, qt: Tensor, x: Tensor, kvd: int, keys: Keys) -> int:
    """x += O (the attention of the queries qt, freed): x's records."""
    o = attend(bd, qt, kvd, keys)
    free(bd, qt)
    rec = gemm(bd, o, w["o"], w["o_b"], name=name + ".o", out=x, res=x, ln=True)
    free(bd, o)
    return rec


def ffn(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int, park: str = "", vt: Vit = VIT
        ) -> tuple[Tensor, int]:
    """LN2, x parked in DDR (image ``park``, default name.x), up + GELU, x back, down + x: the output's C tiles (x's
    memory freed) and records."""
    park = park or name + ".x"
    lv = x.lv
    # (the down GEMM reads the hidden panels and writes y: 506 of the memtile's 510 KB; layers alternate between x low,
    # the LN2 panels at the top and the hidden from the bottom (y then high), and x high, the LN2 panels at the bottom
    # and the hidden at the top (y then low))
    low = x.addr < D.HEAP // 2
    n = alloc(bd, lv, vt.C, "pn", top=low)
    bd.defer(bd.spill(park, x.addr, x.nbytes), release=[(x.addr, x.size or x.nbytes)])  # (with LN2: the hidden
    # panels take the memtile)
    ln(bd, x, n, w["ln2_g"], w["ln2_b"], name=name + ".ln2", rec=rec, C=vt.C, H=vt.H, eps=vt.EPS)
    bd.table("gelu", gelu_table())
    g = alloc(bd, lv, vt.FFN, "pn", top=not low)
    gemm(bd, k_view(n, w["up"]), w["up"], w["up_b"], name=name + ".up", out=g, kind="gelu")
    free(bd, n)
    y = alloc(bd, lv, vt.C)
    bd.op(bd.load(park, y.addr, y.nbytes), {})
    rec = gemm(bd, g, w["down"], w["down_b"], name=name + ".down", out=y, res=y, ln=True)
    free(bd, g)
    return y, rec


def layer(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int, *, kv_ddr: int,
          keys: Keys, vt: Vit = VIT) -> tuple[Tensor, int]:
    """A ViT layer (module docstring) on the residual's C tiles x with its LayerNorm records at rec (its producer's):
    the output's C tiles (x's memory freed) and their records. kv_ddr: a DDR buffer for the attention's K / V^T slots
    (``kv_bytes``); keys: the level's (``Keys``)."""
    n = alloc(bd, x.lv, vt.C, "pn")
    ln(bd, x, n, w["ln1_g"], w["ln1_b"], name=name + ".ln1", rec=rec, C=vt.C, H=vt.H, eps=vt.EPS)
    qt = kv(bd, w, name, n, kv_ddr, keys, 0, q=True)
    bd.ddr_sync()
    free(bd, n)
    return ffn(bd, w, name, x, attn_out(bd, w, name, qt, x, kv_ddr, keys), vt=vt)


class Bands:
    """A banded level's residual in DDR: per band its C tiles (``DTensor`` image ``name.b``) and LayerNorm records
    (image ``name.rec.b``)."""

    def __init__(self, lv: Level, name: str, C: int = C):
        self.lv, self.name, self.C = lv, name, C
        self.x = DTensor(lv, C, name)

    def load(self, bd: Builder, b: int) -> tuple[Tensor, int]:
        """Band b's C tiles and records in the memtile (one op of fills)."""
        lv = self.lv
        t = alloc(bd, lv, self.C)
        rec = bd.mem.alloc(ln_rec(lv))
        bd.op(bd.merge(bd.load(self.x.band(b), t.addr, t.nbytes), bd.load(f"{self.name}.rec.{b}", rec, ln_rec(lv))), {})
        return t, rec

    def store(self, bd: Builder, b: int, t: Tensor, rec: int) -> None:
        """Band b from the memtile's C tiles t and records rec (drained with the next op, their bytes released after
        it)."""
        bd.defer(bd.merge(bd.spill(self.x.band(b), t.addr, t.nbytes), bd.spill(f"{self.name}.rec.{b}", rec, ln_rec(self.lv))),
                 release=[(t.addr, t.size or t.nbytes), (rec, ln_rec(self.lv))])


def layer_bands(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Bands, out: Bands, *, kv_ddr: int, keys: Keys,
                park: Callable[[int], str] | None = None, rope: Callable[[Builder, int], Tensor] | None = None,
                vt: Vit = VIT) -> None:
    """A ViT layer on a banded level (each entry's keys in every band): per band LN1, K and V into the slots; then per
    band LN1 again, Q, the attention, O + x and the feed-forward into ``out``'s band (x's bands read twice; Q computed
    in the second pass instead of parked in DDR). park: band b's DDR image for x during its feed-forward (default
    name.x.b); rope: band b's cos | sin tile in the memtile (an op of fills; freed after its K, its Q)."""
    park = park or (lambda b: f"{name}.x.{b}")
    for b in range(x.lv.nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, x.lv, vt.C, "pn")
        ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=f"{name}.ln1", rec=rec, C=vt.C, H=vt.H, eps=vt.EPS)
        free(bd, xb)
        rt = rope(bd, b) if rope else None
        kv(bd, w, name, n, kv_ddr, keys, b, q=False, rope=rt)
        free(bd, n)
        if rt:
            free(bd, rt)
    bd.op({}, {})  # (the last band's V^T drains)
    bd.ddr_sync()
    for b in range(x.lv.nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, x.lv, vt.C, "pn")
        ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=f"{name}.ln1", rec=rec, C=vt.C, H=vt.H, eps=vt.EPS)
        qt = alloc(bd, x.lv, keys.width, "pn")
        rt = rope(bd, b) if rope else None
        _qk(bd, n, w, "q", name, qt, rt)
        free(bd, n)
        if rt:
            free(bd, rt)
        y, rec = ffn(bd, w, name, xb, attn_out(bd, w, name, qt, xb, kv_ddr, keys), park=park(b), vt=vt)
        out.store(bd, b, y, rec)


def kv_bytes(lv: Level, keys: Keys | None = None) -> int:
    """Bytes of a level's K / V^T slots (``layer``'s kv_ddr; of its ``keys``, default: two images apart)."""
    return (keys or Keys("")).attention(lv).kv_bytes()


def xfer(bd: Builder, name: str, image_bytes: int, *, fill: bool, mem: int, nbytes: int, off: int = 0,
         mdims: list | None = None, sdims: tuple | None = None) -> dict[int, ColOps]:
    """Per column a fill (DDR image ``name`` -> memtile at mem) or drain of nbytes: column c's bytes of the image
    ([col][image_bytes]) from off, the memtile walk mdims (``ops._dims`` form), the shim's sdims (``txn.shim_bd``)."""
    base = bd.ddr(name, COLS * image_bytes)
    (bd.reading if fill else bd.written)(name)
    bd_ = {"addr": mem, "nwords": nbytes // 4} | ({"dims": mdims} if mdims else {})
    t = lambda c: MemTask([bd_], ddr=(base + c * image_bytes + off, nbytes, sdims))
    return {c: ColOps(fills=[t(c)]) if fill else ColOps(drains=[t(c)]) for c in range(COLS)}
