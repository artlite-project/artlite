"""MiniCPM-V 4.7's vision encoder (projector ``minicpmv4_7``: a SigLIP2 ViT of 27 layers, the ViT merger after layer 6
(the 16x file; the 4x file has none), the final 2 x 2 merger) on sd15's static design (``operators.sd15``: one design for both models, its ops built by
host-built transactions).

Channels: the hidden 1152 padded to ``C`` = 1280 (passes of 80 on 4 core rows): the residual's 1152 real channels,
then 128 that zero weight rows, biases, gamma and beta keep zero (LayerNorm's 1 / n over the 1152); a head's 72 dims
padded to 80 (Q / K / V rows, O's columns: 16 heads of d 80); the feed-forward's 4304 to ``FFN`` = 4480.

A dispatch (``encode``) runs the whole encoder on two entries of at most 64 final groups each (sd15's two images) or
on one of at most 128 (joint: the entry over both images, its keys one sequence); the official slicer's slices have
60-64 groups, an overview or a small image's slices up to ~90 (``plan``). The levels: ``PRE`` (before the ViT merger:
4 bands of one 64-token tile per column, the residual and its records in DDR per band, ``Bands``) and ``POST`` (one
band, in the memtile from the merger to the final merger). The entries' sizes reach the dispatch as data only: their
patch panels and position rows (``inputs``) and the levels' key masks (``level_keys``), so ``Encoder`` builds two
transactions once (pair, joint) and rewrites those per dispatch.

A layer (``layer``) on a level: LN1 -> K, V (their slots drained to DDR), Q (x log2 e / sqrt 72) -> attention (16
heads, the softmax's max raised past attention sinks, key groups of padding skipped) -> O + x (LayerNorm records) ->
LN2 (x parked in DDR meanwhile) -> up + GELU (panels) -> x back -> down + x (records). Its memtile layout alternates
between two (x low / high), the same every other layer. On ``PRE`` (``layer_bands``) per band: its keys first, then
its queries.
"""

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import torch

from .. import act_lut
from ..design import COLS
from ..overlay.runtime import Transaction
from ..sd15 import core as K
from ..sd15 import design as D
from ..sd15 import layout as L
from ..sd15.build import Builder, Level
from ..sd15.ops import Apply, Attention, Gemm, _dims
from ..sd15.runtime import Sd15Device
from ..sd15.seq import ColOps, MemTask, RowOps
from ..sd15.unet import DTensor, Tensor, alloc, free, gelu_table, gemm, ln_rec

H, C, HD, HP, HEADS = 1152, 1280, 72, 80, 16
FFN_REAL, FFN = 4304, 4480
EPS = 1e-6
LOG2E = 1.4426950408889634
GROUPS = 64  # final groups (4 x 4 patches) of one image of a band: a tile's 8 columns x 8 lanes


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


PRE, POST = vit_level(4), vit_level(1)  # (the levels before and after the ViT merger)


@dataclass(frozen=True)
class Entry:
    """An entry's patch grid (ph x pw, multiples of 4) on the ViT's levels (``PRE`` before the ViT merger, ``POST``
    after it; sd15's two images: token panel 2 k + image of a tile), laid out so that both 2 x 2 merges are block
    copies: final group G (its 4 x 4 patches; raster order of the ph / 4 x pw / 4 grid) in image ``img`` (joint: image
    G // 64, the entry alone on the levels) at column G % 64 // 8, lane G % 8 of the level after the merger; its window
    k (2 x 2 patches, in the merge's order (0, 0), (0, 1), (1, 0), (1, 1)) in token panel 2 k + image; before the merger
    the window's token s in band s, at its window's place. At most 64 groups (joint: 128)."""

    ph: int
    pw: int
    img: int = 0
    joint: bool = False

    def __post_init__(self) -> None:
        assert self.ph % 4 == 0 and self.pw % 4 == 0, (self.ph, self.pw)
        assert self.groups <= 2 * GROUPS, f"an entry of {self.groups} groups (at most {2 * GROUPS})"

    @property
    def groups(self) -> int:
        return self.ph // 4 * (self.pw // 4)

    @property
    def cells(self) -> int:
        """The images of 64 groups the entry takes."""
        return 2 if self.joint else 1

    def _place(self, g: np.ndarray) -> tuple[np.ndarray, ...]:
        """Final groups -> their image, column and lane."""
        assert self.groups <= self.cells * GROUPS, f"an entry of {self.groups} groups in {self.cells} image(s)"
        return (g // GROUPS if self.joint else np.full_like(g, self.img)), g % GROUPS // 8, g % 8

    def patch_slots(self) -> tuple[np.ndarray, ...]:
        """Per patch (raster order): its band before the merger, column, panel of its image (0 .. 3), lane and image."""
        r, c = np.divmod(np.arange(self.ph * self.pw), self.pw)
        img, col, ln = self._place(r // 4 * (self.pw // 4) + c // 4)
        return (r % 2) * 2 + c % 2, col, (r // 2 % 2) * 2 + c // 2 % 2, ln, img

    def fill_slots(self) -> tuple[np.ndarray, ...]:
        """``patch_slots`` of every group of the entry's images (64 each): a group past the entry's takes the patches of
        group G mod groups (its values typical: bfp16 blocks over tokens, V^T's, mix real and padding groups; the
        padding's keys masked), and per slot its patch's index."""
        n, gw = self.cells * GROUPS, self.pw // 4
        g = np.repeat(np.arange(n), 16)
        k = np.tile(np.arange(16), n)  # (the patch of the group: row k // 4, column k % 4)
        src = g % self.groups
        r, c = 4 * (src // gw) + k // 4, 4 * (src % gw) + k % 4
        img, col, ln = self._place(g)
        return (r % 2) * 2 + c % 2, col, (r // 2 % 2) * 2 + c // 2 % 2, ln, img, r * self.pw + c

    def window_slots(self) -> tuple[np.ndarray, ...]:
        """Per window of the ViT merger's output (raster order of the ph / 2 x pw / 2 grid): its band (0), column,
        panel of its image, lane and image."""
        h, w = self.ph // 2, self.pw // 2
        r, c = np.divmod(np.arange(h * w), w)
        img, col, ln = self._place(r // 2 * (w // 2) + c // 2)
        return np.zeros_like(r), col, (r % 2) * 2 + c % 2, ln, img


def place(entries: list[Entry]) -> list[Entry]:
    """The entries of one dispatch on the levels: two of at most 64 groups (images 0 and 1), or one alone (joint)."""
    if len(entries) == 1:
        return [replace(entries[0], img=0, joint=True)]
    assert len(entries) == 2 and all(e.groups <= GROUPS for e in entries), [e.groups for e in entries]
    return [replace(e, img=i, joint=False) for i, e in enumerate(entries)]


def plan(entries: list[Entry]) -> list[list[int]]:
    """An image's entries (any order) as dispatches (their indices): those of at most 64 groups in pairs, the others
    (and an odd one) alone."""
    small = [i for i, e in enumerate(entries) if e.groups <= GROUPS]
    out = [small[k:k + 2] for k in range(0, len(small), 2)]
    return out + [[i] for i, e in enumerate(entries) if e.groups > GROUPS]


def pack(rows: list[torch.Tensor], slots: list[tuple[np.ndarray, ...]], nb: int) -> torch.Tensor:
    """Entries' rows (n_i, k) at their slots (``Entry.patch_slots`` / ``window_slots``) -> per band and column its
    tile's tokens (nb, COLS, 64, k), zeros elsewhere."""
    out = torch.zeros(nb, COLS, 64, rows[0].shape[1])
    for x, (b, c, k, ln, img) in zip(rows, slots, strict=True):
        out[b, c, (2 * k + img) * 8 + ln] = x.float()
    return out


def unpack(tiles: torch.Tensor, slots: list[tuple[np.ndarray, ...]]) -> list[torch.Tensor]:
    """``pack``'s inverse: per entry its rows."""
    return [tiles[b, c, (2 * k + img) * 8 + ln] for b, c, k, ln, img in slots]


def key_valid(entries: list[Entry], nb: int, slots: str) -> tuple[np.ndarray, np.ndarray]:
    """Per image the validity of its keys in the attention's key order ([band][column][panel][lane], ``ops.Attention``;
    a joint entry's: image 0's keys, then image 1's, for both) on a level of nb bands: ``slots`` "patch" (before the
    merger) or "window" (after it)."""
    ok = np.zeros((2, nb, COLS, 4, 8), bool)
    for e in entries:
        b, c, k, ln, img = e.patch_slots() if slots == "patch" else e.window_slots()
        ok[img, b, c, k, ln] = True
    if any(e.joint for e in entries):
        return ok.reshape(-1), ok.reshape(-1)
    return ok[0].reshape(-1), ok[1].reshape(-1)


@dataclass(frozen=True)
class Keys:
    """A level's attention keys: per image their validity (``key_valid``; None: all), joint (one sequence of both
    images' keys: a lone entry's); ``name``: the DDR image of the level's attention command (shared by its layers:
    ``Encoder`` rewrites its key masks per dispatch)."""

    name: str
    valid: tuple | None = None
    joint: bool = False

    def attention(self, lv: Level, **kw: int) -> Attention:
        return Attention(np=lv.np, d=HP, nbands=lv.nb, heads=HEADS, raise_max=True, valid=self.valid, joint=self.joint,
                         **{"q": 0, "o": 0, "kv": 0} | kw)


def _heads(w: torch.Tensor) -> torch.Tensor:
    """Rows of 16 heads of 72 -> 16 heads of 80 (zero rows after each head's)."""
    out = torch.zeros((HEADS * HP,) + w.shape[1:], dtype=w.dtype)
    out.view(HEADS, HP, *w.shape[1:])[:, :HD] = w.reshape(HEADS, HD, *w.shape[1:])
    return out


def _pad(w: torch.Tensor, rows: int, cols: int = 0) -> torch.Tensor:
    out = torch.zeros((rows, cols) if cols else (rows,), dtype=w.dtype)
    out[tuple(slice(0, n) for n in w.shape)] = w
    return out


def _attn_weights(get: Callable[[str], torch.Tensor], p: str) -> dict[str, torch.Tensor]:
    """The padded q, k, v (rows by head; q x log2 e / sqrt 72), o (columns by head) of prefix p, their biases."""
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
    return w


def layer_weights(get: Callable[[str], torch.Tensor], il: int) -> dict[str, torch.Tensor]:
    """Layer il's padded weights (float32) by role: q, k, v (rows by head), o (columns by head), up, down and their
    biases (``*_b``), ln1 / ln2 gamma and beta (``*_g``, ``*_b``); ``get(name)``: a tensor of the mmproj as
    (rows, cols) or (cols,)."""
    p = f"v.blk.{il}."
    w = _attn_weights(get, p)
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


def attend(bd: Builder, q: Tensor, kv: int, keys: Keys) -> Tensor:
    """The attention of the scaled queries' panels q on the K / V^T slots at DDR kv (their level's ``keys``): the
    output panels."""
    o = alloc(bd, q.lv, C, "pn")
    for t in range(max(q.lv.np // 8, 1)):  # (an op per tile)
        at = keys.attention(q.lv, q=q.addr, o=o.addr, kv=kv, tile=t)
        cmd = bd.image(keys.name, at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd))
    return o


def _kv(bd: Builder, w: dict[str, torch.Tensor], name: str, n: Tensor, kv: int, keys: Keys, band: int, *,
        q: bool) -> Tensor | None:
    """K and V of the LN1 panels n (their slots' drains: band ``band``'s keys), and Q (q: its panels; else None, V^T's
    drains with the next op)."""
    lv = n.lv
    kt, vt = alloc(bd, lv, C, "pn"), alloc(bd, lv, C, "pn")
    at = keys.attention(lv, kv=kv)
    gemm(bd, n, w["k"], w["k_b"], name=name + ".k", out=kt, kind="bfp16")
    gemm(bd, n, w["v"], w["v_b"], name=name + ".v", out=vt, kind="vt",
         extra={c: at.kv_drains(kt.addr, vt.addr, c, part="k", band=band) for c in range(COLS)})
    free(bd, kt)
    vd = {c: at.kv_drains(kt.addr, vt.addr, c, part="v", band=band) for c in range(COLS)}
    if not q:
        bd.defer(vd, release=[(vt.addr, vt.size)])
        return None
    qt = alloc(bd, lv, C, "pn")
    gemm(bd, n, w["q"], w["q_b"], name=name + ".q", out=qt, kind="bfp16", extra=vd)
    free(bd, vt)
    return qt


def _attn_out(bd: Builder, w: dict[str, torch.Tensor], name: str, qt: Tensor, x: Tensor, kv: int, keys: Keys) -> int:
    """x += O (the attention of the queries qt, freed): x's records."""
    o = attend(bd, qt, kv, keys)
    free(bd, qt)
    rec = gemm(bd, o, w["o"], w["o_b"], name=name + ".o", out=x, res=x, ln=True)
    free(bd, o)
    return rec


def _ffn(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int, park: str = "") -> tuple[Tensor, int]:
    """LN2, x parked in DDR (image ``park``, default name.x), up + GELU, x back, down + x: the output's C tiles (x's
    memory freed) and records."""
    park = park or name + ".x"
    lv = x.lv
    # (the down GEMM reads the hidden panels and writes y: 506 of the memtile's 510 KB; layers alternate between x low,
    # the LN2 panels at the top and the hidden from the bottom (y then high), and x high, the LN2 panels at the bottom
    # and the hidden at the top (y then low))
    low = x.addr < D.HEAP // 2
    n = alloc(bd, lv, C, "pn", top=low)
    bd.defer(bd.spill(park, x.addr, x.nbytes), release=[(x.addr, x.size or x.nbytes)])  # (with LN2: the hidden
    # panels take the memtile)
    ln(bd, x, n, w["ln2_g"], w["ln2_b"], name=name + ".ln2", rec=rec)
    bd.table("gelu", gelu_table())
    g = alloc(bd, lv, FFN, "pn", top=not low)
    gemm(bd, n, w["up"], w["up_b"], name=name + ".up", out=g, kind="gelu")
    free(bd, n)
    y = alloc(bd, lv, C)
    bd.op(bd.load(park, y.addr, y.nbytes), {})
    rec = gemm(bd, g, w["down"], w["down_b"], name=name + ".down", out=y, res=y, ln=True)
    free(bd, g)
    return y, rec


def layer(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int, *, kv: int,
          keys: Keys) -> tuple[Tensor, int]:
    """A ViT layer (module docstring) on the residual's C tiles x with its LayerNorm records at rec (its producer's):
    the output's C tiles (x's memory freed) and their records. kv: a DDR buffer for the attention's K / V^T slots
    (``kv_bytes``); keys: the level's (``Keys``)."""
    n = alloc(bd, x.lv, C, "pn")
    ln(bd, x, n, w["ln1_g"], w["ln1_b"], name=name + ".ln1", rec=rec)
    qt = _kv(bd, w, name, n, kv, keys, 0, q=True)
    bd.ddr_sync()
    free(bd, n)
    return _ffn(bd, w, name, x, _attn_out(bd, w, name, qt, x, kv, keys))


class Bands:
    """A banded level's residual in DDR: per band its C tiles (``DTensor`` image ``name.b``) and LayerNorm records
    (image ``name.rec.b``)."""

    def __init__(self, lv: Level, name: str):
        self.lv, self.name = lv, name
        self.x = DTensor(lv, C, name)

    def load(self, bd: Builder, b: int) -> tuple[Tensor, int]:
        """Band b's C tiles and records in the memtile (one op of fills)."""
        lv = self.lv
        t = alloc(bd, lv, C)
        rec = bd.mem.alloc(ln_rec(lv))
        bd.op(bd.merge(bd.load(self.x.band(b), t.addr, t.nbytes), bd.load(f"{self.name}.rec.{b}", rec, ln_rec(lv))), {})
        return t, rec

    def store(self, bd: Builder, b: int, t: Tensor, rec: int) -> None:
        """Band b from the memtile's C tiles t and records rec (drained with the next op, their bytes released after
        it)."""
        bd.defer(bd.merge(bd.spill(self.x.band(b), t.addr, t.nbytes), bd.spill(f"{self.name}.rec.{b}", rec, ln_rec(self.lv))),
                 release=[(t.addr, t.size or t.nbytes), (rec, ln_rec(self.lv))])


def layer_bands(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Bands, out: Bands, *, kv: int, keys: Keys) -> None:
    """A ViT layer on a banded level (each entry's keys in every band): per band LN1, K and V into the slots; then per
    band LN1 again, Q, the attention, O + x and the feed-forward into ``out``'s band (x's bands read twice; Q computed
    in the second pass instead of parked in DDR)."""
    for b in range(x.lv.nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, x.lv, C, "pn")
        ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=f"{name}.ln1", rec=rec)
        free(bd, xb)
        _kv(bd, w, name, n, kv, keys, b, q=False)
        free(bd, n)
    bd.op({}, {})  # (the last band's V^T drains)
    bd.ddr_sync()
    for b in range(x.lv.nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, x.lv, C, "pn")
        ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=f"{name}.ln1", rec=rec)
        qt = alloc(bd, x.lv, C, "pn")
        gemm(bd, n, w["q"], w["q_b"], name=name + ".q", out=qt, kind="bfp16")
        free(bd, n)
        y, rec = _ffn(bd, w, name, xb, _attn_out(bd, w, name, qt, xb, kv, keys), park=f"{name}.x.{b}")
        out.store(bd, b, y, rec)


def kv_bytes(lv: Level) -> int:
    """Bytes of a level's K / V^T slots (``layer``'s kv; joint or not)."""
    return Keys("").attention(lv).kv_bytes()


def patch_weights(get: Callable[[str], torch.Tensor]) -> dict[str, torch.Tensor]:
    """The patch embedding as a GEMM (588 = 3 x 14 x 14 inputs (c, ky, kx) padded to 640 -> C) and its bias."""
    w = get("v.patch_embd.weight").float().reshape(H, -1)
    return {"w": _pad(w, C, 640), "b": _pad(get("v.patch_embd.bias").float().reshape(-1), C)}


def patches(pixels: np.ndarray | torch.Tensor, patch: int = 14) -> torch.Tensor:
    """An entry's pixels (3, H, W) -> its patches (ph pw, 640): (c, ky, kx) of each, raster order, zero-padded."""
    x = torch.as_tensor(np.asarray(pixels), dtype=torch.float32)
    ph, pw = x.shape[1] // patch, x.shape[2] // patch
    p = x.reshape(3, ph, patch, pw, patch).permute(1, 3, 0, 2, 4).reshape(ph * pw, -1)
    return torch.nn.functional.pad(p, (0, 640 - p.shape[1]))


def patch_embed(bd: Builder, w: dict[str, torch.Tensor], name: str) -> Bands:
    """The patch embedding of a dispatch's entries into the level before the merger: per band a GEMM of the host's
    patch panels (DDR image name.pix.b) with the position rows (name.pos.b) as its residual -> the residual's bands and
    records (``inputs`` writes those images)."""
    lv = PRE
    out = Bands(lv, name)
    pn, cn = lv.pn(640), lv.ct(C)
    for b in range(lv.nb):
        pi, ri = bd.ddr(f"{name}.pix.{b}", COLS * pn), bd.ddr(f"{name}.pos.{b}", COLS * cn)
        pt = alloc(bd, lv, 640, "pn")
        x = alloc(bd, lv, C)
        bd.op({c: ColOps(fills=[MemTask([{"addr": pt.addr, "nwords": pn // 4}], ddr=(pi + c * pn, pn, None)),
                               MemTask([{"addr": x.addr, "nwords": cn // 4}], ddr=(ri + c * cn, cn, None))])
               for c in range(COLS)}, {})
        rec = gemm(bd, pt, w["w"], w["b"], name=name + ".gemm", out=x, res=x, ln=True)
        free(bd, pt)
        out.store(bd, b, x, rec)
    return out


def inputs(entries: list[Entry], pix: list[torch.Tensor], pos: list[torch.Tensor], name: str) -> dict[str, np.ndarray]:
    """``patch_embed``'s DDR images for a dispatch's placed entries (``place``): their patches (``patches``: pix) and
    position rows (pos: (n, C)) at every group of their images (``Entry.fill_slots``)."""
    fills = [e.fill_slots() for e in entries]
    slots = [f[:5] for f in fills]
    xp = pack([p[f[5]] for p, f in zip(pix, fills, strict=True)], slots, PRE.nb)
    xr = pack([p[f[5]] for p, f in zip(pos, fills, strict=True)], slots, PRE.nb)
    out = {}
    for b in range(PRE.nb):
        out[f"{name}.pix.{b}"] = np.concatenate([L.panels(xp[b, c]) for c in range(COLS)])
        out[f"{name}.pos.{b}"] = np.concatenate([L.ctiles(xr[b, c], PRE.m) for c in range(COLS)])
    return out


@dataclass
class WindowAttention:
    """The ViT merger's attention inside its 2 x 2 windows (``core`` K_WATTN, role w) for the windows of two token
    panels per column (a window per lane; its 4 tokens the 4 sources): the memtile panel tensors of the sources' Q
    (x log2 e / sqrt 72) and K at q, k, their V^T at vt, the outputs at o; source s's [kb 160][panel 2][72 B] at
    base + s stride. Core row r: heads 4 r .. 4 r + 3, per head its A objects Q, K, V^T (a panel per half: [d-block 16
    (10 used)][source 4]) and its output [panel][d-block 10][source 4]."""

    q: int
    k: int
    vt: int
    o: int
    stride: int

    def command(self) -> np.ndarray:
        cmd = np.zeros(D.Q // 4, np.int32)
        cmd[[K.C_KIND, K.C_NUNIT]] = (K.K_WATTN, HEADS // 4)
        return cmd.view(np.uint8)

    def col_ops(self) -> ColOps:
        co = ColOps()
        sw = self.stride // 4
        walk = _dims((18, 1), (4, sw), (16, 36), (2, 18))  # (a panel, the sources, 16 d-blocks, the panels: 9216 B)
        head = 10 * 36  # (words: a head's 10 d-blocks)
        for r in range(4):
            h0 = 4 * r * head
            sends = [MemTask([{"addr": a + 4 * h0, "nwords": D.A_OBJ // 4, "dims": walk, "iteration": (head, 4)}
                              for a in (self.q, self.k, self.vt)], repeat=4)]
            recvs = [MemTask([{"addr": self.o + 4 * h0, "nwords": 1440, "dims": _dims((18, 1), (4, sw), (10, 36), (2, 18)),
                               "iteration": (head, 4)}], repeat=4)]
            co.sends[r], co.recvs[r] = sends, recvs
        return co

    def row_ops(self, cmd: int) -> dict[int, RowOps]:
        return {r: RowOps(runs=[(cmd, D.Q)], b_objects=1, parity=False, role="w") for r in range(4)}


MERGED, HID = 4 * C, 17280  # (a merged token's channels: 4 x 1280; the ViT merger's hidden, 17216 padded)


def _blocks4(v: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """A merged token's 4 x 1152 channels (along axis) -> 4 x 1280 (each source's padded)."""
    v = v.float().movedim(axis, -1)
    out = torch.zeros(*v.shape[:-1], 4, C)
    out[..., :H] = v.reshape(*v.shape[:-1], 4, H)
    return out.reshape(*v.shape[:-1], MERGED).movedim(-1, axis)


def merger_weights(get: Callable[[str], torch.Tensor]) -> dict[str, torch.Tensor]:
    """The ViT merger's padded weights: its attention's (``_attn_weights``) and ln1; ds_ln (4608 -> 5120, per source),
    up (5120 -> 17280) and down (17280 + 5120 -> 1280: the mean of the merged token's 4 sources as its last K, 1 / 4
    per source channel)."""
    p = "v.vit_merger."
    w = _attn_weights(get, p)
    w["ln1_g"], w["ln1_b"] = (_pad(get(p + f"ln1.{n}").float().reshape(-1), C) for n in ("weight", "bias"))
    w["ds_g"], w["ds_b"] = (_blocks4(get(p + f"ds_ln.{n}").reshape(-1)) for n in ("weight", "bias"))
    w["up"] = _pad(_blocks4(get(p + "ds_ffn_up.weight"), 1), HID, MERGED)
    w["up_b"] = _pad(get(p + "ds_ffn_up.bias").float().reshape(-1), HID)
    mean = torch.zeros(C, MERGED)
    for k in range(4):
        mean[torch.arange(H), C * k + torch.arange(H)] = 0.25
    w["down"] = torch.cat([_pad(get(p + "ds_ffn_down.weight").float(), C, HID), mean], 1)
    w["down_b"] = _pad(get(p + "ds_ffn_down.bias").float().reshape(-1), C)
    return w


def _xfer(bd: Builder, name: str, image_bytes: int, *, fill: bool, mem: int, nbytes: int, off: int = 0,
          mdims: list | None = None, sdims: tuple | None = None) -> dict[int, ColOps]:
    """Per column a fill (DDR image ``name`` -> memtile at mem) or drain of nbytes: column c's bytes of the image
    ([col][image_bytes]) from off, the memtile walk mdims (``ops._dims`` form), the shim's sdims (``txn.shim_bd``)."""
    base = bd.ddr(name, COLS * image_bytes)
    (bd.reading if fill else bd.written)(name)
    bd_ = {"addr": mem, "nwords": nbytes // 4} | ({"dims": mdims} if mdims else {})
    t = lambda c: MemTask([bd_], ddr=(base + c * image_bytes + off, nbytes, sdims))
    return {c: ColOps(fills=[t(c)]) if fill else ColOps(drains=[t(c)]) for c in range(COLS)}


def merger(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Bands) -> tuple[Tensor, int]:
    """The ViT merger (module docstring) from the level before it (x: ``PRE``) into ``POST``'s band, left in the
    memtile: per source band LN1 -> Q, K, V^T panels (to DDR); per merged tile (two token panels: 8 windows of each
    image per column) the window attention of the 4 sources' rows -> o's rows (to DDR); per source band O + x (to DDR);
    per merged tile its 4 sources' rows of that as one tile of 16 merged tokens (5120 channels), their records and
    panels (the mean's K, to DDR), LayerNorm(4608) (to DDR: the band's panels); up + GELU on the band's 64 merged
    tokens a column (m 64: its 100 MB of weights streamed once, not per tile of 16), the hidden to DDR in chunks; per
    merged tile its hidden and mean panels back, down + the mean -> the tile's rows of the band (DDR); the band's C
    tiles and LayerNorm records."""
    lv, lp, l16 = x.lv, POST, vit_level(1, 16)
    pn, ct = lv.pn(C), lv.ct(C)  # (bytes per column: a band's panels, C tiles)
    ob = L.ct_bytes(64)  # (an 80-channel block's C-tile object in a band)
    rows = 2 * 72  # (a kb's two token panels)
    src = C // 8 * rows  # (a source's rows: [kb 160][2][72])
    pad = 1024  # (past each source's rows: the window attention's walk reads 16 d-blocks a head)
    x2, y2 = DTensor(lv, C, name + ".x2"), DTensor(lp, C, name + ".y")
    for s in range(4):  # (Q, K, V^T of each source band)
        xb, rec = x.load(bd, s)
        n = alloc(bd, lv, C, "pn")
        ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=name + ".ln1", rec=rec)
        free(bd, xb)
        for role, kind in (("q", "bfp16"), ("k", "bfp16"), ("v", "vt")):
            o = alloc(bd, lv, C, "pn")
            gemm(bd, n, w[role], w[role + "_b"], name=f"{name}.{role}", out=o, kind=kind)
            bd.defer(_xfer(bd, f"{name}.{role}.{s}", pn, fill=False, mem=o.addr, nbytes=pn), release=[(o.addr, o.size)])
        free(bd, n)
    for j in range(4):  # (the window attention of merged tile j: token panels 2 j, 2 j + 1 of the sources)
        base = {r: bd.mem.alloc(4 * (src + pad)) for r in ("q", "k", "v", "o")}
        sd = (36, C // 8, 144, C // 8 * 36)  # (per kb its two panels' 36 words, 576 B apart)
        for r in ("q", "k", "v"):  # (an op per role: a memtile channel's pool holds 8 BDs)
            bd.op(bd.merge(*[_xfer(bd, f"{name}.{r}.{s}", pn, fill=True, mem=base[r] + s * (src + pad), nbytes=src,
                                   off=2 * j * 72, sdims=sd) for s in range(4)]), {})
        wa = WindowAttention(q=base["q"], k=base["k"], vt=base["v"], o=base["o"], stride=src + pad)
        bd.op({c: wa.col_ops() for c in range(COLS)}, wa.row_ops(bd.image(name + ".wcmd", wa.command())))
        drains = [_xfer(bd, f"{name}.o.{s}", pn, fill=False, mem=base["o"] + s * (src + pad), nbytes=src,
                        off=2 * j * 72, sdims=sd) for s in range(4)]
        bd.op(bd.merge(*drains), {})
        for a in base.values():
            bd.mem.release(a, 4 * (src + pad))
    for s in range(4):  # (O + x of each source band)
        xb, rec = x.load(bd, s)
        bd.mem.release(rec, ln_rec(lv))
        o = alloc(bd, lv, C, "pn")
        bd.op(_xfer(bd, f"{name}.o.{s}", pn, fill=True, mem=o.addr, nbytes=pn), {})
        gemm(bd, o, w["o"], w["o_b"], name=name + ".o", out=xb, res=xb)
        free(bd, o)
        x2.store(bd, s, xb, defer=False)
    pn16, pn64 = l16.pn(MERGED), lp.pn(MERGED)  # (bytes per column: a merged tile's panels, the band's)
    for j in range(4):  # (merged tile j's LayerNorm(4608) panels into the band's (DDR: its token panels 2 j, 2 j + 1)
        # and its panels for the mean's K (DDR))
        xp = alloc(bd, l16, MERGED, "pn")
        nl = alloc(bd, l16, MERGED, "pn")
        mc = _merged(bd, [(x2.band(s), 2 * j) for s in range(4)], name)
        rec = bd.mem.alloc(ln_rec(l16))
        for h in range(2):  # (the panels and records in two copies of 8 passes: a command's biases reach 14; one record
            # over both)
            bd.gemm(Gemm(m=16, ntile=1, K=0, N=MERGED, x=0, y=xp.addr, out="bfp16", res=mc.addr, stats="ln", rec=rec,
                         p0=8 * h, npo=8, stf=(K.ST_HOLD, K.ST_KEEP)[h]), None, None, name=f"{name}.mrec{h}")
        for h in range(2):  # (LayerNorm(4608) in two applies of 2560 channels: 640 a core)
            a = Apply(m=16, nblk=8, mode="ln", x=mc.addr, y=nl.addr, blk0=32 * h, y_kb0=320 * h, rec=rec, C=4 * H,
                      eps=EPS, y_np=nl.np)
            bd.apply(a, w["ds_g"], w["ds_b"], name=f"{name}.ds_ln{h}")
        bd.mem.release(rec, ln_rec(l16))
        free(bd, mc)
        bd.op(bd.merge(_xfer(bd, name + ".nl", pn64, fill=False, mem=nl.addr, nbytes=pn16, off=2 * j * 72,
                             sdims=(36, MERGED // 8, 144, MERGED // 8 * 144)),
                       _xfer(bd, name + ".xp", 4 * pn16, fill=False, mem=xp.addr, nbytes=pn16, off=j * pn16)), {})
        free(bd, nl)
        free(bd, xp)
    # up + GELU on the band's 64 merged tokens a column (its weights streamed once), the hidden to DDR in chunks of 960
    hb = HID // 8 * lp.pn(8)  # (a column's hidden panels: [kb 2160][token panel 8][72 B])
    bd.table("gelu", gelu_table())
    nl = alloc(bd, lp, MERGED, "pn")
    spare = bd.mem.alloc(D.A_OBJ)  # (the GEMM's A walk reads past its panels' end)
    bd.op(_xfer(bd, name + ".nl", pn64, fill=True, mem=nl.addr, nbytes=pn64), {})
    nc = 960  # (3 passes a row)
    g = alloc(bd, lp, nc, "pn")
    for c in range(HID // nc):
        gemm(bd, nl, w["up"][nc * c:nc * (c + 1)], w["up_b"][nc * c:nc * (c + 1)], name=f"{name}.up{c}", out=g,
             kind="gelu")
        bd.op(_xfer(bd, name + ".hid", hb, fill=False, mem=g.addr, nbytes=g.nbytes, off=c * g.nbytes), {})
    free(bd, g)
    bd.mem.release(spare, D.A_OBJ)
    free(bd, nl)
    for j in range(4):  # (merged tile j's down + the mean of 4: its hidden and mean panels from DDR)
        spare = bd.mem.alloc(4096, top=True)  # (the down GEMM's A walk reads past its panels' end)
        xp = alloc(bd, l16, MERGED, "pn", top=True)
        hid = alloc(bd, l16, HID, "pn", top=True)  # (directly below xp: the down GEMM's K is [hidden | xp])
        assert hid.addr + hid.size == xp.addr, (hid.addr, hid.size, xp.addr)
        bd.op(bd.merge(_xfer(bd, name + ".hid", hb, fill=True, mem=hid.addr, nbytes=hid.nbytes, off=2 * j * 72,
                             sdims=(36, HID // 24, 144, HID // 24 * 144)),
                       _xfer(bd, name + ".xp", 4 * pn16, fill=True, mem=xp.addr, nbytes=pn16, off=j * pn16)), {})
        y = alloc(bd, l16, C)
        gemm(bd, Tensor(l16, HID + MERGED, hid.addr, "pn"), w["down"], w["down_b"], name=name + ".down", out=y)
        free(bd, hid)
        free(bd, xp)
        bd.mem.release(spare, 4096)
        ct16 = l16.ct(80)
        bd.op(_xfer(bd, y2.band(0), ct, fill=False, mem=y.addr, nbytes=16 * 2560, off=2 * j * 1280,
                    mdims=_dims((640, 1), (16, ct16 // 4)), sdims=(640, 16, ob // 4, 16 * ob // 4)), {})
        free(bd, y)
    xt = alloc(bd, lp, C)
    bd.op(_xfer(bd, y2.band(0), ct, fill=True, mem=xt.addr, nbytes=ct), {})
    return xt, ln_records(bd, xt, name=name + ".rec")


FUP = 4800  # (the final merger's hidden: 4608 padded to 15 passes a row)
OUT = 1024  # (the language model's width)


def final_weights(get: Callable[[str], torch.Tensor]) -> dict[str, torch.Tensor]:
    """The final merger's padded weights: post_ln, the identity (post_ln's panels back to C tiles), input_norm (4608 ->
    5120, per source), up (5120 -> 4800), down (4800 -> 1280: 1024 real)."""
    w = {"post_g": _pad(get("v.post_ln.weight").float().reshape(-1), C), "post_b": _pad(get("v.post_ln.bias").float().reshape(-1), C)}
    w["eye"] = torch.eye(C)
    w["in_g"], w["in_b"] = (_blocks4(get(f"mm.input_norm.{n}").reshape(-1)) for n in ("weight", "bias"))
    w["up"] = _pad(_blocks4(get("mm.up.weight"), 1), FUP, MERGED)
    w["up_b"] = _pad(get("mm.up.bias").float().reshape(-1), FUP)
    w["down"] = _pad(get("mm.down.weight").float(), C, FUP)
    w["down_b"] = _pad(get("mm.down.bias").float().reshape(-1), C)
    return w


def _post(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int) -> Tensor:
    """post_ln of a band's C tiles x (its records at rec; x freed), its panels back to C tiles by an identity GEMM."""
    n = alloc(bd, x.lv, C, "pn")
    ln(bd, x, n, w["post_g"], w["post_b"], name=name + ".post", rec=rec)
    free(bd, x)
    y = alloc(bd, x.lv, C)
    gemm(bd, n, w["eye"], None, name=name + ".eye", out=y)
    free(bd, n)
    return y


def _merged(bd: Builder, src: list[tuple[str, int]], name: str) -> Tensor:
    """A tile of 16 merged tokens (5120 channels: C tiles of m 16) from 4 sources' token panels (DDR images of a band's
    C tiles, ``src``: per source its image and its first of two token panels): block copies."""
    l16, ct, ob = vit_level(1, 16), POST.ct(C), L.ct_bytes(64)
    ct16 = l16.ct(80)  # (an 80-channel block's object at m 16: 2560 B of 2 rows and padding)
    mc = alloc(bd, l16, MERGED)
    bd.op(bd.merge(*[_xfer(bd, img, ct, fill=True, mem=mc.addr + 16 * s * ct16, nbytes=16 * 2560, off=tp * 1280,
                           mdims=_dims((640, 1), (16, ct16 // 4)), sdims=(640, 16, ob // 4, 16 * ob // 4))
                     for s, (img, tp) in enumerate(src)]), {})
    return mc


def _final_mlp(bd: Builder, w: dict[str, torch.Tensor], name: str, mc: Tensor) -> Tensor:
    """The final merger's MLP on a tile of 16 merged tokens (mc, freed): their records, LayerNorm(4608), up + GELU
    (erf), down -> the embeddings' C tiles (m 16, 1280 channels: 1024 real)."""
    l16 = mc.lv
    nl = alloc(bd, l16, MERGED, "pn")
    rec = bd.mem.alloc(ln_rec(l16))
    for h in range(2):  # (records over both copies; their panels into nl, overwritten by the LayerNorm)
        bd.gemm(Gemm(m=16, ntile=1, K=0, N=MERGED, x=0, y=nl.addr, out="bfp16", res=mc.addr, stats="ln", rec=rec,
                     p0=8 * h, npo=8, stf=(K.ST_HOLD, K.ST_KEEP)[h]), None, None, name=f"{name}.mrec{h}")
    for h in range(2):
        a = Apply(m=16, nblk=8, mode="ln", x=mc.addr, y=nl.addr, blk0=32 * h, y_kb0=320 * h, rec=rec, C=4 * H, eps=EPS,
                  y_np=nl.np)
        bd.apply(a, w["in_g"], w["in_b"], name=f"{name}.in_ln{h}")
    bd.mem.release(rec, ln_rec(l16))
    free(bd, mc)
    bd.table("gelu_erf", gelu_table(act_lut.GELU_ERF))
    g = alloc(bd, l16, FUP, "pn")
    n3 = FUP // 3  # (3 GEMMs of 5 passes a row: a command's biases reach 14)
    for k in range(3):
        gemm(bd, nl, w["up"][n3 * k:n3 * (k + 1)], w["up_b"][n3 * k:n3 * (k + 1)], name=f"{name}.up{k}",
             out=Tensor(l16, n3, g.addr + n3 // 8 * l16.np * 72 * k, "pn"), kind="gelu")
    free(bd, nl)
    e = alloc(bd, l16, C)
    gemm(bd, g, w["down"], w["down_b"], name=name + ".down", out=e)
    free(bd, g)
    return e


def final(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Tensor, rec: int) -> str:
    """The final merger of ``POST``'s band (x, its records at rec; freed): post_ln (``_post``, to DDR), the band's 4
    windows of each group as one tile of 16 merged tokens (``_merged``: a DDR round trip of block copies), the MLP
    (``_final_mlp``) -> the embeddings: DDR image ``name.out`` ([col][16 x 1280 channels as C tiles of 16];
    ``embeddings`` reads it)."""
    y = _post(bd, w, name, x, rec)
    bd.op(_xfer(bd, f"{name}.p", y.nbytes, fill=False, mem=y.addr, nbytes=y.nbytes), {})
    free(bd, y)
    e = _final_mlp(bd, w, name, _merged(bd, [(f"{name}.p", 2 * k) for k in range(4)], name))
    l16 = e.lv
    bd.op(_xfer(bd, name + ".out", l16.ct(C), fill=False, mem=e.addr, nbytes=l16.ct(C)), {})
    free(bd, e)
    return name + ".out"


def final_bands(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Bands) -> str:
    """The final merger of ``PRE``'s bands (the 4x file, without the ViT merger: an output token per window, its 4
    tokens in bands 0 .. 3): per band post_ln (``_post``, to DDR); per merged tile k (window k of both images' groups:
    token panels 2 k, 2 k + 1 of the bands) the 4 bands' rows as one tile of 16 merged tokens, the MLP -> the
    embeddings: DDR image ``name.out`` ([col][tile 4][16 x 1280 channels as C tiles of 16])."""
    p = DTensor(x.lv, C, name + ".p")
    for s in range(4):
        y = _post(bd, w, name, *x.load(bd, s))
        p.store(bd, s, y, defer=False)
    t16 = vit_level(1, 16).ct(C)
    for k in range(4):
        e = _final_mlp(bd, w, name, _merged(bd, [(p.band(s), 2 * k) for s in range(4)], name))
        bd.op(_xfer(bd, name + ".out", 4 * t16, fill=False, mem=e.addr, nbytes=t16, off=k * t16), {})
        free(bd, e)
    return name + ".out"


def embeddings(npu: Sd15Device, image: str, entries: list[Entry], merge: int = 4) -> list[torch.Tensor]:
    """The final merger's output (``final``'s image; merge 2: ``final_bands``') -> per placed entry its embeddings
    (tokens, 1024) in raster order of its final grid (merge 4: its groups; 2: its windows)."""
    l16 = vit_level(1, 16)
    nt = 1 if merge == 4 else 4
    raw = npu.read(image, COLS * nt * l16.ct(C)).reshape(COLS, nt, -1)
    tiles = torch.stack([torch.stack([L.from_ctiles(raw[c, t], 16, C, 16)[:, :OUT] for t in range(nt)])
                         for c in range(COLS)])  # ([col][tile] (16 tokens: image 0's 8 lanes, then image 1's), 1024)
    out = []
    for e in entries:
        if merge == 4:
            img, col, lane = e._place(np.arange(e.groups))
            out.append(tiles[col, 0, 8 * img + lane])
        else:
            _, col, k, lane, img = e.window_slots()
            out.append(tiles[col, k, 8 * img + lane])
    return out


def weights(get: Callable[[str], torch.Tensor], layers: int, insert: int) -> dict:
    """Every padded weight of the encoder: "patch", per layer il its ``layer_weights``, "merger" (after layer
    ``insert``; -1: none, the 4x file), "final"."""
    return {"patch": patch_weights(get), "merger": merger_weights(get) if insert >= 0 else None,
            "final": final_weights(get), **{il: layer_weights(get, il) for il in range(layers)}, "insert": insert,
            "layers": layers}


def level_keys(name: str, entries: list[Entry] | None = None, *, joint: bool) -> tuple[Keys, Keys]:
    """The keys of the levels before and after the ViT merger: their attention commands' images ``name.pre`` /
    ``name.post`` (joint or not: ``attention`` of another key count), the placed entries' validity (None: a build's
    placeholder, every key valid)."""
    v = (None, None) if entries is None else (key_valid(entries, PRE.nb, "patch"), key_valid(entries, POST.nb, "window"))
    return Keys(name + ".pre", v[0], joint), Keys(name + ".post", v[1], joint)


def encode(bd: Builder, w: dict, *, joint: bool, name: str = "v") -> str:
    """One dispatch of the whole encoder (``weights``) on two entries of at most 64 groups each, or one of at most 128
    (joint): the patch embedding (its inputs: ``inputs``, DDR images name.x0.pix.b / .pos.b), layers 0 .. insert on
    ``PRE``'s bands, the ViT merger into ``POST``'s band, the later layers on it in the memtile, the final merger
    (without the ViT merger, the 4x file: every layer on ``PRE``, ``final_bands``); returns the final merger's image
    (``embeddings`` reads it). The levels' attention commands (``level_keys``: name .joint / .pair) hold their key masks
    (``Encoder`` writes them per dispatch)."""
    pre, post = level_keys(f"{name}.{'joint' if joint else 'pair'}", joint=joint)
    x = patch_embed(bd, w["patch"], name + ".x0")
    kv = bd.ddr(name + ".kv", kv_bytes(PRE))
    for il in range(w["insert"] + 1 if w["insert"] >= 0 else w["layers"]):
        y = Bands(PRE, f"{name}.x{il + 1}")
        layer_bands(bd, w[il], f"{name}.l{il}", x, y, kv=kv, keys=pre)
        x = y
    if w["insert"] < 0:
        return final_bands(bd, w["final"], name + ".f", x)
    xt, rec = merger(bd, w["merger"], name + ".mg", x)
    for il in range(w["insert"] + 1, w["layers"]):
        xt, rec = layer(bd, w[il], f"{name}.l{il}", xt, rec, kv=kv, keys=post)
    return final(bd, w["final"], name + ".f", xt, rec)


class Encoder:
    """The encoder (``encode``) as dispatches on a Sd15Device: two transactions built once over the same weight images
    (``Builder.fork``), one for a pair of entries of at most 64 groups each, one for a lone entry (joint). Per dispatch
    (``plan``: an image's entries) the host writes the entries' patch panels and position rows (``inputs``) and the
    levels' key masks into their attention commands (``level_keys``), runs it and reads the embeddings."""

    def __init__(self, npu: Sd15Device, w: dict, pos: torch.Tensor, *, patch: int, name: str = "v"):
        """w: ``weights``; pos: the learned position embedding (70 x 70 rows, 1152) in bf16 values; patch: pixels."""
        self.npu, self.name, self.patch = npu, name, patch
        self.merge = 4 if w["insert"] >= 0 else 2  # (patches per output token per side)
        self.pos = torch.nn.functional.pad(pos.float(), (0, C - H))
        bd = Builder(npu)
        self.words, self.addrs = {}, {}
        for joint, b in ((False, bd), (True, bd.fork())):
            self.out = encode(b, w, joint=joint, name=name)
            self.words[joint], self.addrs[joint] = b.finish(), b.s.t.addrs
        self.images = list(bd.images)  # (both builders': ``fork``)
        self.tr: dict[bool, Transaction] = {}

    def state(self) -> dict:
        """What ``restored`` needs besides the device's images: the transactions (their shim addresses as
        relocations)."""
        return {"words": self.words, "rel": {j: self.npu.relocations(a) for j, a in self.addrs.items()}, "out": self.out,
                "merge": self.merge}

    @classmethod
    def restored(cls, npu: Sd15Device, state: dict, pos: torch.Tensor, *, patch: int, name: str = "v") -> "Encoder":
        """An Encoder of ``state()`` on a device whose images were restored (``Sd15Device.restore``)."""
        enc = cls.__new__(cls)
        enc.npu, enc.name, enc.patch, enc.out, enc.merge, enc.tr = npu, name, patch, state["out"], state["merge"], {}
        enc.pos = torch.nn.functional.pad(pos.float(), (0, C - H))
        enc.words = {j: npu.relocate(wd, state["rel"][j]) for j, wd in state["words"].items()}
        return enc

    def _host(self, ents: list[Entry], pixels: list[np.ndarray]) -> dict[str, np.ndarray]:
        """A dispatch's DDR images: its placed entries' patch panels and position rows, its levels' commands."""
        pos = [self.pos[torch.from_numpy(positions(e.ph, e.pw))] for e in ents]
        out = inputs(ents, [patches(p, self.patch) for p in pixels], pos, self.name + ".x0")
        joint = ents[0].joint
        keys = level_keys(f"{self.name}.{'joint' if joint else 'pair'}", ents, joint=joint)
        for k, lv in list(zip(keys, (PRE, POST), strict=True))[:2 if self.merge == 4 else 1]:  # (4x: no POST level)
            out[k.name] = k.attention(lv).command()
        return out

    def __call__(self, pixels: list[np.ndarray]) -> list[torch.Tensor]:
        """An image's entries (their pixels (3, H, W), normalized) -> their embeddings (tokens, 1024), in order (a
        dispatch's DDR images prepared on a thread while the one before runs)."""
        entries = [Entry(p.shape[1] // self.patch, p.shape[2] // self.patch) for p in pixels]
        out: list[torch.Tensor | None] = [None] * len(entries)
        work = [(idx, place([entries[i] for i in idx])) for idx in plan(entries)]
        with ThreadPoolExecutor(1) as pool:
            host = [pool.submit(self._host, ents, [pixels[i] for i in idx]) for idx, ents in work]
            for (idx, ents), h in zip(work, host, strict=True):
                for n, data in h.result().items():
                    self.npu.write(n, data)
                joint = ents[0].joint
                if joint not in self.tr:
                    self.tr[joint] = Transaction(self.npu.dev, self.words[joint])
                self.npu.run(self.tr[joint])
                for i, e in zip(idx, embeddings(self.npu, self.out, ents, self.merge), strict=True):
                    out[i] = e
        return out


def positions(ph: int, pw: int) -> np.ndarray:
    """The learned position embedding's rows of a ph x pw patch grid (clip.cpp: buckets of a 70 x 70 grid)."""
    bh = np.floor(70.0 * np.arange(ph) / ph).astype(np.int64)
    bw = np.floor(70.0 * np.arange(pw) / pw).astype(np.int64)
    return (bh[:, None] * 70 + bw[None, :]).reshape(-1)
