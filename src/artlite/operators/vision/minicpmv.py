"""MiniCPM-V 4.7's vision encoder (projector ``minicpmv4_7``: a SigLIP2 ViT of 27 layers, the ViT merger after layer 6,
the final 2 x 2 merger) on sd15's static design (``operators.sd15``: one design for both models, its ops built by
host-built transactions).

Channels: the hidden 1152 padded to ``C`` = 1280 (passes of 80 on 4 core rows): the residual's 1152 real channels,
then 128 that zero weight rows, biases, gamma and beta keep zero (LayerNorm's 1 / n over the 1152); a head's 72 dims
padded to 80 (Q / K / V rows, O's columns: 16 heads of d 80); the feed-forward's 4304 to ``FFN`` = 4480.

A layer (``layer``) on a level of two entries (sd15's two images): LN1 -> K, V (their slots drained to DDR), Q
(x log2 e / sqrt 72) -> attention (16 heads, the softmax's max raised past attention sinks) -> O + x (LayerNorm
records) -> LN2 (x parked in DDR meanwhile) -> up + GELU (panels) -> x back -> down + x (records). Its memtile
layout alternates between two (x low / high), the same every other layer. On a banded level (``layer_bands``: the
layers before the ViT merger, 1024 tokens an entry) the residual and its records are in DDR per band (``Bands``).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..design import COLS
from ..sd15 import core as K
from ..sd15 import design as D
from ..sd15 import layout as L
from ..sd15.build import Builder, Level
from ..sd15.ops import Apply, Attention, Gemm, _dims
from ..sd15.seq import ColOps, MemTask, RowOps
from ..sd15.unet import DTensor, Tensor, alloc, free, gelu_table, gemm, ln_rec

H, C, HD, HP, HEADS = 1152, 1280, 72, 80, 16
FFN_REAL, FFN = 4304, 4480
EPS = 1e-6
LOG2E = 1.4426950408889634
GROUPS = 64  # final groups (4 x 4 patches) of an entry per band: a tile's 8 columns x 8 lanes


@dataclass(frozen=True)
class VitLevel(Level):
    """A ViT level: ``nb`` bands of one tile per column of ``tm`` tokens (64; 16: the merges' tiles of merged tokens;
    both entries: token panel tp of entry tp % 2)."""

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
class Entry:
    """An entry's patch grid (ph x pw, multiples of 4) on the ViT's levels, laid out so that both 2 x 2 merges are block
    copies: final group G (its 4 x 4 patches; raster order of the ph / 4 x pw / 4 grid) in band G // 64 of the levels
    after the ViT merger, column G % 64 // 8, lane G % 8; its window s' (2 x 2 patches, in the merge's order (0, 0),
    (0, 1), (1, 0), (1, 1)) in the entry's token panel s' (panel 2 s' + the entry's image of the tile), the window's
    token s in band 4 (G // 64) + s before the merger, at its window's place. ``bands``: the levels' bands after the
    merger (4 x before it); a pair of entries takes the larger's."""

    ph: int
    pw: int

    def __post_init__(self) -> None:
        assert self.ph % 4 == 0 and self.pw % 4 == 0, (self.ph, self.pw)

    @property
    def groups(self) -> int:
        return self.ph // 4 * (self.pw // 4)

    @property
    def bands(self) -> int:
        return -(-self.groups // GROUPS)

    def patch_slots(self) -> tuple[np.ndarray, ...]:
        """Per patch (raster order): its band before the merger, column, panel of its entry (0 .. 3) and lane."""
        r, c = np.divmod(np.arange(self.ph * self.pw), self.pw)
        g = r // 4 * (self.pw // 4) + c // 4
        return 4 * (g // GROUPS) + (r % 2) * 2 + c % 2, g % GROUPS // 8, (r // 2 % 2) * 2 + c // 2 % 2, g % 8

    def window_slots(self) -> tuple[np.ndarray, ...]:
        """Per window of the ViT merger's output (raster order of the ph / 2 x pw / 2 grid): its band, column, panel of
        its entry and lane."""
        h, w = self.ph // 2, self.pw // 2
        r, c = np.divmod(np.arange(h * w), w)
        g = r // 2 * (w // 2) + c // 2
        return g // GROUPS, g % GROUPS // 8, (r % 2) * 2 + c % 2, g % 8


def pack(rows: list[torch.Tensor], slots: list[tuple[np.ndarray, ...]], nb: int) -> torch.Tensor:
    """Two entries' rows (n_i, k) at their slots (``Entry.patch_slots`` / ``window_slots``) -> per band and column its
    tile's tokens (nb, COLS, 64, k), zeros elsewhere."""
    out = torch.zeros(nb, COLS, 64, rows[0].shape[1])
    for img, (x, (b, c, k, ln)) in enumerate(zip(rows, slots, strict=True)):
        out[b, c, (2 * k + img) * 8 + ln] = x.float()
    return out


def unpack(tiles: torch.Tensor, slots: list[tuple[np.ndarray, ...]]) -> list[torch.Tensor]:
    """``pack``'s inverse: per entry its rows."""
    return [tiles[b, c, (2 * k + img) * 8 + ln] for img, (b, c, k, ln) in enumerate(slots)]


def key_valid(entries: list[Entry], nb: int, slots: str) -> tuple[np.ndarray, np.ndarray]:
    """Per entry its keys' validity in the attention's key order ([band][column][panel][lane], ``ops.Attention``) on
    a level of nb bands: ``slots`` "patch" (before the merger) or "window" (after it)."""
    out = []
    for e in entries:
        ok = np.zeros((nb, COLS, 4, 8), bool)
        b, c, k, ln = e.patch_slots() if slots == "patch" else e.window_slots()
        ok[b, c, k, ln] = True
        out.append(ok.reshape(-1))
    return tuple(out)


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


def attend(bd: Builder, q: Tensor, kv: int, *, name: str, valid: tuple | None = None) -> Tensor:
    """The attention of the scaled queries' panels q on the K / V^T slots at DDR kv (every key of their entry; valid:
    per entry its keys' validity, ``key_valid``): the output panels."""
    o = alloc(bd, q.lv, C, "pn")
    for t in range(max(q.lv.np // 8, 1)):  # (an op per tile)
        at = Attention(np=q.lv.np, q=q.addr, o=o.addr, kv=kv, d=HP, nbands=q.lv.nb, tile=t, heads=HEADS, raise_max=True,
                       valid=valid)
        cmd = bd.image(name, at.command())
        bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(cmd))
    return o


def _kv(bd: Builder, w: dict[str, torch.Tensor], name: str, n: Tensor, kv: int, band: int, *, q: bool) -> Tensor | None:
    """K and V of the LN1 panels n (their slots' drains: band ``band``'s keys), and Q (q: its panels; else None, V^T's
    drains with the next op)."""
    lv = n.lv
    kt, vt = alloc(bd, lv, C, "pn"), alloc(bd, lv, C, "pn")
    at = Attention(np=lv.np, q=0, o=0, kv=kv, d=HP, nbands=lv.nb, heads=HEADS)
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


def _attn_out(bd: Builder, w: dict[str, torch.Tensor], name: str, qt: Tensor, x: Tensor, kv: int,
              valid: tuple | None) -> int:
    """x += O (the attention of the queries qt, freed): x's records."""
    o = attend(bd, qt, kv, name=name + ".attn", valid=valid)
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
          valid: tuple | None = None) -> tuple[Tensor, int]:
    """A ViT layer (module docstring) on the residual's C tiles x with its LayerNorm records at rec (its producer's):
    the output's C tiles (x's memory freed) and their records. kv: a DDR buffer for the attention's K / V^T slots
    (``kv_bytes``); valid: the entries' keys' validity (``key_valid``)."""
    n = alloc(bd, x.lv, C, "pn")
    ln(bd, x, n, w["ln1_g"], w["ln1_b"], name=name + ".ln1", rec=rec)
    qt = _kv(bd, w, name, n, kv, 0, q=True)
    bd.ddr_sync()
    free(bd, n)
    return _ffn(bd, w, name, x, _attn_out(bd, w, name, qt, x, kv, valid))


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


def layer_bands(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Bands, out: Bands, *, kv: int,
                valid: tuple | None = None) -> None:
    """A ViT layer on a banded level (each entry's keys in every band): per band LN1, K and V into the slots; then per
    band LN1 again, Q, the attention, O + x and the feed-forward into ``out``'s band (x's bands read twice; Q computed
    in the second pass instead of parked in DDR)."""
    for b in range(x.lv.nb):
        xb, rec = x.load(bd, b)
        n = alloc(bd, x.lv, C, "pn")
        ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=f"{name}.ln1", rec=rec)
        free(bd, xb)
        _kv(bd, w, name, n, kv, b, q=False)
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
        y, rec = _ffn(bd, w, name, xb, _attn_out(bd, w, name, qt, xb, kv, valid), park=f"{name}.x.{b}")
        out.store(bd, b, y, rec)


def kv_bytes(lv: Level) -> int:
    """Bytes of a level's K / V^T slots (``layer``'s kv)."""
    return Attention(np=lv.np, q=0, o=0, kv=0, d=HP, nbands=lv.nb, heads=HEADS).kv_bytes()


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


def patch_embed(bd: Builder, w: dict[str, torch.Tensor], entries: list[Entry], pix: list[torch.Tensor],
                pos: list[torch.Tensor], lv: VitLevel, name: str) -> Bands:
    """The two entries' patch embeddings (``patches``: pix; their position embedding rows: pos, (n, C)) into the
    banded level lv before the merger: per band a GEMM of the host's patch panels with the position rows as its
    residual -> the residual's bands and records."""
    slots = [e.patch_slots() for e in entries]
    xp, xr = pack(pix, slots, lv.nb), pack(pos, slots, lv.nb)
    out = Bands(lv, name)
    for b in range(lv.nb):
        pi = bd.image(f"{name}.pix.{b}", np.concatenate([L.panels(xp[b, c]) for c in range(COLS)]))
        ri = bd.image(f"{name}.pos.{b}", np.concatenate([L.ctiles(xr[b, c], lv.m) for c in range(COLS)]))
        pt = alloc(bd, lv, 640, "pn")
        x = alloc(bd, lv, C)
        pn, cn = lv.pn(640), lv.ct(C)
        bd.op({c: ColOps(fills=[MemTask([{"addr": pt.addr, "nwords": pn // 4}], ddr=(pi + c * pn, pn, None)),
                               MemTask([{"addr": x.addr, "nwords": cn // 4}], ddr=(ri + c * cn, cn, None))])
               for c in range(COLS)}, {})
        rec = gemm(bd, pt, w["w"], w["b"], name=name + ".gemm", out=x, res=x, ln=True)
        free(bd, pt)
        out.store(bd, b, x, rec)
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


def merger(bd: Builder, w: dict[str, torch.Tensor], name: str, x: Bands, out: Bands) -> None:
    """The ViT merger (module docstring) from the level before it (x: 4 bands per band of out) into out's level: per
    band of out, per source band LN1 -> Q, K, V^T panels (to DDR); per merged tile (two token panels: 8 windows of each
    entry per column) the window attention of the 4 sources' rows -> o's rows (to DDR); per source band O + x (to DDR);
    per merged tile its 4 sources' rows of that as one tile of 16 merged tokens (5120 channels), their records and
    panels (the mean's K), LayerNorm(4608), up + GELU, down + the mean -> the tile's rows of out's band; the band's
    records."""
    lv, lp, l16 = x.lv, out.lv, vit_level(1, 16)
    pn, ct = lv.pn(C), lv.ct(C)  # (bytes per column: a band's panels, C tiles)
    ob = L.ct_bytes(64)  # (an 80-channel block's C-tile object in a band)
    rows = 2 * 72  # (a kb's two token panels)
    src = C // 8 * rows  # (a source's rows: [kb 160][2][72])
    pad = 1024  # (past each source's rows: the window attention's walk reads 16 d-blocks a head)
    x2 = DTensor(lv, C, name + ".x2")
    for t in range(lp.nb):
        for s in range(4):  # (Q, K, V^T of each source band)
            b = 4 * t + s
            xb, rec = x.load(bd, b)
            n = alloc(bd, lv, C, "pn")
            ln(bd, xb, n, w["ln1_g"], w["ln1_b"], name=name + ".ln1", rec=rec)
            free(bd, xb)
            for role, kind in (("q", "bfp16"), ("k", "bfp16"), ("v", "vt")):
                o = alloc(bd, lv, C, "pn")
                gemm(bd, n, w[role], w[role + "_b"], name=f"{name}.{role}", out=o, kind=kind)
                bd.defer(_xfer(bd, f"{name}.{role}.{b}", pn, fill=False, mem=o.addr, nbytes=pn), release=[(o.addr, o.size)])
            free(bd, n)
        for j in range(4):  # (the window attention of merged tile j: token panels 2 j, 2 j + 1 of the sources)
            base = {r: bd.mem.alloc(4 * (src + pad)) for r in ("q", "k", "v", "o")}
            sd = (36, C // 8, 144, C // 8 * 36)  # (per kb its two panels' 36 words, 576 B apart)
            for r in ("q", "k", "v"):  # (an op per role: a memtile channel's pool holds 8 BDs)
                bd.op(bd.merge(*[_xfer(bd, f"{name}.{r}.{4 * t + s}", pn, fill=True, mem=base[r] + s * (src + pad),
                                       nbytes=src, off=2 * j * 72, sdims=sd) for s in range(4)]), {})
            wa = WindowAttention(q=base["q"], k=base["k"], vt=base["v"], o=base["o"], stride=src + pad)
            bd.op({c: wa.col_ops() for c in range(COLS)}, wa.row_ops(bd.image(name + ".wcmd", wa.command())))
            drains = [_xfer(bd, f"{name}.o.{4 * t + s}", pn, fill=False, mem=base["o"] + s * (src + pad), nbytes=src,
                            off=2 * j * 72, sdims=sd) for s in range(4)]
            bd.op(bd.merge(*drains), {})
            for a in base.values():
                bd.mem.release(a, 4 * (src + pad))
        for s in range(4):  # (O + x of each source band)
            b = 4 * t + s
            xb, rec = x.load(bd, b)
            bd.mem.release(rec, ln_rec(lv))
            o = alloc(bd, lv, C, "pn")
            bd.op(_xfer(bd, f"{name}.o.{b}", pn, fill=True, mem=o.addr, nbytes=pn), {})
            gemm(bd, o, w["o"], w["o_b"], name=name + ".o", out=xb, res=xb)
            free(bd, o)
            x2.store(bd, b, xb, defer=False)
        for j in range(4):  # (merged tile j's MLP)
            spare = bd.mem.alloc(4096, top=True)  # (the down GEMM's A walk reads past its panels' end)
            xp = alloc(bd, l16, MERGED, "pn", top=True)
            nl = alloc(bd, l16, MERGED, "pn")  # (below mc: mc's memory and what follows it then takes the hidden)
            mc = alloc(bd, l16, MERGED)
            ct16 = l16.ct(80)  # (an 80-channel block's object at m 16: 2560 B of 2 rows and padding)
            fills = [_xfer(bd, x2.band(4 * t + s), ct, fill=True, mem=mc.addr + 16 * s * ct16, nbytes=16 * 2560,
                           off=2 * j * 1280, mdims=_dims((640, 1), (16, ct16 // 4)), sdims=(640, 16, ob // 4, 16 * ob // 4))
                     for s in range(4)]
            bd.op(bd.merge(*fills), {})
            rec = bd.mem.alloc(ln_rec(l16))
            for h in range(2):  # (the panels and records in two copies of 8 passes: a command's biases reach 14; one
                # record over both)
                bd.gemm(Gemm(m=16, ntile=1, K=0, N=MERGED, x=0, y=xp.addr, out="bfp16", res=mc.addr, stats="ln", rec=rec,
                             p0=8 * h, npo=8, stf=(K.ST_HOLD, K.ST_KEEP)[h]), None, None, name=f"{name}.mrec{h}")
            for h in range(2):  # (LayerNorm(4608) in two applies of 2560 channels: 640 a core)
                a = Apply(m=16, nblk=8, mode="ln", x=mc.addr, y=nl.addr, blk0=32 * h, y_kb0=320 * h, rec=rec, C=4 * H,
                          eps=EPS, y_np=nl.np)
                bd.apply(a, w["ds_g"], w["ds_b"], name=f"{name}.ds_ln{h}")
            bd.mem.release(rec, ln_rec(l16))
            free(bd, mc)
            hid = alloc(bd, l16, HID, "pn", top=True)  # (directly below xp: the down GEMM's K is [hidden | xp])
            assert hid.addr + hid.size == xp.addr, (hid.addr, hid.size, xp.addr)
            bd.table("gelu", gelu_table())
            n6 = HID // 6  # (6 GEMMs of 9 passes a row: at most 16)
            for k in range(6):
                gemm(bd, nl, w["up"][n6 * k:n6 * (k + 1)], w["up_b"][n6 * k:n6 * (k + 1)], name=f"{name}.up{k}",
                     out=Tensor(l16, n6, hid.addr + n6 // 8 * l16.np * 72 * k, "pn"), kind="gelu")
            free(bd, nl)
            y = alloc(bd, l16, C)
            gemm(bd, Tensor(l16, HID + MERGED, hid.addr, "pn"), w["down"], w["down_b"], name=name + ".down", out=y)
            free(bd, hid)
            free(bd, xp)
            bd.mem.release(spare, 4096)
            bd.op(_xfer(bd, out.x.band(t), ct, fill=False, mem=y.addr, nbytes=16 * 2560, off=2 * j * 1280,
                        mdims=_dims((640, 1), (16, ct16 // 4)), sdims=(640, 16, ob // 4, 16 * ob // 4)), {})
            free(bd, y)
        xt = alloc(bd, lp, C)  # (the band's records)
        bd.op(_xfer(bd, out.x.band(t), ct, fill=True, mem=xt.addr, nbytes=ct), {})
        out.store(bd, t, xt, ln_records(bd, xt, name=name + ".rec"))
