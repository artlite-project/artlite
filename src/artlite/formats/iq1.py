"""IQ1_S and IQ1_M tiles (50 and 56 B per row and super-block, the GGUF blocks' sizes; R <= 32 rows in 4 row groups of
8, rows past R zero): per super-block a unit of ``UNITS[fmt]`` B (4 per 1024-column tile, nothing else in the tile)
holding, the 8-k blocks of a row in panel order e = [32-k group i 8][8-k panel l 4][row group g 4][r 8] (128 panels of 8
rows x 8 k: row 8 g + r's 8-k block 4 i + l, one grid entry):

- LO (at 0, 1 KB): [e] the grid indices' low bytes;
- HB (at 1024; IQ1_S 384 B, IQ1_M 512 B): [half q 2][bit b][word m 16] u32: bit n of word (q, b, m) = bit 8 + b of the
  index of entry 512 q + 32 m + n (b 0 .. 2), IQ1_M's b 3: that block's delta is negative;
- SC (IQ1_S at 1408, 128 B: [g 4][r 8][i' 4] the nibbles s | dneg << 3 of groups 2 i' (low) and 2 i' + 1, s the 3-bit
  scale, dneg the group's delta sign; IQ1_M at 1536, 256 B: [g 4][r 8][i 8] the scales of the group's 16-k halves, s0 |
  s1 << 4);
- D (IQ1_S at 1536, IQ1_M at 1792, 64 B): [row] bf16(d) (IQ1_M's d: the fp16 of the GGUF scale words' top nibbles).

w = d (2 s + 1) (g + delta / 8): g the index's grid value (``gguf_model.iq1_grid``: 8 values per index, in {-1, 0, 1}),
delta = -1 where dneg, else 1; s per 32-k group (IQ1_S) or 16-k half (IQ1_M), delta per group (IQ1_S) or 8-k block
(IQ1_M).

The grid as a gather table (``grid_words``): entry i = [A | B] (16 bits), A's bit j = value j of the entry is nonzero,
B's = it is -1; 2048 entries, 4 KB: no room for the 4x16 gathers' two copies of each 32-byte half (``common.TAB``'s
two 2 KB slots per core half), so the table holds every entry once. Index bit 10 picks the slot, bits 0 .. 7 the entry's
line and place (byte 4 lo), bit 8 the slot's upper 1 KB, bit 9 the line's 32-byte half; a gather lane reads the half of
its parity, so a row's entry is gathered by two lanes (the even one reads half 0, the odd one half 1) and bit 9 picks.

Decode (``mmv/iq1``): per super-block the scales 2 s + 1 as bf16 and d (``_iq1_ss``; IQ1_S: also the deltas, through the
32-k sum panels: per row group yu = XS (delta (2 s + 1) / 8) on the MMUL, which the half's scaling starts from), then per
half (4 groups) the gather offsets (``_iq1_off``: 4 lo + 1024 b8 + D b10, D the core half's second table copy's distance
from its first, each twice), the weight panels (``_iq1_w``: per two panels 8 4x16 gathers, the candidates in order by
``T16_16x4_lo``, the masks [Ae | Be | Ao | Bo] of the 16 rows by one ``T8_16x4``, b9 picking a row's half by one select,
the panel = sel(0, k, A) signed by B: IQ1_S k = 1 (g), IQ1_M k = 8 from the deltas +-1 (8 g + delta: at exponent 130, (8
g + delta) / 8)), the MMUL (``_iq1_x``: per 16-k half h of a group c = bf16 of its two panels' sums) and y += hilo(yu + sum
c (2 s + 1)) d (``_iq1_sc``; 8 slots: ``_k23sc8`` per row group).

Prefill (``gemm/IQ1_*``): GemmQd on super-block units, IQ2's structure (``iq2``): per k-tile the offsets and the scales
S = bf16(d (2 s + 1) / 8), per panel the codes by the decode's gathers and pick, w = S (8 g + delta) rounded once."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from .common import TAB, TAB_BYTES, DqKernels, Slot, TileFormat, _bf16_bytes, dq_kernel, i32

FORMATS = (("IQ1_S", 17, 50), ("IQ1_M", 18, 56))  # (name, fid, GGUF bytes per 256)
ROWS = 32  # rows of a tile (4 row groups)
UNITS = {"IQ1_S": 1600, "IQ1_M": 1856}  # bytes of a super-block's unit
LO, HB = 0, 1024  # the units' first parts (module docstring)
SC = {"IQ1_S": 1408, "IQ1_M": 1536}
D = {"IQ1_S": 1536, "IQ1_M": 1792}
PLANES = {"IQ1_S": 3, "IQ1_M": 4}  # HB's bit planes


def grid_words() -> np.ndarray:
    """The grid as the gather table's two slots (``Slot.mmv_table``): (2, 512) uint32, slot s holding the entries of
    index bit 10 = s: entry i (A | B << 8) at byte 64 L + 32 h + 2 p of its slot, L = (i >> 4 & 15) + 16 (i >> 8 & 1), h
    = i >> 9 & 1, p = i & 15 (module docstring)."""
    from ..gguf_model import iq1_grid

    g = iq1_grid().astype(np.int32)
    bits = (1 << np.arange(8)).astype(np.uint16)
    e = ((g != 0) * bits).sum(-1, dtype=np.uint16) | (((g < 0) * bits).sum(-1, dtype=np.uint16) << 8)
    i = np.arange(2048)
    pos = 32 * ((i >> 4 & 15) + 16 * (i >> 8 & 1)) + 16 * (i >> 9 & 1) + (i & 15)  # (the halfword in its slot)
    out = np.zeros((2, 1024), np.uint16)
    out[i >> 10, pos] = e
    return out.view(np.uint32)


def entries(a: np.ndarray, nsb: int) -> np.ndarray:
    """(32, nsb, 8 i, 4 l) per-row values of the 8-k blocks -> (nsb, 1024) in panel order."""
    return a.reshape(4, 8, nsb, 8, 4).transpose(2, 3, 4, 0, 1).reshape(nsb, 1024)


def planes(bits: np.ndarray) -> np.ndarray:
    """(nsb, B, 1024) bits in panel order -> (nsb, 2 q, B, 16 m) uint32 words (HB)."""
    nsb, B = bits.shape[:2]
    w = (bits.reshape(nsb, B, 2, 16, 32).astype(np.uint32) << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    return np.ascontiguousarray(w.transpose(0, 2, 1, 3))


# the decode GEMV (``decode.gemv``; the scaling of 8 slots: ``mmv/k23``)
_MMV = templating.source("artlite/formats/iq1.mmv.cc.jinja")

# GemmQd (prefill) on super-block units: tiles of 32 rows (``iq2``'s structure)
_GEMM = templating.source("artlite/formats/iq1.gemm.cc.jinja")


class Iq1Slot(Slot):
    formats = tuple(TileFormat(name, fid, nb, ROWS, 1, nb, 0, planes=1, unit_fixed=UNITS[name], dq_row_cycles=24.0,
                               mmv=name.lower().replace("_", "")) for name, fid, nb in FORMATS)
    sources: ClassVar[dict[str, str]] = {"mmv/iq1": _MMV, **{f"gemm/{name}": _GEMM for name, _, _ in FORMATS}}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R <= ROWS, R
        m = fmt == "IQ1_M"
        idx = np.zeros((ROWS, nsb, 8, 4), np.uint16)
        idx[:R] = f["idx"].reshape(R, nsb, 8, 4)
        e = entries(idx, nsb)
        units = np.zeros((nsb, UNITS[fmt]), np.uint8)
        units[:, LO:LO + 1024] = e & 0xFF
        bits = [(e >> b) & 1 for b in (8, 9, 10)]
        if m:
            dn = np.zeros((ROWS, nsb, 8, 4), np.uint16)
            dn[:R] = f["dneg"].reshape(R, nsb, 8, 4)
            bits.append(entries(dn, nsb))
        units[:, HB:HB + 128 * PLANES[fmt]] = planes(np.stack(bits, 1)).view(np.uint8).reshape(nsb, -1)
        if m:  # ([g][r][i] bytes: s0 | s1 << 4)
            sc = np.zeros((ROWS, nsb, 8, 2), np.uint8)
            sc[:R] = f["sc"].reshape(R, nsb, 8, 2)
            b = sc[..., 0] | sc[..., 1] << 4  # (row, sb, i)
            units[:, SC[fmt]:SC[fmt] + 256] = b.transpose(1, 0, 2).reshape(nsb, 256)
        else:  # ([g][r][i'] bytes: the nibbles s | dneg << 3 of groups 2 i', 2 i' + 1)
            n = np.zeros((ROWS, nsb, 4, 2), np.uint8)
            n[:R] = (f["sc"] | f["dneg"] << 3).reshape(R, nsb, 4, 2)
            units[:, SC[fmt]:SC[fmt] + 128] = (n[..., 0] | n[..., 1] << 4).transpose(1, 0, 2).reshape(nsb, 128)
        d = np.zeros((ROWS, nsb), np.float32)
        d[:R] = f["d"]
        units[:, D[fmt]:D[fmt] + 64] = _bf16_bytes(np.ascontiguousarray(d.T)[..., None]).reshape(nsb, 64)
        return [units.reshape(-1)]

    # GemmQd (prefill): super-block units

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return UNITS[fmt]

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        return tiles <= 4  # (MMUL columns <= 128: a gate / up pass of a whole SiLU group)

    def _ws(self, op: object) -> dict[str, int]:
        """The prefill dequant's scratch (bytes, ``gemm/IQ1_*``'s): the lists' and the scales' pitch LP (rows), the lists
        WL [ks 8][2 LP] u16, the words WM [16-row block][ks 8][2] u32, the scales WS [i'' 2][h 2][LP] bf16, a call's codes WU
        [16-row block][ks 8][rh 2][64]; its size."""
        LP = -(-max(op.n, ROWS * op.tiles) // 16) * 16
        WL = 0
        WM = WL + 32 * LP
        WS = WM + 4 * op.n
        WU = WS + 8 * LP
        return {"LP": LP, "NS": LP, "WL": WL, "WM": WM, "WS": WS, "WU": WU, "size": WU + 64 * op.nm}

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the lists, words, scales, a call's codes (``_ws``); hb: unused; the grid table's two slots (``gemm_qd``)
        return {"ws": (1, self._ws(op)["size"]), "hb": (1, 64), "tab": (1, TAB_BYTES), "tabb": (1, TAB_BYTES)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == ROWS and op.n % 16 == 0 and op.nm % 16 == 0, (op.R, op.n, op.nm)
        assert op.unit_pitch == UNITS[op.fmt], op.unit_pitch
        tab_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, grid_words()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        return {"UP": op.unit_pitch, "UNIT": UNITS[op.fmt], "F": int(op.fmt == "IQ1_M"), "B": PLANES[op.fmt],
                "SCO": SC[op.fmt], "DO": D[op.fmt], **self._ws(op)}

    # the decode GEMV

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/iq1")

    def mmv_table(self, fmt: str) -> np.ndarray:
        return grid_words()

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == ROWS, R  # (4 row groups: the kernels' layout)
        return {"UNIT1": UNITS, "SC1": SC, "D1": D, "TAB": TAB}


SLOT = Iq1Slot()
