"""IQ2_XXS, IQ2_XS and IQ2_S tiles (66, 74 and 82 B per row and super-block, the GGUF blocks' sizes; R <= 24 rows in 3
row groups of 8, rows past R zero): per super-block a unit of ``UNIT`` B (4 per 1024-column tile, nothing else in the
tile) holding, the 8-k blocks of a row in panel order p = [32-k group i 8][8-k panel l 4][row group g 3] (96 panels of 8
rows x 8 k: row 8 g + r's 8-k block 4 i + l, one grid entry):

- LO (at 0, 768 B): [p][r 8] the grid indices' low bytes;
- SG (at 768, 1 KB): [i][row 32][l 4] the 8-k blocks' signs: IQ2_XXS and IQ2_XS the prefix XOR c of the 7-bit sign
  field (c_j = f_0 ^ .. ^ f_j: the 8 signs, the 8th the parity of the field's, are c ^ (c << 1)), IQ2_XXS with bit l of
  its 32-group's scale ls at bit 7 (IQ3_XXS's sign / scale words); IQ2_S the 8 signs (bit j: value j negative);
- SC (at 1792, 256 B; IQ2_XS, IQ2_S): [i][g 4][r 8] the GGUF scales byte (16-group 2 i's scale in the low nibble,
  2 i + 1's in the high);
- HB (at 2048, 192 B; IQ2_XS, IQ2_S): [half q 2][bit b 2][word m 12] u32: bit n of word (q, b, m) = bit 8 + b of the
  index of entry 32 m + n of the half's panels (i in 4 q .. 4 q + 3, in panel order);
- D (at 2240, 48 B): [row] bf16(d) / 2 (IQ2_XXS) or / 8.

w = D S q: S = (2 ls + 1) / 4 of the 32-group (IQ2_XXS), 2 sc + 1 of the 16-group (IQ2_XS, IQ2_S); q = the index's
grid value (``gguf_model.iq2_grid``: 8 values per index, in {8, 25, 43}) negated where its sign is set.

The grids as gather tables (``grid_words``): entry i = [A | B] (16 bits), A's bit k = value k of the entry >= 25, B's =
value k is 43, so that g = 43 where B, else 25 where A, else 8.

Decode (``mmv/iq2``): per super-block the signs as masks and the scales (``_iq2_ss``; IQ2_XXS and IQ2_XS through
IQ3_XXS's ``_iq3_signs``), then per half (4 groups) the gather offsets (``_iq2_off``: 4 lo + 1024 b8 + D b9, IQ2_XXS 16 (lo
& 127) + D b7, u16), the weight panels (IQ2_XS, IQ2_S ``_iq2_w``: a panel's 8 entries by two 4x16 gathers, its A and B
bytes as 64-bit masks by one byte transpose ``T8_8x8``, g = sel(sel(8, 25, A), 43, B); IQ2_XXS ``_iq2_wv``: its 8 rows'
values by two 4x64 gathers from ``value_words``; the signs by a masked subtract), the MMUL (``_iq2_x``: per 16-k half of a
group c = bf16 of its two panels' sums) and the scaling (``_iq2_sc``: y += hilo(sum c S) D). The decode's tables: IQ2_XXS's
values as its halves (index bit 7, 2 KB each), IQ2_S's masks as its halves (index bit 9, 2 KB each), one half in each copy
slot of the core half (``common.TAB``), every lane at the first copy and D its distance to the second; IQ2_XS's masks 2
KB per copy (lanes 0, 1 of a gather from one copy, 2, 3 from the other). The prefill's (``grid_words``): the masks, IQ2_XXS
1 KB.
"""

from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from ..operators import templating
from .common import TAB, TAB_BYTES, DqKernels, Slot, TileFormat, _bf16_bytes, dq_kernel, i32, table_copies

FORMATS = (("IQ2_XXS", 14, 66), ("IQ2_XS", 15, 74), ("IQ2_S", 16, 82))  # (name, fid, GGUF bytes per 256)
UNIT = 2304  # bytes of a super-block's unit
ROWS = 24  # rows of a tile (3 row groups)
LO, SG, SC, HB, D = 0, 768, 1792, 2048, 2240  # the unit's parts (module docstring)


def grid_words(fmt: str) -> np.ndarray:
    """The grid as gather tables (uint32 words, ``Slot.mmv_table``): entry i = A | B << 8 (module docstring) at byte 4 i of
    a table whose 32-byte groups of 16 entries are stored twice; IQ2_S: (2, 512) words, its halves (entries 512 h ..) in
    copy h."""
    from ..gguf_model import iq2_grid

    g = iq2_grid(fmt).astype(np.int32)
    bits = (1 << np.arange(8)).astype(np.uint16)
    e = ((g >= 25) * bits).sum(-1, dtype=np.uint16) | (((g == 43) * bits).sum(-1, dtype=np.uint16) << 8)
    tables = e.reshape(-1, 512) if e.size > 512 else e.reshape(1, -1)
    out = np.stack([np.concatenate([t.reshape(-1, 16)] * 2, 1).reshape(-1).view(np.uint32) for t in tables])
    return out if len(out) > 1 else out[0]


def value_words() -> np.ndarray:
    """IQ2_XXS's grid as the decode's 4x64 gather table (``Slot.mmv_table``; the prefill's: ``grid_words``): (2, 512)
    uint32, slot s holding the entries of index bit 7 = s, entry i's 8 values as bytes at byte 64 G + 8 e of its slot and
    again 32 further (a 4x64 lane reads the 32-byte half of its parity), G = (i & 127) >> 2, e = i & 3: offset 16 (i &
    127) + D b7."""
    from ..gguf_model import iq2_grid

    g = iq2_grid("IQ2_XXS").astype(np.uint8).reshape(2, 32, 1, 4, 8)  # (slot, group, -, entry, values)
    return np.ascontiguousarray(np.concatenate([g, g], 2)).reshape(2, 2048).view(np.uint32)


def prefix_xor7(f: np.ndarray) -> np.ndarray:
    """7-bit sign fields -> their prefix XOR c (uint8, bit j = f_0 ^ .. ^ f_j)."""
    c = f.astype(np.uint8)
    for k in (1, 2, 4):
        c ^= c << k
    return c & 0x7F


def panel_order(a: np.ndarray, nsb: int) -> np.ndarray:
    """(24, nsb, 8 i, 4 l) per-row values of the 8-k blocks -> (nsb, 96 panels x 8 rows) in panel order."""
    return a.reshape(3, 8, nsb, 8, 4).transpose(2, 3, 4, 0, 1).reshape(nsb, 768)


# the decode GEMV (``decode.gemv``; the scaling by D: ``mmv/k23``, the signs of IQ2_XXS and IQ2_XS: ``mmv/iq3signs``)
_MMV = templating.source("artlite/formats/iq2.mmv.cc.jinja")

# GemmQd (prefill) on super-block units: tiles of 24 rows, each 8-row half of the MMUL in one tile
_GEMM = templating.source("artlite/formats/iq2.gemm.cc.jinja")


class Iq2Slot(Slot):
    formats = tuple(TileFormat(name, fid, nb, ROWS, 1, nb, 0, planes=2, unit_fixed=UNIT, dq_row_cycles=11.0,
                               mmv=name.lower().replace("_", "")) for name, fid, nb in FORMATS)
    sources: ClassVar[dict[str, str]] = {"mmv/iq2": _MMV, **{f"gemm/{name}": _GEMM for name, _, _ in FORMATS}}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R <= ROWS, R
        idx = np.zeros((ROWS, nsb, 8, 4), np.uint16)
        idx[:R] = f["idx"].reshape(R, nsb, 8, 4)
        units = np.zeros((nsb, UNIT), np.uint8)
        units[:, LO:LO + 768] = panel_order(idx & 0xFF, nsb)
        sg = np.zeros((32, nsb, 8, 4), np.uint8)  # [row][sb][i][l]
        sg[:R] = f["signs"].reshape(R, nsb, 8, 4)
        if fmt != "IQ2_S":
            sg = prefix_xor7(sg)
        if fmt == "IQ2_XXS":
            ls = np.zeros((32, nsb, 8), np.uint8)
            ls[:R] = f["ls"].reshape(R, nsb, 8)
            sg |= ((ls[..., None] >> np.arange(4, dtype=np.uint8)) & 1) << 7
        units[:, SG:SG + 1024] = sg.transpose(1, 2, 0, 3).reshape(nsb, 1024)
        if fmt != "IQ2_XXS":
            sc = np.zeros((32, nsb, 8, 2), np.uint8)
            sc[:R] = f["sc"].reshape(R, nsb, 8, 2)
            units[:, SC:SC + 256] = (sc[..., 0] | sc[..., 1] << 4).transpose(1, 2, 0).reshape(nsb, 256)
            hb = np.stack([panel_order((idx >> (8 + b)) & 1, nsb) for b in range(2)], 1)  # (nsb, b, 768)
            words = (hb.reshape(nsb, 2, 2, 12, 32).astype(np.uint32) << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
            units[:, HB:HB + 192] = np.ascontiguousarray(words.transpose(0, 2, 1, 3)).view(np.uint8).reshape(nsb, 192)
        d = np.zeros((ROWS, nsb), np.float32)
        d[:R] = f["d"]
        d = d.astype(bfloat16).astype(np.float32) / (2 if fmt == "IQ2_XXS" else 8)  # (exact)
        units[:, D:D + 48] = _bf16_bytes(np.ascontiguousarray(d.T)[..., None]).reshape(nsb, 48)
        return [units.reshape(-1)]

    # GemmQd (prefill): super-block units

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return UNIT

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 6 units in the dequantizing core's L1: 27 KB, a gate / up pass of a whole SiLU
        # group; MMUL columns <= 144)
        return tiles <= 6

    def _ws(self, op: object) -> dict[str, int]:
        """The prefill dequant's scratch (bytes, ``gemm/IQ2_*``'s): the lists' and the scales' pitch LP (rows: the pass's
        tiles' or the MMUL's), the lists WL [ks 8][LP] u16, the masks WM [16-row block][i'' 2][64], the scales WS [i'' 2]
        [h 2][LP] bf16, a call's codes WU [16-row block][i'' 2][8][64]; its size."""
        LP = -(-max(op.n, ROWS * op.tiles) // 16) * 16
        WL = 0
        WM = WL + 16 * LP
        WS = WM + 8 * op.n
        WU = WS + 8 * LP
        return {"LP": LP, "NS": LP, "WL": WL, "WM": WM, "WS": WS, "WU": WU, "size": WU + 64 * op.nm}

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the lists, masks, scales, a call's codes (``_ws``); hb: unused; the grid table's two copies (in two 16 KB
        # regions: ``gemm_qd``)
        return {"ws": (1, self._ws(op)["size"]), "hb": (1, 64), "tab": (1, TAB_BYTES), "tabb": (1, TAB_BYTES)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == ROWS and op.n % 16 == 0 and op.nm % 16 == 0, (op.R, op.n, op.nm)
        assert op.unit_pitch == UNIT, op.unit_pitch
        tab_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
        words = grid_words(op.fmt)
        tables = np.zeros((2, TAB_BYTES // 4), np.uint32)
        for i, w in enumerate(table_copies(words)):
            tables[i, :w.size] = w
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, tables))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        hsrc = [(h // 3, h % 3) if h // 3 < op.tiles else None for h in range(op.n // 8)]
        return {"UP": op.unit_pitch, "F2": [f for f, _, _ in FORMATS].index(op.fmt), "hsrc": hsrc, **self._ws(op)}

    # the decode GEMV

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/iq3signs", "mmv/iq2")

    def mmv_table(self, fmt: str) -> np.ndarray:
        return value_words() if fmt == "IQ2_XXS" else grid_words(fmt)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert 16 < R <= ROWS, R  # (3 row groups: the kernels' layout)
        return {"UNIT2": UNIT, "TAB": TAB}


SLOT = Iq2Slot()
