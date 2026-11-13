"""IQ3_XXS tiles (98 B per row and super-block, the GGUF block's size; R <= 24 rows in 3 row groups of 8): per super-block a
unit of ``unit_bytes`` (``Iq3Slot.unit``: R 23 2320 B, 4 per 1024-column tile, nothing else in the tile) holding

- the grid indices [32-k group i 8][panel pair h 2][8-k panel l' 2][row group g][row 8][2] (the GGUF block's qs: each
  row's two indices of panel l = 2 h + l'; a panel's rows one list of 8 G rows; rows past R zero);
- the sign / scale words [i][row]: byte l = the prefix XOR c of group i's 7-bit sign field l (k 32 i + 8 l ..; c_j =
  f_0 ^ .. ^ f_j: the 8 signs, the 8th the parity of the field's, are c ^ (c << 1)) | bit l of its scale ls at bit 7;
- d [row] bf16;

in the order [indices | sign / scale words | d] (even super-blocks), [d | sign / scale words | indices] (odd ones: the
unit is 16 B off a 32-byte line, so the indices of every super-block start on one).

w = d (2 ls + 1) / 4 q, q = the index's grid value (``gguf_model.iq3xxs_grid``: 4 values per index, in {4 .. 62})
negated where its sign is set.

Decode (``mmv/IQ3_XXS``): a weight panel (8 rows x 8 k) is 16 grid entries read by the gather unit (``load_4x32``: four
lanes per instruction, entry i at byte 8 i of a table whose 32-byte groups of 8 entries are stored twice; lanes 0, 1
of an instruction from one copy, 2, 3 from the other, the copies in two 16 KB regions: in one region a gather waits a
cycle), its signs applied by a masked subtract (``vaddsub8``: the 8 signs of a row one byte of the 64-bit mask): ~6.5
cycles per panel, the move slot's 5 operations and the gathers' memory ports; then the MMUL against the x panels.
A core's half tile (two super-blocks) takes ~2300 cycles. The table (``grid_words``) is each core's buffer at ``TAB``,
resident or written by the format's page (``decode.design``'s tables); the panel loop takes its two copies as pointer
parameters (llvm-aie gives the gathers a memory operand only then)."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import k23
from .common import TAB, TAB_BYTES, DqKernels, Slot, TileFormat, dq_kernel, i32


def grid_words() -> np.ndarray:
    """A copy of the grid table for the gathers (TAB_BYTES / 4 uint32): entry i (its 4 values, int8) at byte 8 i, i.e.
    per group of 8 entries their 32 bytes twice."""
    from ..gguf_model import iq3xxs_grid

    g = np.ascontiguousarray(iq3xxs_grid()).view(np.uint32).reshape(32, 8)
    return np.concatenate([g, g], 1).reshape(-1)


def ss_words(signs: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """(R, n groups x 4) 7-bit sign fields, (R, n) scales -> (R, n) uint32 words: byte l = the prefix XOR of field l |
    bit l of ls << 7."""
    s = signs.reshape(*ls.shape, 4).astype(np.uint32)
    for k in (1, 2, 4):
        s ^= s << k
    s &= 0x7F
    bits = (ls[..., None].astype(np.uint32) >> np.arange(4, dtype=np.uint32)) & 1
    return ((s | bits << 7) << (8 * np.arange(4, dtype=np.uint32))).sum(-1, dtype=np.uint32)


# the decode GEMV (``decode.gemv``; the header's realignment and the scaling by d: ``mmv/k23``)
_MMV = templating.source("artlite/formats/iq3.mmv.cc.jinja")

# the MMUL and the scaling of IQ3_XXS and IQ3_S (their weight panels [i][l 4][g 3][64 B], scales [i][g 4][r 8])
_XSC = templating.source("artlite/formats/iq3.xsc.cc.jinja")

# one weight panel from its 16 grid entries' addresses and the signs (before ``mmv/IQ3_XXS``, which calls it)
_PANEL = templating.source("artlite/formats/iq3.panel.cc.jinja")


# GemmQd (prefill) on super-block units: the pass's rows compacted (``_compact``: tile t's rows from R t on)
_GEMM = templating.source("artlite/formats/iq3.gemm.cc.jinja")


class Iq3Slot(Slot):
    formats = (TileFormat("IQ3_XXS", 10, 98, 24, 1, 98, 0, planes=3, pad8_row=64, unit_align=16, dq_row_cycles=11.0,
                          mmv="iq3"),)
    sources: ClassVar[dict[str, str]] = {"mmv/IQ3_XXS": _MMV, "mmv/iq3panel": _PANEL, "mmv/iq3c": _XSC,
                                         "gemm/IQ3_XXS": _GEMM}

    @staticmethod
    def unit(R: int) -> tuple[int, int, int, int]:
        """(unit, indices, sign / scale words, d) bytes of a super-block's unit of R rows (module docstring)."""
        I, S, D = 512 * -(-R // 8), 32 * R, 2 * R
        return -(-(I + S + D) // 16) * 16, I, S, D

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert nsb % 2 == 0, nsb
        U, I, S, D = self.unit(R)
        d = k23.bf16_u16(f["d"]).astype(np.uint16)  # (R, nsb)
        G = -(-R // 8)
        idx = np.zeros((8 * G, nsb, 8, 4, 2), np.uint8)  # [row][sb][i][l][2]
        idx[:R] = f["idx"].reshape(R, nsb, 8, 4, 2)
        ix = idx.reshape(G, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6).reshape(nsb, I)  # [sb][i][h][l'][g][r][2]
        ss = ss_words(f["signs"], f["ls"]).reshape(R, nsb, 8).transpose(1, 2, 0)  # [sb][i][row]
        units = np.zeros((nsb, U), np.uint8)
        for sb in range(nsb):
            parts = [ix[sb], np.ascontiguousarray(ss[sb]).view(np.uint8).reshape(-1),
                     np.ascontiguousarray(d[:, sb]).view(np.uint8)]
            if sb % 2:  # (odd: [d | ss | ix], ix at the unit's end)
                units[sb, :D], units[sb, U - I - S:U - I], units[sb, U - I:] = parts[2], parts[1], parts[0]
            else:
                units[sb, :I], units[sb, I:I + S], units[sb, I + S:I + S + D] = parts
        return [units.reshape(-1)]

    # GemmQd (prefill): super-block units

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return self.unit(rows)[0]

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 6 units in the dequantizing core's L1: 28 KB, a gate / up pass of a whole SiLU
        # group; MMUL columns <= 128)
        return tiles <= 6

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: S, the signs, a call's codes, the index lists (module docstring of gemm/IQ3_XXS); hb: d; the grid
        # table's two copies (in two 16 KB regions: ``gemm_qd``)
        return {"ws": (1, self._ws(op)["size"]), "hb": (1, max(2 * op.n, 64)), "tab": (1, TAB_BYTES),
                "tabb": (1, TAB_BYTES)}

    def _ws(self, op: object) -> dict[str, int]:
        """Byte offsets in the prefill dequant's scratch: the codes (WU: a dq call's, or a unit's sign / scale words),
        the index lists (WL); its size."""
        WU = 48 * op.n
        WL = WU + max(64 * op.nm, 32 * op.n)
        return {"WU": WU, "WL": WL, "size": WL + 8 * self.list_bytes(op.n)}

    @staticmethod
    def list_bytes(n: int) -> int:
        """Bytes of a compacted index list of n rows in the prefill dequant's scratch (whole 64-byte lines)."""
        return -(-2 * n // 64) * 64

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == 23 and op.n % 16 == 0 and op.nm % 16 == 0, (op.R, op.n)
        tab_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, grid_words()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        U, I, S, _ = self.unit(op.R)
        R, T = op.R, op.tiles

        def seg(q0: int, q1: int) -> list[tuple[int, int, int]]:  # (tile, first row, end) of compacted rows q0 .. q1
            return [(t, max(q0, R * t), min(q1, R * (t + 1))) for t in range(q0 // R, min(T, -(-q1 // R)))]

        def mask(lo: int, hi: int) -> int:  # (bytes lo .. hi - 1 of 64)
            return ((1 << hi) - 1) & ~((1 << lo) - 1)

        ws = self._ws(op)
        return {"UP": op.unit_pitch, "U3": U, "I3": I, "S3": S, "LS": self.list_bytes(op.n), "WU": ws["WU"],
                "WL": ws["WL"], "seg": seg, "mask": mask}

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/iq3panel", "mmv/iq3c", f"mmv/{fmt}")

    def mmv_table(self, fmt: str) -> np.ndarray:
        return grid_words()

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert 16 < R <= 24, R  # (3 row groups: the kernels' layout)
        U, I, S, _ = self.unit(R)
        return {"RI3": R, "TAB": TAB, "U3": U, "I3": I, "S3": S}


SLOT = Iq3Slot()
