"""IQ3_S tiles (R = 17 .. 20 rows in 3 row groups of 8, the third of r2 = R - 16 rows; 110 B per row and super-block,
the GGUF block's size): per super-block a unit of ``UNIT`` B (4 per 1024-column tile, nothing else in the tile) holding

- A (1536 B): the grid indices' low bytes as records [32-k group i 8][panel pair h 2][8-k panel l' 2] of 48 B: rows 0
  .. 7 (row r's two indices of panel l = 2 h + l' at 2 r, 2 r + 1), rows 8 .. 15, rows 16 .. R - 1 (2 r2 B), zeros, then
  at byte 40 the panels' 9th index bits, a 16-bit mask per row group g (bit 2 r + j: row 8 g + r's index j), zeros;
- B (768 B): the signs as records [i 8][panel l 4] of 24 B: rows 0 .. 7's sign bytes (bit j: value 8 l + j of group i
  negative), rows 8 .. 15, rows 16 .. R - 1, zeros, then at byte 20 a 4-byte slot of C;
- C (the 32 slots of B, slot n in record n): slot r < R row r's scale bytes (the GGUF block's: groups 2 j, 2 j + 1 in
  byte j's low and high nibble), slot R + s rows 2 s and 2 s + 1's d (bf16), zeros past them.

So each weight panel (8 rows x 8 k: 16 indices) has its index bytes in 16 contiguous bytes and its signs in 8 (rows past
R: other bytes, whose panel rows the kernels drop). w = d (1 + 2 sc) q, q = the 9-bit index's grid value
(``gguf_model.iq3s_grid``: 4 values per index, in {1, 3, .., 15}) negated where its sign is set.

Decode (``mmv/IQ3_S``): per super-block the panels' 9th bits as bytes 0 / 128 (``_iq3s_hb``: per 4 records their 12
masks as 3 x 64 bits), then per 4 panels (64 index bytes) the gather offsets 4 (qs + 256 qh) on the MMUL (``mul_elem`` +
``mac_elem``; a select from a mask in the gather loop leaves it unpipelined), 4 gathers of 16-bit entries per panel
(``grid_halves``: the grid's 4 values as nibbles, 2 KB per copy at ``common.TAB``), the nibbles unpacked, the signs by
``vaddsub8`` from the tile's masks; the MMUL and the scaling as IQ3_XXS's (``mmv/iq3c``), the scales 1 + 2 sc and d
from C's slots (``_iq3s_ss``)."""

from typing import ClassVar

import numpy as np

from . import k23
from .common import TAB, Slot, TileFormat

UNIT = 2304  # bytes of a super-block's unit (A + B)


def grid_halves() -> np.ndarray:
    """The grid as 16-bit entries for the decode's 4x16 gathers (``TAB_BYTES`` / 4 uint32): entry i (its 4 values,
    one nibble each, value j in nibble j) at byte 4 i of a table whose 32-byte groups of 16 entries are stored twice."""
    from ..gguf_model import iq3s_grid

    g = iq3s_grid().astype(np.uint16)
    e = g[:, 0] | g[:, 1] << 4 | g[:, 2] << 8 | g[:, 3] << 12
    return np.concatenate([e.reshape(32, 16)] * 2, 1).reshape(-1).view(np.uint32)


_MMV = r"""// ---- IQ3_S (formats/iq3s.py: {{ R3S }} rows, 3 row groups; rows past R: junk): per super-block a unit of 2304 B,
// [A: index records [i][h][l' 2] of 48 B (rows' low index bytes; at 40 the 9th bits, a 16-bit mask per row group) | B:
// sign records [i][l 4] of 24 B (8 B per row group's panel; at 20 a slot of C: the rows' scale bytes, then d)]. Core H:
// super-blocks 2 H, 2 H + 1; per super-block u: the 9th bits as bytes (_iq3s_hb), the weight panels (_iq3s_w), the
// scales (_iq3s_ss), the MMUL and y += hilo(sum_i c_i (1 + 2 sc_i)) d (mmv/iq3c).
// Scratch: w / c [i 8][l 4][g 3][64 B] (0 .. 6144) | hb [panel 96][16] (6144; then C's slots st (6144 .. 6272) and sct
// [i][g 4][r 8] bf16 (6272)) | the table's addresses (8000 .. 8008)

// The panels' 9th index bits (A's masks at 48 n + 40: [g0 | g1 | g2 | 0] 16 bits each) -> hb [panel][16]: 0 or 128 per
// index, panels in A's order (per 4 records 12 panels, their masks as 3 x 64 bits)
__attribute__((noinline)) void {{ p }}_iq3s_hb(const uint8_t *__restrict ix, uint8_t *__restrict hb)
{
    const v64uint8 k128 = broadcast_to_v64uint8((unsigned char)128), z = broadcast_zero_to_v64uint8();
    const uint64_t *__restrict pm = (const uint64_t *)(ix + 40);
    v64uint8 *__restrict po = (v64uint8 *)hb;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 8; t++) {
        const uint64_t L0 = pm[0], L1 = pm[6], L2 = pm[12], L3 = pm[18];  // (records 6 words apart)
        po[0] = ::sel(z, k128, L0 | L1 << 48);
        po[1] = ::sel(z, k128, L1 >> 16 | L2 << 32);
        po[2] = ::sel(z, k128, L2 >> 32 | L3 << 16);
        pm += 24;
        po += 3;
    }
}

// a panel from its 16 entries' addresses (lanes 2 r, 2 r + 1: row r's two) and its signs m (bit 8 r + j: value j of row
// r negative): 4 gathers, the entries' nibbles unpacked, zero - w where a sign is set
[[clang::always_inline]] inline v64int8 {{ p }}_iq3s_panel(v16int32 a, v2int32 m)
{
    const v64uint8 u = unpack((v64uint4){{ p }}_g16(a));
    return __builtin_aie2p_vaddsub8(broadcast_zero_to_v64int8(), (v64int8)u, m);
}

// The 96 weight panels of a super-block (ix: A, 4 panels per 64 bytes; hb: their 9th bits as bytes; sg: B, a panel's
// signs at 8 p) -> wp [i][l][g 3][64 B]
__attribute__((noinline)) void {{ p }}_iq3s_w(const uint8_t *__restrict ix, const uint8_t *__restrict hb,
                                              const uint8_t *__restrict sg, const uint32_t *__restrict tab_ab,
                                              const uint32_t *__restrict tab_cd, uint8_t *__restrict wp)
{
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v64int8 four = broadcast_to_v64int8(4), eight = broadcast_to_v64int8(8);
    const v64uint8 __aie_dm_resource_a *__restrict pi = (const v64uint8 __aie_dm_resource_a *)ix;
    const v64uint8 __aie_dm_resource_d *__restrict ph = (const v64uint8 __aie_dm_resource_d *)hb;
    const v8int32 __aie_dm_resource_b *__restrict ps = (const v8int32 __aie_dm_resource_b *)sg;
    v64int8 __aie_dm_resource_c *__restrict pw = (v64int8 __aie_dm_resource_c *)wp;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 24; t++) {
        const v64acc32 q = mac_elem_64_2(*ph++, eight, mul_elem_64_2(*pi++, four));  // (4 (qs + 256 qh) of 4 panels)
        const v16int32 sv = set_v16int32(0, *ps++);
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++)
            pw[u] = {{ p }}_iq3s_panel(add((v16int32)extract_v16acc32(q, u), b16), ext_v2int32(sv, u));
        pw += 4;
    }
}

// C's slots (B + 24 n + 20) -> st [32] words; sct [i][g 4][r 8] = 1 + 2 sc (bf16, exact: bf16(128 + 1 + 2 sc) by its bits
// 0x4300 | (1 + 2 sc), minus 128) from st's [row 24][4 B] (rows past R: other slots), per row group a transpose
__attribute__((noinline)) void {{ p }}_iq3s_ss(const uint8_t *__restrict b, int32_t *__restrict st, bfloat16 *__restrict sct)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict pc = (const int32_t *)(b + 20);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int n = 0; n < 32; n++)
        st[n] = pc[6 * n];
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4301);
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k128(aie::broadcast<float, 32>(-128.0f));
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < 3; g++) {
        const aie::vector<uint8, 64> v = aie::unpack(aie::load_v<64>((const uint4 *)(st + 8 * g)));  // ([r 8][i 8])
        const aie::vector<uint8, 64> v2 = aie::add(v, v);
        const aie::vector<bfloat16, 32> s0 = aie::mac(k128, aie::add(aie::unpack(v2.template extract<32>(0)), k43).template cast_to<bfloat16>(), one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> s1 = aie::mac(k128, aie::add(aie::unpack(v2.template extract<32>(1)), k43).template cast_to<bfloat16>(), one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> t = aie::transpose(aie::concat(s0, s1), 8, 8);  // ([i][r])
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < 8; i++)
            aie::store_v(sct + 32 * i + 8 * g, t.template extract<8>(i));
    }
}

// super-blocks 2 H, 2 H + 1 of an IQ3_S tile: y[g][32] (rows 8 g ..) += their rows against xp (their 64 panels)
template <int H>
__attribute__((noinline)) void {{ p }}_iq3s(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                            const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    uint8_t *hb = scr + 6144;
    int32_t *st = (int32_t *)(scr + 6144);
    bfloat16 *sct = (bfloat16 *)(scr + 6272);
    // (the table copies' addresses through a volatile word: constants at the call would specialize _iq3s_w, and the
    // gathers' memory operands need pointers)
    volatile int32_t *ta = (volatile int32_t *)(scr + 8000);
    ta[0] = H ? {{ TAB[1][0] }} : {{ TAB[0][0] }};
    ta[1] = H ? {{ TAB[1][1] }} : {{ TAB[0][1] }};
    const uint32_t *tab_ab = (const uint32_t *)(intptr_t)ta[0], *tab_cd = (const uint32_t *)(intptr_t)ta[1];
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        uint8_t *un = a + {{ UNIT }} * (2 * H + u);
        {{ p }}_iq3s_hb(un, hb);
        {{ p }}_iq3s_w(un, hb, un + 1536, tab_ab, tab_cd, scr);
        {{ p }}_iq3s_ss(un + 1536, st, sct);
        {{ p }}_iq3_x(xp + {{ 32 * PANEL }} * u, scr, (bfloat16 *)scr);
        {{ p }}_iq3_sc((const bfloat16 *)scr, sct, (const bfloat16 *)(st + {{ R3S }}), y);
    }
}
"""


class Iq3sSlot(Slot):
    formats = (TileFormat("IQ3_S", 13, 110, 20, 1, 110, 0, planes=3, unit_fixed=UNIT, mmv="iq3s"),)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert 16 < R <= 20, R
        r2 = R - 16
        idx = np.zeros((24, nsb, 8, 2, 2, 2), np.uint16)  # [row][sb][i][h][l'][j]
        idx[:R] = f["idx"].reshape(R, nsb, 8, 2, 2, 2)
        a = np.zeros((nsb, 8, 2, 2, 48), np.uint8)
        low = (idx & 0xFF).astype(np.uint8).reshape(3, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6)  # [..][g][r][j]
        a[..., :32] = low[:, :, :, :, :2].reshape(nsb, 8, 2, 2, 32)
        a[..., 32:32 + 2 * r2] = low[:, :, :, :, 2, :r2].reshape(nsb, 8, 2, 2, -1)
        hi = (idx >> 8).astype(np.uint16).reshape(3, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6)
        mask = (hi.reshape(nsb, 8, 2, 2, 3, 16) << np.arange(16, dtype=np.uint16)).sum(-1, dtype=np.uint16)
        a[..., 40:46] = mask.view(np.uint8).reshape(nsb, 8, 2, 2, 6)
        sg = np.zeros((24, nsb, 8, 4), np.uint8)  # [row][sb][i][l]
        sg[:R] = f["signs"].reshape(R, nsb, 8, 4)
        b = np.zeros((nsb, 32, 24), np.uint8)
        s = sg.transpose(1, 2, 3, 0).reshape(nsb, 32, 24)  # [sb][i l][row]
        b[..., :16 + r2] = s[..., :16 + r2]
        c = np.zeros((nsb, 32, 4), np.uint8)
        sc = f["sc"].reshape(R, nsb, 4, 2)
        c[:, :R] = (sc[..., 0] | sc[..., 1] << 4).transpose(1, 0, 2)
        d = np.zeros((2 * -(-R // 2), nsb), np.uint16)
        d[:R] = k23.bf16_u16(f["d"]).astype(np.uint16)
        c[:, R:R + d.shape[0] // 2] = np.ascontiguousarray(d.T).view(np.uint8).reshape(nsb, -1, 4)
        b[..., 20:] = c
        return [np.concatenate([a.reshape(nsb, -1), b.reshape(nsb, -1)], 1).reshape(-1)]

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return UNIT

    sources: ClassVar[dict[str, str]] = {"mmv/IQ3_S": _MMV}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/g16", "mmv/iq3c", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == 20, R  # (d of rows 2 s, 2 s + 1 at slot R + s: 16-byte aligned for the kernels' loads)
        return {"R3S": R, "TAB": TAB, "UNIT": UNIT}

    def mmv_table(self, fmt: str) -> np.ndarray:
        return grid_halves()


SLOT = Iq3sSlot()
