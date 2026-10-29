"""Shared pieces of the tile-format slots: the ``TileFormat`` record, the ``Slot`` interface (the hooks the
operators call), host packing helpers and the kernel-source fragments several slots include."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

KC = 2048  # a K chunk of 8 super-blocks (the GEMM operators' default; the images' tiles: 1024 or 512)
SB = 256  # super-block (8 groups of 32)
KT = 64  # the prefill GEMMs' k-tile (a super-block: 4 k-tiles of 2 groups)


@dataclass(frozen=True)
class TileFormat:
    """name: GGUF-style name; fid: id in the overlay's config words; sb_bytes: bytes per row
    and super-block; max_rows / row_step: row counts a tile may have; slice_row: code bytes per
    row of one super-block slice; header_sb: per-row header bytes per super-block (Q6_K: plus a d
    plane of 2 B per row and super-block, padded to 32 B); planes: the codes of a super-block slice as 32-byte planes per row (a row's 8 words of one step:
    Q4 nibbles 4, Q5_K 5, Q6_K 6; word-major Q2_K 2, Q3_K 3: 2 planes words per k-tile; 0: no codes); plane_major:
    the slice is [plane][row][32 B] (else [row][plane][32 B]): the prefill GEMM's memtile gathers either into
    [plane][row] (``gemm_q``);
    fixed_rows: rows of a fixed-size tile (BFP16: 8-row MMUL panels; the object size follows); pad8_row: code bytes per row
    and super-block stored for the rows rounded up to a multiple of 8 (IQ3_XXS: its indices in 8-row panels); unit_align:
    each super-block's slice padded to a multiple of these bytes (``unit_bytes``; IQ3_XXS: a super-block unit holds the
    slice's indices, sign / scale words and d); ws_planes: GemmQ's expanded
    scale planes (tile-major formats); dq_row_cycles: GemmQd's planar dequant, cycles per row (MMUL
    column) and k-tile on the dequantizing core (measured; ``gemm_qd.k_tile_cycles``). Methods take
    ``nsb``, the super-blocks of a tile (8: 2048 columns); word_major: the tile is stored [word][row][4 B] (header words, then each
    super-block's code words: a row group's word is one contiguous load), else [row][...] parts; the prefill GEMM's
    memtile gathers a k-tile's words into [word][row]; mmv: the name of the decode GEMV's kernel (``decode.gemv``:
    ``{p}_<mmv><H>`` from the slot's ``mmv/<fmt>`` source; "": the decode does not read these tiles)."""

    name: str
    fid: int
    sb_bytes: int
    max_rows: int
    row_step: int
    slice_row: int
    header_sb: int
    planes: int = 0
    plane_major: bool = False
    fixed_rows: int | None = None
    pad8_row: int = 0
    unit_align: int = 0
    header_pad32: bool = False
    ws_planes: int = 0
    dq_row_cycles: float | None = None
    word_major: bool = False
    mmv: str = ""

    @property
    def dequant(self) -> bool:
        """Integer codes the GEMMs dequantize (BFP16 tiles already hold MMUL operands)."""
        return self.planes > 0

    def row_bytes(self, nsb: int = 8) -> int:
        return self.sb_bytes * nsb

    def tile_bytes(self, nsb: int = 8) -> int | None:
        """Fixed overlay object size (None: chosen from the model's formats)."""
        return self.fixed_rows * self.row_bytes(nsb) if self.fixed_rows else None

    def header_bytes(self, rows: int, nsb: int = 8) -> int:
        """Bytes of the per-row scale planes at the start of a tile."""
        pad = -(-2 * nsb * rows // 32) * 32 if self.header_pad32 else 0
        return pad + self.header_sb * nsb * rows

    def slice_bytes(self, rows: int) -> int:
        """Bytes of the codes of one super-block (all rows) after the header."""
        return self.slice_row * rows

    def unit_bytes(self, rows: int) -> int:
        """Bytes of one super-block's slice as stored (its padding rows and ``unit_align`` included)."""
        n = self.slice_bytes(rows) + self.pad8_row * (-rows % 8)
        return -(-n // self.unit_align) * self.unit_align if self.unit_align else n

    def used_bytes(self, rows: int, nsb: int = 8) -> int:
        return self.header_bytes(rows, nsb) + nsb * self.unit_bytes(rows)

    def rows(self, tile_bytes: int, nsb: int = 8, max_rows: int = 0) -> int:
        """Rows per tile of ``tile_bytes`` (0: does not fit), at most ``max_rows`` (default: the format's)."""
        r = min(max_rows or self.max_rows, tile_bytes // self.row_bytes(nsb))
        r -= r % self.row_step
        while r > 0 and self.used_bytes(r, nsb) > tile_bytes:
            r -= self.row_step
        return r


# kernel declaration of an operator: (name suffix, argument types) -> its ``aie.iron.Kernel`` "{p}_<suffix>"
KernelFn = Callable[[str, list], object]
i32 = np.int32


@dataclass(frozen=True)
class DqKernels:
    """The dequantizing core's kernels of a GemmQd (``Slot.qd_kernels``). Per K chunk its loop calls
    ``hdr(b, hb if collect else ws, [unit,] kt)`` per header unit and k-tile slot, ``chunk(hb, ws)`` once,
    then per super-block ``sb(hb, ws, sb)`` and per k-tile ``dq(b, ws, d, sb, kt, [half,] [ex])`` (GemmQ ``hg`` > 1:
    the header unit once per group of K chunks, ``sb(hb, ws, sb, q)`` with the chunk q in the group); ``ex_ty``:
    the type of a buffer of the core that ``dq`` takes last. The optional kernels and the ``ex`` buffer are
    the Worker's arguments after ``hb``, in this order. ``units`` (``Slot.unit_bytes``): no header kernel (``hdr``
    None); per unit ``sb(b, ws, hb, sb)`` on the weight object, then ``dq`` per k-tile on the same object. ``tables``:
    (type, words) of a lookup table the core holds twice (buffers "tab", "tabb" of ``Slot.qd_scratch``, in two 16 KB
    regions: IQ3_XXS's grid for the gathers), passed to ``dq`` after ``ex``."""

    hdr: object
    dq: object
    collect: bool = False
    unit: bool = True
    sb: object = None
    chunk: object = None
    ex_ty: object = None
    units: bool = False
    tables: tuple[object, np.ndarray] | None = None


class Slot:
    """One family of tile formats: its ``TileFormat`` records, host packing, kernel-source fragments (``sources``:
    Jinja sources by include name; ``"gemm/<fmt>"``: the prefill GEMM's dequant kernels; ``"mmv/<fmt>"``: the decode
    GEMV's kernels) and the hooks the GEMM operators and the decode GEMV call. The defaults are those of a format without the hook's feature. A slot must not import the
    operators (they import the slots)."""

    formats: tuple[TileFormat, ...] = ()
    field = "q"  # the GGUF field (``QTensor.fields``) whose shape gives the rows and columns
    sources: ClassVar[dict[str, str]] = {}

    def format(self, fmt: str) -> TileFormat:
        return next(f for f in self.formats if f.name == fmt)

    # host side

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        """The parts of a tile (``pack_tile``) from the fields of R rows x nsb super-blocks."""
        raise NotImplementedError(fmt)

    # prefill GEMM paths (``gemm_q``)

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        """GemmQd reads these tiles through the planar gather."""
        f = self.format(fmt)
        return f.plane_major or f.word_major

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        """GemmQ and GemmQd read these tiles through the tile-major permutation."""
        return False

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        """GemmQd's planar passes may hold ``tiles`` tiles of ``rows`` rows (the planner's other limits aside)."""
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        """Planar tiles of super-block units (a unit holds one super-block's codes and scales of every row, nothing
        else is in the tile): the unit's bytes, else 0. GemmQd's weight object is then one unit of the pass's tiles,
        each tile's padded to whole 32-byte lines (``unit_pitch``); the dequantizing core calls ``sb(b, ws, hb, sb)``
        once per unit and ``dq`` per k-tile on it (``DqKernels.units``), and reads no header units."""
        return 0

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        """Planar tiles whose gathered rows hold tile row r's header at row r too."""
        return False

    def tile_major_dims(self, op: object) -> tuple[list, list] | None:
        """(memtile write, read dims) of the tile-major permutation; None: gemm_q's 16-row nibble tiles."""
        return None

    def gemm_q_header(self, op: object, kernel: KernelFn, B_l1_ty: object, ws_ty: object,
                      st_ty: object) -> tuple[object, object]:
        """GemmQ's header kernels (hdr, expand): ``hdr(b, ws, plane, kt)`` per header plane and k-tile slot,
        or with an expand kernel ``hdr(b, st, kt)`` collecting the slots in the staging buffer and
        ``expand(st, ws)`` once per K chunk (``gemm_q_collects``)."""
        return kernel("hdr", [B_l1_ty, ws_ty, i32, i32]), None

    gemm_q_collects = False  # GemmQ's header kernels collect the header's k-tile slots in the staging buffer

    # GemmQd's dequantizing core (``gemm_qd``)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        """(count, bytes) of the dequantizing core's buffers besides B and D: the scale workspace ``ws``, the
        header slots ``hb`` (and Q5_K's ``ex``)."""
        return {"ws": (1, max(1024 * op.nsb * op.ws_planes * op.tq // 4, 64)), "hb": (1, KT * op.n * 2)}

    def qd_kernels(self, op: object, kernel: KernelFn, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        """The dequantizing core's kernels (declared in this order)."""
        return DqKernels(hdr=kernel("hdr", [B_l1_ty, ws_ty, i32, i32]), dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty))

    def qd_template_vars(self, op: object) -> dict[str, int]:
        """Extra variables of GemmQd's kernel template."""
        return {}

    # the decode GEMV (``decode.gemv``: two cores per tile)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        """The decode GEMV's kernels take tiles of nsb super-blocks."""
        return False

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        """The designs' token slots the decode GEMV's kernels take (``decode.gemv``: 8, x as single bfp16 rows)."""
        return (4,)

    def mmv_split(self, fmt: str, nsb: int) -> int:
        """Rows of a row-split tile (core H takes rows R / 2 H .. of every super-block), or 0: the super-block split
        (core H takes super-blocks nsb / 2 H .. of every row)."""
        return 0

    def mmv_helper_y(self, fmt: str, nsb: int) -> int:
        """Where the second core's rows go in a row-split tile's output (floats; 0: added to the first core's)."""
        return 0

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        """The sources the decode GEMV includes for a format over nsb super-blocks, in order (shared parts first:
        once per kernel)."""
        return (f"mmv/{fmt}",)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        """Variables of the ``mmv/<fmt>`` source for tiles of tile_bytes over nsb super-blocks."""
        return {}


def dq_kernel(op: object, kernel: KernelFn, B_l1_ty: object, ws_ty: object, D_ty: object,
              extra: list | None = None) -> object:
    """GemmQd's ``dq(b, ws, d, sb, kt, [half,] *extra)``."""
    return kernel("dq", [B_l1_ty, ws_ty, D_ty, i32, i32] + [i32] * (op.halves - 1) + (extra or []))


def collected_header(kernel: KernelFn, B_l1_ty: object, ws_ty: object, hb_ty: object,
                     unit: bool = False, chunk: bool = False) -> tuple[object, object]:
    """GemmQd's header kernels of formats whose header slots are collected (``_hcp``, with the unit index if
    ``unit``) and expanded per super-block (``_hdr_sb``, with the K chunk in the unit if ``chunk``: GemmQ ``hg``)."""
    return (kernel("hcp", [B_l1_ty, hb_ty, i32] + ([i32] if unit else [])),
            kernel("hdr_sb", [hb_ty, ws_ty, i32] + ([i32] if chunk else [])))


MMV_SLOTS = 4  # token slots of the decode GEMV's x panels (``decode.gemv``)


# ----------------------------------------------------------------------
# Host-side packing (lossless re-layout of GGUF fields)
# ----------------------------------------------------------------------


def _word_order(nsb: int) -> np.ndarray:
    """k of each position of one packed row of nsb super-blocks (nibble granular)."""
    sb, s, i, j = np.meshgrid(np.arange(nsb), np.arange(4), np.arange(8), np.arange(8), indexing="ij")
    return (256 * sb + 32 * i + 8 * s + j).reshape(-1)


def _nibbles(u: np.ndarray) -> np.ndarray:
    """(R, n) codes in [0, 15] -> (R, n / 2) bytes, low nibble first."""
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def _bf16_bytes(x: np.ndarray) -> np.ndarray:
    """float array (..., n) -> (..., 2n) uint8 of the bf16 values (round to nearest even)."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).bfloat16()
    return t.view(torch.int16).numpy().view(np.uint8).reshape(*x.shape[:-1], -1)


def sb_major(b: np.ndarray, R: int, n: int) -> np.ndarray:
    """(R, n * m) row bytes -> n slices of (R, m)."""
    return b.reshape(R, n, -1).transpose(1, 0, 2)


# ----------------------------------------------------------------------
# Kernel-source fragments shared by several slots
# ----------------------------------------------------------------------

SOURCES = {
# the decode GEMV's scale tables and per-super-block scaling of the formats with a bf16 scale per 32-k group (Q4_0, Q4_1,
# Q8_0)
"mmv/dtab": r"""// ---- Q4_0, Q4_1, Q8_0: per 32-k group i one exact sum c = bf16(C[hi rows] + C[lo rows]), y += c d (d exact in bf16).

// The scales d [row][sb][i] (rows of 8 NSB bf16) of G row groups and NS super-blocks from d -> tab [sb][g][i][r] (64
// bf16 per (sb, g))
template <int G, int NS = {{ NSB }}>
__attribute__((noinline)) void {{ p }}_dtab(const bfloat16 *__restrict d, bfloat16 *__restrict tab)
{
    bfloat16 *__restrict pt = tab;
    AIE_LOOP_NO_UNROLL AIE_LOOP_REG_REWRITE_BASIC  // (II 11; the default rewrite: 13)
    for (int t = 0; t < NS * G; t++) {
        const bfloat16 *q = d + {{ 64 * NSB }} * (t % G) + 8 * (t / G);  // (row group t % G, super-block t / G)
        const aie::vector<bfloat16, 64> v = aie::concat(
            aie::load_v<8>(q), aie::load_v<8>(q + {{ 8 * NSB }}), aie::load_v<8>(q + {{ 16 * NSB }}), aie::load_v<8>(q + {{ 24 * NSB }}),
            aie::load_v<8>(q + {{ 32 * NSB }}), aie::load_v<8>(q + {{ 40 * NSB }}), aie::load_v<8>(q + {{ 48 * NSB }}), aie::load_v<8>(q + {{ 56 * NSB }}));
        aie::store_v(pt, aie::transpose(v, 8, 8));  // [r][i] -> [i][r]
        pt += 64;
    }
}

// One super-block: y[g] += sum over the 8 groups i of c_g,i (c [i][g][32]) times tab [g][i][r] (G row groups)
template <int G>
__attribute__((noinline)) void {{ p }}_sc1(const bfloat16 *__restrict c, const bfloat16 *__restrict tab, float *__restrict y)
{
    aie::accum<accfloat, 32> Y0(aie::load_v<32>(y)), Y1(aie::load_v<32>(y + 32));
    const bfloat16 __aie_dm_resource_a *__restrict pc = (const bfloat16 __aie_dm_resource_a *)c;
    const bfloat16 __aie_dm_resource_b *__restrict ph = (const bfloat16 __aie_dm_resource_b *)tab;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 8; i++) {
        Y0 = aie::mac(Y0, aie::load_v<32>(pc), aie::load_v<8>(ph).template grow_replicate<32>());
        if constexpr (G == 2)
            Y1 = aie::mac(Y1, aie::load_v<32>(pc + 32), aie::load_v<8>(ph + 64).template grow_replicate<32>());
        pc += 32 * G;
        ph += 8;
    }
    aie::store_v(y, Y0.template to_vector<float>());
    if constexpr (G == 2)
        aie::store_v(y + 32, Y1.template to_vector<float>());
}
""",
# the 6-bit scale expansion of Q4_K and Q5_K (``outs``: Q4_K's scattered group stores)
"gemm/plane8": r"""// lo / hi: the 6-bit values' low nibbles and high crumbs (shifted by hsh) [row 16][group 8]; sw: the
// rows' scale words (shifted by ssh: the bf16 in the low half) -> out (group n at out + outs[n])
[[clang::always_inline]] inline void {{ p }}_plane8(aie::vector<uint8, 128> lo, aie::vector<uint8, 128> hi,
                                                  aie::vector<uint32, 16> sw, unsigned hsh, unsigned ssh,
                                                  bfloat16 *__restrict out)
{
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
    const aie::vector<bfloat16, 64> k128 = aie::broadcast<bfloat16, 64>(-128.0f);
    aie::vector<uint8, 128> c = aie::bit_and(aie::downshift(hi, hsh), aie::broadcast<uint8, 128>(3));
    aie::vector<uint8, 128> v = aie::bit_or(lo, aie::upshift(c, 4));  // 6-bit values [row 16][group 8]
    aie::vector<bfloat16, 16> s16 = aie::filter_even(aie::downshift(sw, ssh).template cast_to<bfloat16>(), 1);  // row scales
    aie::vector<bfloat16, 64> S0, S1;
    al::expand16(s16, S0, S1);
    // (128 + v) s - 128 s = v s exact in fp32, one rounding
    aie::vector<bfloat16, 64> r0 =
        aie::mac(aie::mul(S0, k128), al::zipw(v.template extract<64>(0), c43), S0).template to_vector<bfloat16>();
    aie::vector<bfloat16, 64> r1 =
        aie::mac(aie::mul(S1, k128), al::zipw(v.template extract<64>(1), c43), S1).template to_vector<bfloat16>();
    auto [e1, o1] = aie::interleave_unzip(r0, r1, 1);
    auto [e2, o2] = aie::interleave_unzip(e1, o1, 1);
    auto [e3, o3] = aie::interleave_unzip(e2, o2, 1);
{% if outs is not defined %}
    aie::store_v(out, e3);  // [group n 8][row 16]: groups 0 .. 3, 4 .. 7
    aie::store_v(out + 64, o3);
{% else %}
    aie::vector<bfloat16, 128> tr = aie::concat(e3, o3);  // [group n 8][row 16]
{% for o in outs %}
    aie::store_v(out + {{ o }}, tr.template extract<16>({{ loop.index0 }}));
{% endfor %}
{% endif %}
}

""",
# GemmQ / GemmQd's dequant of the tile-major Q4 tiles (Q4_0 / Q4_1 of 16 rows x 2048, Q4_K); BFP16 kernels render it
# too, unused
"gemm/dq_tile_major": r"""// Dequantize k-tile kt of super-block sb: b ({{ tq // 2 }} KB) = {{ 16 * tq }} packed panels [tile {{ tq }}][k-step
// (il, s)][half][row][word] -> stage = the same panels as bfp16ebs8 d (q - 8), the MAC's B operands (GemmQd: the
// dequantizing core's output; GemmQ: a core's half of the k-tile, tq = 2 of the pass's 4 tiles)
void {{ p }}_dq(const uint8_t *__restrict b, const bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt)
{
{% if "ev_dq" in ablate or "ev_kt" in ablate %}
    event0();
{% endif %}
{% if "dq" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    // scales of groups 2 kt, 2 kt + 1 of super-block sb: header slot
    // (kt'' = 2 (sb % 2) + kt / 2, il'' = kt % 2, s'' = sb / 2), h = group % 2
{% if nsb == 8 %}
    const bfloat16 __aie_dm_resource_c *__restrict hd = (const bfloat16 __aie_dm_resource_c *)
        (ws + ((2 * (sb & 1) + (kt >> 1)) * 2 + (kt & 1)) * {{ 128 * tq }} + (sb >> 1) * {{ 32 * tq }});
{% else %}
    // ([group / 2][tile][group % 2][row 16]: _hdr_q4k of 1024-column tiles)
    const bfloat16 __aie_dm_resource_c *__restrict hd = (const bfloat16 __aie_dm_resource_c *)(ws + (sb * 4 + kt) * {{ 32 * tq }});
{% endif %}
    const uint4 __aie_dm_resource_a *__restrict pb = (const uint4 __aie_dm_resource_a *)b;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)stage);
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
{% if fmt in ("Q4_1", "Q4_K") %}
    // w = d q + m = (128 + q) d + (m - 128 d), exact in fp32 (Q4_K: d = d sc, m = -dmin mn)
    const bfloat16 __aie_dm_resource_d *__restrict hm = (const bfloat16 __aie_dm_resource_d *)(hd + {{ 128 * tq * nsb }});
    const aie::vector<bfloat16, 64> k128 = aie::broadcast<bfloat16, 64>(-128.0f);
{% else %}
    // w = d (q - 8) = (128 + q) d - 136 d, exact in fp32
    const aie::vector<bfloat16, 64> k136 = aie::broadcast<bfloat16, 64>(-136.0f);
{% endif %}
    // (L_RM_PORT-bound: shuffles + the stream's pushes / flushes, II 40 per 8 panels; post-RA only: 41)
    for (int it = 0; it < {{ 2 * tq }}; it++) {  // (tile, group) = (it / 2, 2 kt + it % 2)
        aie::vector<bfloat16, 64> D0, D1;
        al::expand16_bc(aie::load_v<16>(hd), D0, D1);
        hd += 16;
{% if fmt in ("Q4_1", "Q4_K") %}
        aie::vector<bfloat16, 64> M0, M1;
        al::expand16_bc(aie::load_v<16>(hm), M0, M1);
        hm += 16;
        aie::accum<accfloat, 64> C0 = aie::mac(aie::accum<accfloat, 64>(M0), D0, k128);  // (no ones vector: one spill less)
        aie::accum<accfloat, 64> C1 = aie::mac(aie::accum<accfloat, 64>(M1), D1, k128);
{% else %}
        aie::accum<accfloat, 64> C0 = aie::mul(D0, k136);
        aie::accum<accfloat, 64> C1 = aie::mul(D1, k136);
{% endif %}
        AIE_LOOP_UNROLL_FULL
        for (int st = 0; st < 4; st++) {
            aie::vector<uint8, 64> q0 = aie::unpack(aie::load_v<64>(pb));
            aie::vector<uint8, 64> q1 = aie::unpack(aie::load_v<64>(pb + 32));
            pb += 64;
            out.push(aie::mac(C0, al::zipw(q0, c43), D0).template to_vector<bfp16ebs8>());
            out.push(aie::mac(C1, al::zipw(q1, c43), D1).template to_vector<bfp16ebs8>());
        }
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}
""",
}
