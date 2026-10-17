"""Decode GEMV on the MMUL for several token slots (the decode overlays' main and helper cores, ``overlay.decode``):
the tiles of the weight image (``formats``) against bfp16 panels of up to 4 slots' x, with exact integer weight panels
(BFP16 tiles: their own panels). Two cores share a tile: the super-block split (every format's tiles of 1024
columns: one core takes super-blocks 0, 1 (``{p}_mmv<0>``), the other 2, 3 (``{p}_mmv<1>``)) or the row split
(Gemma's 512-column Q4_0 / Q8_0 tiles: each core its half of the rows of every super-block, ``row_split``); a solo
format's tiles (``render(solo=)``) the second core alone.

This module is the frame: the shared helpers, the x and sum panels and the dispatcher ``{p}_mmv<H>``. The formats'
kernels and template variables are their slots' (``Slot.mmv_*``, ``sources["mmv/<fmt>"]``: ``formats.q4k`` etc.). The
host model of the arithmetic and a two-core harness (``MmvOperator``) are the tests' (``tests/rtc/mmv.py``).

A tile covers a K chunk of ``nsb`` super-blocks (``Chunk``: 256 nsb k).

x panels: per k-block (8 consecutive k) one 8x8 bfp16 panel, rows a = [hi of slot 0 .. 3 | lo of slot
0 .. 3] (hi = bfp16(x), lo = bfp16(x - hi): ~14 bits below each block's maximum), 72 B, in k order (a super-block's
32 panels in a row); a chunk has 32 nsb. Sum panels: the slots' sums of x over 32-k sub-blocks
(per super-block one panel [a][sub-block 8]) and over 16-k ones (two per super-block), as hi + lo rows, in slots of
two panels (``XS_SLOT``): slot j < nsb / 2: the 32-k panels of super-blocks 2 j, 2 j + 1; slot nsb / 2 + sb: the two
16-k panels of super-block sb.

A weight panel is 8 rows x 8 k of the tile's integer codes as bfp16 mantissas at exponent 133 (value = code), so the
MMUL sums exact products: C[a][r] = sum_k X[a][k] W[r][k] in fp32. The codes of a row are words of 8 consecutive k
(``formats``' layouts), so 8 rows' words are transposed into panels first (``_tr8``: two shuffle stages per 8x8
words). Per sub-block (32 k, Q6_K, Q2_K, Q3_K: 16 k) the partial sums of the hi and lo rows are added and rounded
once to bf16 (c, 32 lanes [slot 4][r 8]), with the codes centered (q - 8, - 16, - 32: the sum panels against -z e_i
start each chain; Q2_K, Q3_K: x against a constant -z panel), so that c stays small; then y += c (d sc) with d sc
(exact in fp32) as bf16 hi + lo (``_sc``; Q2_K, Q3_K: the integer scales, ``formats.k23``). Each core
keeps its transposed codes, its tables and partial sums in its scratch (``SCR_BYTES``; Q4_K: the codes in place).

Output: y [g 2][slot 4][r 8] fp32 (rows 8 g ..), accumulated.
"""

from ..formats import FORMATS, kernel_template, slot
from ..formats import SLOTS as FORMAT_SLOTS
from ..formats.common import MMV_SLOTS

SLOTS = MMV_SLOTS  # token slots of an x panel
KB = 8  # k per block
PANEL = 72  # bytes of a bfp16 8x8 panel
XS_SLOT = 192  # bytes of a slot of sum panels (two panels, 64-byte aligned: a panel stream must start aligned)
Y_FLOATS = 64  # a tile's output: [g 2][slot 4][r 8] fp32 (tiles of up to 32 rows in the super-block split: [g 4], 128)
SCR_BYTES = 8192  # a core's scratch: transposed codes, tables, partial sums


def supported(fmt: str, nsb: int) -> bool:
    """The decode GEMV reads tiles of this format over nsb super-blocks (its slot's kernels)."""
    return bool(FORMATS[fmt].mmv) and slot(fmt).mmv_supports(fmt, nsb)


def row_split(fmt: str, nsb: int) -> int:
    """Rows of a row-split tile (both cores take all super-blocks, the second core the second half of the rows), or 0:
    the super-block split (``Slot.mmv_split``)."""
    return slot(fmt).mmv_split(fmt, nsb)


def row_splits(nsb: int) -> dict[str, int]:
    """{format: rows} of the formats read row-split over nsb super-blocks (an image's ``max_rows``)."""
    return {f: row_split(f, nsb) for f in FORMATS if supported(f, nsb) and row_split(f, nsb)}


def helper_y(fmt: str, nsb: int) -> int:
    """Where the second core's rows start in a row-split tile's output (floats: [g][slot][r]; 0: added to the first
    core's, or Q8_0's: its helper writes its row group 1 itself)."""
    return slot(fmt).mmv_helper_y(fmt, nsb)


def tile_rows_of(fmt: str, tile_bytes: int, nsb: int) -> int:
    """Rows of a format's tile in the decode GEMV (a row-split format's are fixed: ``row_split``)."""
    rs = row_split(fmt, nsb)
    r = FORMATS[fmt].rows(tile_bytes, nsb, rs)
    assert not rs or r == rs, (fmt, tile_bytes, nsb, r)
    return r


class Chunk:
    """The K chunk of a tile: ``nsb`` super-blocks, ``k`` columns, the bytes of its x panels (``xp``) and of its sum
    panels (``xs``: nsb / 2 slots of 32-k panels, nsb of 16-k ones)."""

    def __init__(self, nsb: int = 4):
        assert nsb % 2 == 0, nsb
        self.nsb, self.k = nsb, 256 * nsb
        self.xp = self.k // KB * PANEL
        self.xs = (nsb // 2 + nsb) * XS_SLOT


_SRC = r"""
namespace {

[[clang::always_inline]] inline v8int8 {{ p }}_e8(int e)
{
    v2int32 v;
    v[0] = e * 0x01010101;
    v[1] = e * 0x01010101;
    return (v8int8)v;
}

// 8 rows' 32-byte chunks (row r at p + r ST) -> o0 .. o3 = [panel 2 k | panel 2 k + 1], panel c = word c of the 8 rows
template <int ST, typename T>
[[clang::always_inline]] inline void {{ p }}_tr8(const T *p, v16int32 &o0, v16int32 &o1, v16int32 &o2, v16int32 &o3)
{
    const v16int32 a0 = concat(*(const v8int32 *)(p), *(const v8int32 *)(p + ST));
    const v16int32 a1 = concat(*(const v8int32 *)(p + 2 * ST), *(const v8int32 *)(p + 3 * ST));
    const v16int32 a2 = concat(*(const v8int32 *)(p + 4 * ST), *(const v8int32 *)(p + 5 * ST));
    const v16int32 a3 = concat(*(const v8int32 *)(p + 6 * ST), *(const v8int32 *)(p + 7 * ST));
    const v16int32 t0 = shuffle(a0, a1, T32_4x8_lo), t1 = shuffle(a0, a1, T32_4x8_hi);
    const v16int32 t2 = shuffle(a2, a3, T32_4x8_lo), t3 = shuffle(a2, a3, T32_4x8_hi);
    o0 = shuffle(t0, t2, T128_2x4_lo);
    o1 = shuffle(t0, t2, T128_2x4_hi);
    o2 = shuffle(t1, t3, T128_2x4_lo);
    o3 = shuffle(t1, t3, T128_2x4_hi);
}

// the 8 panels of o0 .. o3 -> p + c ST (panel c)
template <int ST, typename T>
[[clang::always_inline]] inline void {{ p }}_put8(T *p, const v16int32 &o0, const v16int32 &o1, const v16int32 &o2,
                                                  const v16int32 &o3)
{
    *(v8int32 *)(p) = extract_v8int32(o0, 0);
    *(v8int32 *)(p + ST) = extract_v8int32(o0, 1);
    *(v8int32 *)(p + 2 * ST) = extract_v8int32(o1, 0);
    *(v8int32 *)(p + 3 * ST) = extract_v8int32(o1, 1);
    *(v8int32 *)(p + 4 * ST) = extract_v8int32(o2, 0);
    *(v8int32 *)(p + 5 * ST) = extract_v8int32(o2, 1);
    *(v8int32 *)(p + 6 * ST) = extract_v8int32(o3, 0);
    *(v8int32 *)(p + 7 * ST) = extract_v8int32(o3, 1);
}

// a weight panel of 32 B of nibbles (8 rows x 8 k) unpacked by the load, at exponent e (value = code 2^(e - 133))
template <typename T>
[[clang::always_inline]] inline v64bfp16ebs8 {{ p }}_wu(const T *p, int e = 0x85)
{
    v64bfp16ebs8 n;
    n.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_b *)p);
    n.exponent = {{ p }}_e8(e);
    return n;
}

// a constant panel -z e_i (column i): the chain start of sub-block i, -z XS_i (exponent e: -z 2^(e - 133))
template <typename T>
[[clang::always_inline]] inline v64bfp16ebs8 {{ p }}_np(const T *p, int e = 0x85)
{
    v64bfp16ebs8 n;
    n.mantissa = *(const v64int8 __aie_dm_resource_d *)p;
    n.exponent = {{ p }}_e8(e);
    return n;
}

// 8 bf16 (one per row r) -> 64 lanes [a 8][r 8]
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_rep8v(aie::vector<bfloat16, 8> v)
{
    const v32bfloat16 b = (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)v.template grow<32>(), 0);
    return aie::concat(aie::vector<bfloat16, 32>(b), aie::vector<bfloat16, 32>(b));
}

// 8 bf16 (one per row r) -> 32 lanes [slot 4][r 8]
template <typename T>
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_rep4(const T *p)
{
    return aie::vector<bfloat16, 32>((v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)aie::load_v<8>(p).template grow<32>(), 0));
}

// the 64 lanes [a][r] of an MMUL result -> bf16 of the hi and lo rows' sums: 32 lanes [slot][r]
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_sum2(v64accfloat C)
{
    const aie::accum<accfloat, 64> a(C);
    return aie::add(a.template extract<32>(0), a.template extract<32>(1)).template to_vector<bfloat16>();
}

// T (exact integers, [i][r]) times D ([i][r] = d_r) exactly in fp32 -> tab: bf16 hi, lo (64 each)
[[clang::always_inline]] inline void {{ p }}_td(v64accfloat T, aie::vector<bfloat16, 64> D, bfloat16 *__restrict tab)
{
    const aie::accum<accfloat, 64> TD = aie::mul(aie::accum<accfloat, 64>(T).template to_vector<bfloat16>(), D);
    const aie::vector<bfloat16, 64> th = TD.template to_vector<bfloat16>();
    aie::store_v(tab, th);
    aie::store_v(tab + 64, aie::msc(TD, th, aie::broadcast<bfloat16, 64>(1.0f)).template to_vector<bfloat16>());
}

// acc + M [a][r] as bf16 hi + lo times s ([a][r])
[[clang::always_inline]] inline aie::accum<accfloat, 64> {{ p }}_mo(v64accfloat M, aie::vector<bfloat16, 64> s,
                                                                    aie::accum<accfloat, 64> acc)
{
    const aie::accum<accfloat, 64> m(M);
    const aie::vector<bfloat16, 64> h = m.template to_vector<bfloat16>();
    return aie::mac(aie::mac(acc, h, s), aie::msc(m, h, aie::broadcast<bfloat16, 64>(1.0f)).template to_vector<bfloat16>(), s);
}

// y[0 .. 32) += the hi and lo rows' sums of a (64 lanes [a][r])
[[clang::always_inline]] inline void {{ p }}_yadd(float *__restrict y, aie::accum<accfloat, 64> a)
{
    const aie::vector<float, 64> v = a.template to_vector<float>();
    aie::store_v(y, aie::add(aie::load_v<32>(y), aie::add(v.template extract<32>(0), v.template extract<32>(1))));
}

// the 8 values of the rows at lanes sbl, sbl + 2, .. of v
[[clang::always_inline]] inline aie::vector<bfloat16, 8> {{ p }}_pick(aie::vector<bfloat16, 16> v, int sbl)
{
    return aie::filter_even(aie::shuffle_down(v, sbl), 1);
}

// -z e_i panels (mantissas): [z 8, 16, 32][i][64]
alignas(64) static const int8_t {{ p }}_nz[3][8][64] = {
{% for z in (8, 16, 32) %}
    {
{% for i in range(8) %}
        { {% for l in range(64) %}{{ -z if l % 8 == i else 0 }}, {% endfor %}},
{% endfor %}
    },
{% endfor %}
};

// Pass 4 of every format: y[g] += sum over 8 sub-blocks of c_g (TD hi + lo)_g; c: of g 0 at c + CS n, g 1 at + 32;
// tab: TD hi of g 0 at tab + TN n, lo at + 64, g 1 at + TG (one copy for every format: runtime strides)
__attribute__((noinline)) void {{ p }}_sc(const bfloat16 *__restrict c, const bfloat16 *__restrict tab, float *__restrict y,
                                          int TG, int TN, int CS)
{
    constexpr int N = 8;
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 32> Y0(aie::load_v<32>(y)), Y1(aie::load_v<32>(y + 32));
    const bfloat16 __aie_dm_resource_a *__restrict pc = (const bfloat16 __aie_dm_resource_a *)c;
    const bfloat16 __aie_dm_resource_b *__restrict ph = (const bfloat16 __aie_dm_resource_b *)tab;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < N; i++) {
        const aie::vector<bfloat16, 32> c0 = aie::load_v<32>(pc), c1 = aie::load_v<32>(pc + 32);
        Y0 = aie::mac(aie::mac(Y0, c0, {{ p }}_rep4(ph)), c0, {{ p }}_rep4(ph + 64));
        Y1 = aie::mac(aie::mac(Y1, c1, {{ p }}_rep4(ph + TG)), c1, {{ p }}_rep4(ph + TG + 64));
        pc += CS;
        ph += TN;
    }
    aie::store_v(y, Y0.template to_vector<float>());
    aie::store_v(y + 32, Y1.template to_vector<float>());
}

{% for part in parts %}

{% include part %}
{% endfor %}

// One tile of format fmt, super-blocks 2 H, 2 H + 1: y[g][32] += their rows against xp (the 64 panels of those
// super-blocks) and xs (the chunk's sum panels)
{% if solo %}
// (a solo format, ``render(solo=)``: core 1 runs both halves of its tiles, core 0 none)
{% endif %}
{% if rowsplit %}
// (a row-split format, ``row_split``: rows 16 H .. of all its super-blocks, xp the chunk's panels; Q8_0: its row
// group H into y[32 H ..])
{% endif %}
template <int H>
__attribute__((noinline)) void {{ p }}_mmv(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr,
                                           int32_t fmt)
{
{% for f in fmts %}
{% if f in solo %}
    if (fmt == {{ fid[f] }}) {
        if constexpr (H == 1) {
            {{ p }}_{{ fn[f] }}<0>(a, xp - {{ -solo_x0 }}, xs, y, scr);
            {{ p }}_{{ fn[f] }}<1>(a, xp, xs, y, scr);
        }
        return;
    }
{% else %}
    if (fmt == {{ fid[f] }})
        {{ p }}_{{ fn[f] }}<H>(a, xp, xs, y, scr);
{% endif %}
{% endfor %}
}

}  // namespace
"""


_TEMPLATE = kernel_template(_SRC)


def render(p: str, fmts: list[str], *, tile_bytes: int = 9280, nsb: int = 4, solo: tuple[str, ...] = (),
           solo_x0: int = 0) -> str:
    """The kernel source of ``{p}_mmv<H>`` for the tile formats ``fmts`` (tiles of ``tile_bytes`` over nsb
    super-blocks): the frame and the slots' ``mmv/<fmt>`` sources (``Slot.mmv_parts``). ``solo``: formats whose tiles
    core 1 runs alone, both halves (its xp: the second half's panels, as always; the first half's at xp + solo_x0,
    default: right before them), core 0 none: program memory of core 0 (the main core) for a rare format."""
    assert all(supported(f, nsb) for f in fmts), (fmts, nsb)
    assert all(f in fmts and not row_split(f, nsb) for f in solo), (solo, fmts)
    parts: list[str] = []
    var: dict[str, int] = {}
    for s in FORMAT_SLOTS:
        for f in s.formats:
            if f.name in fmts:
                parts += [x for x in s.mmv_parts(f.name, nsb) if x not in parts]
                var.update(s.mmv_vars(f.name, tile_bytes, nsb))
    return _TEMPLATE.render(p=p, fmts=sorted(fmts), solo=solo, solo_x0=solo_x0 or -(32 * nsb // 2 * PANEL), parts=parts,
                            rowsplit=any(row_split(f, nsb) for f in fmts),
                            fid={f: FORMATS[f].fid for f in FORMATS}, fn={f: FORMATS[f].mmv for f in FORMATS}, NSB=nsb,
                            PANEL=PANEL, XS_SLOT=XS_SLOT, **var)
