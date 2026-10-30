"""Kernel source of the decode overlay's main and helper cores (``design.DecodeOverlayOperator._render`` renders it
with the design's constants and the metadata, flag and state words of ``decode``): per X object ``begin`` (a phase's
metadata, a residual slice, another core's object), the input strips (``strip`` / ``strip_n``: the helper's commands;
``nbegin``: a norm-weight object; ``psums``: a header's partial sums), ``input`` / ``inv`` (the sum panels, 1 / rms), per
weight tile ``cmd`` / ``tile`` / ``collect`` (the main core's half, the helper's, the row epilogue: norm scale, GU and
PLI activations, residual adds), ``emit`` (C objects), and the helper's ``htask`` (half tiles, ``hstrip`` /
``hpstrip`` x panels, ``hinput``). The GEMV kernels themselves: ``gemv.render``."""

from ..operators.base import JINJA_ENV

SRC = JINJA_ENV.from_string(r"""
// The main core's workspace (offsets in bytes: DecodeOverlayOperator.workspace())
#define WS(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// The event window of a stats build (DecodeOverlayOperator(win=)): event0 / event1 around region K
template <int K>
struct {{ p }}_Win {
    [[clang::always_inline]] {{ p }}_Win() { if constexpr (K == {{ WIN }}) event0(); }
    [[clang::always_inline]] ~{{ p }}_Win() { if constexpr (K == {{ WIN }}) event1(); }
};

alignas(64) static const int16_t {{ p }}_iota[64] = { {% for i in range(64) %}{{ i }}, {% endfor %}};

// n 64-byte vectors of zeros at p, of src at dst (64-byte aligned): one copy for the cold paths (a template's inline
// loop per call site costs ~150 B of program memory)
__attribute__((noinline, minsize)) void {{ p }}_zv(int32_t *__restrict p, int n)
{
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i++)
        aie::store_v(p + 16 * i, aie::zeros<int32, 16>());
}

__attribute__((noinline, minsize)) void {{ p }}_cv(int32_t *__restrict dst, const int32_t *__restrict src, int n)
{
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i++)
        aie::store_v(dst + 16 * i, aie::load_v<16>(src + 16 * i));
}

// dst[0 .. R) = o[0 .. R) (R <= N <= 32) through an aligned 64-element window
template <unsigned N>
__attribute__((noinline)) void {{ p }}_put_bf16(bfloat16 *__restrict dst, aie::vector<bfloat16, N> o, int R)
{
    bfloat16 *__restrict pw = (bfloat16 *)((uintptr_t)dst & ~(uintptr_t)63);
    const int sh = (int)(dst - pw);
    aie::vector<bfloat16, 64> v = aie::zeros<bfloat16, 64>();
    v.insert(0, o);
    v = aie::shuffle_up_fill(v, aie::zeros<bfloat16, 64>(), sh);
    const aie::vector<int16, 64> li = aie::load_v<64>({{ p }}_iota);
    const aie::mask<64> m = aie::ge(li, (int16)sh) & aie::lt(li, (int16)(sh + R));
    aie::store_v(pw, aie::select(aie::load_v<64>(pw), v, m));
}

// the block of row tile rt: its index, first row tile and first output row (blocks' valid rows in order)
[[clang::always_inline]] inline int {{ p }}_block(const int32_t *__restrict st, int rt, int &rt0, int &row0)
{
    int b = 0;
    rt0 = 0;
    row0 = 0;
    AIE_LOOP_NO_UNROLL
    while (b + 1 < st[{{ S_NB }}] && rt >= st[{{ S_B }} + 3 * b]) {
        rt0 = st[{{ S_B }} + 3 * b];
        row0 += st[{{ S_B }} + 3 * b + 2];
        b++;
    }
    return b;
}

[[clang::always_inline]] inline int {{ p }}_rows(int fmt)
{
{% for f, r in rows %}
    if (fmt == {{ fid[f] }})
        return {{ r }};
{% endfor %}
    return 0;
}
{% if rowsplit %}

// a row-split format's tile (the helper takes its second half of the rows: gemv.row_split), where the helper's rows
// go in the tile's output (gemv.helper_y)
[[clang::always_inline]] inline int {{ p }}_rs(int fmt)
{
    return {% for f in rows_split %}fmt == {{ fid[f] }}{{ " || " if not loop.last else "" }}{% endfor %};
}

[[clang::always_inline]] inline int {{ p }}_hy(int fmt)
{
{% for f, y in helper_y %}
    if (fmt == {{ fid[f] }})
        return {{ y }};
{% endfor %}
    return 0;
}
{% endif %}

// B_j: the 8x8 panel [n][k] of ones in row n = j (value 1: mantissa 1 at exponent 133): A B_j^T sums each row of A
// into lane j
// B_j (block j ones, the others zero; exponent 133: values 1) as bfp16 panels in memory order ([E, m0 .. m7] per block):
// popped from a stream, not assembled from a mantissa load and a constant exponent (that pins the register pair the
// next iteration's loads reuse: RecMII 33)
alignas(64) static const uint8_t {{ p }}_ohp[8 * 72] = {
{% for j in range(8) %}
    {% for i in range(8) %}133, {% for m in range(8) %}{{ 1 if i == j else 0 }}, {% endfor %}{% endfor %}

{% endfor %}
};

alignas(64) static const int8_t {{ p }}_onehot[8][64] = {
{% for j in range(8) %}
    { {% for l in range(64) %}{{ 1 if l // 8 == j else 0 }}, {% endfor %}},
{% endfor %}
};

// The 4 slots' 8 values of an input k block of SP slots ([SP][8] bf16 at x). The lanes of the slots past SP are
// undefined: every use of x is per slot (rows of the panels and the MMUL, lanes of the squares), and their outputs go
// nowhere
template <int SP, typename T>
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_kb(const T *x)
{
    if constexpr (SP == 4)
        return aie::load_v<32>(x);
    else
        return aie::load_v<8 * SP>(x).template grow<32>();
}

// _kb of a runtime count of slots (x 16-byte aligned): four 16-byte loads, the lanes past the slots the next k blocks'
// values (undefined as _kb's)
template <typename T>
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_kbr(const T *x)
{
    return aie::concat(aie::load_v<8>(x), aie::load_v<8>(x + 8), aie::load_v<8>(x + 16), aie::load_v<8>(x + 24));
}

{% if SLOTS == 8 %}
// The helper: a strip of {{ OBJ_K // 8 }} k blocks of x ([k block][slot SP][8] bf16, times the norm weights w (bf16 per
// k) when NORM) -> their x panels at xp (per k block the slots' x as single bfp16, ``gemv``) and the slots' 16-k sums
// at t ([128-k group][slot 8][16-k block 8] fp32: the panels' rows summed on the MMUL against B_j); sq_ += the squares
// ([slot 8][8] lanes). The rows and lanes of the slots past SP are undefined (as _kb's)
template <bool NORM, int SP>
__attribute__((noinline)) void {{ p }}_hstrip(const bfloat16 *__restrict x, const bfloat16 *__restrict w,
                                              float *__restrict sq_, uint8_t *__restrict xp, float *__restrict t)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BOut<> o((bfp16ebs8 *)xp);
    aie::accum<accfloat, 64> sq(aie::load_v<64>(sq_));
    const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)x;
    const bfloat16 __aie_dm_resource_b *__restrict pw = (const bfloat16 __aie_dm_resource_b *)w;
    float *__restrict pt = t;
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < {{ OBJ_K // 128 }}; g++) {
        v64accfloat T = (v64accfloat)aie::zeros<accfloat, 64>();
        const int8_t __aie_dm_resource_d *__restrict pb = (const int8_t __aie_dm_resource_d *){{ p }}_onehot[0];
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < 8; j++) {  // 16-k block j: k blocks 2 j, 2 j + 1
            aie::vector<bfloat16, 64> v0, v1;
            if constexpr (SP == 8) {
                v0 = aie::load_v<64>(px), v1 = aie::load_v<64>(px + 64);
            } else {
                v0 = aie::load_v<8 * SP>(px).template grow<64>(), v1 = aie::load_v<8 * SP>(px + 8 * SP).template grow<64>();
            }
            px += 16 * SP;
            sq = aie::mac_square(aie::mac_square(sq, v0), v1);
            aie::accum<accfloat, 64> x0, x1;
            if constexpr (NORM) {  // (the 8 weights of a k block for every slot)
                const aie::vector<bfloat16, 16> w2 = aie::load_v<16>(pw);
                pw += 16;
                x0 = aie::mul(v0, w2.template extract<8>(0).template grow_replicate<64>());
                x1 = aie::mul(v1, w2.template extract<8>(1).template grow_replicate<64>());
            } else {
                x0 = aie::accum<accfloat, 64>(v0);
                x1 = aie::accum<accfloat, 64>(v1);
            }
            const al::BV p0 = x0.template to_vector<bfp16ebs8>(), p1 = x1.template to_vector<bfp16ebs8>();
            o.push(p0);
            o.push(p1);
            v64bfp16ebs8 b;
            b.mantissa = *(const v64int8 __aie_dm_resource_d *)pb;
            b.exponent = al::e8(0x85);
            pb += 64;
            T = mac_8x8_8x8T((v64bfp16ebs8)p0, 1, b, 0, T);
            T = mac_8x8_8x8T((v64bfp16ebs8)p1, 1, b, 0, T);
        }
        aie::store_v(pt, aie::accum<accfloat, 64>(T).template to_vector<float>());  // (rows: the slots)
        pt += 64;
    }
    aie::store_v(sq_, sq.template to_vector<float>());
}

// the sum panels of a chunk's super-block sb from its 16-k sums t (128 fp32 [128-k group 2][slot 8][16-k block 8])
// -> xs (its 32-k panels, hi then lo, at the slot of its pair; its two 16-k panels, hi and lo each, at slot 2 + sb)
__attribute__((noinline)) void {{ p }}_xs_sb(const float *__restrict t, uint8_t *__restrict xs, int sb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::BV hi, lo;
    {
        al::BOut<> o((bfp16ebs8 *)(xs + {{ gv.XS_SLOT }} * ({{ gv.NSB // 2 }} + sb)));
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(t)), nI, hi, lo);
        o.push(hi);
        o.push(lo);
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(t + 64)), nI, hi, lo);
        o.push(hi);
        o.push(lo);
    }
    // the 32-k sums [slot 8][32-k block 8]: block 4 g + i of a slot = t_g[slot][2 i] + t_g[slot][2 i + 1]
    aie::vector<float, 32> s[2];
    for (int g = 0; g < 2; g++) {
        const aie::vector<float, 32> a = aie::load_v<32>(t + 64 * g), b = aie::load_v<32>(t + 64 * g + 32);
        s[g] = aie::concat(aie::add(aie::filter_even(a, 1), aie::filter_odd(a, 1)),
                           aie::add(aie::filter_even(b, 1), aie::filter_odd(b, 1)));  // ([slot 8][4])
    }
    const auto [z0, z1] = aie::interleave_zip(s[0], s[1], 4);
    al::BOut<> o((bfp16ebs8 *)(xs + {{ gv.XS_SLOT }} * (sb >> 1) + {{ 2 * gv.PANEL }} * (sb & 1)));
    al::split_hilo(aie::accum<accfloat, 64>(aie::concat(z0, z1)), nI, hi, lo);
    o.push(hi);
    o.push(lo);
}
{% else %}
// The helper: a strip of {{ OBJ_K // 8 }} k blocks of x ([k block][slot sp][8] bf16, times the norm weights w (bf16 per
// k) when NORM) -> their x panels at xp ([hi of the slots | lo of the slots] per k block) and the slots' 16-k sums at t
// ([128-k group][slot 4][16-k block 8] fp32: the panels' rows summed on the MMUL against B_j); sq_ += the squares
// ([slot 4][8] lanes). (II ~17 per 16 k: the panel stream's push / flush chain)
template <bool NORM>
__attribute__((noinline)) void {{ p }}_hstrip(const bfloat16 *__restrict x, const bfloat16 *__restrict w,
                                              float *__restrict sq_, uint8_t *__restrict xp, float *__restrict t, int sp)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::BOut<> o((bfp16ebs8 *)xp);
    aie::accum<accfloat, 32> sq(aie::load_v<32>(sq_));
    const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)x;
    const bfloat16 __aie_dm_resource_b *__restrict pw = (const bfloat16 __aie_dm_resource_b *)w;
    float *__restrict pt = t;
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < {{ OBJ_K // 128 }}; g++) {
        v64accfloat T = (v64accfloat)aie::zeros<accfloat, 64>();
        const int8_t __aie_dm_resource_d *__restrict pb = (const int8_t __aie_dm_resource_d *){{ p }}_onehot[0];
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < 8; j++) {  // 16-k block j: k blocks 2 j, 2 j + 1
            const aie::vector<bfloat16, 32> v0 = {{ p }}_kbr(px), v1 = {{ p }}_kbr(px + 8 * sp);
            px += 16 * sp;
            sq = aie::mac_square(aie::mac_square(sq, v0), v1);
            aie::accum<accfloat, 32> x0, x1;
            if constexpr (NORM) {  // (the 8 weights of a k block for every slot)
                const v16int32 w2 = (v16int32)(v32bfloat16)aie::load_v<16>(pw).template grow<32>();
                pw += 16;
                x0 = aie::mul(v0, aie::vector<bfloat16, 32>((v32bfloat16)broadcast_elem_128(w2, 0)));
                x1 = aie::mul(v1, aie::vector<bfloat16, 32>((v32bfloat16)broadcast_elem_128(w2, 1)));
            } else {
                x0 = aie::accum<accfloat, 32>(v0);
                x1 = aie::accum<accfloat, 32>(v1);
            }
            const aie::accum<accfloat, 64> a = aie::concat(x0, x1);
            al::MM8 l(a);
            l.mac(a.template to_vector<bfp16ebs8>(), aie::op_transpose(nI));  // a - hi exactly
            const aie::accum<accfloat, 64> lo = l.to_accum();
            const al::BV p0 = aie::concat(x0, lo.template extract<32>(0)).template to_vector<bfp16ebs8>();
            const al::BV p1 = aie::concat(x1, lo.template extract<32>(1)).template to_vector<bfp16ebs8>();
            o.push(p0);
            o.push(p1);
            v64bfp16ebs8 b;
            b.mantissa = *(const v64int8 __aie_dm_resource_d *)pb;
            b.exponent = al::e8(0x85);
            pb += 64;
            T = mac_8x8_8x8T((v64bfp16ebs8)p0, 1, b, 0, T);
            T = mac_8x8_8x8T((v64bfp16ebs8)p1, 1, b, 0, T);
        }
        const aie::accum<accfloat, 64> ts(T);  // rows: the slots' hi sums, then their lo sums
        aie::store_v(pt, aie::add(ts.template extract<32>(0), ts.template extract<32>(1)).template to_vector<float>());
        pt += 32;
    }
    aie::store_v(sq_, sq.template to_vector<float>());
}

// the sum panels of a chunk's super-block sb from its 16-k sums t (64 fp32 [128-k group 2][slot 4][16-k block 8])
// -> xs (its 32-k panel at the slot of its pair, its two 16-k panels at slot 2 + sb)
__attribute__((noinline)) void {{ p }}_xs_sb(const float *__restrict t, uint8_t *__restrict xs, int sb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    const aie::vector<float, 32> t0 = aie::load_v<32>(t), t1 = aie::load_v<32>(t + 32);
    {
        al::BOut<> o((bfp16ebs8 *)(xs + {{ gv.XS_SLOT }} * ({{ gv.NSB // 2 }} + sb)));
        o.push(al::hilo4(aie::accum<accfloat, 32>(t0), nI));
        o.push(al::hilo4(aie::accum<accfloat, 32>(t1), nI));
    }
    // the 32-k sums [slot 4][32-k block 8]: block 4 g + i of a slot = t_g[slot][2 i] + t_g[slot][2 i + 1]
    const aie::vector<float, 16> s0 = aie::add(aie::filter_even(t0, 1), aie::filter_odd(t0, 1));
    const aie::vector<float, 16> s1 = aie::add(aie::filter_even(t1, 1), aie::filter_odd(t1, 1));
    const auto [z0, z1] = aie::interleave_zip(s0, s1, 4);
    al::BOut<> o((bfp16ebs8 *)(xs + {{ gv.XS_SLOT }} * (sb >> 1) + {{ gv.PANEL }} * (sb & 1)));
    o.push(al::hilo4(aie::accum<accfloat, 32>(aie::concat(z0, z1)), nI));
}
{% endif %}

// The helper: an input's sum panels (nsb super-blocks: T16 at t) -> xs; with norm the slots' 1 / rms -> r (8 lanes
// per slot)
__attribute__((noinline)) void {{ p }}_hinput(uint8_t *__restrict xs, const float *__restrict t, const float *__restrict sq,
                                              float *__restrict r, int nsb, int norm{{ ", float ih" if post else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < nsb; s++)  // (super-block s: chunk s / {{ gv.NSB }})
        {{ p }}_xs_sb(t + {{ 16 * SLOTS }} * s, xs + {{ gv.XS_CHUNK }} * (s >> {{ gv.NSB.bit_length() - 1 }}), s & {{ gv.NSB - 1 }});
    if (!norm)
        return;
{% if SLOTS == 8 %}
    const aie::vector<float, 8> ss = al::group8<false>(aie::load_v<64>(sq));
    aie::store_v(r, al::bcast8(al::inv_rms<{{ H }}, 8>(ss.template grow<16>(), {{ eps }}f).template extract<8>(0)));
{% else %}
    const aie::vector<float, 8> ss = al::group8<false>(aie::concat(aie::load_v<32>(sq), aie::zeros<float, 32>()));
{% endif %}
{% if SLOTS == 8 %}
{% elif post %}
    aie::store_v(r, al::rows4(al::inv_rms<{{ SLOTS }}>(ss.template grow<16>(), ih, {{ eps }}f)));
{% else %}
    aie::store_v(r, al::rows4(al::inv_rms<{{ H }}, {{ SLOTS }}>(ss.template grow<16>(), {{ eps }}f)));
{% endif %}
}

{% if post %}
// res[s][j0 .. j0 + n) = (res + pend ivv wp) scale for every slot (the exact residual update of a header phase; n a
// multiple of 8, wp: n bf16 of w_post)
__attribute__((noinline)) void {{ p }}_pupd(const bfloat16 *__restrict wp, int32_t *__restrict ws, int j0, int n)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict r = WS(float, {{ w_.res }}) + j0;
    const float *__restrict v = WS(float, {{ w_.pend }}) + j0;
    aie::vector<bfloat16, 32> ivh, ivl;
    al::split2<32>(aie::load_v<32>(WS(float, {{ w_.ivv }})), ivh, ivl);
    const aie::vector<bfloat16, 32> sc = aie::broadcast<bfloat16, 32>(
        __builtin_bit_cast(bfloat16, (uint16_t)WS(int32_t, {{ w_.st }})[{{ S_SCALE }}]));
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < n; j += 8) {
        const aie::vector<float, 32> rv = aie::concat(aie::load_v<8>(r + j), aie::load_v<8>(r + {{ RES_PITCH }} + j),
                                                      aie::load_v<8>(r + {{ 2 * RES_PITCH }} + j), aie::load_v<8>(r + {{ 3 * RES_PITCH }} + j));
        const aie::vector<float, 32> vv = aie::concat(aie::load_v<8>(v + j), aie::load_v<8>(v + {{ RES_PITCH }} + j),
                                                      aie::load_v<8>(v + {{ 2 * RES_PITCH }} + j), aie::load_v<8>(v + {{ 3 * RES_PITCH }} + j));
        const aie::vector<bfloat16, 8> w8 = aie::load_v<8>(wp + j);
        aie::vector<bfloat16, 32> ah, al;  // (a = v wp; res + a ivv: al::mulff's three products onto res)
        al::split2<32>(al::mulc<32>(vv, w8.template grow_replicate<32>()).template to_vector<float>(), ah, al);
        const aie::vector<float, 32> n = aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 32>(rv), ah, ivh), ah, ivl), al, ivh)
                                             .template to_vector<float>();
        const aie::vector<float, 32> rn = al::mulc<32>(n, sc).template to_vector<float>();
        aie::store_v(r + j, rn.template extract<8>(0));
        aie::store_v(r + {{ RES_PITCH }} + j, rn.template extract<8>(1));
        aie::store_v(r + {{ 2 * RES_PITCH }} + j, rn.template extract<8>(2));
        aie::store_v(r + {{ 3 * RES_PITCH }} + j, rn.template extract<8>(3));
    }
}

// r + a iv (fp32) for a = its bf16 hi + lo exactly (a product of two bf16), iv = iv_hi + iv_lo: three products onto r
[[clang::always_inline]] inline aie::vector<float, 32> {{ p }}_rvw(aie::vector<bfloat16, 32> r, aie::vector<float, 32> a,
                                                                  aie::vector<bfloat16, 32> iv_hi, aie::vector<bfloat16, 32> iv_lo)
{
    const aie::vector<bfloat16, 32> a_hi = aie::accum<accfloat, 32>(a).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> a_lo = aie::accum<accfloat, 32>(
        aie::sub(a, aie::accum<accfloat, 32>(a_hi).template to_vector<float>())).template to_vector<bfloat16>();
    return aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 32>(r), a_hi, iv_hi), a_hi, iv_lo), a_lo, iv_hi).template to_vector<float>();
}

// slot 0's 8 values of a k block in slot 1's lanes, the others zero (an F_KPAIR header's mirrored strip: its slot 1 is zero
// in DDR)
template <typename T>
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_kbm(const T *x)
{
    return aie::concat(aie::shuffle_up_rotate(aie::load_v<16>(x), 8), aie::zeros<bfloat16, 16>());  // ([s0 | 0] -> [0 | s0])
}

// hpstrip's first loop: per 16-k block n = r + v ivv wp of its two k blocks ([k block][v | r][slot SP][8] at x, wp per
// k) -> nb [hi | lo] of each ([slot 4][8] bf16), sq += bf16(n)^2 (MIR: slot 0's values in slot 1's lanes)
template <int SP, bool MIR>
[[clang::always_inline]] inline void {{ p }}_hpn(const bfloat16 __aie_dm_resource_a *__restrict x,
                                                 const bfloat16 __aie_dm_resource_b *__restrict wp,
                                                 aie::vector<bfloat16, 32> iv_hi, aie::vector<bfloat16, 32> iv_lo,
                                                 aie::accum<accfloat, 32> &sq, bfloat16 *__restrict nb)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < 8; j++) {
        aie::vector<bfloat16, 32> v0, r0, v1, r1;
        if constexpr (MIR) {
            v0 = {{ p }}_kbm(x), r0 = {{ p }}_kbm(x + 8 * SP), v1 = {{ p }}_kbm(x + 16 * SP), r1 = {{ p }}_kbm(x + 24 * SP);
        } else {
            v0 = {{ p }}_kb<SP>(x), r0 = {{ p }}_kb<SP>(x + 8 * SP), v1 = {{ p }}_kb<SP>(x + 16 * SP), r1 = {{ p }}_kb<SP>(x + 24 * SP);
        }
        x += 32 * SP;
        const v16int32 p2 = (v16int32)(v32bfloat16)aie::load_v<16>(wp).template grow<32>();
        wp += 16;
        const aie::vector<bfloat16, 32> wp0((v32bfloat16)broadcast_elem_128(p2, 0)), wp1((v32bfloat16)broadcast_elem_128(p2, 1));
        const aie::vector<float, 32> n0 = {{ p }}_rvw(r0, aie::mul(v0, wp0).template to_vector<float>(), iv_hi, iv_lo);
        const aie::vector<float, 32> n1 = {{ p }}_rvw(r1, aie::mul(v1, wp1).template to_vector<float>(), iv_hi, iv_lo);
        const aie::vector<bfloat16, 32> h0 = aie::accum<accfloat, 32>(n0).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> h1 = aie::accum<accfloat, 32>(n1).template to_vector<bfloat16>();
        sq = aie::mac_square(aie::mac_square(sq, h0), h1);
        aie::store_v(nb, h0);
        aie::store_v(nb + 32, aie::accum<accfloat, 32>(aie::sub(n0, aie::accum<accfloat, 32>(h0).template to_vector<float>())).template to_vector<bfloat16>());
        aie::store_v(nb + 64, h1);
        aie::store_v(nb + 96, aie::accum<accfloat, 32>(aie::sub(n1, aie::accum<accfloat, 32>(h1).template to_vector<float>())).template to_vector<bfloat16>());
        nb += 128;
    }
}

// hpstrip's second loop: x = (n_hi + n_lo) w1 (nb, w per k) -> the x panels at xp and the 16-k sums at t (as _hstrip)
[[clang::always_inline]] inline void {{ p }}_hpx(const bfloat16 __aie_dm_resource_c *__restrict nb,
                                                 const bfloat16 __aie_dm_resource_b *__restrict w,
                                                 uint8_t *__restrict xp, float *__restrict t)
{
    const al::BV nI = al::neg_eye();
    al::BOut<> o((bfp16ebs8 *)xp);
    v64accfloat T = (v64accfloat)aie::zeros<accfloat, 64>();
    al::BInR<aie_dm_resource::d> bs((const bfp16ebs8 *){{ p }}_ohp);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < 8; j++) {
        const aie::vector<bfloat16, 32> h0 = aie::load_v<32>(nb), l0 = aie::load_v<32>(nb + 32);
        const aie::vector<bfloat16, 32> h1 = aie::load_v<32>(nb + 64), l1 = aie::load_v<32>(nb + 96);
        nb += 128;
        const v16int32 w2 = (v16int32)(v32bfloat16)aie::load_v<16>(w).template grow<32>();
        w += 16;
        const aie::vector<bfloat16, 32> w0((v32bfloat16)broadcast_elem_128(w2, 0)), w1((v32bfloat16)broadcast_elem_128(w2, 1));
        const aie::accum<accfloat, 32> x0 = aie::mac(aie::mul(h0, w0), l0, w0);
        const aie::accum<accfloat, 32> x1 = aie::mac(aie::mul(h1, w1), l1, w1);
        const aie::accum<accfloat, 64> a = aie::concat(x0, x1);
        al::MM8 l(a);
        l.mac(a.template to_vector<bfp16ebs8>(), aie::op_transpose(nI));  // a - hi exactly
        const aie::accum<accfloat, 64> lo = l.to_accum();
        const al::BV q0 = aie::concat(x0, lo.template extract<32>(0)).template to_vector<bfp16ebs8>();
        const al::BV q1 = aie::concat(x1, lo.template extract<32>(1)).template to_vector<bfp16ebs8>();
        o.push(q0);
        o.push(q1);
        const v64bfp16ebs8 b = (v64bfp16ebs8)bs.pop();
        T = mac_8x8_8x8T((v64bfp16ebs8)q0, 1, b, 0, T);
        T = mac_8x8_8x8T((v64bfp16ebs8)q1, 1, b, 0, T);
    }
    const aie::accum<accfloat, 64> ts(T);  // rows: the slots' hi sums, then their lo sums
    aie::store_v(t, aie::add(ts.template extract<32>(0), ts.template extract<32>(1)).template to_vector<float>());
}

// The helper, a header phase's input strip: {{ POST_K // 8 }} k blocks of [v | r] ([k block][v | r][slot SP][8] bf16) with
// the norm object's w1 and w_post at w, wp (bf16 per k) and the slots' 1 / rms(v) at ivv ([slot][8] fp32): x = (r + v
// ivv wp) w1 -> the x panels and 16-k sums as _hstrip; sq_ += bf16(r + v ivv wp)^2. Per 128-k group two loops through
// nb (per 16-k block [hi | lo] of its two k blocks; as one loop: II 72, the register file): _hpn, then _hpx, each with
// its buffers as distinct restrict parameters. MIR (F_KPAIR): slot 0's values and 1 / rms(v) as slot 1's
template <int SP, bool MIR = false>
__attribute__((noinline)) void {{ p }}_hpstrip(const bfloat16 *__restrict x, const bfloat16 *__restrict w,
                                               const bfloat16 *__restrict wp, const float *__restrict ivv,
                                               float *__restrict sq_, uint8_t *__restrict xp, float *__restrict t,
                                               bfloat16 *__restrict nb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 32> sq(aie::load_v<32>(sq_));
    aie::vector<float, 32> iv = aie::load_v<32>(ivv);
    if constexpr (MIR)
        iv = aie::concat(iv.template extract<8>(0), iv.template extract<8>(0), iv.template extract<8>(0), iv.template extract<8>(0));
    const aie::vector<bfloat16, 32> iv_hi = aie::accum<accfloat, 32>(iv).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> iv_lo = aie::accum<accfloat, 32>(
        aie::sub(iv, aie::accum<accfloat, 32>(iv_hi).template to_vector<float>())).template to_vector<bfloat16>();
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < {{ POST_K // 128 }}; g++) {
        {{ p }}_hpn<SP, MIR>((const bfloat16 __aie_dm_resource_a *)(x + 256 * SP * g), (const bfloat16 __aie_dm_resource_b *)(wp + 128 * g),
                        iv_hi, iv_lo, sq, nb);
        {{ p }}_hpx((const bfloat16 __aie_dm_resource_c *)nb, (const bfloat16 __aie_dm_resource_b *)(w + 128 * g),
                    xp + {{ 2 * 8 * 72 }} * g, t + 32 * g);
    }
    aie::store_v(sq_, sq.template to_vector<float>());
}

{% endif %}
{% set lut = km.act(km.SILU if act == "silu" else km.GELU) %}
{{ lut.src }}

// A complete row tile of every slot (yt [g {{ YT // 32 }}][slot 4][r 8], rows 8 g + r): F_NORM: times the slots' 1 / rms{{ " (moe: the nf floats of the tile's row blocks)" if moe else "" }}; F_GU
// ({{ "per 16 rows: " if rowsplit else "" }}g 0 gate, g 1 up): {{ act }}(bf16(gate)) up into g 0 (kmath.act: the gather LUT, through gs){{ "; F_PLI: gelu(bf16(y)) pli" if pli else "" }}
__attribute__((noinline)) void {{ p }}_rows(int32_t *__restrict ws, int flags{{ ", int ng" if act == "silu" and wide else "" }}{{ ", int nf" if moe else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict yt = WS(float, {{ w_.yt }});
{% if moe %}
    if (flags & {{ F_NORM + F_WS }}) {  // (F_WS: the slots' scales)
        const aie::vector<float, 32> iv = aie::load_v<32>(WS(float, (flags & {{ F_NORM }}) ? {{ w_.inv }} : {{ w_.wsc }}));
{% else %}
    if (flags & {{ F_NORM }}) {
{% if SLOTS == 8 %}
        const aie::vector<float, 32> iv0 = aie::load_v<32>(WS(float, {{ w_.inv }})), iv1 = aie::load_v<32>(WS(float, {{ w_.inv }}) + 32);
        AIE_LOOP_NO_UNROLL
        for (int g = 0; g < {{ YT }}; g += 64) {  // ([g][slot 8][r 8]: slots 0 .. 3, then 4 .. 7)
            aie::store_v(yt + g, al::mulff<32>(aie::load_v<32>(yt + g), iv0));
            aie::store_v(yt + g + 32, al::mulff<32>(aie::load_v<32>(yt + g + 32), iv1));
        }
    }
{% else %}
        const aie::vector<float, 32> iv = aie::load_v<32>(WS(float, {{ w_.inv }}));
{% endif %}
{% endif %}
{% if SLOTS == 4 %}
        AIE_LOOP_NO_UNROLL
        for (int g = 0; g < {{ "nf" if moe else YT }}; g += 32)
            aie::store_v(yt + g, al::mulff<32>(aie::load_v<32>(yt + g), iv));
    }
{% endif %}
{% if act == "silu" and wide %}
    if (flags & {{ F_GU }}) {
        bfloat16 *__restrict gs = WS(bfloat16, {{ w_.gs }});
        AIE_LOOP_NO_UNROLL
        for (int h = 0; h < 64 * ng; h += 64) {  // (the ng windows of [8 gate | 8 up] rows completed)
            aie::store_v(gs, aie::accum<accfloat, 32>(aie::load_v<32>(yt + h)).template to_vector<bfloat16>());
            {{ lut }}<32>(gs, gs + 32);
            aie::store_v(yt + h, al::mulc<32>(aie::load_v<32>(yt + h + 32), aie::load_v<32>(gs + 32)).template to_vector<float>());
        }
    }
{% elif act == "silu" and SLOTS == 8 %}
    if (flags & {{ F_GU }}) {  // (gate: yt's g 0 [slot 8][r 8], up: g 1)
        bfloat16 *__restrict gs = WS(bfloat16, {{ w_.gs }});
        aie::store_v(gs, aie::accum<accfloat, 32>(aie::load_v<32>(yt)).template to_vector<bfloat16>());
        aie::store_v(gs + 32, aie::accum<accfloat, 32>(aie::load_v<32>(yt + 32)).template to_vector<bfloat16>());
        {{ lut }}<64>(gs, gs + 64);
        aie::store_v(yt, al::mulc<32>(aie::load_v<32>(yt + 64), aie::load_v<32>(gs + 64)).template to_vector<float>());
        aie::store_v(yt + 32, al::mulc<32>(aie::load_v<32>(yt + 96), aie::load_v<32>(gs + 96)).template to_vector<float>());
    }
{% elif act == "silu" %}
    if (flags & {{ F_GU }}) {
        bfloat16 *__restrict gs = WS(bfloat16, {{ w_.gs }});
        aie::store_v(gs, aie::accum<accfloat, 32>(aie::load_v<32>(yt)).template to_vector<bfloat16>());
        {{ lut }}<32>(gs, gs + 32);
        aie::store_v(yt, al::mulc<32>(aie::load_v<32>(yt + 32), aie::load_v<32>(gs + 32)).template to_vector<float>());
    }
{% else %}
    bfloat16 *__restrict gs = WS(bfloat16, {{ w_.gs }});
    if (flags & {{ F_GU }}) {
        AIE_LOOP_NO_UNROLL
        for (int h = 0; h < {{ YT }}; h += 64) {
            aie::store_v(gs, aie::accum<accfloat, 32>(aie::load_v<32>(yt + h)).template to_vector<bfloat16>());
            {{ lut }}<32>(gs, gs + 32);
            aie::store_v(yt + h, al::mulc<32>(aie::load_v<32>(yt + h + 32), aie::load_v<32>(gs + 32)).template to_vector<float>());
        }
    }
{% endif %}
{% if pli %}
    if (flags & {{ F_PLI }}) {  // (pli [g][slot][r] bf16)
        const bfloat16 *__restrict pl = WS(bfloat16, {{ w_.pli }});
        AIE_LOOP_NO_UNROLL
        for (int g = 0; g < {{ YT }}; g += 32) {
            aie::store_v(gs, aie::accum<accfloat, 32>(aie::load_v<32>(yt + g)).template to_vector<bfloat16>());
            {{ lut }}<32>(gs, gs + 32);
            aie::store_v(yt + g, aie::mul(aie::load_v<32>(gs + 32), aie::load_v<32>(pl + g)).template to_vector<float>());
        }
    }
{% endif %}
}

{% if SLOTS == 8 %}
// Slot s's residual slice (fp32): slots 0 .. 3 in the workspace, 4 .. 7 in hres (the helper's memory)
[[clang::always_inline]] inline float *{{ p }}_res(int32_t *ws, float *hres, int s)
{
    return s < {{ RES_SLOTS }} ? WS(float, {{ w_.res }}) + {{ RES_PITCH }} * s : hres + {{ RES_PITCH }} * (s - {{ RES_SLOTS }});
}

{% endif %}
// Slot s's rows 0 .. nv - 1 of a complete row tile (after _rows): into the residual at row q, or the staging buffer at
// n; returns the values staged{{ " (h: the tile's rows 16 h ..)" if wide and not moe else "" }}{{ " (h: the tile's block of 16 rows h; moe: a B_PAIR block's pairs' sums, B_KEEP its first one kept; F_HOUT into the hold buffer, F_ACCUM into acc)" if moe else "" }}
__attribute__((noinline)) int {{ p }}_slot(int32_t *__restrict ws, int s, int flags, int fw, int nv, int q, int n{{ ", int h" if wide or moe else "" }}{{ ", bfloat16 *__restrict acc" if moe else "" }}{{ ", float *__restrict hres" if SLOTS == 8 else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict yt = WS(float, {{ w_.yt }}){{ " + 64 * h" if wide or moe else "" }};
    aie::vector<float, 16> y = aie::concat(aie::load_v<8>(yt + 8 * s), aie::load_v<8>(yt + {{ 8 * SLOTS }} + 8 * s));  // rows 0 .. 15
{% if moe %}
    if (fw & {{ B_PAIR }}) {  // (rows [hi, lo] of nv / 2 values: their sums, fp32 as two bf16 slots each)
        const aie::vector<float, 8> pv = aie::add(aie::filter_even(y, 1), aie::filter_odd(y, 1));
        if (fw & {{ B_KEEP }}) {
            WS(float, {{ w_.st }})[{{ S_SG }} + s] = pv[0];
            return 0;
        }
        {{ p }}_put_bf16(WS(bfloat16, {{ w_.stg }}) + {{ 2 * C_LEN }} * s + n, pv.template cast_to<bfloat16>(), nv);
        return nv;
    }
    if (flags & {{ F_HOUT }}) {  // (a gate / up tile's 8 outputs: k block n / 8 of the hold buffer's [h | h])
        bfloat16 *__restrict ph = WS(bfloat16, {{ w_.hold }}) + 32 * (n >> 3) + 8 * s;
        const aie::vector<bfloat16, 8> o = aie::accum<accfloat, 16>(y).template to_vector<bfloat16>().template extract<8>(0);
        aie::store_v(ph, o);
        aie::store_v(ph + {{ 32 * (OBJ_K // 16) }}, o);
        return nv;
    }
{% if not wide %}
    if (flags & {{ F_ACCUM }}) {  // (into acc [slot][H] bf16 at row q: a window of 32 at its 16-aligned base, in two
        // 32-byte halves (a 64-byte access is 64-byte aligned); F_ACC0 stores)
        bfloat16 *__restrict pa = acc + {{ H }} * s + (q & ~15);
        const int sh = q & 15;
        aie::vector<float, 32> v = aie::zeros<float, 32>();
        v.insert(0, y);
        v = aie::shuffle_up_fill(v, aie::zeros<float, 32>(), sh);
        const aie::vector<bfloat16, 32> old = aie::concat(aie::load_v<16>(pa), aie::load_v<16>(pa + 16));
        const aie::accum<accfloat, 32> sum = (flags & {{ F_ACC0 }}) ? aie::accum<accfloat, 32>(v)
                                                                   : aie::add(aie::accum<accfloat, 32>(old), v);
        const aie::vector<int16, 32> li = aie::load_v<32>({{ p }}_iota);
        const aie::vector<bfloat16, 32> nw = aie::select(old, sum.template to_vector<bfloat16>(),
                                                         aie::ge(li, (int16)sh) & aie::lt(li, (int16)(sh + nv)));
        aie::store_v(pa, nw.template extract<16>(0));
        aie::store_v(pa + 16, nw.template extract<16>(1));
        return 0;
    }
{% endif %}
{% endif %}
{% if post %}
    if (flags & {{ F_RESID + F_VOUT }}) {  // (a window of 32 floats at the 16-aligned base; F_VOUT: into pend)
        float *__restrict pr = WS(float, {{ w_.res }}) + ((flags & {{ F_VOUT }}) ? {{ w_.pend - w_.res }} / 4 : 0) + {{ RES_PITCH }} * s + (q & ~15);
{% else %}
    if (flags & {{ F_RESID }}) {  // (a window of 32 floats at the 16-aligned base)
{% if SLOTS == 8 %}
        float *__restrict pr = {{ p }}_res(ws, hres, s) + (q & ~15);
{% else %}
        float *__restrict pr = WS(float, {{ w_.res }}) + {{ RES_PITCH }} * s + (q & ~15);
{% endif %}
{% endif %}
        y = aie::select(y, aie::zeros<float, 16>(), aie::mask<16>::from_uint32(0xFFFFu << nv));
        aie::vector<float, 32> v = aie::zeros<float, 32>();
        v.insert(0, y);
        v = aie::shuffle_up_fill(v, aie::zeros<float, 32>(), q & 15);
        aie::store_v(pr, aie::add(aie::load_v<32>(pr), v));
        return 0;
    }
    bfloat16 *__restrict ps = WS(bfloat16, {{ w_.stg }}) + {{ 2 * C_LEN }} * s + n;
    if (fw & {{ B_F32 }}) {  // fp32 rows: two bf16 slots each
        {{ p }}_put_bf16(ps, y.template extract<8>(0).template cast_to<bfloat16>(), nv >= 8 ? 16 : 2 * nv);
        if (nv > 8)
            {{ p }}_put_bf16(ps + 16, y.template extract<8>(1).template cast_to<bfloat16>(), 2 * (nv - 8));
        return 2 * nv;
    }
    {{ p }}_put_bf16(ps, aie::accum<accfloat, 16>(y).template to_vector<bfloat16>(), nv);
    return nv;
}
{% if SLOTS == 8 %}

// _slot of the slots s < S in one call (their rows 2 C_LEN apart in the staging, their residual windows at the same
// row: one shift and mask for all)
__attribute__((noinline)) int {{ p }}_slots(int32_t *__restrict ws, int S, int flags, int fw, int nv, int q, int n,
                                            float *__restrict hres)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict yt = WS(float, {{ w_.yt }});
    if (flags & {{ F_RESID }}) {  // (a window of 32 floats at the 16-aligned base)
        const int b = q & ~15, sh = q & 15;
        const aie::mask<16> m = aie::mask<16>::from_uint32(0xFFFFu << nv);
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < S; s++) {
            aie::vector<float, 16> y = aie::concat(aie::load_v<8>(yt + 8 * s), aie::load_v<8>(yt + 64 + 8 * s));
            y = aie::select(y, aie::zeros<float, 16>(), m);
            aie::vector<float, 32> v = aie::zeros<float, 32>();
            v.insert(0, y);
            v = aie::shuffle_up_fill(v, aie::zeros<float, 32>(), sh);
            float *__restrict pr = {{ p }}_res(ws, hres, s) + b;
            aie::store_v(pr, aie::add(aie::load_v<32>(pr), v));
        }
        return 0;
    }
    bfloat16 *__restrict stg = WS(bfloat16, {{ w_.stg }});
    if (fw & {{ B_F32 }}) {  // fp32 rows: two bf16 slots each
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < S; s++)
            {{ p }}_slot(ws, s, flags, fw, nv, q, n, hres);
        return 2 * nv;
    }
    bfloat16 *__restrict pw = (bfloat16 *)((uintptr_t)(stg + n) & ~(uintptr_t)63);  // (put_bf16's window)
    const int sh = (int)((stg + n) - pw);
    const aie::vector<int16, 64> li = aie::load_v<64>({{ p }}_iota);
    const aie::mask<64> m = aie::ge(li, (int16)sh) & aie::lt(li, (int16)(sh + nv));
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const aie::vector<float, 16> y = aie::concat(aie::load_v<8>(yt + 8 * s), aie::load_v<8>(yt + 64 + 8 * s));
        aie::vector<bfloat16, 64> v = aie::zeros<bfloat16, 64>();
        v.insert(0, aie::accum<accfloat, 16>(y).template to_vector<bfloat16>());
        v = aie::shuffle_up_fill(v, aie::zeros<bfloat16, 64>(), sh);
        bfloat16 *__restrict pd = pw + {{ 2 * C_LEN }} * s;
        aie::store_v(pd, aie::select(aie::load_v<64>(pd), v, m));
    }
    return nv;
}
{% endif %}
{% if moe and wide %}

// (an expert's down, F_ACCUM) rows 0 .. nv - 1 of a tile of G row groups of R <= 32 rows (group j's in yt's block j) of
// every slot into acc at row q ([slot][H] bf16; F_ACC0 stores): _slot's sums, a group at a time (its 32 rows shifted
// onto a window of 48 at its 16-aligned base: lanes 0 .. 31, then 32 .. 47 where its rows reach them)
__attribute__((noinline)) void {{ p }}_accum(int32_t *__restrict ws, int S, int flags, int nv, int q, int R, int G,
                                            bfloat16 *__restrict acc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict yt = WS(float, {{ w_.yt }});
    const aie::vector<int16, 32> li = aie::load_v<32>({{ p }}_iota);
    const bool st0 = flags & {{ F_ACC0 }};
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        AIE_LOOP_NO_UNROLL
        for (int j = 0; j < G; j++) {
            const int k = nv - R * j > R ? R : nv - R * j, qj = q + R * j, sh = qj & 15;
            if (k <= 0)
                break;
            const float *__restrict yb = yt + {{ YB }} * j + 8 * s;
            const aie::vector<float, 32> y = aie::concat(aie::load_v<8>(yb), aie::load_v<8>(yb + 32), aie::load_v<8>(yb + 64),
                                                         aie::load_v<8>(yb + 96));  // rows 0 .. 31
            bfloat16 *__restrict pa = acc + {{ H }} * s + (qj & ~15);
            const aie::vector<float, 32> v0 = aie::shuffle_up_fill(y, aie::zeros<float, 32>(), sh);
            const aie::vector<bfloat16, 32> o0 = aie::concat(aie::load_v<16>(pa), aie::load_v<16>(pa + 16));
            const aie::accum<accfloat, 32> s0 = st0 ? aie::accum<accfloat, 32>(v0) : aie::add(aie::accum<accfloat, 32>(o0), v0);
            const aie::vector<bfloat16, 32> n0 = aie::select(o0, s0.template to_vector<bfloat16>(),
                                                             aie::ge(li, (int16)sh) & aie::lt(li, (int16)(sh + k)));
            aie::store_v(pa, n0.template extract<16>(0));
            aie::store_v(pa + 16, n0.template extract<16>(1));
            if (sh + k > 32) {  // (rows 32 - sh .. k - 1)
                const aie::vector<float, 16> v1 = aie::shuffle_down_fill(y, aie::zeros<float, 32>(), 32 - sh).template extract<16>(0);
                const aie::vector<bfloat16, 16> o1 = aie::load_v<16>(pa + 32);
                const aie::accum<accfloat, 16> s1 = st0 ? aie::accum<accfloat, 16>(v1) : aie::add(aie::accum<accfloat, 16>(o1), v1);
                aie::store_v(pa + 32, aie::select(o1, s1.template to_vector<bfloat16>(),
                                                  aie::lt(li.template extract<16>(0), (int16)(sh + k - 32))));
            }
        }
    }
}
{% endif %}
{% if gu_win and wide %}

// F_GU tiles of R != 16 rows (gu_win: their [8 gate | 8 up] row groups cross tiles; R <= 32): the tile's R rows of each
// slot onto the slot's window (gw [slot][64] fp32, st[S_GN] rows); per 16 rows it holds, the window's 16 w .. 16 w + 15
// as [8 gate | 8 up] into yt's rows 16 w .. (then act(gate) up: ``rows``), and the window moves down by 16 per window.
// A SiLU group's 8th window (its 128 rows: the rest of its last tile is padding, ``decode.image.gate_up_rows``) empties
// it. Returns the windows completed: 0, 1 or 2 (8 outputs each).
__attribute__((noinline)) int {{ p }}_guwin(int32_t *__restrict ws, int S, int R)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    float *__restrict yt = WS(float, {{ w_.yt }});
    float *__restrict gw = WS(float, {{ w_.gw }});
    const int gn = st[{{ S_GN }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {  // (the window's halves: 32 rows each; shifts of 1024-bit vectors)
        aie::vector<float, 32> y = aie::concat(aie::load_v<8>(yt + 8 * s), aie::load_v<8>(yt + 32 + 8 * s),
                                               aie::load_v<8>(yt + 64 + 8 * s), aie::load_v<8>(yt + 96 + 8 * s));
        y = aie::select(y, aie::zeros<float, 32>(), aie::mask<32>::from_uint32(R >= 32 ? 0u : 0xFFFFFFFFu << R));
        aie::store_v(gw + 64 * s, aie::add(aie::load_v<32>(gw + 64 * s), aie::shuffle_up_fill(y, aie::zeros<float, 32>(), gn)));
        if (gn)
            aie::store_v(gw + 64 * s + 32, aie::add(aie::load_v<32>(gw + 64 * s + 32),
                                                    aie::shuffle_down_fill(y, aie::zeros<float, 32>(), 32 - gn)));
    }
    const int k = st[{{ S_GK }}], have = (gn + R) >> 4;
    const int nw = have < 8 - k ? have : 8 - k, last = k + nw == 8;  // (the SiLU group's 8th: its window empties)
    if (nw == 0) {
        st[{{ S_GN }}] = gn + R;
        return 0;
    }
    st[{{ S_GK }}] = last ? 0 : k + nw;
    st[{{ S_GN }}] = last ? 0 : gn + R - 16 * nw;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {  // (the other slots' rows go nowhere)
        const aie::vector<float, 32> w0 = aie::load_v<32>(gw + 64 * s), w1 = aie::load_v<32>(gw + 64 * s + 32);
        aie::store_v(yt + 8 * s, w0.template extract<8>(0));
        aie::store_v(yt + 32 + 8 * s, w0.template extract<8>(1));
        aie::store_v(yt + 64 + 8 * s, w0.template extract<8>(2));
        aie::store_v(yt + 96 + 8 * s, w0.template extract<8>(3));
        const aie::vector<float, 32> n0 = nw == 1 ? aie::concat(w0.template extract<16>(1), w1.template extract<16>(0)) : w1;
        const aie::vector<float, 32> n1 = nw == 1 ? aie::concat(w1.template extract<16>(1), aie::zeros<float, 16>()) : aie::zeros<float, 32>();
        aie::store_v(gw + 64 * s, last ? aie::zeros<float, 32>() : n0);
        aie::store_v(gw + 64 * s + 32, last ? aie::zeros<float, 32>() : n1);
    }
    {{ p }}_rows(ws, {{ F_GU }}, nw{{ ", 0" if moe else "" }});
    return nw;
}
{% elif gu_win %}

// F_GU tiles of R < 16 rows (gu_win: their [8 gate | 8 up] row groups cross tiles): the tile's R rows of each slot onto
// the slot's window (gw [slot][32] fp32, st[S_GN] rows); once it holds 16, [8 gate | 8 up] into yt as a 16-row tile's
// (then act(gate) up: ``rows``) and the window moves down by 16. A SiLU group's 8th (its 128 rows: the rest of its last
// tile is padding, ``decode.image.gate_up_rows``) empties it. Returns the rows to stage: 8, or 0.
__attribute__((noinline)) int {{ p }}_guwin(int32_t *__restrict ws, int S, int R)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    float *__restrict yt = WS(float, {{ w_.yt }});
    float *__restrict gw = WS(float, {{ w_.gw }});
    const int gn = st[{{ S_GN }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        aie::vector<float, 16> y = aie::concat(aie::load_v<8>(yt + 8 * s), aie::load_v<8>(yt + 32 + 8 * s));  // rows 0 .. 15
        y = aie::select(y, aie::zeros<float, 16>(), aie::mask<16>::from_uint32(0xFFFFu << R));
        aie::vector<float, 32> v = aie::zeros<float, 32>();
        v.insert(0, y);
        v = aie::shuffle_up_fill(v, aie::zeros<float, 32>(), gn);
        aie::store_v(gw + 32 * s, aie::add(aie::load_v<32>(gw + 32 * s), v));
    }
    if (gn + R < 16) {
        st[{{ S_GN }}] = gn + R;
        return 0;
    }
    const int k = st[{{ S_GK }}] + 1, last = k == 8;  // (the SiLU group's 8th: its window empties)
    st[{{ S_GK }}] = last ? 0 : k;
    st[{{ S_GN }}] = last ? 0 : gn + R - 16;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {  // (the other slots' rows go nowhere)
        const aie::vector<float, 32> w = aie::load_v<32>(gw + 32 * s);
        aie::store_v(yt + 8 * s, w.template extract<8>(0));
        aie::store_v(yt + 32 + 8 * s, w.template extract<8>(1));
        aie::store_v(gw + 32 * s, last ? aie::zeros<float, 32>() : aie::concat(w.template extract<16>(1), aie::zeros<float, 16>()));
    }
    {{ p }}_rows(ws, {{ F_GU }}{{ ", 0" if moe else "" }});
    return 8;
}
{% endif %}

}  // namespace

extern "C" {

// A new X object: a slot's residual slice, another core's object, or the phase's metadata; the helper's command cmd
// (K_BEGIN: its ``hbegin`` copies the residual slice, or takes a phase's metadata: the trip counts and the phase's
// state; else K_NOP)
__attribute__((minsize)) void {{ p }}_begin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t *__restrict cmd,
                                           int32_t col)
{
{% if paged %}
    col -= {{ COLS }};  // (paged designs pass COLS + the column: ``decode``)
{% endif %}
{% if WIN == 8 %}
    event1();
{% endif %}
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int flags = m[{{ M_FLAGS }}];
{% if WIN == 9 %}
    if ((flags & {{ F_MAIN }}) && !(flags & {{ F_LOADR }}))
        event1();
{% endif %}
    st[{{ S_FLAGS }}] = flags;
    for (int i = 0; i < {{ S_FLAGS }}; i++)
        st[i] = 0;
{% if moe %}
    st[{{ S_NSUB }}] = 0;
{% endif %}
{% if post %}
    st[{{ S_NPS }}] = 0;
{% endif %}
    cmd[{{ C_KIND }}] = {{ K_NOP }};
    if (!(flags & {{ F_MAIN }})) {  // another core's: skip it, the X and A objects that follow it; relay its outputs
        st[{{ S_NSKIP }}] = m[{{ M_SKIP }}];
        st[{{ S_NASKIP }}] = m[{{ M_NASKIP }}];
        st[{{ S_NR }}] = m[{{ M_NREL }}];
{% if gate %}
        {{ p }}_gbegin(m, st, col);  // (the model's: the gates its relay applies)
{% endif %}
{% if help_flag %}
        if (flags & {{ help_flag }}) {  // (the model core's work the helper shares)
            cmd[1] = m[{{ help_word }}];
            cmd[{{ C_KIND }}] = {{ K_HELP }};
        }
{% endif %}
{% if tok %}
        if (flags & {{ F_TOK }})  // a layer kind's token object: the column's cache objects of its attentions
            st[{{ S_NAS }} + m[{{ M_TOK_KIND }}]] = m[{{ M_TOK_NG }} + col];
        else if (m[{{ M_NASKIP }}] < 0)
            st[{{ S_NASKIP }}] = st[{{ S_NAS }} - 1 - m[{{ M_NASKIP }}]];
{% endif %}
        return;
    }
    cmd[0] = (int32_t)(uintptr_t)x + ({{ SHM }});
    cmd[1] = (int32_t)(uintptr_t)ws + ({{ SHM }});
    cmd[2] = col;
    cmd[{{ C_KIND }}] = {{ K_BEGIN }};
    if (flags & {{ F_LOADR }})  // (a slot's residual slice: hbegin copies it)
        return;
{% if WIN in (5, 6, 7) %}
{% if WIN == 6 %}
    if (flags & {{ F_POST }}) {
{% elif WIN == 7 %}
    if (!(flags & {{ F_POST }})) {
{% else %}
    {
{% endif %}
        st[63] = 1;  // (tile: the window's end)
        event0();
    }
{% endif %}
}

{% if post %}

// A header phase's partial sums object (the 8 columns' C objects of v's sums of squares: [column][slot 4][8 lanes]
// fp32): ivv = the slots' 1 / rms(v) (8 lanes per slot)
__attribute__((minsize)) void {{ p }}_psums(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float ih = __builtin_bit_cast(float, WS(int32_t, {{ w_.st }})[{{ S_IH }}]);
    const float *__restrict ps = (const float *)(x + {{ 2 * META }});
    aie::vector<float, 32> t = aie::load_v<32>(ps);
    AIE_LOOP_NO_UNROLL
    for (int c = 1; c < {{ COLS }}; c++)
        t = aie::add(t, aie::load_v<32>(ps + 32 * c));
    const aie::vector<float, 8> ss = al::group8<false>(aie::concat(t, aie::zeros<float, 32>()));
    aie::store_v(WS(float, {{ w_.ivv }}), al::rows4(al::inv_rms<{{ SLOTS }}>(ss.template grow<16>(), ih, {{ eps }}f)));
}
{% endif %}

// A norm-weight object: the input objects that use its weights
__attribute__((minsize)) void {{ p }}_nbegin(const uint8_t *__restrict a, int32_t *__restrict ws)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int left = st[{{ S_NX }}] - st[{{ S_XD }}], n = left > {{ NX_PER_NORM }} ? {{ NX_PER_NORM }} : left;
{% if packed %}
    const int per = st[{{ S_NXS }}];
{% if SLOTS == 8 %}
    st[{{ S_NXN }}] = (n + per - 1) >> ((per >> 1) - (per >> 3));  // (input objects of per = 1, 2, 4, 8 strips)
{% else %}
    st[{{ S_NXN }}] = (n + per - 1) >> (per >> 1);  // (input objects of per = 1, 2, 4 strips)
{% endif %}
{% else %}
    st[{{ S_NXN }}] = n;
{% endif %}
{% if post %}
    if (st[{{ S_FLAGS }}] & {{ F_POST }}) {  // [w1 | w_post] of {{ NX_PER_NORM * POST_K }} k: the column's res slice where they lie
        const int h8 = st[{{ S_H8 }}], k0 = st[{{ S_XD }}] * {{ POST_K }}, c0 = h8 * st[{{ S_COL }}];
        const int lo = c0 > k0 ? c0 : k0, hi = c0 + h8 < k0 + {{ NX_PER_NORM * POST_K }} ? c0 + h8 : k0 + {{ NX_PER_NORM * POST_K }};
        if (lo < hi)
            {{ p }}_pupd((const bfloat16 *)a + {{ NX_PER_NORM * POST_K }} + (lo - k0), ws, lo - c0, hi - lo);
    }
{% endif %}
{% if pli %}
    if ((st[{{ S_FLAGS }}] & {{ F_PLI }}) && st[{{ S_XD }}] == 0) {  // (the phase's first) the column's per-layer
        // inputs pl = bf16(p / rms(p) w + e), w its weights / sqrt 2 after [w1 | w_post] (bf16 hi [32] | lo [32])
        aie::set_rounding(aie::rounding_mode::conv_even);
        bfloat16 *__restrict pl = WS(bfloat16, {{ w_.pli }});
        const bfloat16 *__restrict ivb = WS(bfloat16, {{ w_.pli + 4 * 4 * SLOTS * 8 }});
        const aie::vector<bfloat16, 32> ih = aie::load_v<32>(ivb), il = aie::load_v<32>(ivb + 32);
        const bfloat16 *__restrict w = (const bfloat16 *)(a + {{ 4 * NX_PER_NORM * POST_K }});
        const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
        AIE_LOOP_NO_UNROLL
        for (int g = 0; g < 4; g++) {
            const aie::vector<bfloat16, 8> h8 = aie::load_v<8>(w + 8 * g), l8 = aie::load_v<8>(w + 32 + 8 * g);
            const aie::vector<bfloat16, 32> wh = aie::concat(h8, h8, h8, h8), wl = aie::concat(l8, l8, l8, l8);
            const aie::accum<accfloat, 32> c = aie::mac(aie::mac(aie::mul(ih, wh), ih, wl), il, wh);  // (1 / rms w)
            const aie::vector<bfloat16, 32> ch = c.template to_vector<bfloat16>();
            const aie::vector<bfloat16, 32> cl = aie::msc(c, ch, one).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 32> pv = aie::load_v<32>(pl + 32 * g);
            const aie::accum<accfloat, 32> y = aie::mac(aie::mac(aie::mul(aie::load_v<32>(pl + {{ 4 * SLOTS * 8 }} + 32 * g), one),
                                                                 pv, ch), pv, cl);
            aie::store_v(pl + 32 * g, y.template to_vector<bfloat16>());
        }
    }
{% endif %}
}

{% if packed %}
// Strip d's k blocks in its input object x: 2 OBJ_K SP bytes (plain: [k block][SP][8]; a header's: [k block][v | r][SP][8]
// of POST_K k) at (d mod strips per object) of them
[[clang::always_inline]] inline const bfloat16 *{{ p }}_strip_x(const bfloat16 *x, const int32_t *__restrict st, int d)
{
    return x + {{ 2 * META }} + {{ OBJ_K }} * st[{{ S_SP }}] * (d & (st[{{ S_NXS }}] - 1));
}

{% endif %}
// A strip d of the input object x under the norm-weight object w / without one (w = 0): the helper's command (x and w as
// the helper sees them: the main core's memory is its south neighbour's). (One copy for strip_n and strip)
__attribute__((noinline, minsize)) void {{ p }}_post(const bfloat16 *x, const uint8_t *w, int32_t *__restrict ws,
                                                  int32_t *__restrict cmd)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int d = st[{{ S_XD }}];
    st[{{ S_XD }}] = d + 1;
{% if packed %}
    if (d >= st[{{ S_NX }}]) {  // (the last object's strips past the input)
        cmd[{{ C_KIND }}] = {{ K_NOP }};
        return;
    }
    cmd[0] = (int32_t)(uintptr_t){{ p }}_strip_x(x, st, d) + ({{ SHM }});
    cmd[{{ C_SP }}] = st[{{ S_SP }}];
{% else %}
    cmd[0] = (int32_t)(uintptr_t)(x + {{ 2 * META }}) + ({{ SHM }});
{% endif %}
    cmd[1] = w ? (int32_t)(uintptr_t)(w + {{ 2 * OBJ_K }} * (d & {{ NX_PER_NORM - 1 }})) + ({{ SHM }}) : 0;
    cmd[2] = d;
    cmd[{{ C_KIND }}] = {{ K_STRIP }};
{% if moe %}
    cmd[3] = st[{{ S_XC0 }}];
    if (st[{{ S_FLAGS }}] & {{ F_COMB }}) {  // (a combine's object d: slot d's partials into the residual)
        cmd[1] = (int32_t)(uintptr_t)ws + ({{ SHM }});
        cmd[{{ C_KIND }}] = {{ K_ACC }};
    }
{% endif %}
}

__attribute__((minsize)) void {{ p }}_strip_n(const bfloat16 *__restrict x, const uint8_t *__restrict a, int32_t *__restrict ws,
                     int32_t *__restrict cmd)
{
{% if post %}
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    if (st[{{ S_FLAGS }}] & {{ F_POST }}) {  // a header's input strip: [v | r] under [w1 | w_post]
        const int d = st[{{ S_XD }}];
        st[{{ S_XD }}] = d + 1;
{% if packed %}
        if (d >= st[{{ S_NX }}]) {
            cmd[{{ C_KIND }}] = {{ K_NOP }};
            return;
        }
        const int mir = (st[{{ S_FLAGS }}] & {{ F_KPAIR }}) ? (d & 1) : 0;  // (strip 1 of a K pair: strip 0's values)
        cmd[0] = (int32_t)(uintptr_t){{ p }}_strip_x(x, st, d - mir) + ({{ SHM }});
        cmd[{{ C_SP }}] = st[{{ S_SP }}] + (mir ? {{ C_MIRROR }} : 0);
{% else %}
        cmd[0] = (int32_t)(uintptr_t)(x + {{ 2 * META }}) + ({{ SHM }});
{% endif %}
        cmd[1] = (int32_t)(uintptr_t)(a + {{ 2 * POST_K }} * (d & {{ NX_PER_NORM - 1 }})) + ({{ SHM }});
        cmd[2] = d;
        cmd[3] = (int32_t)(uintptr_t)WS(float, {{ w_.ivv }}) + ({{ SHM }});
        cmd[{{ C_KIND }}] = {{ K_PSTRIP }};
        return;
    }
{% endif %}
    {{ p }}_post(x, a, ws, cmd);
}

__attribute__((minsize)) void {{ p }}_strip(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t *__restrict cmd)
{
    {{ p }}_post(x, nullptr, ws, cmd);
}

// The input is complete: the helper's command for the sum panels (into xs) and the 1 / rms
__attribute__((minsize)) void {{ p }}_input(int32_t *__restrict ws, int32_t *__restrict cmd, uint8_t *__restrict xs)
{
    const int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    cmd[0] = (int32_t)(uintptr_t){{ "(xs + %d * st[%d])" % (gv.XS_CHUNK, S_XC0) if moe else "xs" }} + ({{ SHM }});
    cmd[1] = st[{{ S_FLAGS }}] & {{ F_NORM }};
    cmd[2] = st[{{ S_K8 }}];
{% if post %}
    cmd[3] = st[{{ S_IH }}];
{% endif %}
    cmd[{{ C_KIND }}] = {{ K_INPUT }};
}

// The helper's answer to the input command: the slots' 1 / rms{{ "; a header: the updated res slice goes out next" if post else "" }}
__attribute__((minsize)) void {{ p }}_inv(const float *__restrict r, int32_t *__restrict ws)
{
{% if moe %}
    if (WS(int32_t, {{ w_.st }})[{{ S_FLAGS }}] & {{ F_NORM }})  // (moe: the expert's gate / up reuses a phase's 1 / rms)
{% endif %}
    aie::store_v(WS(float, {{ w_.inv }}), aie::load_v<32>(r));
{% if SLOTS == 8 %}
    aie::store_v(WS(float, {{ w_.inv }}) + 32, aie::load_v<32>(r + 32));
{% endif %}
{% if post %}
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    if (st[{{ S_FLAGS }}] & {{ F_POST }}) {
        st[{{ S_ESRC }}] = 3;
        st[{{ S_EJ }}] = 0;
        st[{{ S_ESLOT }}] = 0;
        st[{{ S_NE }}] = st[{{ S_NS }}] * ((st[{{ S_H8 }}] + {{ C_LEN - 1 }}) >> {{ C_LEN.bit_length() - 1 }});
    }
{% endif %}
}

// A weight tile: the helper's command (its super-blocks: the tile and the sum panels as it sees them; released before
// our half: the halves run at once)
__attribute__((minsize)) void {{ p }}_cmd(uint8_t *__restrict a, int32_t *__restrict ws, int32_t *__restrict cmd, uint8_t *__restrict xs)
{
    const int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int kc = st[{{ S_KC }}];
    cmd[0] = (int32_t)(uintptr_t)a + ({{ SHM }});
    cmd[1] = kc;
    cmd[2] = (int32_t)(uintptr_t)(xs + {{ gv.XS_CHUNK }} * kc) + ({{ SHM }});
    cmd[3] = st[{{ S_FMT }}];
    cmd[{{ C_KIND }}] = {{ K_TILE }};
}

// Our half of the tile
void {{ p }}_tile(uint8_t *__restrict a, int32_t *__restrict ws, uint8_t *__restrict xp, uint8_t *__restrict xs)
{
    const int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int kc = st[{{ S_KC }}];
{% if WIN in (5, 6, 7) %}
    if (st[63]) {
        event1();
        ((int32_t *)st)[63] = 0;
    }
{% endif %}
    {{ p }}_Win<1> w_;
    {{ mp }}_mmv<0>(a, xp + {{ gv.XPHC if xh else gv.XP_CHUNK }} * kc, xs + {{ gv.XS_CHUNK }} * kc, WS(float, {{ w_.yt }}),
                    WS(uint8_t, {{ w_.scr }}), st[{{ S_FMT }}]);
}

// The phase's tiles are through: its last outputs, the residual (F_EMITR) or the staged values' last objects (the
// staging past them is zero)
__attribute__((noinline, minsize)) void {{ p }}_fin(int32_t *__restrict st)
{
{% if WIN in (8, 9) %}
    event0();
{% endif %}
    const int flags = st[{{ S_FLAGS }}], S = st[{{ S_NS }}];
    if (flags & {{ F_EMITR }}) {
{% if post %}
        const int v = (flags & {{ F_VOUT }}) ? 1 : 0;  // (v from pend, then its partial sums)
        st[{{ S_ESRC }}] = 1 + v;
        st[{{ S_EJ }}] = 0;
        st[{{ S_NE }}] = S * ((st[{{ S_H8 }}] + {{ C_LEN - 1 }}) >> {{ C_LEN.bit_length() - 1 }}) + v;
{% else %}
        st[{{ S_ESRC }}] = 1;
        st[{{ S_EJ }}] = 0;
        st[{{ S_NE }}] = S * {{ H8 // C_LEN }};
{% endif %}
        return;
    }
{% if moe %}
    if (flags & {{ F_EMITA }}) {  // (the last down: acc out, [64-row object][slot])
        st[{{ S_ESRC }}] = 4;
        st[{{ S_EJ }}] = 0;
        st[{{ S_NE }}] = S * {{ H // C_LEN }};
        return;
    }
    if (flags & {{ F_HOUT + F_ACCUM }})
        return;
{% endif %}
    const int t = st[{{ S_STG }}];
    st[(flags & {{ F_MSG }}) ? {{ S_NM }} : {{ S_NE }}] = t > {{ C_LEN }} ? 2 * S : (t > 0 ? S : 0);
}

// The helper's partial rows: a complete row tile's rows per slot -> the residual or the staging buffers (st[S_NE]
// C objects / st[S_NM] M objects to emit); after the phase's last tile its last outputs (fin)
void {{ p }}_collect(const float *__restrict r, int32_t *__restrict ws{{ ", bfloat16 *__restrict acc" if moe else "" }}{{ ", float *__restrict hres" if SLOTS == 8 else "" }})
{
    {{ p }}_Win<3> w_;
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    float *__restrict yt = WS(float, {{ w_.yt }});
{% if moe %}
    const int fm = st[{{ S_FMT }}];
    if (fm & {{ B_G4 }})  // (row groups: the helper's rows are rows of their own, after ours: its two groups' blocks)
        al::copy<{{ 2 * YB }}, 32>(yt + {{ 2 * YB }}, r);
    else if (fm & {{ B_G2 }})  // (its group's block)
{% if not wide %}
        for (int i = 0; i < {{ YB }}; i += 32)  // (as al::copy: one tail with the unrolled copy above, 6 cycles more)
            aie::store_v(yt + {{ YB }} + i, aie::load_v<32>(r + i));
{% else %}
        al::copy<{{ YB }}, 32>(yt + {{ YB }}, r);
{% endif %}
    else  // (its partial sums of our block)
        for (int i = 0; i < {{ YB }}; i += 32)
            aie::store_v(yt + i, aie::add(aie::load_v<32>(yt + i), aie::load_v<32>(r + i)));
{% elif rowsplit %}
    float *__restrict yh = yt + {{ p }}_hy(st[{{ S_FMT }}]);  // (a row-split tile: the helper's rows)
    for (int i = 0; i < 64; i += 32)
        aie::store_v(yh + i, aie::add(aie::load_v<32>(yh + i), aie::load_v<32>(r + i)));
{% else %}
    for (int i = 0; i < {{ YN }}; i += 32)
        aie::store_v(yt + i, aie::add(aie::load_v<32>(yt + i), aie::load_v<32>(r + i)));
{% endif %}
    st[{{ S_NE }}] = 0;
    st[{{ S_NM }}] = 0;
    const int tl = st[{{ S_TL }}] - 1;
    st[{{ S_TL }}] = tl;
    const int kc = st[{{ S_KC }}];
    if (kc + 1 < st[{{ S_G }}]) {
        st[{{ S_KC }}] = kc + 1;
        return;
    }
    st[{{ S_KC }}] = {{ "st[%d]" % S_XC0 if moe else "0" }};
    const int rt = st[{{ S_RT }}], nrt = rt + 1 < st[{{ S_NRT }}] ? rt + 1 : 0;
    st[{{ S_RT }}] = nrt;
    int rt0, row0;
    const int b = {{ p }}_block(st, rt, rt0, row0);
    const int fw = st[{{ S_B }} + 3 * b + 1], R = {{ p }}_rows(fw & 255);
    const int bn = b + 1 < st[{{ S_NB }}] && nrt >= st[{{ S_B }} + 3 * b] ? b + 1 : (nrt ? b : 0);  // (the next row tile's)
{% if moe %}
    st[{{ S_FMT }}] = st[{{ S_B }} + 3 * bn + 1] & {{ 255 + B_G2 + B_G4 }};
    const int G = (fw & {{ B_G4 }}) ? 4 : ((fw & {{ B_G2 }}) ? 2 : 1);  // (a tile's outputs: G R rows)
    const int r0 = (rt - rt0) * R * G;
    int nv = st[{{ S_B }} + 3 * b + 2] - r0;
    if (nv > R * G)
        nv = R * G;
{% else %}
    st[{{ S_FMT }}] = st[{{ S_B }} + 3 * bn + 1] & 255;
    const int r0 = (rt - rt0) * R;
    int nv = st[{{ S_B }} + 3 * b + 2] - r0;
    if (nv > R)
        nv = R;
{% endif %}
    const int flags = st[{{ S_FLAGS }}], S = st[{{ S_NS }}];
    const int n = st[{{ S_STG }}];
{% if moe %}
    const int gu = (flags & {{ F_GU }}) | ((fw & {{ B_GU }}) ? {{ F_GU }} : 0);  // (F_GU, or this block's B_GU)
{% if wide %}
    {{ p }}_rows(ws, flags & ~{{ F_GU }}, 0, {{ YB }} * G);
    const int nw = gu ? {{ p }}_guwin(ws, S, R) : 0;  // (gate / up: the windows completed, [8 gate | 8 up] each)
    int add = 0;
    if (flags & {{ F_ACCUM }})  // (an expert's down: its rows into acc)
        {{ p }}_accum(ws, S, flags, nv, row0 + r0, R, G, acc);
    else {
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < S; s++) {  // (per 16 rows: a row group's rows 0 .. 15, 16 .. (block j: halves 2 j, 2 j + 1),
            // a window's 8 outputs)
            int a = 0;
            if (gu) {
                AIE_LOOP_NO_UNROLL
                for (int w = 0; w < nw; w++)
                    a += {{ p }}_slot(ws, s, flags, fw, 8, 0, n + a, w, acc);
            } else {
                AIE_LOOP_NO_UNROLL
                for (int j = 0; j < G; j++) {
                    const int k = nv - R * j > R ? R : nv - R * j;
                    if (k > 0)
                        a += {{ p }}_slot(ws, s, flags, fw, k > 16 ? 16 : k, row0 + r0 + R * j, n + a, 2 * j, acc);
                    if (k > 16)
                        a += {{ p }}_slot(ws, s, flags, fw, k - 16, row0 + r0 + R * j + 16, n + a, 2 * j + 1, acc);
                }
            }
            add = a;
        }
    }
{% else %}
    {{ p }}_rows(ws, flags | gu, {{ YB }} * G);
    int add = 0;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {  // (per block of 16 rows: R rows of a row group, or a tile's rows; F_GU: 8 outputs)
        int a = 0;
        AIE_LOOP_NO_UNROLL
        for (int j = 0; j < G; j++) {
            const int k = gu ? 8 : (nv - R * j > R ? R : nv - R * j);
            if (k > 0)
                a += {{ p }}_slot(ws, s, flags, fw, k, row0 + r0 + R * j, n + a, j, acc);
        }
        add = a;
    }
{% endif %}
    AIE_LOOP_NO_UNROLL
    for (int b = 0; b < G; b++)  // (the tile's row blocks; the others are zero)
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < {{ YB }}; i += 16)
            aie::store_v(yt + {{ YB }} * b + i, aie::zeros<float, 16>());
{% else %}
    {{ p }}_rows(ws, {{ "flags & ~%d" % F_GU if gu_win else "flags" }}{{ ", 2" if act == "silu" and wide else "" }});
{% endif %}
{% if moe %}
{% elif wide %}
    int nv0 = nv > 16 ? 16 : nv, nv1 = nv - nv0;  // (the tile's rows 0 .. 15, 16 .. 31)
{% if gu_win %}
    if (flags & {{ F_GU }}) {  // (the windows completed: [8 gate | 8 up] at rows 0 .., 16 ..)
        const int nw = {{ p }}_guwin(ws, S, R);
        nv0 = nw > 0 ? 8 : 0;
        nv1 = nw > 1 ? 8 : 0;
    }
{% else %}
    if (flags & {{ F_GU }}) {  // ([8 gate | 8 up] per 16 rows)
        nv0 = 8;
        nv1 = R > 16 ? 8 : 0;
    }
{% endif %}
    int add = 0;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        add = {{ p }}_slot(ws, s, flags, fw, nv0, row0 + r0, n, 0);
        if (nv1 > 0)
            add += {{ p }}_slot(ws, s, flags, fw, nv1, row0 + r0 + 16, n + add, 1);
    }
    AIE_LOOP_UNROLL_FULL  // (zv's rolled loop: ~10 cycles a store, on every row tile)
    for (int i = 0; i < {{ YT }}; i += 16)
        aie::store_v(yt + i, aie::zeros<float, 16>());
{% else %}
    if (flags & {{ F_GU }})
{% if gu_win %}
        nv = {{ p }}_guwin(ws, S, R);
{% else %}
        nv = 8;
{% endif %}
{% if SLOTS == 8 %}
    const int add = {{ p }}_slots(ws, S, flags, fw, nv, row0 + r0, n, hres);
{% else %}
    int add = 0;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++)
        add = {{ p }}_slot(ws, s, flags, fw, nv, row0 + r0, n);
{% endif %}
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < {{ YT }}; i += 16)
        aie::store_v(yt + i, aie::zeros<float, 16>());
{% endif %}
    st[{{ S_STG }}] = n + add;
    if (n + add >= {{ C_LEN }}{{ " && !(flags & %d)" % F_HOUT if moe else "" }})
        st[(flags & {{ F_MSG }}) ? {{ S_NM }} : {{ S_NE }}] = S;
    if (tl == 0)
        {{ p }}_fin(st);
}

// The next C (or M) object: slot e's first C_LEN staged values (after the last slot: the staging moves down), or
// the residual's next object of a slot
__attribute__((minsize)) void {{ p }}_emit(int32_t *__restrict ws, bfloat16 *__restrict c{{ ", const bfloat16 *__restrict acc" if moe else "" }}{{ ", float *__restrict hres" if SLOTS == 8 else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int s = st[{{ S_ESLOT }}], S = st[{{ S_NS }}];
    const int last = s + 1 == S;
    st[{{ S_ESLOT }}] = last ? 0 : s + 1;
{% if post %}
    const int src = st[{{ S_ESRC }}];
    if (src) {  // 1: res, 2: pend then its partial sums, 3: res (a header's update) then back to the staging
        const int j = st[{{ S_EJ }}], h8 = st[{{ S_H8 }}], nobj = (h8 + {{ C_LEN - 1 }}) >> {{ C_LEN.bit_length() - 1 }};
        if (j == nobj) {  // (src 2) the column's sums of squares of v: [slot 4][8 lanes] fp32
            const float *__restrict pv = WS(float, {{ w_.pend }});
            aie::accum<accfloat, 32> sq = aie::zeros<accfloat, 32>();  // (v^2: al::mulff's three products of v, v)
            AIE_LOOP_NO_UNROLL
            for (int k = 0; k < h8; k += 8) {
                const aie::vector<float, 32> v = aie::concat(aie::load_v<8>(pv + k), aie::load_v<8>(pv + {{ RES_PITCH }} + k),
                                                             aie::load_v<8>(pv + {{ 2 * RES_PITCH }} + k), aie::load_v<8>(pv + {{ 3 * RES_PITCH }} + k));
                aie::vector<bfloat16, 32> vh, vl;
                al::split2<32>(v, vh, vl);
                sq = aie::mac(aie::mac(aie::mac_square(sq, vh), vh, vl), vl, vh);
            }
            aie::store_v((float *)c, sq.template to_vector<float>());
            return;
        }
        st[{{ S_EJ }}] = last ? j + 1 : j;
        const float *__restrict pr = WS(float, {{ w_.res }}) + (src == 2 ? {{ w_.pend - w_.res }} / 4 : 0) + {{ RES_PITCH }} * s + {{ C_LEN }} * j;
        aie::store_v(c, aie::accum<accfloat, 64>(aie::load_v<64>(pr)).template to_vector<bfloat16>());
        if (src == 3 && last && j + 1 == nobj)
            st[{{ S_ESRC }}] = 0;
        return;
    }
{% else %}
    if (st[{{ S_ESRC }}]) {
        const int j = st[{{ S_EJ }}];
        st[{{ S_EJ }}] = last ? j + 1 : j;
{% if moe %}
        if (st[{{ S_ESRC }}] >= 4) {  // 4: acc ([slot][H] bf16), 5: the routing's objects (S_NS 1)
            aie::store_v(c, aie::load_v<64>(st[{{ S_ESRC }}] == 4 ? acc + {{ H }} * s + {{ C_LEN }} * j
                                                                   : WS(bfloat16, {{ w_.rbuf }}) + {{ C_LEN }} * j));
            return;
        }
{% endif %}
{% if SLOTS == 8 %}
        const float *__restrict pr = {{ p }}_res(ws, hres, s) + {{ C_LEN }} * j;
{% else %}
        const float *__restrict pr = WS(float, {{ w_.res }}) + {{ RES_PITCH }} * s + {{ C_LEN }} * j;
{% endif %}
        aie::store_v(c, aie::accum<accfloat, 64>(aie::load_v<64>(pr)).template to_vector<bfloat16>());
        return;
    }
{% endif %}
    bfloat16 *__restrict stg = WS(bfloat16, {{ w_.stg }});
    aie::store_v(c, aie::load_v<64>(stg + {{ 2 * C_LEN }} * s));
    if (!last)
        return;
    for (int i = 0; i < {{ SLOTS }}; i++) {  // (the staging moves down, zero behind)
        aie::store_v(stg + {{ 2 * C_LEN }} * i, aie::load_v<64>(stg + {{ 2 * C_LEN }} * i + {{ C_LEN }}));
        aie::store_v(stg + {{ 2 * C_LEN }} * i + {{ C_LEN }}, aie::zeros<bfloat16, 64>());
    }
    st[{{ S_STG }}] -= {{ C_LEN }};
}

{% if moe %}
// The helper's command for the next sub-phase of an expert object (its state: ``hsub``)
__attribute__((minsize)) void {{ p }}_sub(int32_t *__restrict ws, int32_t *__restrict cmd)
{
    cmd[1] = (int32_t)(uintptr_t)ws + ({{ SHM }});
    cmd[{{ C_KIND }}] = {{ K_SUB }};
}

// The next strip of a down's input: the hold buffer's [h | h] (the helper's command)
__attribute__((minsize)) void {{ p }}_lstrip(int32_t *__restrict ws, int32_t *__restrict cmd)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int d = st[{{ S_XD }}];
    st[{{ S_XD }}] = d + 1;
    cmd[0] = (int32_t)(uintptr_t)WS(bfloat16, {{ w_.hold }}) + ({{ SHM }});
    cmd[1] = 0;
    cmd[2] = d;
    cmd[3] = st[{{ S_XC0 }}];
    cmd[{{ C_KIND }}] = {{ K_STRIP }};
}

{% endif %}
// An object of the attention core's outputs -> C
__attribute__((minsize)) void {{ p }}_relay(const bfloat16 *__restrict r, bfloat16 *__restrict c)
{
    aie::store_v(c, aie::load_v<{{ C_LEN }}>(r));
}

{% if moe %}
{{ route }}

// The helper, an expert object m: its metadata and the gate / up, down templates (its payload) into the plan, its
// sub-phases (per item and half: gate / up, down; the first down stores acc, the last emits it)
__attribute__((noinline, minsize)) void {{ p }}_hexpert(const int32_t *__restrict m, int32_t *__restrict ws)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    int32_t *__restrict pl = WS(int32_t, {{ w_.plan }});
    {{ p }}_cv(pl, m, {{ 3 * META // 16 }});
    int ns = 0, first = 2;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < m[{{ M_NITEM }}]; i++) {
        const int w = m[{{ M_ITEM }} + {{ ITEM_WORDS }} * i];
        AIE_LOOP_NO_UNROLL
        for (int h = (w >> 8) & 1; h <= ((w >> 9) & 1); h++) {  // (word: down | first << 1 | last << 2 | h << 3 | item << 4)
            pl[{{ PLAN_SUBS }} + ns++] = i << 4 | h << 3;
            pl[{{ PLAN_SUBS }} + ns++] = i << 4 | h << 3 | first | 1;
            first = 0;
        }
    }
    pl[{{ PLAN_SUBS }} + ns - 1] |= 4;
    st[{{ S_NSUB }}] = ns;
    st[{{ S_SUB }}] = 0;
    st[{{ S_NS }}] = m[{{ M_NS }}];
}

__attribute__((noinline, minsize)) void {{ p }}_hbegin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col);

// The helper, the next sub-phase of an expert object (``hexpert``): its metadata from its template (gate / up: on the
// normed phase's x panels and 1 / rms, its outputs into the hold buffer; down: on two strips of the hold buffer in the
// side chunk, its rows times the item's weights into acc)
__attribute__((noinline, minsize)) void {{ p }}_hsub(int32_t *__restrict ws)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    int32_t *__restrict pl = WS(int32_t, {{ w_.plan }});
    const int j = st[{{ S_SUB }}];
    st[{{ S_SUB }}] = j + 1;
    const int w = pl[{{ PLAN_SUBS }} + j], down = w & 1;
    int32_t *__restrict sm = pl + {{ PLAN_META }};
    {{ p }}_cv(sm, pl + {{ META }} + {{ META }} * down, {{ META // 16 }});
    sm[{{ M_FLAGS }}] = down ? {{ F_MAIN + F_WS + F_ACCUM }} | ((w & 2) ? {{ F_ACC0 }} : 0) | ((w & 4) ? {{ F_EMITA }} : 0)
                             : {{ F_MAIN + F_NORM + F_GU + F_HOUT }};
    sm[{{ M_NS }}] = pl[{{ M_NS }}];
    sm[{{ M_NX }}] = 0;
    sm[{{ M_XC0 }}] = down ? {{ SIDE }} : 0;
    {{ p }}_hbegin((const bfloat16 *)sm, ws, 0);
    st[{{ S_FLAGS }}] = sm[{{ M_FLAGS }}];
    st[{{ S_NIN }}] = down;
    st[{{ S_NLS }}] = down ? 2 : 0;
    st[{{ S_NE }}] = 0;
    st[{{ S_NM }}] = 0;
    if (down) {  // (the item's weights per slot)
        const int32_t *__restrict ei = pl + {{ M_ITEM }} + {{ ITEM_WORDS }} * (w >> 4);
        float *__restrict wsc = WS(float, {{ w_.wsc }});
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < {{ SLOTS }}; s++)
            aie::store_v(wsc + 8 * s, aie::broadcast<float, 8>(__builtin_bit_cast(float, ei[1 + s])));
    }
}

// The helper, a combine's object: slot d's partials of the columns ([column][H8] bf16 at x) into the main core's
// residual slice of the slot (fp32)
__attribute__((noinline)) void {{ p }}_hacc(const bfloat16 *__restrict x, int32_t *__restrict ws, int d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict r = WS(float, {{ w_.res }}) + {{ RES_PITCH }} * d;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ H8 }}; i += 32) {
        aie::accum<accfloat, 32> a(aie::load_v<32>(r + i));
        AIE_LOOP_NO_UNROLL
        for (int c = 0; c < {{ COLS }}; c++)
            a = aie::add(a, aie::accum<accfloat, 32>(aie::load_v<32>(x + {{ H8 }} * c + i)).template to_vector<float>());
        aie::store_v(r + i, a.template to_vector<float>());
    }
}
{% endif %}

// The helper: a phase's metadata object x (in the main core's memory, as its workspace ws) -> the main core's trip
// counts and the phase's state, or a slot's residual slice -> its residual (``begin``'s F_MAIN objects){{ "; the route object (``route``), an expert object (``hexpert``); a combine" if moe else "" }}
__attribute__((noinline, minsize)) void {{ p }}_hbegin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col{{ ", float *__restrict hres" if SLOTS == 8 else "" }})
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    const int flags = m[{{ M_FLAGS }}];
    if (flags & {{ F_LOADR }}) {
{% if SLOTS == 8 %}
        {{ p }}_cv((int32_t *){{ p }}_res(ws, hres, m[{{ M_SLOT }}]), (const int32_t *)(x + {{ 2 * META }}), {{ H8 // 16 }});
{% else %}
        {{ p }}_cv(WS(int32_t, {{ w_.res }}) + {{ RES_PITCH }} * m[{{ M_SLOT }}], (const int32_t *)(x + {{ 2 * META }}), {{ H8 // 16 }});
{% endif %}
        return;
    }
{% if moe %}
    if (flags & {{ F_ROUTE }}) {
        {{ p }}_route(m, ws, col);
        return;
    }
    if (flags & {{ F_EXPERT }}) {
        {{ p }}_hexpert(m, ws);
        return;
    }
{% endif %}
    const int nx = m[{{ M_NX }}];
    st[{{ S_NS }}] = m[{{ M_NS }}];
{% if packed and SLOTS == 8 %}
    const int sp = m[{{ M_SP }}], lper = sp == 1 ? 3 : (sp == 2 ? 2 : (sp == 4 ? 1 : 0));  // (the host's: gdn_out 8)
{% elif packed %}
    const int sp = m[{{ M_NS }}] == 1 ? 1 : (m[{{ M_NS }}] == 2 ? 2 : 4), lper = sp == 1 ? 2 : (sp == 2 ? 1 : 0);
{% endif %}
{% if packed %}
    st[{{ S_SP }}] = sp;
    st[{{ S_NXS }}] = 1 << lper;
    st[{{ S_NXS1 }}] = (1 << lper) - 1;
{% endif %}
    st[{{ S_NX }}] = nx;
    st[{{ S_NIN }}] = 1;
    st[{{ S_K8 }}] = m[{{ M_K }}] >> 3;
    st[{{ S_NT }}] = m[{{ M_NT }}];
    st[{{ S_G }}] = m[{{ M_G }}];
    st[{{ S_NRT }}] = m[{{ M_NRT }}];
    st[{{ S_NB }}] = m[{{ M_NB }}];
    for (int i = 0; i < {{ 3 * NBLK }}; i++)
        st[{{ S_B }} + i] = m[{{ M_B }} + i];
    st[{{ S_FMT }}] = m[{{ M_B + 1 }}] & {{ 255 + B_G2 + B_G4 if moe else 255 }};  // (the first row tile's)
    st[{{ S_TL }}] = m[{{ M_NT }}];
    {{ p }}_zv(WS(int32_t, {{ w_.stg }}), {{ 2 * SLOTS * C_LEN * 2 // 64 }});
    st[{{ S_KC }}] = 0;
    st[{{ S_RT }}] = 0;
    st[{{ S_XD }}] = 0;
    st[{{ S_STG }}] = 0;
    st[{{ S_ESLOT }}] = 0;
    st[{{ S_EJ }}] = 0;
    st[{{ S_ESRC }}] = 0;
{% if moe %}
    const int xc0 = m[{{ M_XC0 }}];  // (the phase's first chunk of x / sum panels)
    st[{{ S_XC0 }}] = xc0;
    st[{{ S_KC }}] = xc0;
    st[{{ S_G }}] = m[{{ M_G }}] + xc0;
    st[{{ S_NIN }}] = nx > 0;
    if (flags & {{ F_COMB }}) {  // (its objects add into the residual, which then goes out)
        st[{{ S_NIN }}] = 0;
        st[{{ S_NE }}] = m[{{ M_NS }}] * {{ H8 // C_LEN }};
        st[{{ S_ESRC }}] = 1;
    }
{% endif %}
{% if post %}
    st[{{ S_NPS }}] = (flags & {{ F_POST }}) ? 1 : 0;
    st[{{ S_SCALE }}] = m[{{ M_SCALE }}];
    st[{{ S_COL }}] = col;
    st[{{ S_H8 }}] = m[{{ M_H8 }}];
    st[{{ S_IH }}] = m[{{ M_IH }}];
    if (flags & {{ F_PZERO }})
        {{ p }}_zv(WS(int32_t, {{ w_.pend }}), {{ SLOTS * RES_PITCH * 4 // 64 }});
    if (flags & {{ F_NORM + F_POST }})
{% else %}
    if (flags & {{ F_NORM }})
{% endif %}
        st[{{ S_NNO }}] = (nx + {{ NX_PER_NORM - 1 }}) / {{ NX_PER_NORM }};
    else
{% if packed %}
        st[{{ S_NXP }}] = (nx + (1 << lper) - 1) >> lper;  // (input objects of 1 << lper strips)
{% else %}
        st[{{ S_NXP }}] = nx;
{% endif %}
{% if pli %}
    if (flags & {{ F_PLI }}) {  // the per-layer inputs ([p: slot][{{ 32 * COLS }}] | [e: slot][{{ 32 * COLS }}] bf16): the
        // column's p and e -> [g][slot][r], the slots' 1 / rms(p) -> [slot][r] as bf16 hi | lo (``nbegin`` completes them)
        const bfloat16 *__restrict src = x + {{ 2 * META }};
        bfloat16 *__restrict pl = WS(bfloat16, {{ w_.pli }});
        float *__restrict ss = WS(float, {{ w_.pli + 4 * 4 * SLOTS * 8 }});  // ([slot][8] partial sums of squares)
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < {{ SLOTS }}; s++) {
            const bfloat16 *__restrict ps = src + {{ 32 * COLS }} * s;
            aie::accum<accfloat, 32> sq = aie::mul(aie::load_v<32>(ps), aie::load_v<32>(ps));
            AIE_LOOP_NO_UNROLL
            for (int i = 32; i < {{ 32 * COLS }}; i += 32) {
                const aie::vector<bfloat16, 32> v = aie::load_v<32>(ps + i);
                sq = aie::mac(sq, v, v);
            }
            const aie::accum<accfloat, 16> h = aie::add(sq.template extract<16>(0), sq.template extract<16>(1));
            aie::store_v(ss + 8 * s, aie::add(h.template extract<8>(0), h.template extract<8>(1)).template to_vector<float>());
            for (int g = 0; g < 4; g++) {
                aie::store_v(pl + 32 * g + 8 * s, aie::load_v<8>(ps + 32 * col + 8 * g));
                aie::store_v(pl + {{ 4 * SLOTS * 8 }} + 32 * g + 8 * s,
                             aie::load_v<8>(ps + {{ 32 * COLS * SLOTS }} + 32 * col + 8 * g));
            }
        }
        const aie::vector<float, 8> t = al::group8<false>(aie::concat(aie::load_v<32>(ss), aie::zeros<float, 32>()));
        aie::vector<bfloat16, 32> ih, il;
        al::split2<32>(al::rows4(al::inv_rms<{{ 32 * COLS }}, {{ SLOTS }}>(t.template grow<16>(), {{ ple_eps }}f)), ih, il);
        aie::store_v((bfloat16 *)ss, ih);
        aie::store_v((bfloat16 *)ss + 32, il);
    }
{% endif %}
{% if gu_win and moe %}
    int gub = flags & {{ F_GU }};
    for (int i = 0; i < m[{{ M_NB }}]; i++)
        gub |= m[{{ M_B + 1 }} + 3 * i] & {{ B_GU }};
    if (gub) {  // (the slots' gate / up windows: ``guwin``; F_GU or a B_GU block)
        st[{{ S_GN }}] = 0;
        st[{{ S_GK }}] = 0;
        {{ p }}_zv(WS(int32_t, {{ w_.gw }}), {{ (64 if wide else 32) * SLOTS * 4 // 64 }});
    }
{% elif gu_win %}
    if (flags & {{ F_GU }}) {  // (the slots' gate / up windows: ``guwin``)
        st[{{ S_GN }}] = 0;
        st[{{ S_GK }}] = 0;
        {{ p }}_zv(WS(int32_t, {{ w_.gw }}), {{ (64 if wide else 32) * SLOTS * 4 // 64 }});
    }
{% endif %}
    {{ p }}_zv(WS(int32_t, {{ w_.yt }}), {{ YT * 4 // 64 }});
}

// The helper: one command of the main core. K_TILE: the partial rows of its super-blocks -> r; K_STRIP: an input
// object (its x panels into xp, its 16-k sums into the scratch); K_INPUT: the sum panels of every chunk (into the main
// core's xs), the slots' 1 / rms -> r (8 lanes per slot); K_BEGIN: a phase's metadata (hbegin)
void {{ p }}_htask(const int32_t *__restrict cmd, float *__restrict r, uint8_t *__restrict scr, float *__restrict sq,
                   uint8_t *__restrict xp{{ ", float *__restrict hres" if SLOTS == 8 else "" }}{{ ", int32_t *__restrict hst" if help_flag else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int kind = cmd[{{ C_KIND }}];
{% if help_flag %}
    hst[0] = kind == {{ K_HELP }} ? cmd[1] : 0;  // (the model core's work after this command: K_HELP)
    if (kind == {{ K_HELP }})
        return;
{% endif %}
    if (kind == {{ K_NOP }})
        return;
    if (kind == {{ K_BEGIN }}) {
        {{ p }}_hbegin((const bfloat16 *)(uintptr_t)cmd[0], (int32_t *)(uintptr_t)cmd[1], cmd[2]{{ ", hres" if SLOTS == 8 else "" }});
        return;
    }
{% if moe %}
    if (kind == {{ K_SUB }}) {
        {{ p }}_hsub((int32_t *)(uintptr_t)cmd[1]);
        return;
    }
    if (kind == {{ K_ACC }}) {
        {{ p }}_hacc((const bfloat16 *)(uintptr_t)cmd[0], (int32_t *)(uintptr_t)cmd[1], cmd[2]);
        return;
    }
{% endif %}
    if (kind == {{ K_TILE }}) {
        {{ p }}_Win<2> w_;
        al::zero<{{ YN }}, 32>(r);
{% if rowsplit %}
        {{ mp }}_mmv<1>((uint8_t *)(uintptr_t)cmd[0], xp + {{ gv.XP_CHUNK }} * cmd[1] + ({{ p }}_rs(cmd[3]) ? 0 : {{ gv.XP_CHUNK // 2 }}),
                        (const uint8_t *)(uintptr_t)cmd[2], r, scr, cmd[3]);
{% else %}
        {{ mp }}_mmv<1>((uint8_t *)(uintptr_t)cmd[0], xp + {{ "%d + %d" % (gv.XPH, gv.XPHC) if xh else "%d + %d" % (gv.XP_CHUNK // 2, gv.XP_CHUNK) }} * cmd[1],
                        (const uint8_t *)(uintptr_t)cmd[2], r, scr, cmd[3]);
{% endif %}
        return;
    }
    if (kind == {{ K_STRIP }}) {  // input object d
        {{ p }}_Win<4> w_;
{% if "strip" in ablate %}
        return;
{% endif %}
        const int d = cmd[2];
        if (d == 0)
            aie::store_v(sq, aie::zeros<float, 32>());
{% if SLOTS == 8 %}
        if (d == 0)
            aie::store_v(sq + 32, aie::zeros<float, 32>());
{% endif %}
        const bfloat16 *x = (const bfloat16 *)(uintptr_t)cmd[0];
{% if xh and SLOTS == 8 %}
        // (strip d: super-block d & 3 of chunk d / 4, in half (d >> 1) & 1)
        uint8_t *xpk = xp + {{ gv.XPH }} * ((d >> 1) & 1) + {{ gv.XPHC }} * (d >> 2) + {{ 32 * gv.PANEL }} * (d & 1);
{% elif xh %}
        uint8_t *xpk = xp + {{ gv.XPH }} * (d & 1) + {{ gv.XPHC }} * {{ "((d >> 1) + cmd[3])" if moe else "(d >> 1)" }};  // (strip d: half d & 1 of chunk d / 2{{ " + the phase's first" if moe else "" }})
{% else %}
        uint8_t *xpk = xp + {{ gv.PANEL * OBJ_K // 8 }} * d;
{% endif %}
        float *t = (float *)scr + {{ OBJ_K * SLOTS // 16 }} * d;
{% if SLOTS == 8 %}
        const bfloat16 *w = (const bfloat16 *)(uintptr_t)cmd[1];
        const int sp = cmd[{{ C_SP }}];
{% for sp in (1, 2, 4, 8) %}
        {{ "else " if not loop.first else "" }}if (sp == {{ sp }}) {
            if (w)
                {{ p }}_hstrip<true, {{ sp }}>(x, w, sq, xpk, t);
            else
                {{ p }}_hstrip<false, {{ sp }}>(x, nullptr, sq, xpk, t);
        }
{% endfor %}
{% else %}
        const int sp = {{ "cmd[%d]" % C_SP if packed else 4 }};
        if (cmd[1])
            {{ p }}_hstrip<true>(x, (const bfloat16 *)(uintptr_t)cmd[1], sq, xpk, t, sp);
        else
            {{ p }}_hstrip<false>(x, nullptr, sq, xpk, t, sp);
{% endif %}
        return;
    }
{% if post %}
    if (kind == {{ K_PSTRIP }}) {  // a header's input object d
        {{ p }}_Win<4> w_;
{% if "strip" in ablate %}
        return;
{% endif %}
        const int d = cmd[2];
        if (d == 0)
            aie::store_v(sq, aie::zeros<float, 32>());
        const bfloat16 *x = (const bfloat16 *)(uintptr_t)cmd[0], *w = (const bfloat16 *)(uintptr_t)cmd[1];
        const float *ivv = (const float *)(uintptr_t)cmd[3];
        uint8_t *xpk = xp + {{ gv.PANEL * POST_K // 8 }} * d;
        float *t = (float *)scr + {{ POST_K // 4 }} * d;
        bfloat16 *nb = (bfloat16 *)(scr + {{ MAX_K }});  // (past the K group's t)
{% if packed %}
        const int sp = cmd[{{ C_SP }}];
        if (sp == 1)
            {{ p }}_hpstrip<1>(x, w, w + {{ NX_PER_NORM * POST_K }}, ivv, sq, xpk, t, nb);
        else if (sp == 2)
            {{ p }}_hpstrip<2>(x, w, w + {{ NX_PER_NORM * POST_K }}, ivv, sq, xpk, t, nb);
        else if (sp == 2 + {{ C_MIRROR }})
            {{ p }}_hpstrip<2, true>(x, w, w + {{ NX_PER_NORM * POST_K }}, ivv, sq, xpk, t, nb);
        else
{% endif %}
            {{ p }}_hpstrip<4>(x, w, w + {{ NX_PER_NORM * POST_K }}, ivv, sq, xpk, t, nb);
        return;
    }
{% endif %}
    {{ p }}_hinput((uint8_t *)(uintptr_t)cmd[0], (const float *)scr, sq, r, cmd[2] >> 5, cmd[1]{{ ", __builtin_bit_cast(float, cmd[3])" if post else "" }});
}

}  // extern "C"
""")
