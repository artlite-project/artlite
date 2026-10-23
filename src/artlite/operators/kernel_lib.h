// Kernel helpers shared by the operators' templates (namespace al); math functions with variants (exp2,
// SiLU, GELU) are generated per contract in kmath.py.
//
// Included by operators.h, so every kernel gets them through the precompiled header, and a fused
// ELF (one translation unit for all its operators) has one copy. Everything is inline: a helper
// costs nothing unless a kernel calls it.
#pragma once

#include <aie_api/aie.hpp>

namespace al {

// ---------------------------------------------------------------------------------------------
// Scalar integer helpers
// ---------------------------------------------------------------------------------------------

// t / D for 0 <= t < 65536 / D: a multiply and a shift (a signed division by a constant is a
// __divsi3 call on this core, an unsigned one a __muldi3 call)
template <int D>
[[clang::always_inline]] inline int div_small(int t)
{
    static_assert(D > 0 && D <= 256, "small divisors");
    if constexpr ((D & (D - 1)) == 0)
        return t >> __builtin_ctz(D);
    else
        return (t * ((65536 + D - 1) / D)) >> 16;
}

// ---------------------------------------------------------------------------------------------
// Copies
// ---------------------------------------------------------------------------------------------

// dst[0 .. N) = src[0 .. N) in vectors of V elements (N a multiple of V). Walked pointers pipeline
// at II 1 (one load, one store per bundle); a loop that indexes both as base + i shares one index
// register between them and ran at II 6-9. (Pointer helpers are templates on the pointee type: a
// __aie_dm_resource_x qualifier is an address space, and a plain T * parameter casts it away.)
template <int N, int V, typename TD, typename TS>
[[clang::always_inline]] inline void copy(TD *__restrict dst, const TS *__restrict src)
{
    static_assert(N % V == 0, "whole vectors");
    for (int i = 0; i < N / V; i++) {
        aie::store_v(dst, aie::load_v<V>(src));
        src += V;
        dst += V;
    }
}

// dst[0 .. N) = 0 in vectors of V elements: walked, II 1 per 64-byte store (indexed dst + i: II 3-4)
template <int N, int V, typename T>
[[clang::always_inline]] inline void zero(T *__restrict dst)
{
    static_assert(N % V == 0, "whole vectors");
    for (int i = 0; i < N / V; i++) {
        aie::store_v(dst, aie::zeros<T, V>());
        dst += V;
    }
}

// ---------------------------------------------------------------------------------------------
// fp32 arithmetic through bf16 hi / lo splits (the vector unit multiplies bf16 into fp32)
// ---------------------------------------------------------------------------------------------

// x * c for an exact bf16 c without the emulated fp32 multiply (~2^-16 rel. error)
template <unsigned N>
[[clang::always_inline]] inline aie::accum<accfloat, N> mulc(aie::vector<float, N> x, aie::vector<bfloat16, N> c)
{
    aie::vector<bfloat16, N> hi = aie::accum<accfloat, N>(x).template to_vector<bfloat16>();
    aie::vector<float, N> r = aie::sub(x, aie::accum<accfloat, N>(hi).template to_vector<float>());
    aie::vector<bfloat16, N> lo = aie::accum<accfloat, N>(r).template to_vector<bfloat16>();
    return aie::mac(aie::mul(hi, c), lo, c);
}

// x * C for an fp32 constant C = Hi + Lo given as its bf16 parts (three products)
template <unsigned N, float Hi, float Lo>
[[clang::always_inline]] inline aie::vector<float, N> mul_hilo(aie::vector<float, N> x)
{
    const aie::vector<bfloat16, N> l_hi = aie::broadcast<bfloat16, N>(Hi);
    const aie::vector<bfloat16, N> l_lo = aie::broadcast<bfloat16, N>(Lo);
    aie::vector<bfloat16, N> hi = aie::accum<accfloat, N>(x).template to_vector<bfloat16>();
    aie::vector<float, N> r = aie::sub(x, aie::accum<accfloat, N>(hi).template to_vector<float>());
    aie::vector<bfloat16, N> lo = aie::accum<accfloat, N>(r).template to_vector<bfloat16>();
    return aie::mac(aie::mac(aie::mul(hi, l_hi), lo, l_hi), hi, l_lo).template to_vector<float>();
}

// x * log2(e)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> log2e(aie::vector<float, N> x)
{
    return mul_hilo<N, 1.4453125f, -0.00261688232421875f>(x);
}

// a * b for fp32 a, b via bf16 hi / lo splits (~2^-16 relative error)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> mulff(aie::vector<float, N> a, aie::vector<float, N> b)
{
    aie::vector<bfloat16, N> a_hi = aie::accum<accfloat, N>(a).template to_vector<bfloat16>();
    aie::vector<bfloat16, N> a_lo = aie::accum<accfloat, N>(
        aie::sub(a, aie::accum<accfloat, N>(a_hi).template to_vector<float>())).template to_vector<bfloat16>();
    aie::vector<bfloat16, N> b_hi = aie::accum<accfloat, N>(b).template to_vector<bfloat16>();
    aie::vector<bfloat16, N> b_lo = aie::accum<accfloat, N>(
        aie::sub(b, aie::accum<accfloat, N>(b_hi).template to_vector<float>())).template to_vector<bfloat16>();
    return aie::mac(aie::mac(aie::mul(a_hi, b_hi), a_hi, b_lo), a_lo, b_hi).template to_vector<float>();
}

// The bf16 hi / lo parts of fp32 x
template <unsigned N>
[[clang::always_inline]] inline void split2(aie::vector<float, N> x, aie::vector<bfloat16, N> &hi, aie::vector<bfloat16, N> &lo)
{
    hi = aie::accum<accfloat, N>(x).template to_vector<bfloat16>();
    lo = aie::accum<accfloat, N>(aie::sub(x, aie::accum<accfloat, N>(hi).template to_vector<float>())).template to_vector<bfloat16>();
}

// A rotation of fp32 pairs: o0 = a c - b s, o1 = a s + b c, each the six products of two mulff in one accumulator
// (each operand split once)
template <unsigned N>
[[clang::always_inline]] inline void rot2(aie::vector<float, N> a, aie::vector<float, N> b, aie::vector<float, N> c,
                                          aie::vector<float, N> s, aie::accum<accfloat, N> &o0, aie::accum<accfloat, N> &o1)
{
    aie::vector<bfloat16, N> ah, al, bh, bl, ch, cl, sh, sl;
    split2<N>(a, ah, al);
    split2<N>(b, bh, bl);
    split2<N>(c, ch, cl);
    split2<N>(s, sh, sl);
    o0 = aie::msc(aie::msc(aie::msc(aie::mac(aie::mac(aie::mul(ah, ch), ah, cl), al, ch), bh, sh), bh, sl), bl, sh);
    o1 = aie::mac(aie::mac(aie::mac(aie::mac(aie::mac(aie::mul(ah, sh), ah, sl), al, sh), bh, ch), bh, cl), bl, ch);
}

// The bf16 hi / lo parts of an fp32 value, in every lane: x * v = x * hi + x * lo to ~2^-16 for a
// bf16 x. (A scale rounded to bf16 alone is off by up to 0.4% in every element at once: a 1 / rms
// that way shifts the scores of a whole K vector.)
template <unsigned N>
[[clang::always_inline]] inline void bf16_hilo(float v, aie::vector<bfloat16, N> &hi, aie::vector<bfloat16, N> &lo)
{
    const aie::vector<float, N> f = aie::broadcast<float, N>(v);
    hi = aie::accum<accfloat, N>(f).template to_vector<bfloat16>();
    lo = aie::accum<accfloat, N>(aie::sub(f, aie::accum<accfloat, N>(hi).template to_vector<float>()))
             .template to_vector<bfloat16>();
}

// 1 / sqrt(ss / N + eps) for the sum of squares ss of N values. The scalar unit has no floating
// point (a scalar product or sum is a soft-float call, ~1 KB of program memory each), so the mean
// and eps go through vector lanes: ss / N as bf16 products (~2^-16; 1 / N is a bf16 for N a power
// of two, else hi / lo parts)
template <unsigned N>
[[clang::always_inline]] inline float inv_rms(float ss, float eps)
{
    aie::accum<accfloat, 16> ms;
    if constexpr ((N & (N - 1)) == 0)
        ms = mulc<16>(aie::broadcast<float, 16>(ss), aie::broadcast<bfloat16, 16>(1.0f / N));
    else
        ms = aie::accum<accfloat, 16>(mulff<16>(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(1.0f / N)));
    ms = aie::add(ms, aie::accum<accfloat, 16>(aie::broadcast<float, 16>(eps)));
    return aie::invsqrt(ms.template to_vector<float>()[0]);
}

// inv_rms of a length known at run time: inv_n = 1 / N (as inv_rms<N> for N not a power of two)
[[clang::always_inline]] inline float inv_rms(float ss, float inv_n, float eps)
{
    aie::accum<accfloat, 16> ms(mulff<16>(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_n)));
    ms = aie::add(ms, aie::accum<accfloat, 16>(aie::broadcast<float, 16>(eps)));
    return aie::invsqrt(ms.template to_vector<float>()[0]);
}

// ---------------------------------------------------------------------------------------------
// Reciprocal, LUT activations (exp2, SiLU, GELU in software: kmath.py; the hardware vexp2 is the
// chord 2^n (1 + f): up to +6% on bf16 inputs; vtanh is a coarse LUT)
// ---------------------------------------------------------------------------------------------

// 1 / x in fp32 to ~2^-16 for normal x > 0: the mantissa's reciprocal (a linear guess on [1, 2), 6%, and three Newton
// steps in mulff), its exponent negated in the bits (no scalar division: a soft-float call)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> recip_fp32(aie::vector<float, N> x)
{
    const aie::vector<int32, N> b = x.template cast_to<int32>();
    const aie::vector<int32, N> e = aie::sub(aie::bit_and(b, aie::broadcast<int32, N>(0x7F800000)),
                                             aie::broadcast<int32, N>(0x3F800000));
    const aie::vector<float, N> m = aie::sub(b, e).template cast_to<float>();  // (in [1, 2))
    const aie::vector<float, N> two = aie::broadcast<float, N>(2.0f);
    aie::vector<float, N> r = aie::sub(aie::broadcast<float, N>(1.4117647f),
                                       mulc<N>(m, aie::broadcast<bfloat16, N>(0.47058824f)).template to_vector<float>());
    for (int i = 0; i < 3; i++)
        r = mulff<N>(r, aie::sub(two, mulff<N>(m, r)));
    return aie::sub(r.template cast_to<int32>(), e).template cast_to<float>();
}

// vtanh / vexp2 of an fp32 accumulator (16 lanes per instruction; the LUTs take the argument truncated
// to bf16)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<bfloat16, N> tanh_acc(aie::accum<accfloat, N> a)
{
    if constexpr (N == 16) {
        return ::tanh(a.to_native());
    } else {
        return aie::concat(tanh_acc<N / 2>(a.template extract<N / 2>(0)), tanh_acc<N / 2>(a.template extract<N / 2>(1)));
    }
}

template <unsigned N>
[[clang::always_inline]] inline aie::vector<bfloat16, N> exp2_acc(aie::accum<accfloat, N> a)
{
    if constexpr (N == 16) {
        return ::exp2(a.to_native());
    } else {
        return aie::concat(exp2_acc<N / 2>(a.template extract<N / 2>(0)), exp2_acc<N / 2>(a.template extract<N / 2>(1)));
    }
}

// silu(x) for |x| < 2.5 as u + s (P0 + P1 s + P2 s^2), s = u^2, u = x / 2 (tanh(u) / u fitted in bf16
// arithmetic; with the vtanh / vexp2 form elsewhere, as in gemm_q's gate/up epilogue: max 0.98 ULP for
// x >= -1.5, mean 0.079 for |x| <= 8, against the LUT form's 9.7 / 0.12). x again through xr (a second load of the same data, not merged with x's by the
// compiler: shorter lifetimes, II 10 instead of 14-18). Returns y where |x| >= 2.5.
[[clang::always_inline]] inline aie::vector<bfloat16, 32> silu_small(aie::vector<bfloat16, 32> x,
                                                                   aie::vector<bfloat16, 32> xr,
                                                                   aie::vector<bfloat16, 32> xm,
                                                                   aie::vector<bfloat16, 32> y)
{
    const aie::vector<bfloat16, 32> kh = aie::broadcast<bfloat16, 32>(0.5f);
    const aie::vector<bfloat16, 32> p0 = aie::broadcast<bfloat16, 32>(0.99609375f);
    const aie::vector<bfloat16, 32> p2 = aie::broadcast<bfloat16, 32>(0.06298828125f);
    const aie::accum<accfloat, 32> p1 = aie::accum<accfloat, 32>(aie::broadcast<float, 32>(-0.30078125f));
    aie::vector<bfloat16, 32> u = aie::mul(x, kh).template to_vector<bfloat16>();
    aie::vector<bfloat16, 32> s = aie::mul(u, u).template to_vector<bfloat16>();
    aie::accum<accfloat, 32> acc = aie::mac(aie::mul(xr, kh), s, p0);
    acc = aie::mac(acc, aie::mul(s, s).template to_vector<bfloat16>(), aie::mac(p1, s, p2).template to_vector<bfloat16>());
    return aie::select(y, acc.template to_vector<bfloat16>(), aie::lt(aie::abs(xm), aie::broadcast<bfloat16, 32>(2.5f)));
}

// ---------------------------------------------------------------------------------------------
// Lane shuffles
// ---------------------------------------------------------------------------------------------

// 8 bf16 -> 64 lanes, each repeated 8 times (lane 8 i + j = s_i)
[[clang::always_inline]] inline aie::vector<bfloat16, 64> expand8(aie::vector<bfloat16, 8> s8)
{
    aie::vector<bfloat16, 32> v = aie::zeros<bfloat16, 32>();
    v.insert(0, s8);
    auto [a, a1] = aie::interleave_zip(v, v, 1);
    auto [b, b1] = aie::interleave_zip(a, a, 1);
    auto [c, c1] = aie::interleave_zip(b, b, 1);
    return aie::concat(c, c1);
}

// 16 bf16 (rows 0..15 of one group) -> the two 64-lane scale vectors of rows 0..7 and 8..15, lane
// 8 i + j = s_i (7 shuffles; upper input lanes unused)
[[clang::always_inline]] inline void expand16(aie::vector<bfloat16, 16> s16, aie::vector<bfloat16, 64> &D0,
                                              aie::vector<bfloat16, 64> &D1)
{
    aie::vector<bfloat16, 32> v = s16.template grow<32>();
    auto [a, a1] = aie::interleave_zip(v, v, 1);  // a: s0 s0 s1 s1 .. s15 s15
    auto [b0, b1] = aie::interleave_zip(a, a, 1);  // b0: rows 0..7 x4, b1: rows 8..15 x4
    auto [c0, c1] = aie::interleave_zip(b0, b0, 1);
    auto [d0, d1] = aie::interleave_zip(b1, b1, 1);
    D0 = aie::concat(c0, c1);
    D1 = aie::concat(d0, d1);
}

// lanes 8 e .. 8 e + 7 of v (e < 4) -> 64 lanes, each repeated 8 times (lane 8 i + j = v[8 e + i]): the 128-bit
// element broadcast 4 times (vextbcst: no L_RM_PORT) and two 8x8 transposes, 2 shuffles instead of the zips' 3 - 4.
// For loops bound by L_RM_PORT (shuffles and a bfp16 stream's pushes / flushes: GemmQ's dequant II 46 -> 40 per 8
// panels); in Q6_K's alu-bound dequant (runtime index math) the zips' schedule was better (II 46 vs 88).
[[clang::always_inline]] inline aie::vector<bfloat16, 64> expand8_of(aie::vector<bfloat16, 32> v, int e)
{
    const v32bfloat16 b = (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)v, e);
    return aie::concat(aie::vector<bfloat16, 32>(shuffle(b, b, T16_8x8_lo)),
                       aie::vector<bfloat16, 32>(shuffle(b, b, T16_8x8_hi)));
}

// expand16 by expand8_of (4 shuffles)
[[clang::always_inline]] inline void expand16_bc(aie::vector<bfloat16, 16> s16, aie::vector<bfloat16, 64> &D0,
                                                 aie::vector<bfloat16, 64> &D1)
{
    const aie::vector<bfloat16, 32> v = s16.template grow<32>();
    D0 = expand8_of(v, 0);
    D1 = expand8_of(v, 1);
}

// integer codes zipped with exponent bytes c -> bf16 (0x43: 128 + q, exact)
template <typename V>
[[clang::always_inline]] inline aie::vector<bfloat16, 64> zipw(V q, aie::vector<uint8, 64> c)
{
    auto [u, v] = aie::interleave_zip(q, c, 1);
    return aie::concat(u, v).template cast_to<bfloat16>();
}

// 64 lanes (q, k) = 8 groups of 8: reduce each group (max or sum) -> 8 lanes (q)
template <bool Max>
[[clang::always_inline]] inline aie::vector<float, 8> group8(aie::vector<float, 64> v)
{
    auto [e1, o1] = aie::interleave_unzip(v.template extract<32>(0), v.template extract<32>(1), 1);
    aie::vector<float, 32> r1 = Max ? aie::max(e1, o1) : aie::add(e1, o1);
    auto [e2, o2] = aie::interleave_unzip(r1.template extract<16>(0), r1.template extract<16>(1), 1);
    aie::vector<float, 16> r2 = Max ? aie::max(e2, o2) : aie::add(e2, o2);
    auto [e3, o3] = aie::interleave_unzip(r2.template extract<8>(0), r2.template extract<8>(1), 1);
    return Max ? aie::max(e3, o3) : aie::add(e3, o3);
}

// 8 lanes -> 64 lanes, lane 8 q + k = v[q]
[[clang::always_inline]] inline aie::vector<float, 64> bcast8(aie::vector<float, 8> v)
{
    aie::vector<float, 16> a = aie::concat(v, v);
    auto [b0, b1] = aie::interleave_zip(a, a, 1);  // q0 q0 q1 q1 ...
    aie::vector<float, 32> b = aie::concat(b0, b1);
    auto [c0, c1] = aie::interleave_zip(b, b, 2);
    aie::vector<float, 64> c = aie::concat(c0, c1);
    auto [d0, d1] = aie::interleave_zip(c, c, 4);
    return aie::concat(d0, d1).template extract<64>(0);
}

// ---------------------------------------------------------------------------------------------
// bfp16ebs8 MMUL operands
// ---------------------------------------------------------------------------------------------

using BV = aie::block_vector<bfp16ebs8, 64>;  // one 8x8 panel
using MM8 = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8>;
// panel streams (sizeof(bfp16ebs8) is 9 on Peano: no pointer arithmetic on bfp16ebs8 *)
template <aie_dm_resource R = aie_dm_resource::none>
using BIn = aie::block_vector_input_buffer_stream<bfp16ebs8, 64, R>;
template <aie_dm_resource R = aie_dm_resource::none>
using BInR = aie::block_vector_restrict_input_buffer_stream<bfp16ebs8, 64, R>;
template <aie_dm_resource R = aie_dm_resource::none>
using BOut = aie::block_vector_output_buffer_stream<bfp16ebs8, 64, R>;

// -I as a bfp16 panel (diagonal mantissa -64 at exponent 127: value m 2^(E - 133) = -1)
[[clang::always_inline]] inline BV neg_eye()
{
    alignas(64) static const int8_t d_[64] = {
        -64, 0, 0, 0, 0, 0, 0, 0, 0, -64, 0, 0, 0, 0, 0, 0, 0, 0, -64, 0, 0, 0, 0, 0, 0, 0, 0, -64, 0, 0, 0, 0,
        0, 0, 0, 0, -64, 0, 0, 0, 0, 0, 0, 0, 0, -64, 0, 0, 0, 0, 0, 0, 0, 0, -64, 0, 0, 0, 0, 0, 0, 0, 0, -64,
    };
    alignas(8) static const int8_t e_[8] = {127, 127, 127, 127, 127, 127, 127, 127};
    v64bfp16ebs8 n;
    n.mantissa = *(const v64int8 *)d_;
    n.exponent = *(const v8int8 *)e_;
    return BV(n);
}

// +I as a bfp16 panel
[[clang::always_inline]] inline BV eye()
{
    alignas(64) static const int8_t d_[64] = {
        64, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0,
        0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 64,
    };
    alignas(8) static const int8_t e_[8] = {127, 127, 127, 127, 127, 127, 127, 127};
    v64bfp16ebs8 n;
    n.mantissa = *(const v64int8 *)d_;
    n.exponent = *(const v8int8 *)e_;
    return BV(n);
}

// x -> hi = bfp16(x), lo = bfp16(x - hi); x - hi is exact on the MMUL as x + hi (-I)^T
[[clang::always_inline]] inline void split_hilo(aie::accum<accfloat, 64> x, const BV &nI, BV &hi, BV &lo)
{
    hi = x.template to_vector<bfp16ebs8>();
    MM8 t(x);
    t.mac(hi, aie::op_transpose(nI));
    lo = t.to_accum().template to_vector<bfp16ebs8>();
}

// x (32 lanes: rows 0 .. 3 of 8) -> the MMUL A panel [bfp16(x); bfp16(x - bfp16(x))]: rows 4 .. 7 the lo parts, ~14
// bits below each block's maximum (decode inputs of 4 token slots, attention queries)
[[clang::always_inline]] inline BV hilo4(aie::accum<accfloat, 32> x, const BV &nI)
{
    const aie::accum<accfloat, 64> xx = aie::concat(x, x);
    MM8 t(xx);
    t.mac(xx.template to_vector<bfp16ebs8>(), aie::op_transpose(nI));  // xx - hi exactly
    return aie::concat(x, t.to_accum().template extract<32>(1)).template to_vector<bfp16ebs8>();
}

// ---------------------------------------------------------------------------------------------
// Rows of 8 lanes (4 rows in 32 lanes: attention rows, token slots)
// ---------------------------------------------------------------------------------------------

// lanes 8 r .. 8 r + 7 = v[r] (r < 4)
[[clang::always_inline]] inline aie::vector<float, 32> rows4(aie::vector<float, 16> v)
{
    aie::vector<float, 32> r = aie::broadcast<float, 32>(v[0]);
    r = aie::select(r, aie::broadcast<float, 32>(v[1]), aie::mask<32>::from_uint32(0x0000FF00u));
    r = aie::select(r, aie::broadcast<float, 32>(v[2]), aie::mask<32>::from_uint32(0x00FF0000u));
    return aie::select(r, aie::broadcast<float, 32>(v[3]), aie::mask<32>::from_uint32(0xFF000000u));
}

// per row r < 4 (lanes 8 r ..): the maximum of its 8 lanes, in all 8
[[clang::always_inline]] inline aie::vector<float, 32> max8(aie::vector<float, 32> v)
{
    v = aie::max(v, aie::shuffle_down_rotate(v, 1));
    v = aie::max(v, aie::shuffle_down_rotate(v, 2));
    v = aie::max(v, aie::shuffle_down_rotate(v, 4));  // lane 8 r: the max of lanes 8 r .. 8 r + 7
    aie::vector<float, 16> m = aie::zeros<float, 16>();
    m[0] = v[0];
    m[1] = v[8];
    m[2] = v[16];
    m[3] = v[24];
    return rows4(m);
}

// per row r < 4 (lanes 8 r ..): the sum of its 8 lanes (lanes 0 .. 3 of the result; the others 1)
[[clang::always_inline]] inline aie::vector<float, 16> sum8(aie::vector<float, 32> v)
{
    v = aie::add(v, aie::shuffle_down_rotate(v, 1));
    v = aie::add(v, aie::shuffle_down_rotate(v, 2));
    v = aie::add(v, aie::shuffle_down_rotate(v, 4));  // lane 8 r: the sum of lanes 8 r .. 8 r + 7
    aie::vector<float, 16> s = aie::broadcast<float, 16>(1.0f);
    s[0] = v[0];
    s[1] = v[8];
    s[2] = v[16];
    s[3] = v[24];
    return s;
}


// ---------------------------------------------------------------------------------------------
// Attention: V records from exact values, partial softmax sums
// ---------------------------------------------------------------------------------------------

// A V record of 8 positions is per d-block a panel of 8 blocks, one per dim over the positions (the K-axis of P V is
// the positions). Re-encoding a record whenever a position joins rounds its earlier positions again with an exponent
// that later values raised: what a position sees would depend on later positions and on rolled-back ones. So an
// open record's values stay exact (bf16 rows [d-block nd][position 8][8]) and its panels are encoded from them, up to
// the position that reads them.

// (nd: the d-blocks of a record. The loops of a run-time nd are pipelined for MIN iterations on (default 32): nd >= MIN;
// a template ND > 0 is the count, known at compile time.)

// A record's V panels -> its rows (exact: a bfp16 value has 8 significant bits)
[[clang::always_inline]] inline void v_rows(const uint8_t *__restrict rec, bfloat16 *__restrict rows, int nd)
{
    BIn<> vi((const bfp16ebs8 *)rec);
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < nd; db++) {
        MM8 t(aie::zeros<accfloat, 64>());
        t.mac(eye(), aie::op_transpose(vi.pop()));  // [position][dim]
        aie::store_v(rows + 64 * db, t.to_accum().template to_vector<bfloat16>());
    }
}

// A position's V (src: 8 nd bf16) into its row of a record's rows (dst: the row's first 8 values)
template <int ND = 0, int MIN = 32>
[[clang::always_inline]] inline void v_put(bfloat16 *__restrict dst, const bfloat16 *__restrict src, int nd)
{
    if constexpr (ND > 0) {
        for (int db = 0; db < ND; db++)
            aie::store_v(dst + 64 * db, aie::load_v<8>(src + 8 * db));
    } else {
        AIE_PREPARE_FOR_POSTPIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(MIN)
        for (int db = 0; db < nd; db++)
            aie::store_v(dst + 64 * db, aie::load_v<8>(src + 8 * db));
    }
}

// rows, those from position n on as zeros -> the record's V panels
template <int MIN = 32>
[[clang::always_inline]] inline void v_enc(const bfloat16 *__restrict rows, uint8_t *__restrict rec, int nd, int n)
{
    BOut<> o((bfp16ebs8 *)rec);
    const aie::mask<64> keep = aie::mask<64>::from_uint64(n < 8 ? (1ull << (8 * n)) - 1 : ~0ull);
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN)
    for (int db = 0; db < nd; db++) {
        const aie::vector<bfloat16, 64> pd = aie::select(aie::zeros<bfloat16, 64>(), aie::load_v<64>(rows + 64 * db), keep);
        o.push(aie::accum<accfloat, 64>(aie::transpose(pd, 8, 8)).template to_vector<bfp16ebs8>());
    }
}

// y = yi + yi (a0 - 1) + yw a1 over n >= MIN blocks of 32 lanes (yi, y: one buffer, read before written; a0 - 1, a1
// per lane as bf16 hi + lo: per term the three products of mulff, all six on yi in one accumulator): two partial
// softmax sums merged, or one rescaled (a1 = 0). A row whose factor is 1 keeps its values exactly (y a0 as a product
// would truncate them to 16 bits: a row's numbers would depend on whether another row of its vector rescaled).
template <int MIN = 32>
[[clang::always_inline]] inline void merge(const float __aie_dm_resource_a *__restrict yi, float *__restrict y,
                                           const float __aie_dm_resource_b *__restrict yw, aie::vector<float, 32> a0,
                                           aie::vector<float, 32> a1, int n)
{
    aie::vector<bfloat16, 32> a0h, a0l, a1h, a1l;
    split2<32>(aie::sub(a0, aie::broadcast<float, 32>(1.0f)), a0h, a0l);
    split2<32>(a1, a1h, a1l);
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN)
    for (int db = 0; db < n; db++) {
        const aie::vector<float, 32> u = aie::load_v<32>(yi), w = aie::load_v<32>(yw);
        yi += 32;
        yw += 32;
        aie::vector<bfloat16, 32> uh, ul, wh, wl;
        split2<32>(u, uh, ul);
        split2<32>(w, wh, wl);
        aie::accum<accfloat, 32> acc(u);
        acc = aie::mac(acc, uh, a0h);
        acc = aie::mac(acc, uh, a0l);
        acc = aie::mac(acc, ul, a0h);
        acc = aie::mac(acc, wh, a1h);
        acc = aie::mac(acc, wh, a1l);
        acc = aie::mac(acc, wl, a1h);
        aie::store_v(y, acc.template to_vector<float>());
        y += 32;
    }
}

}  // namespace al
