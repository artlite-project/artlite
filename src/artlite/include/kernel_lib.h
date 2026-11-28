// Kernel helpers shared by the operators' templates (namespace al); math functions with variants (exp2,
// reciprocal, expm1, softplus, sigmoid, SiLU, GELU) are generated per contract in operators/kmath/.
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
// register between them and ran at II 6-9. Up to 8 loads (of 64 bytes or less) the copy is
// unrolled: the compiler keeps a short constant loop as a zero-overhead loop (~180 bytes, also in
// minsize code), while straight-line code takes ~8 bytes per load and 2-7 cycles less. Not beyond:
// when dst and src are offsets of one base (r, r + 1024) the straight-line accesses take an offset
// register each (a movxm per access) and lose to the loop from ~10 loads on. When both addresses
// need arithmetic, computing src before the call saves a bundle (register allocation).
// (Pointer helpers are templates on the pointee type: a __aie_dm_resource_x qualifier is an
// address space, and a plain T * parameter casts it away.)
template <int N, int V, typename TD, typename TS>
[[clang::always_inline]] inline void copy(TD *__restrict dst, const TS *__restrict src)
{
    static_assert(N % V == 0, "whole vectors");
    constexpr int LOADS = N / V * ((V * int(sizeof(TS)) + 63) / 64);
    AIE_LOOP_UNROLL(LOADS <= 8 ? N / V : 1)
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

// The bf16 hi / lo parts of fp32 x: x = hi + lo to ~2^-16 (x - hi is exact in fp32); of an accumulator, the second
// form, or split2_msc (the lo part as one msc(x, hi, 1): the same values) where that one schedules better
template <unsigned N>
[[clang::always_inline]] inline void split2(aie::vector<float, N> x, aie::vector<bfloat16, N> &hi, aie::vector<bfloat16, N> &lo)
{
    hi = aie::accum<accfloat, N>(x).template to_vector<bfloat16>();
    lo = aie::accum<accfloat, N>(aie::sub(x, aie::accum<accfloat, N>(hi).template to_vector<float>())).template to_vector<bfloat16>();
}

template <unsigned N>
[[clang::always_inline]] inline void split2(aie::accum<accfloat, N> x, aie::vector<bfloat16, N> &hi, aie::vector<bfloat16, N> &lo)
{
    hi = x.template to_vector<bfloat16>();
    lo = aie::accum<accfloat, N>(aie::sub(x.template to_vector<float>(), aie::accum<accfloat, N>(hi).template to_vector<float>()))
             .template to_vector<bfloat16>();
}

template <unsigned N>
[[clang::always_inline]] inline void split2_msc(aie::accum<accfloat, N> x, aie::vector<bfloat16, N> &hi, aie::vector<bfloat16, N> &lo)
{
    hi = x.template to_vector<bfloat16>();
    lo = aie::msc(x, hi, aie::broadcast<bfloat16, N>(1.0f)).template to_vector<bfloat16>();
}

// x * c for an exact bf16 c without the emulated fp32 multiply (~2^-16 rel. error)
template <unsigned N>
[[clang::always_inline]] inline aie::accum<accfloat, N> mulc(aie::vector<float, N> x, aie::vector<bfloat16, N> c)
{
    aie::vector<bfloat16, N> hi, lo;
    split2<N>(x, hi, lo);
    return aie::mac(aie::mul(hi, c), lo, c);
}

// x * C for an fp32 constant C = Hi + Lo given as its bf16 parts (three products)
template <unsigned N, float Hi, float Lo>
[[clang::always_inline]] inline aie::vector<float, N> mul_hilo(aie::vector<float, N> x)
{
    const aie::vector<bfloat16, N> l_hi = aie::broadcast<bfloat16, N>(Hi);
    const aie::vector<bfloat16, N> l_lo = aie::broadcast<bfloat16, N>(Lo);
    aie::vector<bfloat16, N> hi, lo;
    split2<N>(x, hi, lo);
    return aie::mac(aie::mac(aie::mul(hi, l_hi), lo, l_hi), hi, l_lo).template to_vector<float>();
}

// x * log2(e)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> log2e(aie::vector<float, N> x)
{
    return mul_hilo<N, 1.4453125f, -0.00262451171875f>(x);
}

// a * b for fp32 a, b via bf16 hi / lo splits (~2^-16 relative error)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> mulff(aie::vector<float, N> a, aie::vector<float, N> b)
{
    aie::vector<bfloat16, N> a_hi, a_lo, b_hi, b_lo;
    split2<N>(a, a_hi, a_lo);
    split2<N>(b, b_hi, b_lo);
    return aie::mac(aie::mac(aie::mul(a_hi, b_hi), a_hi, b_lo), a_lo, b_hi).template to_vector<float>();
}

// out = in a for N vectors of L fp32 (mulff), walked. in and out may be one buffer (each vector is read before it is
// written): as distinct parameters a store and the next iteration's load are independent (the attention rescale's
// o in place through one pointer: II 31, here 14)
template <int N, unsigned L>
[[clang::always_inline]] inline void scale(const float *__restrict in, float *__restrict out, aie::vector<float, L> a)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < N; i++) {
        aie::store_v(out, mulff<L>(aie::load_v<L>(in), a));
        in += L;
        out += L;
    }
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
    split2<N>(aie::broadcast<float, N>(v), hi, lo);
}

// 1 / sqrt(ms + eps) in lanes 0 .. K - 1 (the others 0): one scalar invsqrt per lane
template <unsigned K>
[[clang::always_inline]] inline aie::vector<float, 16> inv_sqrt_eps(aie::accum<accfloat, 16> ms, float eps)
{
    const aie::vector<float, 16> m = aie::add(ms, aie::accum<accfloat, 16>(aie::broadcast<float, 16>(eps))).template to_vector<float>();
    aie::vector<float, 16> r = aie::zeros<float, 16>();
    AIE_LOOP_UNROLL_FULL
    for (unsigned i = 0; i < K; i++)
        r[i] = aie::invsqrt(m[i]);
    return r;
}

// 1 / sqrt(ss / N + eps) of the sums of squares ss of N values, lanes 0 .. K - 1 (the others 0). The scalar unit has
// no floating point (a scalar product or sum is a soft-float call, ~1 KB of program memory each), so the mean and eps
// go through vector lanes: ss / N as bf16 products (~2^-16; 1 / N is a bf16 for N a power of two, else hi / lo parts).
// Several sums at once share that chain (8 tokens' norms: ~280 cycles less than 8 scalar calls).
template <unsigned N, unsigned K>
[[clang::always_inline]] inline aie::vector<float, 16> inv_rms(aie::vector<float, 16> ss, float eps)
{
    if constexpr ((N & (N - 1)) == 0)
        return inv_sqrt_eps<K>(mulc<16>(ss, aie::broadcast<bfloat16, 16>(1.0f / N)), eps);
    else
        return inv_sqrt_eps<K>(aie::accum<accfloat, 16>(mulff<16>(ss, aie::broadcast<float, 16>(1.0f / N))), eps);
}

// inv_rms of a length known at run time: inv_n = 1 / N (as inv_rms<N> for N not a power of two)
template <unsigned K>
[[clang::always_inline]] inline aie::vector<float, 16> inv_rms(aie::vector<float, 16> ss, float inv_n, float eps)
{
    return inv_sqrt_eps<K>(aie::accum<accfloat, 16>(mulff<16>(ss, aie::broadcast<float, 16>(inv_n))), eps);
}

// one sum
template <unsigned N>
[[clang::always_inline]] inline float inv_rms(float ss, float eps)
{
    return inv_rms<N, 1>(aie::broadcast<float, 16>(ss), eps)[0];
}

[[clang::always_inline]] inline float inv_rms(float ss, float inv_n, float eps)
{
    return inv_rms<1>(aie::broadcast<float, 16>(ss), inv_n, eps)[0];
}

// ---------------------------------------------------------------------------------------------
// LUT activations (exp2, reciprocal, SiLU, GELU, sigmoid in software: operators/kmath/; the hardware vexp2
// is the chord 2^n (1 + f): up to +6% on bf16 inputs; vtanh is a coarse LUT)
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// Vectors from 128-bit pieces
// ---------------------------------------------------------------------------------------------

// aie::concat with 128-bit pieces (8 bf16 / int16, 16 int8, 4 fp32 / int32) built by vshuffles: T128_2x4_lo of two
// pieces (padded to 512 bits) holds [a b] in its low 256 bits and T256_2x2_lo of two such pairs [a b c d], so 2 pieces
// take 1 vshuffle, 4 take 3 and 8 take 6. aie::concat takes a vshift per piece (aie_api's shift_bytes chain), a
// vector's inserts of 128-bit pieces vshifts and vsels (Peano's insert), and joining 256-bit results adds moves: 8 rows
// into 64 lanes cost 8 vshifts + 2 vmovs or 4 vshifts + 4 vsels + 2 vmovs, all move-slot ops (shifts, selects and
// shuffles also take the rm port). Other pieces go to aie::concat (256-bit pieces are register halves).
namespace detail {
template <typename T, unsigned N>
[[clang::always_inline]] inline v16int32 pad128(aie::vector<T, N> v)
{
    return set_v16int32(0, (v4int32)v.to_native());
}

template <typename T, unsigned N>
[[clang::always_inline]] inline v16int32 cat2(aie::vector<T, N> a, aie::vector<T, N> b)
{
    return ::shuffle(pad128(a), pad128(b), T128_2x4_lo);
}

template <typename T, unsigned N>
[[clang::always_inline]] inline aie::vector<T, 4 * N> cat4(aie::vector<T, N> a, aie::vector<T, N> b,
                                                           aie::vector<T, N> c, aie::vector<T, N> d)
{
    return aie::vector<int32, 16>(::shuffle(cat2(a, b), cat2(c, d), T256_2x2_lo)).template cast_to<T>();
}
}  // namespace detail

template <typename T, unsigned N, typename... V>
[[clang::always_inline]] inline auto concat(aie::vector<T, N> a, V... v)
{
    constexpr unsigned K = 1 + sizeof...(V);
    if constexpr (sizeof(T) * N != 16) {
        return aie::concat(a, v...);
    } else {
        static_assert(K == 2 || K == 4 || K == 8, "2, 4 or 8 pieces");
        const aie::vector<T, N> p[K] = {a, v...};
        if constexpr (K == 2)
            return aie::vector<int32, 16>(detail::cat2(p[0], p[1])).template cast_to<T>().template extract<2 * N>(0);
        else if constexpr (K == 4)
            return detail::cat4(p[0], p[1], p[2], p[3]);
        else
            return aie::concat(detail::cat4(p[0], p[1], p[2], p[3]), detail::cat4(p[4], p[5], p[6], p[7]));
    }
}

namespace detail {
template <int E, typename T, int... I>
[[clang::always_inline]] inline auto gather(const T *p, int pitch, std::integer_sequence<int, I...>)
{
    return al::concat(aie::load_v<E>(p + I * pitch)...);
}
}  // namespace detail

// K pieces of E elements at p, p + pitch, ..., p + (K - 1) pitch (pitch in elements; K rows of a panel) -> lanes
// [piece K][E]: al::concat of the loads
template <int K, int E, typename T>
[[clang::always_inline]] inline auto gather(const T *p, int pitch)
{
    return detail::gather<E>(p, pitch, std::make_integer_sequence<int, K>());
}

// The gather bases of a table's two copies: lanes 4 n, 4 n + 1 hold tab_ab, lanes 4 n + 2, 4 n + 3 tab_cd (the
// pointers stay the callers' values: a gather's memory operand needs a base that traces to a pointer)
[[clang::always_inline]] inline v16int32 table_b16(const uint32_t *tab_ab, const uint32_t *tab_cd)
{
    return ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
}

// ---------------------------------------------------------------------------------------------
// fp32 maximum
// ---------------------------------------------------------------------------------------------

// aie::max of fp32 vectors is emulated (an accfloat subtraction, its sign as a bf16 compare, a select: per 16 lanes 4
// mv-slot ops, a vec op, a conversion and ~10 cycles). Exact from integer compares instead: the signed order of fp32
// bits is the float order unless both are negative, where it is the reverse (max(-0, +0) = +0; no NaNs).

// 3 ops: vmax_lt.32's cmp (a < b), flipped where its maximum is negative (both are), selects
[[clang::always_inline]] inline v16int32 fmax16(v16int32 a, v16int32 b)
{
    unsigned int cmp;
    const v16int32 hi = ::max_lt(a, b, cmp);
    return ::sel(a, b, cmp ^ ::lt(hi, ::broadcast_zero_to_v16int32()));
}

// 4 ops and a cycle shorter (the minimum beside the compare, no scalar xor after it): the steps of a reduction, where
// the chain is the cost
[[clang::always_inline]] inline v16int32 fmax16_chain(v16int32 a, v16int32 b)
{
    const v16int32 hi = ::max(a, b), lo = ::min(a, b);
    return ::sel(hi, lo, ::lt(hi, ::broadcast_zero_to_v16int32()));
}

// elementwise, any N (straight-line halves: minsize code keeps even a 2-trip loop rolled)
template <unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> fmax(aie::vector<float, N> a, aie::vector<float, N> b)
{
    if constexpr (N <= 16) {
        const v16int32 x = (v16int32)a.template grow<16>().template cast_to<int32>(),
                       y = (v16int32)b.template grow<16>().template cast_to<int32>();
        return aie::vector<int32, 16>(fmax16(x, y)).template cast_to<float>().template extract<N>(0);
    } else {
        return aie::concat(fmax<N / 2>(a.template extract<N / 2>(0), b.template extract<N / 2>(0)),
                           fmax<N / 2>(a.template extract<N / 2>(1), b.template extract<N / 2>(1)));
    }
}

// ---------------------------------------------------------------------------------------------
// Lane shuffles
// ---------------------------------------------------------------------------------------------

// 16 bf16 (rows 0..15 of one group) -> the two 64-lane scale vectors of rows 0..7 and 8..15, lane
// 8 i + j = s_i (7 shuffles; upper input lanes unused). expand16_bc schedules better in the dequant loops (Q4_0 / Q4_1 /
// Q5_K: 5-13% per call); in Q4_K's header pass the two are mixed (+7% / -2%).
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
// element broadcast 4 times (vextbcst: no L_RM_PORT) and two 8x8 transposes, 2 shuffles instead of the zips' 3 - 4
// (loops bound by L_RM_PORT, shuffles and a bfp16 stream's pushes / flushes: GemmQ's dequant II 46 -> 40 per 8 panels).
[[clang::always_inline]] inline aie::vector<bfloat16, 64> expand8_of(aie::vector<bfloat16, 32> v, int e)
{
    const v32bfloat16 b = (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)v, e);
    return aie::concat(aie::vector<bfloat16, 32>(shuffle(b, b, T16_8x8_lo)),
                       aie::vector<bfloat16, 32>(shuffle(b, b, T16_8x8_hi)));
}

// 8 bf16 -> 64 lanes, each repeated 8 times (lane 8 i + j = s_i): expand8_of's 2 shuffles (three zips cost 20-60 more
// cycles per call in gdn's and Q2_K's loops)
[[clang::always_inline]] inline aie::vector<bfloat16, 64> expand8(aie::vector<bfloat16, 8> s8)
{
    return expand8_of(s8.template grow<32>(), 0);
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

// a reduction step of group8: the maximum (fmax16_chain per 16 lanes) or the sum
template <bool Max, unsigned N>
[[clang::always_inline]] inline aie::vector<float, N> max_or_add(aie::vector<float, N> a, aie::vector<float, N> b)
{
    if constexpr (!Max) {
        return aie::add(a, b);
    } else if constexpr (N <= 16) {
        const v16int32 x = (v16int32)a.template grow<16>().template cast_to<int32>(),
                       y = (v16int32)b.template grow<16>().template cast_to<int32>();
        return aie::vector<int32, 16>(fmax16_chain(x, y)).template cast_to<float>().template extract<N>(0);
    } else {
        return aie::concat(max_or_add<Max, N / 2>(a.template extract<N / 2>(0), b.template extract<N / 2>(0)),
                           max_or_add<Max, N / 2>(a.template extract<N / 2>(1), b.template extract<N / 2>(1)));
    }
}

// 64 lanes (q, k) = 8 groups of 8: reduce each group (max or sum) -> 8 lanes (q). Each group's maximum in its 8 lanes:
// max8 (one transposed chain, not group8 then bcast8)
template <bool Max>
[[clang::always_inline]] inline aie::vector<float, 8> group8(aie::vector<float, 64> v)
{
    auto [e1, o1] = aie::interleave_unzip(v.template extract<32>(0), v.template extract<32>(1), 1);
    aie::vector<float, 32> r1 = max_or_add<Max>(e1, o1);
    auto [e2, o2] = aie::interleave_unzip(r1.template extract<16>(0), r1.template extract<16>(1), 1);
    aie::vector<float, 16> r2 = max_or_add<Max>(e2, o2);
    auto [e3, o3] = aie::interleave_unzip(r2.template extract<8>(0), r2.template extract<8>(1), 1);
    return max_or_add<Max>(e3, o3);
}

// 4 values repeated (lane 4 i + k = value k, as broadcast_elem_128 leaves them) -> rows of 8 (lanes 8 k .. 8 k + 7 =
// value k): T32_8x4_lo / hi, lane 8 k + i = lane 4 i + k
[[clang::always_inline]] inline aie::vector<float, 32> spread4(v16int32 x)
{
    using V = aie::vector<int32, 16>;
    return aie::concat(V(shuffle(x, x, T32_8x4_lo)), V(shuffle(x, x, T32_8x4_hi))).template cast_to<float>();
}

// 8 lanes -> 64 lanes, lane 8 q + k = v[q]: v[0 .. 3] and v[4 .. 7] each as a 128-bit element 4 times (vextbcst),
// spread (6 ops; three zips took 7 shuffles and more cycles)
[[clang::always_inline]] inline aie::vector<float, 64> bcast8(aie::vector<float, 8> v)
{
    const v16int32 g = (v16int32)(v16float)v.template grow<16>();
    return aie::concat(spread4(broadcast_elem_128(g, 0)), spread4(broadcast_elem_128(g, 1)));
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

// The 8 block exponents of a panel, all e (a mantissa m then has the value m 2^(e - 133))
[[clang::always_inline]] inline v8int8 e8(int e)
{
    v2int32 v;
    v[0] = e * 0x01010101;
    v[1] = e * 0x01010101;
    return (v8int8)v;
}

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

// Rotations of a whole register cross rows, and extracting a lane per row costs a vextbcst and a select each. The
// transposes do not: T16_4x8 (one register) and T32_4x8_lo / hi (a pair) put row r, column c in lane 4 c + r, so a row
// lies on the lanes = r mod 4 and rotations by multiples of 4 lanes reduce rows without mixing them; T16_8x4 / T32_8x4
// (lane 8 k + i = lane 4 i + k) bring 4 values, each in the lanes of its residue, back to rows of 8. No vshuffle mode
// keeps a 16-bit lane in its row of 8 and moves every lane to another pair, so bf16 needs at least 5 permutes and 3
// maxima (max8 takes 8 mv-slot ops; the extraction form took 14).

// lanes 8 r .. 8 r + 7 = v[4 E + r] (r < 4): those 4 values as a 128-bit element 4 times (vextbcst), spread
template <int E = 0>
[[clang::always_inline]] inline aie::vector<float, 32> rows4(aie::vector<float, 16> v)
{
    return spread4(broadcast_elem_128((v16int32)(v16float)v, E));
}

// per row r (lanes 8 r .. 8 r + 7): the maximum of its 8 lanes, in all 8. bf16, N = 16 or 32 (one register):
// rotations by 16, 8, 4 lanes of the transpose. fp32, N = 32: the maximum of a pair's two transposes (columns 0 .. 3,
// 4 .. 7), then rotations by 8 and 4. N = 64: two such pairs' maxima, columns c and c + 2 of both by T256_2x2_lo / hi
// (lanes 0 .. 7 / 8 .. 15 of each), c and c + 1 by a rotation of 4 lanes (rows 0 .. 3 in lanes 0 .. 3, rows 4 .. 7 in
// lanes 8 .. 11)
template <typename T, unsigned N>
[[clang::always_inline]] inline aie::vector<T, N> max8(aie::vector<T, N> v)
{
    if constexpr (std::is_same_v<T, bfloat16>) {
        static_assert(N == 16 || N == 32);
        v32bfloat16 x = shuffle((v32bfloat16)v.template grow<32>(), T16_4x8);
        x = ::max(x, shift_bytes(x, x, 32));
        x = ::max(x, shift_bytes(x, x, 16));
        x = ::max(x, shift_bytes(x, x, 8));
        return aie::vector<bfloat16, 32>(shuffle(x, T16_8x4)).template extract<N>(0);
    } else if constexpr (N == 32) {
        const aie::vector<int32, 32> i = v.template cast_to<int32>();
        const v16int32 a = i.template extract<16>(0), b = i.template extract<16>(1);
        v16int32 x = fmax16_chain(shuffle(a, b, T32_4x8_lo), shuffle(a, b, T32_4x8_hi));
        x = fmax16_chain(x, shift_bytes(x, x, 32));
        return spread4(fmax16_chain(x, shift_bytes(x, x, 16)));
    } else {
        static_assert(N == 64);
        const aie::vector<int32, 64> i = v.template cast_to<int32>();
        const v16int32 r0 = i.template extract<16>(0), r1 = i.template extract<16>(1), r2 = i.template extract<16>(2),
                       r3 = i.template extract<16>(3);
        const v16int32 s01 = fmax16_chain(shuffle(r0, r1, T32_4x8_lo), shuffle(r0, r1, T32_4x8_hi));
        const v16int32 s23 = fmax16_chain(shuffle(r2, r3, T32_4x8_lo), shuffle(r2, r3, T32_4x8_hi));
        const v16int32 t = fmax16_chain(shuffle(s01, s23, T256_2x2_lo), shuffle(s01, s23, T256_2x2_hi));
        const v16int32 m = fmax16_chain(t, shift_bytes(t, t, 16));
        return aie::concat(spread4(broadcast_elem_128(m, 0)), spread4(broadcast_elem_128(m, 2)));
    }
}

// per row r < 4 (lanes 8 r ..): the sum of its 8 lanes in lane 4 r (rows4_inv's layout; the other lanes partial sums),
// added as ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)): the even and odd lanes of the pair (T32_16x2: lane 4 r + m holds
// row r's pair m), then rotations by 1 and 2 lanes within each row's 4
[[clang::always_inline]] inline aie::vector<float, 16> sum8(aie::vector<float, 32> v)
{
    using V = aie::vector<float, 16>;
    const v16float a = v.template extract<16>(0), b = v.template extract<16>(1);
    V s = aie::add(V(shuffle(a, b, T32_16x2_lo)), V(shuffle(a, b, T32_16x2_hi)));
    s = aie::add(s, V(shift_bytes((v16float)s, (v16float)s, 4)));
    return aie::add(s, V(shift_bytes((v16float)s, (v16float)s, 8)));
}

// lanes 8 r .. 8 r + 7 = 1 / v[4 r] (r < 4, sum8's layout): 4 scalar inverses (aie::inv of a vector takes one per lane),
// each broadcast from its register, two rows a register by a select
[[clang::always_inline]] inline aie::vector<float, 32> rows4_inv(aie::vector<float, 16> v)
{
    const auto pair = [&](int r) __attribute__((always_inline)) {
        const aie::vector<float, 16> lo = aie::broadcast<float, 16>(aie::inv(v[4 * r])),
                                     hi = aie::broadcast<float, 16>(aie::inv(v[4 * r + 4]));
        return aie::select(lo, hi, aie::mask<16>::from_uint32(0xFF00u));
    };
    return aie::concat(pair(0), pair(2));
}

// the maximum of lanes 0 .. K - 1 of 16 fp32 lanes (K a power of 2), in all 16: rotations by K / 2 .. 1 leave it in
// lane 0 (in every lane for K = 16), then a broadcast
template <unsigned K = 16>
[[clang::always_inline]] inline aie::vector<float, 16> max16(aie::vector<float, 16> v)
{
    static_assert(K >= 2 && K <= 16 && (K & (K - 1)) == 0);
    v16int32 x = (v16int32)v.template cast_to<int32>();
    if constexpr (K == 16)
        x = fmax16_chain(x, shift_bytes(x, x, 32));
    if constexpr (K >= 8)
        x = fmax16_chain(x, shift_bytes(x, x, 16));
    if constexpr (K >= 4)
        x = fmax16_chain(x, shift_bytes(x, x, 8));
    const aie::vector<float, 16> m = aie::vector<int32, 16>(fmax16_chain(x, shift_bytes(x, x, 4))).template cast_to<float>();
    if constexpr (K == 16)
        return m;
    else
        return aie::broadcast<float, 16>(m[0]);
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

// row (ND * 8 fp32) -> row r of the stage [d-block ND][r 8][8]
template <int ND>
[[clang::always_inline]] inline void stage_row(const float *__restrict row, float *__restrict st, int r)
{
    float *__restrict d = st + 8 * r;
    for (int db = 0; db < ND; db++) {
        aie::store_v(d, aie::load_v<8>(row));
        row += 8;
        d += 64;
    }
}

// the stage [d-block ND][r 8][8] -> per d-block the hi panel and the lo panel: one stream (hi, lo per d-block), or
// two ([hi panels ND | lo panels ND], the lo ones at out + 72 ND)
template <int ND, bool INTERLEAVED>
[[clang::always_inline]] inline void encode_hilo(const float *__restrict st, uint8_t *__restrict out)
{
    const BV nI = neg_eye();
    BOut<> ho((bfp16ebs8 *)out);
    if constexpr (INTERLEAVED) {
        for (int db = 0; db < ND; db++) {
            BV h, l;
            split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
            ho.push(h);
            ho.push(l);
            st += 64;
        }
    } else {
        BOut<> lo((bfp16ebs8 *)(out + 72 * ND));
        for (int db = 0; db < ND; db++) {
            BV h, l;
            split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
            ho.push(h);
            lo.push(l);
            st += 64;
        }
    }
}

// the stage [d-block ND][r 8][8] -> per d-block the single bfp16 panel
template <int ND>
[[clang::always_inline]] inline void encode_panel(const float *__restrict st, uint8_t *__restrict out)
{
    BOut<> o((bfp16ebs8 *)out);
    for (int db = 0; db < ND; db++) {
        o.push(aie::accum<accfloat, 64>(aie::load_v<64>(st)).template to_vector<bfp16ebs8>());
        st += 64;
    }
}

// The V panel of d-block db of 8 tokens' rows (bf16, HD per row): per dim of the 8 a block over the positions
template <int HD>
[[clang::always_inline]] inline BV vpan(const bfloat16 *__restrict v, int db)
{
    const aie::vector<bfloat16, 64> pd = gather<8, 8>(v + 8 * db, HD);  // [position][dim]
    return aie::accum<accfloat, 64>(aie::transpose(pd, 8, 8)).template to_vector<bfp16ebs8>();
}

// C += A (hi + lo) B (hi + lo)^T: three products (a, b: hi and lo panels)
[[clang::always_inline]] inline void mm3(MM8 &c, const BV &ah, const BV &al_, const BV &bh, const BV &bl)
{
    c.mac(ah, aie::op_transpose(bh));
    c.mac(ah, aie::op_transpose(bl));
    c.mac(al_, aie::op_transpose(bh));
}

// x (64 fp32) -> the panels hi, lo (bfp16 of the 8 x 8 block, 8 values per block along the lanes' fast axis) pushed at out
[[clang::always_inline]] inline void push2(aie::accum<accfloat, 64> x, const BV &nI, uint8_t *out)
{
    BV h, l;
    split_hilo(x, nI, h, l);
    BOut<> po((bfp16ebs8 *)out);
    po.push(h);
    po.push(l);
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

// 16 consecutive rows of a run of tiles of R rows (8 <= R <= 16), each tile's rows in lanes 0 .. R - 1 of a vector:
// rows A .. R - 1 of x's tile (A < R), then the first rows of y's (the next tile's); constant lane shifts
template <int R, int A, typename T>
[[clang::always_inline]] inline aie::vector<T, 16> join_rows(aie::vector<T, 16> x, aie::vector<T, 16> y)
{
    static_assert(8 <= R && R <= 16 && 0 <= A && A < R && 16 - (R - A) <= R);
    if constexpr (R == 16 && A == 0)
        return x;
    else if constexpr (R == 16)
        return aie::shuffle_down_fill(x, y, A);
    else
        return aie::shuffle_down_fill(aie::shuffle_up(x, 16 - R), y, A + 16 - R);
}

}  // namespace al
