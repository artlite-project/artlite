#pragma once

// The bfp16ebs8 blocks of 16 fp32 values (``operators.bfp16.blocks``) as AVX-512 code: for the weight packers
// (tile_pack.cpp, slot_pack.cpp), compiled with -march=znver5 (CMakeLists.txt).

#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace artlite::pack {

/// 16 values (two blocks of 8) -> their bfp16ebs8 blocks as ``operators.bfp16.blocks`` rounds them: the exponent of
/// the block's maximum (at least 2^-120), mantissas rint(x 2^(6 - e)), a positive maximum rounded up to 128 raises
/// the exponent (a negative one stays at -128), an all-zero block has exponent byte 0.
inline void bfp16_blocks(__m512 x, uint8_t* blk0, uint8_t* blk1) {
  const __m512i bits = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7FFFFFFF));
  __m512i mx = _mm512_max_epu32(bits, _mm512_shuffle_epi32(bits, _MM_PERM_CDAB));
  mx = _mm512_max_epu32(mx, _mm512_shuffle_epi32(mx, _MM_PERM_BADC));
  mx = _mm512_max_epu32(mx, _mm512_shuffle_i32x4(mx, mx, _MM_SHUFFLE(2, 3, 0, 1)));  // (per 8 lanes)
  __m512i e = _mm512_max_epi32(_mm512_sub_epi32(_mm512_srli_epi32(mx, 23), _mm512_set1_epi32(127)),
                               _mm512_set1_epi32(-120));
  auto mant = [&](__m512i ee) {
    const __m512 scale = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(133), ee), 23));
    return _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(x, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  };
  __m512i m = mant(e);
  const __mmask16 over = _mm512_cmpgt_epi32_mask(m, _mm512_set1_epi32(127));
  if (over) {
    const __mmask16 raise = static_cast<__mmask16>(((over & 0xFF) ? 0xFF : 0) | ((over & 0xFF00) ? 0xFF00 : 0));
    e = _mm512_mask_add_epi32(e, raise, e, _mm512_set1_epi32(1));
    m = _mm512_mask_mov_epi32(m, raise, mant(e));
  }
  const __m128i mb = _mm512_cvtepi32_epi8(m);
  const int e0 = _mm512_cvtsi512_si32(e), e1 = _mm_extract_epi32(_mm512_extracti32x4_epi32(e, 2), 0);
  const uint32_t a0 = static_cast<uint32_t>(_mm512_cvtsi512_si32(mx));
  const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(_mm512_extracti32x4_epi32(mx, 2), 0));
  blk0[0] = a0 ? static_cast<uint8_t>(e0 + 127) : 0;
  blk1[0] = a1 ? static_cast<uint8_t>(e1 + 127) : 0;
  std::memcpy(blk0 + 1, &mb, 8);
  std::memcpy(blk1 + 1, reinterpret_cast<const uint8_t*>(&mb) + 8, 8);
}

/// The values of two bfp16ebs8 blocks (``operators.bfp16.block_values``): m 2^(E - 133).
inline __m512 bfp16_values(const uint8_t* blk0, const uint8_t* blk1) {
  int64_t m0, m1;
  std::memcpy(&m0, blk0 + 1, 8);
  std::memcpy(&m1, blk1 + 1, 8);
  const __m512 m = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_set_epi64x(m1, m0)));
  const __m256 e0 = _mm256_set1_ps(float(blk0[0]) - 133.0f), e1 = _mm256_set1_ps(float(blk1[0]) - 133.0f);
  return _mm512_scalef_ps(m, _mm512_insertf32x8(_mm512_castps256_ps512(e0), e1, 1));
}

}  // namespace artlite::pack
