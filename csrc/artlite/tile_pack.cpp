// The tile packers of ``artlite.operators.formats`` (q4k.py, q5k.py, q6k.py, q4.py, q8.py, q2k.py, q3k.py, bf16.py:
// ``pack``) on GGUF blocks (and F32 rows: BFP16 planes), with AVX-512 VBMI / GFNI for the K-quants' code planes. A super-block's 256 codes go out in the tiles' word-major order
// (``formats.common._word_order``: [sb][s][i][j] = k 256 sb + 32 i + 8 s + j, two codes per byte, the lower k in the
// low nibble), so byte 32 s + 4 i + jj of a super-block's low-nibble planes holds codes 32 i + 8 s + 2 jj (+ 1).
// Compiled with -march=znver5 (CMakeLists.txt).

#include "artlite/tile_pack.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <unordered_map>

#include "artlite/error.h"
#include "artlite/platform.h"

namespace artlite {

namespace {

constexpr int kMaxNsb = 16;
// per 256 columns: Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, Q4_1, Q2_K, Q3_K, BF16, F16, F32
constexpr size_t kBlockBytes[] = {144, 176, 210, 144, 272, 160, 84, 110, 512, 512, 1024};
constexpr int64_t kItemRows = 2048;                // rows of a work item
constexpr size_t kPrefetchWindow = size_t(2) << 30;
constexpr size_t kPrefetchPiece = size_t(8) << 20;
constexpr unsigned kPrefetchThreads = 16;

alignas(64) const uint8_t kZeroRow[kMaxNsb * 1024] = {};  // a padding row's blocks (-1)

using Idx = std::array<uint8_t, 64>;

/// vpermi2b indices of a super-block's low-nibble planes (bytes 64 half .. 64 half + 63) from the nibble pairs of
/// its 128 code bytes (``nibble_pairs``: byte 2 m + h = nibble h of code bytes 2 m and 2 m + 1). Q4_K / Q5_K: code
/// 32 i + l is nibble i % 2 of byte 32 (i / 2) + l; Q6_K: nibble (i / 2) % 2 of byte 64 (i / 4) + 32 (i % 2) + l.
constexpr Idx low_planes(bool q6k, int half) {
  Idx idx{};
  for (int o = 0; o < 64; o++) {
    int b = 64 * half + o, s = b >> 5, i = (b >> 2) & 7, jj = b & 3;
    int byte = q6k ? 64 * (i >> 2) + 32 * (i & 1) + 8 * s + 2 * jj : 32 * (i >> 1) + 8 * s + 2 * jj;
    int nib = q6k ? (i >> 1) & 1 : i & 1;
    idx[o] = static_cast<uint8_t>(byte + nib);  // (2 m + h with m = byte / 2)
  }
  return idx;
}

constexpr Idx kQ4Lo = low_planes(false, 0), kQ4Hi = low_planes(false, 1);
constexpr Idx kQ6Lo = low_planes(true, 0), kQ6Hi = low_planes(true, 1);

/// Q5_K's high-bit plane: byte 4 i + jj = bits i of qh[8 s + 2 jj] (s = 0 .. 3: the low nibble) and of
/// qh[8 s + 2 jj + 1] (the high one), a transposed 8 x 8 bit matrix per jj (GF2P8AFFINEQB with the identity as
/// data: result byte i bit k = matrix byte 7 - k bit i). kQ5Gather: the matrices (byte 7 - k of qword jj = row k
/// = qh[8 (k % 4) + 2 jj + k / 4]); kQ5Out: byte 4 i + jj <- byte 8 jj + i of the transposes.
constexpr Idx q5_gather() {
  Idx idx{};
  for (int jj = 0; jj < 4; jj++)
    for (int b = 0; b < 8; b++) {
      int k = 7 - b;
      idx[8 * jj + b] = static_cast<uint8_t>(8 * (k & 3) + 2 * jj + (k >> 2));
    }
  return idx;
}

constexpr Idx q5_out() {
  Idx idx{};
  for (int i = 0; i < 8; i++)
    for (int jj = 0; jj < 4; jj++) idx[4 * i + jj] = static_cast<uint8_t>(8 * jj + i);
  return idx;
}

constexpr Idx kQ5Gather = q5_gather(), kQ5Out = q5_out();

/// Q6_K's high-crumb planes h = 0, 1 (byte 32 h + 4 i + jj, i = 4 n + m): crumbs m of a = qh[32 n + 16 h + 2 jj],
/// b = a + 8 bytes, a' = a + 1 byte, b' = a + 9 bytes at bits 0, 2, 4, 6. Per (n, h, jj) a qword of the rows
/// [a a b b a' a' b' b'] transposed: byte 2 m + e bit 2 t + e = bit 2 m + e of input t, so byte m of the result =
/// (byte 2 m & 0x55) | (byte 2 m + 1 & 0xAA). kQ6Gather[n]: the matrices of qwords 4 h + jj (rows reversed, as
/// for Q5_K); kQ6Out: byte 32 h + 4 i + jj <- byte 64 n + 8 (4 h + jj) + 2 m of the combined transposes.
constexpr Idx q6_gather(int n) {
  constexpr int off[4] = {0, 8, 1, 9};
  Idx idx{};
  for (int q = 0; q < 8; q++)
    for (int b = 0; b < 8; b++) {
      int h = q >> 2, jj = q & 3, k = 7 - b;
      idx[8 * q + b] = static_cast<uint8_t>(32 * n + 16 * h + 2 * jj + off[k >> 1]);
    }
  return idx;
}

constexpr Idx q6_out() {
  Idx idx{};
  for (int h = 0; h < 2; h++)
    for (int i = 0; i < 8; i++)
      for (int jj = 0; jj < 4; jj++)
        idx[32 * h + 4 * i + jj] = static_cast<uint8_t>(64 * (i >> 2) + 8 * (4 * h + jj) + 2 * (i & 3));
  return idx;
}

constexpr Idx kQ6Gather0 = q6_gather(0), kQ6Gather1 = q6_gather(1), kQ6Out = q6_out();

inline __m512i load_idx(const Idx& idx) { return _mm512_loadu_si512(idx.data()); }

/// 64 code bytes -> per 16-bit lane m: byte 0 = the low nibbles of bytes 2 m, 2 m + 1 (low first), byte 1 = their
/// high nibbles (the middle nibbles of the lane swapped).
inline __m512i nibble_pairs(__m512i w) {
  const __m512i m_mid = _mm512_set1_epi16(0x00F0), m_keep = _mm512_set1_epi16(static_cast<short>(0xF00F));
  __m512i mid = _mm512_ternarylogic_epi64(m_mid, _mm512_srli_epi16(w, 4), _mm512_slli_epi16(w, 4), 0xCA);
  return _mm512_ternarylogic_epi64(m_keep, w, mid, 0xCA);  // (a ? b : c)
}

/// bf16 bits of an fp16 value (as fp32, rounded to nearest even), negated first with ``neg``.
inline uint16_t bf16_of(const uint8_t* f16, bool neg = false) {
  uint16_t h;
  std::memcpy(&h, f16, 2);
  float f = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));  // (_cvtsh_ss: not in MSVC)
  if (neg) f = -f;
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

inline void put16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }

/// The 6-bit scales and mins of a Q4_K / Q5_K super-block (its 12 scale bytes).
inline void scales_k(const uint8_t* s, uint8_t* sc, uint8_t* mn) {
  for (int j = 0; j < 4; j++) {
    sc[j] = s[j] & 63;
    mn[j] = s[j + 4] & 63;
    sc[j + 4] = (s[j + 8] & 0xF) | ((s[j] >> 6) << 4);
    mn[j + 4] = (s[j + 8] >> 4) | ((s[j + 4] >> 6) << 4);
  }
}

inline uint8_t nib(uint8_t lo, uint8_t hi) { return static_cast<uint8_t>((lo & 15) | (hi << 4)); }

/// Q4_K (q4k.py): per row [sl | ml | sh | mh | d | -dmin] (16 nsb bytes), then [sb][row] 128 code bytes.
void tile_q4k(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  const __m512i lo = load_idx(kQ4Lo), hi = load_idx(kQ4Hi);
  const int g = 8 * nsb;
  uint8_t* codes = t + 16 * nsb * R;
  for (int r = 0; r < R; r++) {
    uint8_t sc[8 * kMaxNsb], mn[8 * kMaxNsb];
    uint8_t* h = t + 16 * nsb * r;
    for (int sb = 0; sb < nsb; sb++) {
      const uint8_t* b = src[r] + 144 * sb;
      scales_k(b + 4, sc + 8 * sb, mn + 8 * sb);
      put16(h + 12 * nsb + 2 * sb, bf16_of(b));
      put16(h + 14 * nsb + 2 * sb, bf16_of(b + 2, true));
      __m512i x0 = nibble_pairs(_mm512_loadu_si512(b + 16)), x1 = nibble_pairs(_mm512_loadu_si512(b + 80));
      uint8_t* c = codes + (size_t(sb) * R + r) * 128;
      _mm512_storeu_si512(c, _mm512_permutex2var_epi8(x0, lo, x1));
      _mm512_storeu_si512(c + 64, _mm512_permutex2var_epi8(x0, hi, x1));
    }
    for (int k = 0; k < g / 2; k++) {
      h[k] = nib(sc[2 * k], sc[2 * k + 1] & 15);
      h[4 * nsb + k] = nib(mn[2 * k], mn[2 * k + 1] & 15);
    }
    auto top = [&](const uint8_t* v, int k) { return static_cast<uint8_t>((v[k] >> 4) + 4 * (v[k + g / 2] >> 4)); };
    for (int k = 0; k < g / 4; k++) {
      h[8 * nsb + k] = nib(top(sc, 2 * k), top(sc, 2 * k + 1));
      h[10 * nsb + k] = nib(top(mn, 2 * k), top(mn, 2 * k + 1));
    }
  }
}

/// Q5_K (q5k.py): [pair][row] 32-byte records [sc lo 2 x 4 | mn lo 2 x 4 | crumbs 2 x 4 | d 2 | -dmin 2], then
/// [sb][plane: 4 low-nibble planes, the high bits][row] 32 bytes.
void tile_q5k(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  const __m512i lo = load_idx(kQ4Lo), hi = load_idx(kQ4Hi);
  const __m256i gather = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kQ5Gather.data()));
  const __m256i order = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kQ5Out.data()));
  const __m256i ident = _mm256_set1_epi64x(0x8040201008040201);
  uint8_t* codes = t + 16 * nsb * R;
  for (int r = 0; r < R; r++) {
    uint8_t sc[8 * kMaxNsb], mn[8 * kMaxNsb];
    for (int sb = 0; sb < nsb; sb++) {
      const uint8_t* b = src[r] + 176 * sb;
      scales_k(b + 4, sc + 8 * sb, mn + 8 * sb);
      uint8_t* rec = t + (size_t(sb / 2) * R + r) * 32;
      put16(rec + 24 + 2 * (sb & 1), bf16_of(b));
      put16(rec + 28 + 2 * (sb & 1), bf16_of(b + 2, true));
      __m512i x0 = nibble_pairs(_mm512_loadu_si512(b + 48)), x1 = nibble_pairs(_mm512_loadu_si512(b + 112));
      __m512i p01 = _mm512_permutex2var_epi8(x0, lo, x1), p23 = _mm512_permutex2var_epi8(x0, hi, x1);
      auto plane = [&](int p) { return reinterpret_cast<__m256i*>(codes + ((size_t(sb) * 5 + p) * R + r) * 32); };
      _mm256_storeu_si256(plane(0), _mm512_castsi512_si256(p01));
      _mm256_storeu_si256(plane(1), _mm512_extracti64x4_epi64(p01, 1));
      _mm256_storeu_si256(plane(2), _mm512_castsi512_si256(p23));
      _mm256_storeu_si256(plane(3), _mm512_extracti64x4_epi64(p23, 1));
      __m256i m = _mm256_permutexvar_epi8(gather, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16)));
      _mm256_storeu_si256(plane(4), _mm256_permutexvar_epi8(order, _mm256_gf2p8affine_epi64_epi8(ident, m, 0)));
    }
    for (int sb = 0; sb < nsb; sb++) {
      uint8_t* rec = t + (size_t(sb / 2) * R + r) * 32 + 4 * (sb & 1);
      const uint8_t *s = sc + 8 * sb, *n = mn + 8 * sb;
      for (int k = 0; k < 4; k++) {
        rec[k] = nib(s[2 * k], s[2 * k + 1] & 15);
        rec[8 + k] = nib(n[2 * k], n[2 * k + 1] & 15);
        rec[16 + k] = nib((s[2 * k] >> 4) | (n[2 * k] >> 4) << 2, (s[2 * k + 1] >> 4) | (n[2 * k + 1] >> 4) << 2);
      }
    }
  }
}

/// Q6_K (q6k.py): [row][sb] bf16 d (padded to 32 bytes), [pair][row] 32-byte scale planes ([sb % 2][h][i] = sc[2 i +
/// h]), then [half super-block 2 sb + h][plane: low-nibble planes s = 2 h, 2 h + 1, the high crumbs][row] 32 bytes.
void tile_q6k(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  const __m512i lo = load_idx(kQ6Lo), hi = load_idx(kQ6Hi);
  const __m512i g0 = load_idx(kQ6Gather0), g1 = load_idx(kQ6Gather1), order = load_idx(kQ6Out);
  const __m512i ident = _mm512_set1_epi64(0x8040201008040201), even = _mm512_set1_epi16(0x0055);
  const size_t n_d = (2 * size_t(nsb) * R + 31) / 32 * 32;
  uint8_t* planes = t + n_d;
  uint8_t* halves = planes + 16 * nsb * R;
  for (int r = 0; r < R; r++) {
    for (int sb = 0; sb < nsb; sb++) {
      const uint8_t* b = src[r] + 210 * sb;
      put16(t + 2 * (size_t(nsb) * r + sb), bf16_of(b + 208));
      __m512i x0 = nibble_pairs(_mm512_loadu_si512(b)), x1 = nibble_pairs(_mm512_loadu_si512(b + 64));
      __m512i p01 = _mm512_permutex2var_epi8(x0, lo, x1), p23 = _mm512_permutex2var_epi8(x0, hi, x1);
      auto plane = [&](int h, int p) {
        return reinterpret_cast<__m256i*>(halves + (((2 * size_t(sb) + h) * 3 + p) * R + r) * 32);
      };
      _mm256_storeu_si256(plane(0, 0), _mm512_castsi512_si256(p01));
      _mm256_storeu_si256(plane(0, 1), _mm512_extracti64x4_epi64(p01, 1));
      _mm256_storeu_si256(plane(1, 0), _mm512_castsi512_si256(p23));
      _mm256_storeu_si256(plane(1, 1), _mm512_extracti64x4_epi64(p23, 1));
      __m512i qh = _mm512_loadu_si512(b + 128);
      __m512i t0 = _mm512_gf2p8affine_epi64_epi8(ident, _mm512_permutexvar_epi8(g0, qh), 0);
      __m512i t1 = _mm512_gf2p8affine_epi64_epi8(ident, _mm512_permutexvar_epi8(g1, qh), 0);
      t0 = _mm512_ternarylogic_epi64(even, t0, _mm512_srli_epi16(t0, 8), 0xCA);
      t1 = _mm512_ternarylogic_epi64(even, t1, _mm512_srli_epi16(t1, 8), 0xCA);
      __m512i crumbs = _mm512_permutex2var_epi8(t0, order, t1);
      _mm256_storeu_si256(plane(0, 2), _mm512_castsi512_si256(crumbs));
      _mm256_storeu_si256(plane(1, 2), _mm512_extracti64x4_epi64(crumbs, 1));
    }
    for (int p = 0; p < nsb / 2; p++) {
      uint8_t* dst = planes + (size_t(p) * R + r) * 32;
      for (int e = 0; e < 2; e++) {
        const uint8_t* s = src[r] + 210 * (2 * p + e) + 192;
        for (int h = 0; h < 2; h++)
          for (int i = 0; i < 8; i++) dst[16 * e + 8 * h + i] = s[2 * i + h];
      }
    }
  }
}

/// Q4_0 (q4.py): per row [sb][group 8] bf16 d (16 nsb bytes), then [sb][row] 128 code bytes in word order: byte
/// 32 s + 4 i + jj holds codes 8 s + 2 jj (low nibble), + 1 of group i (a GGUF block's byte e holds codes e, e + 16).
void tile_q4_0(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  uint8_t* codes = t + 16 * nsb * R;
  for (int r = 0; r < R; r++) {
    uint8_t* h = t + 16 * nsb * r;
    for (int sb = 0; sb < nsb; sb++) {
      uint8_t* c = codes + (size_t(sb) * R + r) * 128;
      for (int i = 0; i < 8; i++) {
        const uint8_t* b = src[r] + 144 * sb + 18 * i;
        put16(h + 16 * sb + 2 * i, bf16_of(b));
        const uint8_t* q = b + 2;
        for (int s = 0; s < 4; s++)
          for (int jj = 0; jj < 4; jj++) {
            const int e = 8 * s + 2 * jj;
            c[32 * s + 4 * i + jj] = e < 16 ? nib(q[e], q[e + 1]) : static_cast<uint8_t>((q[e - 16] >> 4) | (q[e - 15] & 0xF0));
          }
      }
    }
  }
}

/// Q8_0 (q8.py): per row [sb][group 8] bf16 d (16 nsb bytes), then the codes as the decode's MMUL panels:
/// [sb][row group R / 8][group i][step s][row 8][8 bytes] (codes 8 s .. 8 s + 7 of group i).
void tile_q8_0(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  uint8_t* codes = t + 16 * nsb * R;
  for (int r = 0; r < R; r++) {
    uint8_t* h = t + 16 * nsb * r;
    for (int sb = 0; sb < nsb; sb++)
      for (int i = 0; i < 8; i++) {
        const uint8_t* b = src[r] + 272 * sb + 34 * i;
        put16(h + 16 * sb + 2 * i, bf16_of(b));
        uint8_t* c = codes + size_t(sb) * R * 256 + (r / 8) * 2048 + 256 * i + 8 * (r % 8);
        for (int s = 0; s < 4; s++) std::memcpy(c + 64 * s, b + 2 + 8 * s, 8);
      }
  }
}

/// Q4_1 (q4.py): per row [sb][group 8] bf16 d, then m (16 nsb bytes each), then the codes as Q4_0's.
void tile_q4_1(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  uint8_t* m = t + 16 * nsb * R;
  uint8_t* codes = t + 32 * nsb * R;
  for (int r = 0; r < R; r++) {
    uint8_t* h = t + 16 * nsb * r;
    for (int sb = 0; sb < nsb; sb++) {
      uint8_t* c = codes + (size_t(sb) * R + r) * 128;
      for (int i = 0; i < 8; i++) {
        const uint8_t* b = src[r] + 160 * sb + 20 * i;
        put16(h + 16 * sb + 2 * i, bf16_of(b));
        put16(m + 16 * nsb * r + 16 * sb + 2 * i, bf16_of(b + 2));
        const uint8_t* q = b + 4;
        for (int s = 0; s < 4; s++)
          for (int jj = 0; jj < 4; jj++) {
            const int e = 8 * s + 2 * jj;
            c[32 * s + 4 * i + jj] = e < 16 ? nib(q[e], q[e + 1]) : static_cast<uint8_t>((q[e - 16] >> 4) | (q[e - 15] & 0xF0));
          }
      }
    }
  }
}

/// The 2-bit code of element e of a Q2_K / Q3_K block's qs (element 128 n + 32 j + 16 h + l: bits 2 j of
/// qs[32 n + 16 h + l]).
inline uint32_t k23_code(const uint8_t* qs, int e) {
  const int n = e >> 7, j = (e >> 5) & 3, hl = e & 31;
  return (qs[32 * n + hl] >> (2 * j)) & 3;
}

/// A word of 8 nibbles, nibble b = v[b].
inline uint32_t nib_word(const uint32_t* v) {
  uint32_t w = 0;
  for (int b = 0; b < 8; b++) w |= (v[b] & 15) << (4 * b);
  return w;
}

/// Header words per row of Q2_K (q2k.py ``header_words``) and Q3_K (q3k.py's).
inline int q2k_header_words(int nsb) { return 5 * nsb; }
inline int q3k_header_words(int nsb) { return (7 * nsb + 1) / 2; }

/// Q2_K (q2k.py, word-major [word][row] 4 bytes): header words [sb][sc h 0, sc h 1, mn h 0, mn h 1] (nibble s: sub-block
/// 8 h + s) | [sb] bf16 d | bf16 dmin << 16, then per super-block 16 code words, word j:
/// nibble b = c(16 j + b) + 4 c(16 j + 8 + b).
void tile_q2k(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  const int nw = q2k_header_words(nsb);
  uint32_t* hw = reinterpret_cast<uint32_t*>(t);
  uint32_t* cw = hw + size_t(nw) * R;
  for (int r = 0; r < R; r++) {
    for (int w = 5 * nsb; w < nw; w++) hw[size_t(w) * R + r] = 0;
    for (int sb = 0; sb < nsb; sb++) {
      const uint8_t* b = src[r] + 84 * sb;
      const uint8_t *sc = b, *qs = b + 16;
      for (int u = 0; u < 4; u++) {  // (sc | mn, h)
        uint32_t v[8];
        for (int s = 0; s < 8; s++) v[s] = u < 2 ? sc[8 * (u & 1) + s] & 15 : sc[8 * (u & 1) + s] >> 4;
        hw[size_t(4 * sb + u) * R + r] = nib_word(v);
      }
      hw[size_t(4 * nsb + sb) * R + r] = uint32_t(bf16_of(b + 80)) | (uint32_t(bf16_of(b + 82)) << 16);
      for (int j = 0; j < 16; j++) {
        uint32_t v[8];
        for (int bb = 0; bb < 8; bb++) v[bb] = k23_code(qs, 16 * j + bb) + 4 * k23_code(qs, 16 * j + 8 + bb);
        cw[(size_t(sb) * 16 + j) * R + r] = nib_word(v);
      }
    }
  }
}

/// Q3_K (q3k.py, word-major): header words [sb][h 2] the low nibbles of u = sc + 32 of sub-blocks 8 h .. | [sb] the
/// high crumbs, nibble s = c(s) + 4 c(s + 8) | [sb pair] bf16 d of both, then per super-block
/// 24 code words, word 3 p + t (t < 3): nibble b = q(32 p + 8 t + b) + 8 (bit t of q(32 p + 24 + b)).
void tile_q3k(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  const int nw = q3k_header_words(nsb);
  uint32_t* hw = reinterpret_cast<uint32_t*>(t);
  uint32_t* cw = hw + size_t(nw) * R;
  for (int r = 0; r < R; r++) {
    for (int w = 3 * nsb + nsb / 2; w < nw; w++) hw[size_t(w) * R + r] = 0;
    uint16_t d[kMaxNsb];
    for (int sb = 0; sb < nsb; sb++) {
      const uint8_t* b = src[r] + 110 * sb;
      const uint8_t *hm = b, *qs = b + 32, *s = b + 96;
      d[sb] = bf16_of(b + 108);
      uint32_t u[16];  // the 6-bit scales: low nibbles of bytes 0 .. 7 then their high nibbles, the high crumbs of
                       // scale 4 k + i in bits 2 k of byte 8 + i
      for (int i = 0; i < 16; i++) u[i] = (i < 8 ? s[i] & 15 : s[i - 8] >> 4) | (((s[8 + (i & 3)] >> (2 * (i >> 2))) & 3) << 4);
      for (int h = 0; h < 2; h++) hw[size_t(2 * sb + h) * R + r] = nib_word(u + 8 * h);
      uint32_t c[8];
      for (int s2 = 0; s2 < 8; s2++) c[s2] = (u[s2] >> 4) + 4 * (u[s2 + 8] >> 4);
      hw[size_t(2 * nsb + sb) * R + r] = nib_word(c);
      auto q = [&](int e) {
        const int n = e >> 7, j = (e >> 5) & 3, hl = e & 31;
        return k23_code(qs, e) | (((hm[hl] >> (4 * n + j)) & 1) << 2);
      };
      for (int p = 0; p < 8; p++)
        for (int tt = 0; tt < 3; tt++) {
          uint32_t v[8];
          for (int bb = 0; bb < 8; bb++) v[bb] = q(32 * p + 8 * tt + bb) + 8 * ((q(32 * p + 24 + bb) >> tt) & 1);
          cw[(size_t(sb) * 24 + 3 * p + tt) * R + r] = nib_word(v);
        }
    }
    for (int pp = 0; pp < nsb / 2; pp++)
      hw[size_t(3 * nsb + pp) * R + r] = uint32_t(d[2 * pp]) | (uint32_t(d[2 * pp + 1]) << 16);
  }
}

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

/// BFP16 (bf16.py) of BF16 or F16 rows: 8 rows of [k8][row][exp, 8 int8] blocks.
template <bool F16>
void tile_bfp16(uint8_t* t, const uint8_t* const* src, int R, int nsb) {
  const int nk8 = 32 * nsb;
  for (int r = 0; r < R; r++)
    for (int k = 0; k < nk8; k += 2) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src[r] + 16 * k));
      const __m512 x = F16 ? _mm512_cvtph_ps(v) : _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
      bfp16_blocks(x, t + (size_t(k) * R + r) * 9, t + (size_t(k + 1) * R + r) * 9);
    }
}

/// BFP16 planes of F32 rows (bf16.py): row r's plane lo[r] (0: bfp16(w), 1: bfp16(w - plane 0); w - plane 0 is exact
/// in fp32).
void tile_bfp16_f32(uint8_t* t, const uint8_t* const* src, const uint8_t* lo, int R, int nsb) {
  const int nk8 = 32 * nsb;
  for (int r = 0; r < R; r++)
    for (int k = 0; k < nk8; k += 2) {
      uint8_t *b0 = t + (size_t(k) * R + r) * 9, *b1 = t + (size_t(k + 1) * R + r) * 9;
      const __m512 x = _mm512_loadu_ps(src[r] + 32 * k);
      bfp16_blocks(x, b0, b1);
      if (lo[r]) bfp16_blocks(_mm512_sub_ps(x, bfp16_values(b0, b1)), b0, b1);
    }
}

size_t used_bytes(TileFormat f, int R, int nsb) {
  switch (f) {
    case TileFormat::Q4_K:
      return 144 * size_t(nsb) * R;
    case TileFormat::Q5_K:
      return 176 * size_t(nsb) * R;
    case TileFormat::Q6_K:
      return (2 * size_t(nsb) * R + 31) / 32 * 32 + 208 * size_t(nsb) * R;
    case TileFormat::Q4_0:
      return 144 * size_t(nsb) * R;
    case TileFormat::Q8_0:
      return 272 * size_t(nsb) * R;
    case TileFormat::Q4_1:
      return 160 * size_t(nsb) * R;
    case TileFormat::Q2_K:
      return 4 * (q2k_header_words(nsb) + 16 * size_t(nsb)) * R;
    case TileFormat::Q3_K:
      return 4 * (q3k_header_words(nsb) + 24 * size_t(nsb)) * R;
    case TileFormat::BFP16_BF16:
    case TileFormat::BFP16_F16:
    case TileFormat::BFP16_F32:
      return 288 * size_t(nsb) * R;
  }
  return 0;
}

/// Copy a tile to its place in the image (64-byte aligned both: non-temporal stores).
void put_tile(uint8_t* dst, const uint8_t* tile, size_t S) {
  if (reinterpret_cast<uintptr_t>(dst) % 64 || S % 64) {
    std::memcpy(dst, tile, S);
    return;
  }
  for (size_t i = 0; i < S; i += 64)
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_load_si512(tile + i));
}

struct Item {
  int block, col;
  int64_t t0, nt;  // row tiles
};

/// The whole pages inside [p, p + n) (MADV_DONTNEED must not touch a neighbour's) or around it (MADV_WILLNEED).
std::pair<uintptr_t, size_t> pages(const uint8_t* p, size_t n, bool inside) {
  const uintptr_t ps = static_cast<uintptr_t>(platform::page_size());
  uintptr_t a = reinterpret_cast<uintptr_t>(p), b = a + n;
  a = inside ? (a + ps - 1) / ps * ps : a / ps * ps;
  b = inside ? b / ps * ps : (b + ps - 1) / ps * ps;
  return {a, b > a ? b - a : 0};
}

}  // namespace

TileFormat tile_format(std::string_view name) {
  if (name == "Q4_K") return TileFormat::Q4_K;
  if (name == "Q5_K") return TileFormat::Q5_K;
  if (name == "Q6_K") return TileFormat::Q6_K;
  if (name == "Q4_0") return TileFormat::Q4_0;
  if (name == "Q8_0") return TileFormat::Q8_0;
  if (name == "Q4_1") return TileFormat::Q4_1;
  if (name == "Q2_K") return TileFormat::Q2_K;
  if (name == "Q3_K") return TileFormat::Q3_K;
  if (name == "BFP16/BF16") return TileFormat::BFP16_BF16;
  if (name == "BFP16/F16") return TileFormat::BFP16_F16;
  if (name == "BFP16/F32") return TileFormat::BFP16_F32;
  throw Error(std::format("pack_tiles: no native packer for {}", name));
}

size_t block_bytes(TileFormat f) { return kBlockBytes[static_cast<int>(f)]; }

void pack_tiles(uint8_t* out, size_t out_size, const std::vector<PackBlock>& blocks, const PackOptions& opt) {
  // checks, the work items, the tensors' item counts (release) in order of first use (prefetch)
  struct Tensor {
    const uint8_t* data;
    size_t bytes;
    int64_t left;                  // items (release)
    std::atomic<int64_t> pending;  // pieces not read yet (prefetch)
  };
  std::deque<Tensor> tensors;
  std::unordered_map<const uint8_t*, size_t> tensor_of;
  std::vector<std::vector<size_t>> block_tensors(blocks.size());
  std::vector<Item> items;
  size_t max_S = 64;
  for (size_t bi = 0; bi < blocks.size(); bi++) {
    const PackBlock& b = blocks[bi];
    const size_t row_bytes = size_t(b.row_sb) * kBlockBytes[static_cast<int>(b.fmt)];
    const bool pairs = b.fmt <= TileFormat::Q6_K || b.fmt == TileFormat::Q3_K;  // (scales of super-block pairs)
    const bool bfp = b.fmt >= TileFormat::BFP16_BF16, f32 = b.fmt == TileFormat::BFP16_F32;
    const int64_t tile_rows = int64_t(b.R) * b.G;  // (entries of rows per row tile)
    if (b.R <= 0 || b.G <= 0 || b.nsb <= 0 || b.nsb > kMaxNsb || (b.nsb % 2 && pairs) || b.n % tile_rows ||
        tile_rows > 256 || (b.fmt == TileFormat::Q8_0 && b.R % 8) || (bfp && b.R != 8) || b.n_kc <= 0 ||
        b.kc_group <= 0 || b.n_kc % b.kc_group || used_bytes(b.fmt, b.R, b.nsb) > b.S || b.row_sb <= 0 ||
        b.row_sb > b.n_kc * b.nsb || (b.row_sb % 2 && pairs && b.G == 1) ||
        (b.G > 1 && (b.n_kc != 1 || b.row_sb * b.G != b.nsb || f32)) ||
        b.col_stride < size_t(b.n_rt) * b.n_kc * b.S || b.rt0 < 0 || b.rt0 + b.n / tile_rows > b.n_rt ||
        b.offset + (b.cols - 1) * b.col_stride + size_t(b.n_rt) * b.n_kc * b.S > out_size)
      throw Error(std::format("pack_tiles: block {}: inconsistent geometry (R {}, G {}, S {}, nsb {}, row_sb {}, n {}, "
                              "n_kc {} / {}, row tiles {} + {} of {}, {} B at {} of {})", bi, b.R, b.G, b.S, b.nsb,
                              b.row_sb, b.n, b.n_kc, b.kc_group, b.rt0, b.n / tile_rows, b.n_rt,
                              b.cols * b.col_stride, b.offset, out_size));
    int64_t total = 0;
    for (auto [data, rows] : b.tensors) {
      total += rows;
      auto [it, added] = tensor_of.try_emplace(data, tensors.size());
      if (added) tensors.emplace_back(data, size_t(rows) * row_bytes, 0, 0);
      block_tensors[bi].push_back(it->second);
    }
    const int64_t indices = f32 ? 2 * total : total;  // (F32: two planes per row)
    for (int64_t i = 0; i < b.cols * b.n; i++)
      if (b.rows[i] < -1 || b.rows[i] >= indices)
        throw Error(std::format("pack_tiles: block {}: row {} of {} stacked rows", bi, b.rows[i], indices));
    max_S = std::max(max_S, b.S);
    const int64_t step = std::max<int64_t>(1, kItemRows / tile_rows), tiles = b.n / tile_rows;
    for (int c = 0; c < b.cols; c++)
      for (int64_t t0 = 0; t0 < tiles; t0 += step) {
        items.push_back({static_cast<int>(bi), c, t0, std::min(step, tiles - t0)});
        for (size_t t : block_tensors[bi]) tensors[t].left++;
      }
  }

  std::atomic<size_t> next{0}, done_bytes{0};
  std::atomic<size_t> needed{0};  // (tensors 0 .. needed - 1: a claimed item reads them; the prefetchers never hold
                                  // them back)
  std::atomic<bool> failed{false}, finished{false};
  std::mutex mutex;  // (the tensors' counts, the first error)
  std::exception_ptr error;

  auto worker = [&] {
    auto tile = platform::aligned_buffer<uint8_t>(64, (max_S + 63) / 64 * 64);
    std::vector<uint8_t> part;  // (the rows of a chunk past a row's end: its super-blocks, then zero ones)
    try {
      if (!tile) throw std::bad_alloc();
      const uint8_t* src[256];
      const uint8_t* base[256];
      uint8_t lo[256];
      for (size_t k; !failed && (k = next++) < items.size();) {
        const Item& it = items[k];
        const PackBlock& b = blocks[it.block];
        const bool f32 = b.fmt == TileFormat::BFP16_F32;
        const int tile_rows = b.R * b.G;
        const size_t bb = kBlockBytes[static_cast<int>(b.fmt)], row_bytes = size_t(b.row_sb) * bb;
        for (size_t t : block_tensors[it.block]) {
          size_t n = needed;
          while (n < t + 1 && !needed.compare_exchange_weak(n, t + 1)) {
          }
        }
        for (size_t t : block_tensors[it.block])  // (read by the prefetchers: no page fault reads it in 128 KB)
          while (tensors[t].pending > 0 && !failed) std::this_thread::sleep_for(std::chrono::microseconds(100));
        std::memset(tile.get(), 0, b.S);  // (the padding stays zero: every tile writes the same bytes)
        uint8_t* col = out + b.offset + it.col * b.col_stride;
        for (int64_t rt = it.t0; rt < it.t0 + it.nt; rt++) {
          const int64_t* rows = b.rows + it.col * b.n + rt * tile_rows;
          for (int r = 0; r < tile_rows; r++) {
            base[r] = nullptr;
            int64_t i = rows[r];
            lo[r] = f32 && i >= 0 ? i & 1 : 0;
            if (f32 && i >= 0) i >>= 1;
            for (auto [data, n] : b.tensors) {
              if (i < 0) break;
              if (i < n) {
                base[r] = data + i * row_bytes;
                break;
              }
              i -= n;
            }
          }
          for (int kc = 0; kc < b.n_kc; kc++) {
            const int have = std::clamp(b.row_sb - kc * b.nsb, 0, b.nsb);  // (the chunk's super-blocks in a row)
            if ((have < b.nsb || b.G > 1) && part.size() < size_t(b.R) * b.nsb * bb)
              part.assign(size_t(b.R) * b.nsb * bb, 0);
            for (int r = 0; r < b.R; r++) {
              if (b.G > 1) {  // (a tile row of G rows of row_sb super-blocks)
                uint8_t* p = part.data() + size_t(r) * b.nsb * bb;
                for (int j = 0; j < b.G; j++)
                  std::memcpy(p + j * row_bytes, base[b.G * r + j] ? base[b.G * r + j] : kZeroRow, row_bytes);
                src[r] = p;
              } else if (!base[r] || have == 0) {
                src[r] = kZeroRow;
              } else if (have == b.nsb) {
                src[r] = base[r] + kc * b.nsb * bb;
              } else {
                uint8_t* p = part.data() + size_t(r) * b.nsb * bb;
                std::memcpy(p, base[r] + kc * b.nsb * bb, have * bb);
                std::memset(p + have * bb, 0, (b.nsb - have) * bb);
                src[r] = p;
              }
            }
            switch (b.fmt) {
              case TileFormat::Q4_K:
                tile_q4k(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q5_K:
                tile_q5k(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q6_K:
                tile_q6k(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q4_0:
                tile_q4_0(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q8_0:
                tile_q8_0(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q4_1:
                tile_q4_1(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q2_K:
                tile_q2k(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::Q3_K:
                tile_q3k(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::BFP16_BF16:
                tile_bfp16<false>(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::BFP16_F16:
                tile_bfp16<true>(tile.get(), src, b.R, b.nsb);
                break;
              case TileFormat::BFP16_F32:
                tile_bfp16_f32(tile.get(), src, lo, b.R, b.nsb);
                break;
            }
            const int64_t g = b.kc_group, idx = (kc / g * b.n_rt + b.rt0 + rt) * g + kc % g;
            put_tile(col + idx * b.S, tile.get(), b.S);
          }
        }
        done_bytes += size_t(it.nt) * tile_rows * row_bytes;
        if (opt.release) {
          std::vector<size_t> drop;
          {
            std::lock_guard lock(mutex);
            for (size_t t : block_tensors[it.block])
              if (--tensors[t].left == 0) drop.push_back(t);
          }
          for (size_t t : drop) {  // (the pages stay in the page cache)
            auto [a, n] = pages(tensors[t].data, tensors[t].bytes, true);
            if (n) platform::drop_pages(reinterpret_cast<void*>(a), n);
          }
        }
      }
      _mm_sfence();
    } catch (...) {
      std::lock_guard lock(mutex);
      if (!error) error = std::current_exception();
      failed = true;
    }
  };

  // prefetch: the tensors in pieces, in order of first use, each read into the page cache by one of kPrefetchThreads
  // threads unless mincore finds it resident (pread: the device's 128 KB readahead makes page faults and
  // MADV_WILLNEED / POPULATE_READ stream at 1.2 - 1.8 GB/s, 16 threads of 8 MB reads at 3.5 GB/s: a cold 27B packs
  // in 5.3 s), at most a window ahead of the workers but for the tensors of claimed items (the window counts the
  // items' rows, which can fall behind the tensors' bytes: workers waiting on a held-back tensor would never finish
  // one); a worker starts an item once its tensors are read
  std::unique_ptr<platform::File> file;
  if (!opt.file.empty()) file = std::make_unique<platform::File>(opt.file);
  const bool prefetching = file != nullptr;
  struct Piece {
    const uint8_t* data;
    size_t bytes, tensor;
  };
  std::vector<Piece> pieces;
  size_t window = kPrefetchWindow;  // (>= a tensor: the workers of its first item wait for all of it)
  for (size_t t = 0; t < tensors.size(); t++) {
    window = std::max(window, tensors[t].bytes);
    for (size_t o = 0; o < tensors[t].bytes; o += kPrefetchPiece)
      pieces.push_back({tensors[t].data + o, std::min(kPrefetchPiece, tensors[t].bytes - o), t});
    tensors[t].pending = prefetching ? -(-int64_t(tensors[t].bytes) / int64_t(kPrefetchPiece)) : 0;
  }
  std::atomic<size_t> next_piece{0}, issued{0};
  auto prefetcher = [&] {
    const size_t ps = platform::page_size();
    auto buf = platform::aligned_buffer(ps, kPrefetchPiece + ps);
    std::vector<unsigned char> resident(kPrefetchPiece / ps + 2);
    for (size_t k; !finished && !failed && (k = next_piece++) < pieces.size();) {
      const Piece& pc = pieces[k];
      while (!finished && !failed && issued > done_bytes + window && pc.tensor >= needed)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      issued += pc.bytes;
      auto [a, len] = pages(pc.data, pc.bytes, false);
      if (buf && platform::needs_reading(reinterpret_cast<void*>(a), len, resident.data()))
        for (size_t got = 0; got < len;) {  // (errors, no buffer: the workers' page faults read it)
          size_t r = 0;
          try {
            r = file->read_at(buf.get(), len - got, a - reinterpret_cast<uintptr_t>(opt.map) + got);
          } catch (const std::system_error&) {
          }
          if (r == 0) break;
          got += r;
        }
      tensors[pc.tensor].pending--;
    }
  };

  unsigned threads = opt.threads ? opt.threads : std::max(1u, std::thread::hardware_concurrency() / 2);
  threads = static_cast<unsigned>(std::min<size_t>(threads, std::max<size_t>(1, items.size())));
  std::vector<std::thread> pool;
  std::vector<std::thread> prefetch;
  if (prefetching)
    for (unsigned i = 0; i < kPrefetchThreads; i++) prefetch.emplace_back(prefetcher);
  for (unsigned i = 1; i < threads; i++) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  finished = true;
  for (auto& th : prefetch) th.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace artlite
