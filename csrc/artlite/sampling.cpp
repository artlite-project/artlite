// Sampling on the host (``models.sampling``): the candidates of a choice in one pass over a vocabulary's logits, the
// choice, a draft's probability. One thread, AVX-512 (-march=znver5: CMakeLists.txt).
//
// The pass compares 32 bf16 (16 fp32) logits per instruction as ordered integer keys against a threshold, the k-th
// largest key of the candidates so far: a logit above it is a candidate. Once the candidates fill a buffer of 4 k,
// the k largest stay and the threshold rises to the k-th, so past the first few thousand logits almost none passes
// and the pass runs at the rate the logits come from memory.

#include "artlite/sampling.h"

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

namespace artlite {

namespace {

// A value's bits as a signed integer with the magnitude bits of the negative ones flipped, plus one for those (-0 as
// +0), compare as the values do; no value has the smallest integer.
int32_t key(uint16_t b) {
  const int32_t s = int16_t(b);
  return (s ^ ((s >> 15) & 0x7FFF)) - (s >> 15);
}
int32_t key(float f) {
  const int32_t s = std::bit_cast<int32_t>(f);
  return (s ^ ((s >> 31) & 0x7FFFFFFF)) - (s >> 31);
}

// A candidate: its key's bits ordered as unsigned, then the complement of its index: the larger, the better (equal
// keys: the lower index).
uint64_t candidate(int32_t k, uint32_t i) { return uint64_t(uint32_t(k) ^ 0x80000000u) << 32 | ~i; }
uint32_t index_of(uint64_t c) { return ~uint32_t(c); }
int32_t key_of(uint64_t c) { return int32_t(uint32_t(c >> 32) ^ 0x80000000u); }

// The candidates of the k largest of x[0 .. n) (k >= 1; ``mask``: of those whose bit is set, nullptr: all), unsorted: at
// most max(4 k, 256), those k among them.
template <class T>
void scan(const T* x, size_t n, size_t k, const uint32_t* mask, std::vector<uint64_t>& c) {
  constexpr bool BF16 = sizeof(T) == 2;
  constexpr size_t L = 64 / sizeof(T);  // lanes of a vector
  const size_t cap = std::max<size_t>(4 * k, 256);
  c.reserve(cap + 4 * L);
  auto splat = [](int32_t t) { return BF16 ? _mm512_set1_epi16(int16_t(t)) : _mm512_set1_epi32(t); };
  int32_t thr = BF16 ? INT16_MIN : INT32_MIN;  // (below every key)
  __m512i t = splat(thr);
  auto keys = [](__m512i v) {
    if constexpr (BF16) {
      const __m512i s = _mm512_srai_epi16(v, 15);
      return _mm512_sub_epi16(_mm512_xor_si512(v, _mm512_and_si512(s, _mm512_set1_epi16(0x7FFF))), s);
    } else {
      const __m512i s = _mm512_srai_epi32(v, 31);
      return _mm512_sub_epi32(_mm512_xor_si512(v, _mm512_and_si512(s, _mm512_set1_epi32(0x7FFFFFFF))), s);
    }
  };
  auto above = [&](__m512i v) -> uint32_t {
    if constexpr (BF16)
      return _mm512_cmpgt_epi16_mask(keys(v), t);
    else
      return _mm512_cmpgt_epi32_mask(keys(v), t);
  };
  auto allowed = [&](size_t i) -> uint32_t {  // (the mask's bits of the L logits from i, a multiple of L)
    if (!mask) return ~0u;
    if constexpr (BF16)
      return mask[i / 32];
    else
      return (mask[i / 32] >> (i % 32)) & 0xFFFFu;
  };
  auto push = [&](size_t i0, uint32_t m) {  // (lanes of a mask taken with an older threshold: the next keep drops them)
    for (; m; m &= m - 1) {
      const size_t i = i0 + std::countr_zero(m);
      c.push_back(candidate(key(x[i]), uint32_t(i)));
    }
    if (c.size() >= cap) {
      std::nth_element(c.begin(), c.begin() + (k - 1), c.end(), std::greater<>());
      c.resize(k);
      thr = key_of(c[k - 1]);
      t = splat(thr);
    }
  };
  size_t i = 0;
  for (; i + 4 * L <= n; i += 4 * L) {
    const uint32_t m0 = above(_mm512_loadu_si512(x + i)) & allowed(i),
                   m1 = above(_mm512_loadu_si512(x + i + L)) & allowed(i + L),
                   m2 = above(_mm512_loadu_si512(x + i + 2 * L)) & allowed(i + 2 * L),
                   m3 = above(_mm512_loadu_si512(x + i + 3 * L)) & allowed(i + 3 * L);
    if (!(m0 | m1 | m2 | m3)) continue;
    push(i, m0);
    push(i + L, m1);
    push(i + 2 * L, m2);
    push(i + 3 * L, m3);
  }
  for (; i < n; i += L) {
    const size_t r = std::min(L, n - i);
    const uint32_t lanes = (r == 32 ? ~0u : (1u << r) - 1) & allowed(i);
    if constexpr (BF16)
      push(i, _mm512_mask_cmpgt_epi16_mask(lanes, keys(_mm512_maskz_loadu_epi16(lanes, x + i)), t));
    else
      push(i, _mm512_mask_cmpgt_epi32_mask(__mmask16(lanes), keys(_mm512_maskz_loadu_epi32(__mmask16(lanes), x + i)),
                                           t));
  }
}

// The candidates of the k largest of the logits (``mask``: of the allowed ones), best first.
const std::vector<uint64_t>& best(Logits x, size_t k, const uint32_t* mask = nullptr) {
  thread_local std::vector<uint64_t> c;
  c.clear();
  if (x.bf16)
    scan(static_cast<const uint16_t*>(x.data), x.n, k, mask, c);
  else
    scan(static_cast<const float*>(x.data), x.n, k, mask, c);
  if (c.size() > k) {
    std::nth_element(c.begin(), c.begin() + (k - 1), c.end(), std::greater<>());
    c.resize(k);
  }
  std::sort(c.begin(), c.end(), std::greater<>());
  return c;
}

float logit(Logits x, size_t i) {
  return x.bf16 ? std::bit_cast<float>(uint32_t(static_cast<const uint16_t*>(x.data)[i]) << 16)
                : static_cast<const float*>(x.data)[i];
}

// 2^t (t <= 0; fp32: 2^f (f = t - round(t), |f| <= 1/2) as its Taylor polynomial of degree 6, relative error < 2e-7,
// times 2^round(t)); t below -150: 0.
__m512 exp2_ps(__m512 t) {
  t = _mm512_max_ps(t, _mm512_set1_ps(-150.0f));
  const __m512 r = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512 f = _mm512_sub_ps(t, r);
  __m512 p = _mm512_set1_ps(1.5403530393381606e-4f);
  for (float c : {1.3333558146428443e-3f, 9.6181291076284772e-3f, 5.5504108664821580e-2f, 2.4022650695910071e-1f,
                  6.9314718055994531e-1f, 1.0f})
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c));
  return _mm512_scalef_ps(p, r);
}

}  // namespace

size_t top_logits(Logits x, size_t k, float cap, const int32_t* at, const float* counts, size_t m, Penalty p,
                  const int32_t* rows, float* values, int32_t* ids, const uint32_t* mask) {
  k = std::min(k, x.n);
  if (!k) return 0;
  // the k + m largest hold the k largest of the others
  const std::vector<uint64_t>& c = best(x, std::min(k + m, x.n), mask);
  auto capped = [&](size_t i) {
    const float v = logit(x, i);
    return cap > 0 ? std::tanh(v / cap) * cap : v;
  };
  struct Cand {
    float v;
    uint32_t i;
  };
  thread_local std::vector<Cand> out;
  out.clear();
  for (uint64_t e : c) {
    if (out.size() == k) break;
    const uint32_t i = index_of(e);
    if (std::find(at, at + m, int32_t(i)) == at + m) out.push_back({capped(i), i});
  }
  if (m) {
    for (size_t j = 0; j < m; j++) {
      if (mask && !(mask[size_t(at[j]) / 32] >> (size_t(at[j]) % 32) & 1)) continue;
      float v = capped(size_t(at[j]));
      v = v <= 0 ? v * p.repeat : v / p.repeat;
      v = v - p.presence - counts[j] * p.frequency;
      out.push_back({v, uint32_t(at[j])});
    }
    std::stable_sort(out.begin(), out.end(), [](const Cand& a, const Cand& b) { return a.v > b.v; });
    out.resize(std::min(out.size(), k));
  }
  for (size_t j = 0; j < out.size(); j++) {
    values[j] = out[j].v;
    ids[j] = rows ? rows[out[j].i] : int32_t(out[j].i);
  }
  return out.size();
}

size_t pick(const float* values, size_t n, float temp, float top_p, float min_p, double u) {
  thread_local std::vector<double> q;
  q.resize(n);
  double mx = -INFINITY, s = 0.0;
  for (size_t i = 0; i < n; i++) mx = std::max(mx, double(values[i]) / temp);
  for (size_t i = 0; i < n; i++) s += q[i] = std::exp(double(values[i]) / temp - mx);
  const bool filter = top_p < 1.0f || min_p > 0.0f;
  double before = 0.0, kept = 0.0;  // (before: the probability of the candidates before i)
  const double floor = min_p * q[0] / s;
  for (size_t i = 0; i < n; i++) {
    const double pi = q[i] / s;
    const bool keep = !filter || i == 0 || (before < top_p && pi >= floor);
    before += pi;
    q[i] = keep ? pi : 0.0;
    kept += q[i];
  }
  const double target = u * kept;
  double acc = 0.0;
  size_t last = 0;
  for (size_t i = 0; i < n; i++) {
    if (q[i] == 0.0) continue;
    acc += q[i];
    last = i;
    if (target < acc) return i;
  }
  return last;
}

std::pair<size_t, float> max_prob(const uint16_t* x, size_t n) {
  if (!n) return {0, 0.0f};
  const size_t i = index_of(best({x, n, true}, 1)[0]);
  const __m512 mx = _mm512_set1_ps(std::bit_cast<float>(uint32_t(x[i]) << 16)), l2e = _mm512_set1_ps(1.4426950408889634f);
  auto term = [&](__m256i b) {  // (16 bf16 -> exp(l - max))
    const __m512 l = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b), 16));
    return exp2_ps(_mm512_mul_ps(_mm512_sub_ps(l, mx), l2e));
  };
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  size_t j = 0;
  for (; j + 64 <= n; j += 64) {
    s0 = _mm512_add_ps(s0, term(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j))));
    s1 = _mm512_add_ps(s1, term(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j + 16))));
    s2 = _mm512_add_ps(s2, term(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j + 32))));
    s3 = _mm512_add_ps(s3, term(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j + 48))));
  }
  for (; j < n; j += 16) {
    const __mmask16 lanes = n - j >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (n - j)) - 1);
    s0 = _mm512_mask_add_ps(s0, lanes, s0, term(_mm256_maskz_loadu_epi16(lanes, x + j)));
  }
  const float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
  return {i, 1.0f / s};
}

float logsumexp(Logits x, float cap) {
  if (!x.n) return -INFINITY;
  const float top = logit(x, index_of(best(x, 1)[0]));
  const float mx = cap > 0 ? std::tanh(top / cap) * cap : top;
  const __m512 vmx = _mm512_set1_ps(mx), l2e = _mm512_set1_ps(1.4426950408889634f);
  const __m512 inv = _mm512_set1_ps(cap > 0 ? 1.0f / cap : 0.0f), vcap = _mm512_set1_ps(cap);
  auto term = [&](__m512 l) {  // (exp(capped l - max); tanh(y) = 1 - 2 / (exp(2 y) + 1), y within +-9: +-1 in fp32)
    if (cap > 0) {
      const __m512 y = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(l, inv), _mm512_set1_ps(-9.0f)), _mm512_set1_ps(9.0f));
      const __m512 e = exp2_ps(_mm512_mul_ps(_mm512_add_ps(y, y), l2e));  // (exp(2 y); 2^t holds for t <= 26 too)
      const __m512 t = _mm512_sub_ps(_mm512_set1_ps(1.0f),
                                     _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, _mm512_set1_ps(1.0f))));
      l = _mm512_mul_ps(t, vcap);
    }
    return exp2_ps(_mm512_mul_ps(_mm512_sub_ps(l, vmx), l2e));
  };
  auto load = [&](size_t j, __mmask16 lanes) {
    if (x.bf16)
      return _mm512_castsi512_ps(_mm512_slli_epi32(
          _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(lanes, static_cast<const uint16_t*>(x.data) + j)), 16));
    return _mm512_maskz_loadu_ps(lanes, static_cast<const float*>(x.data) + j);
  };
  __m512 s0 = _mm512_setzero_ps(), s1 = s0;
  size_t j = 0;
  for (; j + 32 <= x.n; j += 32) {
    s0 = _mm512_add_ps(s0, term(load(j, 0xFFFF)));
    s1 = _mm512_add_ps(s1, term(load(j + 16, 0xFFFF)));
  }
  for (; j < x.n; j += 16) {
    const __mmask16 lanes = x.n - j >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (x.n - j)) - 1);
    s0 = _mm512_mask_add_ps(s0, lanes, s0, term(load(j, lanes)));
  }
  return mx + std::log(_mm512_reduce_add_ps(_mm512_add_ps(s0, s1)));
}

namespace {

// 16 bf16 logits as fp32 (bf16: fp32's top half)
inline __m512 bf16_ps(const uint16_t* x) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x))), 16));
}

// whether logit i is allowed by a mask (nullptr: all are), the mask's bits of the 16 logits from i (a multiple of 16)
inline bool allowed(const uint32_t* mask, size_t i) { return !mask || (mask[i / 32] >> (i % 32) & 1); }
inline __mmask16 lanes16(const uint32_t* mask, size_t i) {
  return mask ? __mmask16(mask[i / 32] >> (i % 32)) : __mmask16(0xFFFF);
}

// the indices of mask k's lanes (from i) appended to ids[c ..] (the first max in all); the new count
inline size_t append(__mmask16 k, size_t i, int32_t* ids, size_t c, size_t max) {
  if (c + 16 <= max) {
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    _mm512_mask_compressstoreu_epi32(ids + c, k, _mm512_add_epi32(iota, _mm512_set1_epi32(int32_t(i))));
  } else {
    size_t at = c;
    for (uint32_t b = k; b && at < max; b &= b - 1) ids[at++] = int32_t(i + size_t(std::countr_zero(b)));
  }
  return c + size_t(std::popcount(uint32_t(k)));
}

}  // namespace

void screen_stats(const uint8_t* raw, size_t n, size_t row_bytes, size_t cols, float* scale, float* bound) {
  // (a Q6_K block: screen_exact's layout; the low nibble of value 128 h + 32 j + l: nibble j / 2 of ql[64 h + 32 (j &
  // 1) + l], its crumb: j of qh[32 h + l])
  for (size_t i = 0; i < n; i++) {
    const uint8_t* b = raw + i * row_bytes;
    double rw = 0.0, ww = 0.0, rr = 0.0;
    for (size_t k = 0; k < cols / 256; k++, b += 210) {
      const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
      uint16_t d16;
      std::memcpy(&d16, b + 208, 2);
      const double d = double(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d16))));
      for (int h = 0; h < 2; h++)
        for (int j = 0; j < 4; j++) {
          const uint8_t* ql = b + 64 * h + 32 * (j & 1);
          const uint8_t* qh = b + 128 + 32 * h;
          for (int half = 0; half < 2; half++) {  // (a sub-block: 16 values of one scale)
            int32_t s_rw = 0, s_ww = 0, s_rr = 0;
            for (int l = 16 * half; l < 16 * half + 16; l++) {
              const int lo = (ql[l] >> (4 * (j >> 1))) & 15, hi = (qh[l] >> (2 * j)) & 3;
              const int r = lo - 8, w = 16 * hi + lo - 32;
              s_rw += r * w;
              s_ww += w * w;
              s_rr += r * r;
            }
            const double a = d * double(sc[8 * h + 2 * j + half]), a2 = a * a;
            rw += a2 * s_rw;
            ww += a2 * s_ww;
            rr += a2 * s_rr;
          }
        }
    }
    const double rho = ww > 0 ? rw / ww : 0.0;
    const double nv = std::max(rr - 2 * rho * rw + rho * rho * ww, 0.0) / double(cols);
    scale[i] = float(1.0 / (1.0 - rho));
    bound[i] = float(std::sqrt(nv) / (1.0 - rho));
  }
}

float screen_lower(const uint16_t* x, const float* scale, const float* bound, float gnorm, size_t n,
                   const uint32_t* mask) {
  const __m512 g = _mm512_set1_ps(gnorm);
  __m512 lo = _mm512_set1_ps(-INFINITY);
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    lo = _mm512_mask_max_ps(lo, lanes16(mask, i), lo,
                            _mm512_fnmadd_ps(_mm512_loadu_ps(bound + i), g, _mm512_mul_ps(bf16_ps(x + i), _mm512_loadu_ps(scale + i))));
  float best = _mm512_reduce_max_ps(lo);
  for (; i < n; i++)
    if (allowed(mask, i)) best = std::max(best, std::bit_cast<float>(uint32_t(x[i]) << 16) * scale[i] - bound[i] * gnorm);
  return best;
}

size_t screen_above(const uint16_t* x, const float* scale, const float* bound, float gnorm, size_t n, float thr,
                    int32_t* ids, size_t max, const uint32_t* mask) {
  const __m512 g = _mm512_set1_ps(gnorm), t = _mm512_set1_ps(thr);
  size_t c = 0, i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 up = _mm512_fmadd_ps(_mm512_loadu_ps(bound + i), g, _mm512_mul_ps(bf16_ps(x + i), _mm512_loadu_ps(scale + i)));
    const __mmask16 k = _mm512_mask_cmp_ps_mask(lanes16(mask, i), up, t, _CMP_GE_OQ);
    if (k) c = append(k, i, ids, c, max);
  }
  for (; i < n; i++)
    if (allowed(mask, i) && std::bit_cast<float>(uint32_t(x[i]) << 16) * scale[i] + bound[i] * gnorm >= thr) {
      if (c < max) ids[c] = int32_t(i);
      c++;
    }
  return c;
}

size_t mask_ids(const uint32_t* mask, size_t n, int32_t* ids, size_t max) {
  size_t c = 0, i = 0;
  for (; i + 16 <= n; i += 16)
    if (const __mmask16 k = lanes16(mask, i)) c = append(k, i, ids, c, max);
  for (; i < n; i++)
    if (allowed(mask, i)) {
      if (c < max) ids[c] = int32_t(i);
      c++;
    }
  return c;
}

void screen_dense(const uint16_t* x, const float* scale, size_t n, float floor, float* a) {
  const __m512 f = _mm512_set1_ps(floor);
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(a + i, _mm512_min_ps(_mm512_mul_ps(bf16_ps(x + i), _mm512_loadu_ps(scale + i)), f));
  for (; i < n; i++) a[i] = std::min(std::bit_cast<float>(uint32_t(x[i]) << 16) * scale[i], floor);
}

void screen_exact(const uint8_t* raw, size_t row_bytes, const int32_t* ids, size_t m, const float* g, size_t cols,
                  float* out) {
  // A Q6_K block (210 B per 256 values): ql 128 B, qh 64 B, 16 int8 scales, fp16 d; value 128 h + 32 j + l: nibble
  // j / 2 of ql[64 h + 32 (j & 1) + l], crumb j of qh[32 h + l]; its scale: sc[8 h + 2 j + l / 16]
  const __m256i m15 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3), c32 = _mm256_set1_epi8(32);
  for (size_t r = 0; r < m; r++) {
    const uint8_t* b = raw + size_t(ids[r]) * row_bytes;
    double total = 0.0;
    for (size_t k = 0; k < cols / 256; k++, b += 210) {
      const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
      const float* gk = g + 256 * k;
      __m512 acc = _mm512_setzero_ps();
      for (int h = 0; h < 2; h++) {
        const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 128 + 32 * h));
        for (int j = 0; j < 4; j++) {
          __m256i ql = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 64 * h + 32 * (j & 1)));
          if (j & 2) ql = _mm256_srli_epi16(ql, 4);
          const __m256i hi = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2 * j)), m3);
          const __m256i q = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(ql, m15), _mm256_slli_epi16(hi, 4)), c32);
          const __m512 v0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q)));
          const __m512 v1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q, 1)));
          const float* gj = gk + 128 * h + 32 * j;
          acc = _mm512_fmadd_ps(_mm512_mul_ps(v0, _mm512_loadu_ps(gj)), _mm512_set1_ps(float(sc[8 * h + 2 * j])), acc);
          acc = _mm512_fmadd_ps(_mm512_mul_ps(v1, _mm512_loadu_ps(gj + 16)), _mm512_set1_ps(float(sc[8 * h + 2 * j + 1])), acc);
        }
      }
      uint16_t d16;
      std::memcpy(&d16, b + 208, 2);
      total += double(_mm512_reduce_add_ps(acc)) * double(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d16))));
    }
    out[r] = float(total);
  }
}

}  // namespace artlite
