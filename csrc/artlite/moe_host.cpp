// An MoE layer's host step in the prefill (``operators/moe/prefill.py``): the routing of a chunk's tokens, its plan of
// expert chunks, the columns' control streams.

#include "artlite/moe_host.h"

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <thread>
#include <vector>

#include "artlite/error.h"

namespace artlite {

namespace {

constexpr size_t NEXT_WORDS = 15;  // (a block's tail: ``moe.ctl.NEXT_WORDS``)

// numpy's pairwise sum of a short row (its order for n < 8 and 8 <= n <= 128)
double row_sum(const double* a, int n) {
  if (n < 8) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
  }
  double r[8];
  for (int j = 0; j < 8; j++) r[j] = a[j];
  int i = 8;
  for (; i + 8 <= n; i += 8)
    for (int j = 0; j < 8; j++) r[j] += a[i + j];
  double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; i++) s += a[i];
  return s;
}

// (the reference: per token an insertion sort of ordered keys; NaN logits and the shapes ``route_rows_avx512`` does
// not take)
void route_rows_scalar(const float* logits, int64_t t0, int64_t t1, int64_t row, int per_col, int64_t pitch, int E,
                       int k, int32_t* ids, float* weights) {
  // per token the k largest keys, descending: a logit's ordered bits, then ~id (ties to the lower id; -0 as +0)
  std::vector<uint64_t> top(k + 1);
  std::vector<double> e(k);
  for (int64_t t = t0; t < t1; t++) {
    const float* p = logits + t * row;
    int nt = 0;
    for (int c = 0, x = 0; c < E / per_col; c++)
      for (int i = 0; i < per_col; i++, x++) {
        const uint32_t u = std::bit_cast<uint32_t>(p[c * pitch + i] + 0.0f);
        const uint64_t key = uint64_t(u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u)) << 32 | uint32_t(~x);
        if (nt == k && key <= top[k - 1]) continue;
        int j = nt < k ? nt++ : k - 1;
        for (; j > 0 && top[j - 1] < key; j--) top[j] = top[j - 1];
        top[j] = key;
      }
    const int32_t first = int32_t(~uint32_t(top[0]));
    const double l0 = p[(first / per_col) * pitch + first % per_col];
    for (int j = 0; j < k; j++) {
      const int32_t x = int32_t(~uint32_t(top[j]));
      ids[t * k + j] = x;
      e[j] = std::exp(double(p[(x / per_col) * pitch + x % per_col]) - l0);
    }
    const double s = row_sum(e.data(), k);
    for (int j = 0; j < k; j++) weights[t * k + j] = float(e[j] / s);
  }
}

// e^x of 8 doubles x <= 0 (the routing's l - l_max): x = n ln 2 + r, |r| <= ln 2 / 2, a degree-12 Taylor polynomial in
// r (relative error < 2^-52), times 2^n (n >= -1022: below, 0)
__m512d exp_nonpos(__m512d x) {
  const __m512d log2e = _mm512_set1_pd(1.4426950408889634), ln2hi = _mm512_set1_pd(6.93147180369123816490e-01),
                ln2lo = _mm512_set1_pd(1.90821492927058770002e-10);
  x = _mm512_max_pd(x, _mm512_set1_pd(-708.0));
  const __m512d n = _mm512_roundscale_pd(_mm512_mul_pd(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m512d r = _mm512_fnmadd_pd(n, ln2lo, _mm512_fnmadd_pd(n, ln2hi, x));
  static constexpr double c[13] = {1.0,          1.0,          1.0 / 2,       1.0 / 6,        1.0 / 24,
                                   1.0 / 120,    1.0 / 720,    1.0 / 5040,    1.0 / 40320,    1.0 / 362880,
                                   1.0 / 3628800, 1.0 / 39916800, 1.0 / 479001600};
  __m512d p = _mm512_set1_pd(c[12]);
  for (int i = 11; i >= 0; i--) p = _mm512_fmadd_pd(p, r, _mm512_set1_pd(c[i]));
  const __m512i e = _mm512_slli_epi64(_mm512_add_epi64(_mm512_cvtpd_epi64(n), _mm512_set1_epi64(1023)), 52);
  return _mm512_mul_pd(p, _mm512_castsi512_pd(e));
}

// route_rows_scalar's routing, AVX-512 (Zen 5): a token's E logits in E / 16 vectors (a column's per_col from c pitch:
// per_col a multiple of 16). Their lane maxima (a max tree) bound the k largest from below: the k-th largest lane
// maximum t has k logits >= t (one per lane), so the k largest are >= t. The logits >= t (k .. ~20), compressed in id
// order with their ids, go through an insertion sort (larger first; a later equal value stays behind: ties to the lower
// id; -0 == +0 as the scalar path's keys). The weights: exp_nonpos (k == 8) in numpy's summation order. A token with a
// NaN logit goes to the scalar path.
void route_rows_avx512(const float* logits, int64_t t0, int64_t t1, int64_t row, int per_col, int64_t pitch, int E,
                       int k, int32_t* ids, float* weights) {
  const int nv = E / 16, vpc = per_col / 16;
  const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  __m512 v[32];
  alignas(64) float cv[512 + 16];
  alignas(64) int32_t ci[512 + 16];
  constexpr int64_t kAhead = 8;  // (tokens prefetched ahead: the logits come from DRAM, which the NPU keeps busy)
  for (int64_t t = t0; t < t1; t++) {
    const float* p = logits + t * row;
    if (t + kAhead < t1)
      for (int j = 0; j < nv; j++)
        _mm_prefetch(reinterpret_cast<const char*>(p + kAhead * row + (j / vpc) * pitch + (j % vpc) * 16), _MM_HINT_T0);
    __mmask16 nan = 0;
    for (int j = 0; j < nv; j++) {
      v[j] = _mm512_loadu_ps(p + (j / vpc) * pitch + (j % vpc) * 16);
      nan |= _mm512_cmp_ps_mask(v[j], v[j], _CMP_UNORD_Q);
    }
    if (nan) {
      route_rows_scalar(logits, t, t + 1, row, per_col, pitch, E, k, ids, weights);
      continue;
    }
    __m512 m[32];  // (a max tree)
    for (int j = 0; j < nv; j++) m[j] = v[j];
    for (int w = nv; w > 1;) {
      const int h = w / 2;
      for (int j = 0; j < h; j++) m[j] = _mm512_max_ps(m[2 * j], m[2 * j + 1]);
      if (w % 2) m[h] = m[w - 1];
      w = h + w % 2;
    }
    // the k-th largest lane maximum: the least of the lanes that fewer than k lanes exceed (each lane against the
    // 15 rotations, independent compares)
    const __m512 lm = m[0];
    __m512i above = _mm512_setzero_si512();
    for (int r = 1; r < 16; r++) {
      const __m512 rot = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_add_epi32(lane, _mm512_set1_epi32(r)),
                                                                _mm512_set1_epi32(15)), lm);
      above = _mm512_mask_add_epi32(above, _mm512_cmp_ps_mask(rot, lm, _CMP_GT_OQ), above, _mm512_set1_epi32(1));
    }
    const __mmask16 top = _mm512_cmplt_epi32_mask(above, _mm512_set1_epi32(k));
    const __m512 th = _mm512_set1_ps(_mm512_mask_reduce_min_ps(top, lm));
    int n = 0;
    for (int j = 0; j < nv; j++) {
      const __mmask16 ge = _mm512_cmp_ps_mask(v[j], th, _CMP_GE_OQ);
      const int x0 = (j / vpc) * per_col + (j % vpc) * 16;
      // (compresses into registers: a compress into memory is microcoded on Zen 4 / 5)
      _mm512_storeu_ps(cv + n, _mm512_maskz_compress_ps(ge, v[j]));
      _mm512_storeu_si512(ci + n, _mm512_maskz_compress_epi32(ge, _mm512_add_epi32(lane, _mm512_set1_epi32(x0))));
      n += std::popcount(unsigned(ge));
    }
    float tv[16];
    int32_t ti[16];
    int nt = 0;
    for (int i = 0; i < n; i++) {
      const float x = cv[i] + 0.0f;
      if (nt == k && !(x > tv[k - 1])) continue;
      int j = nt < k ? nt++ : k - 1;
      for (; j > 0 && tv[j - 1] < x; j--) tv[j] = tv[j - 1], ti[j] = ti[j - 1];
      tv[j] = x, ti[j] = ci[i];
    }
    for (int j = 0; j < k; j++) ids[t * k + j] = ti[j];
    if (k != 8) {
      double e[16];
      for (int j = 0; j < k; j++) e[j] = std::exp(double(tv[j]) - double(tv[0]));
      const double s = row_sum(e, k);
      for (int j = 0; j < k; j++) weights[t * k + j] = float(e[j] / s);
      continue;
    }
    const __m512d l = _mm512_cvtps_pd(_mm256_loadu_ps(tv));
    const __m512d e = exp_nonpos(_mm512_sub_pd(l, _mm512_set1_pd(double(tv[0]))));
    alignas(64) double ev[8];
    _mm512_store_pd(ev, e);
    const double sum = ((ev[0] + ev[1]) + (ev[2] + ev[3])) + ((ev[4] + ev[5]) + (ev[6] + ev[7]));
    _mm256_storeu_ps(weights + t * k, _mm512_cvtpd_ps(_mm512_div_pd(e, _mm512_set1_pd(sum))));
  }
}

void route_rows(const float* logits, int64_t t0, int64_t t1, int64_t row, int per_col, int64_t pitch, int E, int k,
                int32_t* ids, float* weights) {
  if (k <= 16 && per_col % 16 == 0 && E % per_col == 0 && E <= 512)
    route_rows_avx512(logits, t0, t1, row, per_col, pitch, E, k, ids, weights);
  else
    route_rows_scalar(logits, t0, t1, row, per_col, pitch, E, k, ids, weights);
}

}  // namespace

void moe_route(const float* logits, int64_t n, int64_t row, int per_col, int64_t pitch, int E, int k, int32_t* ids,
               float* weights) {
  if (k < 1 || k > E || E % per_col) throw Error(std::format("moe_route: k {} of E {} ({} per column)", k, E, per_col));
  // (threads of 4096 tokens at least: the vector path takes ~0.16 us per token, a thread's start ~0.1 ms)
  const int64_t nt = std::min<int64_t>(n / 4096, std::max(1u, std::thread::hardware_concurrency() / 2));
  if (nt <= 1) {
    route_rows(logits, 0, n, row, per_col, pitch, E, k, ids, weights);
    return;
  }
  std::vector<std::thread> ts;
  for (int64_t t = 0; t < nt; t++)
    ts.emplace_back(route_rows, logits, n * t / nt, n * (t + 1) / nt, row, per_col, pitch, E, k, ids, weights);
  for (auto& t : ts) t.join();
}

void moe_plan(const int32_t* ids, int64_t n, int k, int E, int cols, int cap, int chunk, int32_t* experts,
              int32_t* counts, int32_t* sizes, int32_t* rows, int64_t* pos) {
  const int64_t pairs = n * k;
  std::vector<int64_t> start(E + 1, 0);
  for (int64_t p = 0; p < pairs; p++) {
    const int32_t x = ids[p];
    if (x < 0 || x >= E) throw Error(std::format("moe_plan: expert id {} of {}", x, E));
    start[x + 1]++;
  }
  for (int x = 0; x < E; x++) start[x + 1] += start[x];
  std::vector<int64_t> by(pairs);  // (pairs by expert, in token order)
  {
    std::vector<int64_t> at(start.begin(), start.end() - 1);
    for (int64_t p = 0; p < pairs; p++) by[at[ids[p]]++] = p;
  }
  std::fill(counts, counts + cols, 0);
  int64_t j = 0;  // (chunks dealt)
  for (int x = 0; x < E; x++)
    for (int64_t a = start[x]; a < start[x + 1]; a += chunk) {
      const int64_t b = std::min(a + chunk, start[x + 1]);
      const int c = int(j % cols);
      const int kk = counts[c]++;
      if (kk >= cap) throw Error(std::format("moe_plan: more than {} chunks in a column", cap));
      experts[c * cap + kk] = x;
      sizes[c * cap + kk] = int32_t(b - a);
      int32_t* r = rows + (int64_t(c) * cap + kk) * chunk;
      const int64_t base = (int64_t(c) * cap + kk) * chunk;
      for (int64_t i = a; i < b; i++) {
        r[i - a] = int32_t(by[i] / k);
        pos[by[i]] = base + (i - a);
      }
      for (int64_t i = b - a; i < chunk; i++) r[i] = r[0];
      j++;
    }
}

size_t ctl_assemble(uint32_t* out, size_t capacity, uint64_t base, std::span<const uint32_t> block0,
                    std::span<const CtlTemplate> templates, std::span<const int32_t> blocks, const uint64_t* bases,
                    int n_bases, std::span<const uint32_t> tails, const uint32_t* lock_bits) {
  if (tails.size() != 2 * NEXT_WORDS) throw Error("ctl_assemble: two tails of NEXT_WORDS");
  const size_t n = blocks.size();
  size_t total = block0.size();
  for (size_t i = 0; i < n; i++) {
    const int32_t t = blocks[i];
    if (t < 0 || size_t(t) >= templates.size()) throw Error(std::format("ctl_assemble: template {}", t));
    total += templates[t].words.size() + (i + 1 < n ? NEXT_WORDS : 0);
  }
  if (total > capacity) throw Error(std::format("ctl_assemble: {} words, room for {}", total, capacity));
  std::memcpy(out, block0.data(), 4 * block0.size());
  size_t w = block0.size();
  for (size_t i = 0; i < n; i++) {
    const CtlTemplate& tp = templates[blocks[i]];
    uint32_t* b = out + w;
    std::memcpy(b, tp.words.data(), 4 * tp.words.size());
    const uint64_t* bs = bases + i * n_bases;
    for (size_t s = 0; s + 4 <= tp.slots.size(); s += 4) {
      const int64_t lo = tp.slots[s], hi = tp.slots[s + 1], bi = tp.slots[s + 2], off = tp.slots[s + 3];
      const uint64_t a = bs[bi] + uint64_t(off);
      b[lo] = uint32_t(a);
      if (hi >= 0) b[hi] |= uint32_t(a >> 32);
    }
    w += tp.words.size();
    if (i + 1 == n) break;
    const CtlTemplate& nx = templates[blocks[i + 1]];
    const size_t next = nx.words.size() + (i + 2 < n ? NEXT_WORDS : 0);  // (the next block's length)
    const uint64_t at = base + 4 * (w + NEXT_WORDS);
    uint32_t* tl = out + w;
    std::memcpy(tl, tails.data() + (i % 2) * NEXT_WORDS, 4 * NEXT_WORDS);
    tl[2] = uint32_t(next);
    tl[3] = uint32_t(at);
    tl[4] = uint32_t(at >> 32);
    if (lock_bits) tl[11] |= lock_bits[i + 1];
    w += NEXT_WORDS;
  }
  return w;
}

}  // namespace artlite
