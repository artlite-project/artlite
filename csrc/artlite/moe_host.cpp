// An MoE layer's host step in the prefill (``operators/moe/prefill.py``): the routing of a chunk's tokens, its plan of
// expert chunks, the columns' control streams.

#include "artlite/moe_host.h"

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

void route_rows(const float* logits, int64_t t0, int64_t t1, int64_t row, int per_col, int64_t pitch, int E, int k,
                int32_t* ids, float* weights) {
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

}  // namespace

void moe_route(const float* logits, int64_t n, int64_t row, int per_col, int64_t pitch, int E, int k, int32_t* ids,
               float* weights) {
  if (k < 1 || k > E || E % per_col) throw Error(std::format("moe_route: k {} of E {} ({} per column)", k, E, per_col));
  // (a chunk's tokens on threads of 256 at least)
  const int64_t nt = std::min<int64_t>(n / 256, std::max(1u, std::thread::hardware_concurrency() / 2));
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
