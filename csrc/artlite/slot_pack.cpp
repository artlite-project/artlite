// The vision overlay's weight packer (slot_pack.h): each item a pass's 80 rows over at most 16 k-tiles. Per row the
// part rows' elements of the item's columns go into an fp32 row (a source of another type, or one that strided runs
// gather from, first decoded into a span buffer: the needed columns of the row, contiguous), 16 at a time into two
// bfp16ebs8 blocks at their places in the item's slots, which then stream out to the image. Compiled with
// -march=znver5 (CMakeLists.txt).

#include "artlite/slot_pack.h"

#include <immintrin.h>

#include <algorithm>
#include <format>
#include <memory>
#include <unordered_map>

#include "artlite/bfp16_simd.h"
#include "artlite/error.h"
#include "artlite/memory.h"
#include "artlite/platform.h"

namespace artlite {

namespace {

constexpr int kKt = kTileCols;
constexpr size_t kSlot = kSlotBytes;
constexpr int kItemKt = 16;      // k-tiles of an item (92 KB of slots, twice with hilo)

struct Item {
  int job, pass, kt0, kt1;
};

/// A run of output columns clipped to an item's: k (from the item's first column), n, the source column of its first,
/// the stride.
struct Clip {
  int k, n;
  int64_t src, stride;
};

inline __mmask16 lanes(int64_t n) { return n >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << n) - 1); }

/// d[0 .. n) = scale src[s + i stride] (fp32).
inline void gather(float* d, const float* src, int64_t s, int64_t stride, int n, float scale) {
  const __m512 sc = _mm512_set1_ps(scale);
  const bool one = scale == 1.0f;
  if (stride == 1) {
    for (int j = 0; j < n; j += 16) {
      const __mmask16 m = lanes(n - j);
      __m512 x = _mm512_maskz_loadu_ps(m, src + s + j);
      _mm512_mask_storeu_ps(d + j, m, one ? x : _mm512_mul_ps(x, sc));
    }
    return;
  }
  const __m512i step = _mm512_set1_epi32(static_cast<int>(16 * stride));
  __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(s)),
                                 _mm512_mullo_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                                                    _mm512_set1_epi32(static_cast<int>(stride))));
  for (int j = 0; j < n; j += 16, idx = _mm512_add_epi32(idx, step)) {
    const __mmask16 m = lanes(n - j);
    __m512 x = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m, idx, src, 4);
    _mm512_mask_storeu_ps(d + j, m, one ? x : _mm512_mul_ps(x, sc));
  }
}

/// Elements [lo, hi) of row r of t as fp32 into span (span[0] = element lo; Q8_0 from lo's block: returns the first
/// element decoded).
int64_t decode(const SlotTensor& t, int64_t r, int64_t lo, int64_t hi, float* span) {
  const uint8_t* row = t.data + r * t.row_bytes;
  switch (t.type) {
    case SlotSource::F32:
      std::memcpy(span, row + 4 * lo, 4 * size_t(hi - lo));
      return lo;
    case SlotSource::BF16:
    case SlotSource::F16:
      for (int64_t c = lo; c < hi; c += 16) {
        const __m256i v = _mm256_maskz_loadu_epi16(lanes(hi - c), row + 2 * c);
        _mm512_storeu_ps(span + (c - lo), t.type == SlotSource::F16
                                              ? _mm512_cvtph_ps(v)
                                              : _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16)));
      }
      return lo;
    case SlotSource::Q8_0: {
      const int64_t b0 = lo / 32, b1 = (hi + 31) / 32;
      for (int64_t b = b0; b < b1; b++) {  // (w = d q: fp16 d, int8 q, one fp32 product)
        const uint8_t* blk = row + 34 * b;
        uint16_t h;
        std::memcpy(&h, blk, 2);
        const __m512 d = _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));
        for (int half = 0; half < 2; half++) {
          const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2 + 16 * half));
          _mm512_storeu_ps(span + 32 * (b - b0) + 16 * half, _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q))));
        }
      }
      return 32 * b0;
    }
    case SlotSource::EYE:
      std::fill(span, span + (hi - lo), 0.0f);
      if (r >= lo && r < hi) span[r - lo] = 1.0f;
      return lo;
  }
  return lo;
}

}  // namespace

SlotSource slot_source(std::string_view name) {
  if (name == "F32") return SlotSource::F32;
  if (name == "F16") return SlotSource::F16;
  if (name == "BF16") return SlotSource::BF16;
  if (name == "Q8_0") return SlotSource::Q8_0;
  if (name == "EYE") return SlotSource::EYE;
  throw Error(std::format("pack_slots: no source type {}", name));
}

size_t slot_row_bytes(SlotSource type, int64_t cols) {
  switch (type) {
    case SlotSource::F32:
      return 4 * size_t(cols);
    case SlotSource::Q8_0:
      return 34 * size_t(cols / 32);
    case SlotSource::EYE:
      return 0;
    case SlotSource::F16:
    case SlotSource::BF16:
      return 2 * size_t(cols);
  }
  return 0;
}

void pack_slots(const std::vector<SlotTensor>& tensors, const std::vector<SlotPart>& parts,
                const std::vector<SlotJob>& jobs, const std::vector<pack::MappedFile>& files, unsigned threads,
                bool release) {
  // checks, the items, the tensors in order of first use
  pack::Items run;
  std::unordered_map<int, uint32_t> number;  // (tensor -> its index in run.tensors)
  std::vector<Item> items;
  int64_t max_cols = 0;
  for (size_t ji = 0; ji < jobs.size(); ji++) {
    const SlotJob& j = jobs[ji];
    if (j.npo <= 0 || j.nkt <= 0 || (j.hilo && j.nkt % 2) || j.part0 < 0 || j.n_parts < 0 ||
        size_t(j.part0) + j.n_parts > parts.size())
      throw Error(std::format("pack_slots: job {}: {} passes, {} k-tiles{}, parts {} + {} of {}", ji, j.npo, j.nkt,
                              j.hilo ? " (hilo)" : "", j.part0, j.n_parts, parts.size()));
    const int64_t cols = int64_t(j.hilo ? j.nkt / 2 : j.nkt) * kKt;
    std::vector<uint32_t> ts;
    double elem = 0;  // (source bytes per output element: the read-ahead window)
    for (int pi = j.part0; pi < j.part0 + j.n_parts; pi++) {
      const SlotPart& p = parts[pi];
      if (p.tensor < 0 || size_t(p.tensor) >= tensors.size() || p.n_runs < 0)
        throw Error(std::format("pack_slots: job {}: part {}: tensor {} of {}", ji, pi, p.tensor, tensors.size()));
      const SlotTensor& t = tensors[p.tensor];
      for (int64_t r = 0; r < int64_t(j.npo) * kPassRows; r++)
        if (p.rows[r] < -1 || p.rows[r] >= t.rows)
          throw Error(std::format("pack_slots: job {}: part {}: row {} of a tensor of {}", ji, pi, p.rows[r], t.rows));
      for (int k = 0; k < p.n_runs; k++) {
        const int32_t* u = p.runs + 4 * k;
        if (u[0] < 0 || u[1] <= 0 || u[0] + int64_t(u[1]) > cols || u[2] < 0 || u[3] < 0 ||
            u[2] + int64_t(u[1] - 1) * u[3] >= t.cols)
          throw Error(std::format("pack_slots: job {}: part {}: run [{}, {}) of {} columns from {} step {} of {}", ji,
                                  pi, u[0], u[0] + u[1], cols, u[2], u[3], t.cols));
      }
      max_cols = std::max(max_cols, t.cols);
      elem += double(slot_row_bytes(t.type, 32)) / 32;
      auto [it, added] = number.try_emplace(p.tensor, static_cast<uint32_t>(run.tensors.size()));
      if (added) run.tensors.push_back({t.data, t.type == SlotSource::EYE ? 0 : size_t(t.rows) * t.row_bytes, t.file});
      if (std::find(ts.begin(), ts.end(), it->second) == ts.end()) ts.push_back(it->second);
    }
    const int half = j.hilo ? j.nkt / 2 : j.nkt;
    for (int p = 0; p < j.npo; p++)
      for (int kt = 0; kt < half; kt += kItemKt) {
        items.push_back({static_cast<int>(ji), p, kt, std::min(kt + kItemKt, half)});
        run.add(ts, static_cast<size_t>(kPassRows * (items.back().kt1 - kt) * kKt * elem));
      }
  }

  pack::run_items(run, files, threads, release, [&] {
    auto row = std::shared_ptr<float>(platform::aligned_buffer<float>(64, sizeof(float) * (kItemKt * kKt + 16)).release(),
                                      platform::AlignedFree{});
    auto span = std::shared_ptr<float>(platform::aligned_buffer<float>(64, sizeof(float) * (max_cols + 64)).release(),
                                       platform::AlignedFree{});
    auto out = std::shared_ptr<uint8_t>(platform::aligned_buffer<uint8_t>(64, 2 * kItemKt * kSlot).release(),
                                        platform::AlignedFree{});
    if (!row || !span || !out) throw std::bad_alloc();
    auto clips = std::make_shared<std::vector<Clip>>();
    return std::function<void(size_t)>([&, row, span, out, clips](size_t k) {
      const Item& it = items[k];
      const SlotJob& j = jobs[it.job];
      const int half = j.hilo ? j.nkt / 2 : j.nkt, nkt = it.kt1 - it.kt0;
      const int k0 = it.kt0 * kKt, nk = nkt * kKt;
      // per part: its runs clipped to the item's columns, the source columns they read [lo, hi), whether they all
      // run contiguously
      struct Use {
        const SlotPart* part;
        size_t c0, c1;
        int64_t lo, hi;
        bool direct;
      };
      Use use[8];
      int n_use = 0;
      clips->clear();
      for (int pi = j.part0; pi < j.part0 + j.n_parts; pi++) {
        const SlotPart& p = parts[pi];
        Use u{&p, clips->size(), 0, INT64_MAX, -1, tensors[p.tensor].type == SlotSource::F32};
        for (int r = 0; r < p.n_runs; r++) {
          const int32_t* v = p.runs + 4 * r;
          const int a = std::max<int>(v[0], k0), b = std::min<int>(v[0] + v[1], k0 + nk);
          if (a >= b) continue;
          const int64_t s = v[2] + int64_t(a - v[0]) * v[3];
          clips->push_back({a - k0, b - a, s, v[3]});
          u.lo = std::min(u.lo, s);
          u.hi = std::max(u.hi, s + int64_t(b - a - 1) * v[3] + 1);
          u.direct = u.direct && v[3] == 1;
        }
        u.c1 = clips->size();
        if (u.c1 > u.c0) {
          if (n_use == 8) throw Error("pack_slots: more than 8 parts in a job");
          use[n_use++] = u;
        }
      }
      for (int i = 0; i < kPassRows; i++) {
        float* x = row.get();
        std::fill(x, x + nk, 0.0f);
        for (int ui = 0; ui < n_use; ui++) {
          const Use& u = use[ui];
          const int64_t r = u.part->rows[int64_t(it.pass) * kPassRows + i];
          if (r < 0) continue;
          const SlotTensor& t = tensors[u.part->tensor];
          const float* src;
          int64_t base;  // (the source column of src[0])
          if (u.direct) {
            src = reinterpret_cast<const float*>(t.data + r * t.row_bytes);
            base = 0;
          } else {
            base = decode(t, r, u.lo, u.hi, span.get());
            src = span.get();
          }
          for (size_t c = u.c0; c < u.c1; c++) {
            const Clip& cl = (*clips)[c];
            gather(x + cl.k, src, cl.src - base, cl.stride, cl.n, u.part->scale);
          }
        }
        // the row's blocks: row i = 16-row block i / 16, half (i / 8) % 2, row i % 8 of its slots
        const size_t at = ((size_t(i / 16) * 8 * 2 + (i / 8) % 2) * 8 + i % 8) * 9;
        for (int kt = 0; kt < nkt; kt++)
          for (int s = 0; s < 8; s += 2) {
            uint8_t* b0 = out.get() + kt * kSlot + at + s * 2 * 8 * 9;
            const __m512 v = _mm512_loadu_ps(x + kt * kKt + s * 8);
            pack::bfp16_blocks(v, b0, b0 + 2 * 8 * 9);
            if (j.hilo) {
              uint8_t* l0 = b0 + nkt * kSlot;
              pack::bfp16_blocks(_mm512_sub_ps(v, pack::bfp16_values(b0, b0 + 2 * 8 * 9)), l0, l0 + 2 * 8 * 9);
            }
          }
      }
      uint8_t* dst = j.dst + (size_t(it.pass) * j.nkt + it.kt0) * kSlot;
      memory::stream_copy(dst, out.get(), nkt * kSlot);
      if (j.hilo) memory::stream_copy(dst + half * kSlot, out.get() + nkt * kSlot, nkt * kSlot);
    });
  });
}

}  // namespace artlite
