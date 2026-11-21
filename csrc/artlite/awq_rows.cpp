#include "artlite/awq_rows.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <format>
#include <mutex>
#include <thread>
#include <vector>

#include "artlite/error.h"

namespace artlite {

namespace {

constexpr int kAwqShift[8] = {0, 16, 4, 20, 8, 24, 12, 28};  // (logical column j of a word: bits 4 [0 4 1 5 2 6 3 7][j])
constexpr int64_t kRowBlock = 128;  // rows of a work item (one cache line of a qweight row: 16 words)

/// Rows r0 .. r1 - 1 of the output.
void rows(uint8_t* out, const AwqSource& s, int64_t r0, int64_t r1) {
  const int64_t nb = s.K / 256, words = s.N / 8, G = s.K / s.group;
  const bool five = s.qweight_hi != nullptr;
  const int64_t bb = five ? kAwq5BlockBytes : kAwqBlockBytes, c0 = five ? 48 : 16, hw = s.N / 32;
  const int64_t n = r1 - r0;
  std::vector<int64_t> word(n), col(n);
  std::vector<int> shift(n);
  for (int64_t r = 0; r < n; r++) {
    const int64_t c = col[r] = s.rows_of ? s.rows_of[r0 + r] : r0 + r;
    word[r] = c >> 3;
    shift[r] = kAwqShift[c & 7];
  }
  for (int64_t g = 0; g < G; g++) {
    const int64_t gs = s.groups_of ? s.groups_of[g] : g;
    const int64_t sb = g * s.group / 256, half = (g * s.group % 256) / 128;
    // the group's scale and zero per row (block bytes 2 half, 4 + half)
    const int32_t* zrow = s.qzeros + gs * words;
    const uint16_t* srow = s.scales + gs * s.N;
    for (int64_t r = 0; r < n; r++) {
      uint8_t* b = out + ((r0 + r) * nb + sb) * bb;
      const int64_t c = col[r];
      std::memcpy(b + 2 * half, srow + c, 2);
      uint32_t z = (static_cast<uint32_t>(zrow[word[r]]) >> shift[r]) & 15;
      if (five) z |= ((static_cast<uint32_t>(s.qzeros_hi[gs * hw + (c >> 5)]) >> (c & 31)) & 1) << 4;
      b[4 + half] = static_cast<uint8_t>(z);
    }
    for (int p = 0; p < s.group; p++) {
      const int64_t ch = s.chan ? s.chan[g * s.group + p] : p;
      const int32_t* q = s.qweight + (gs * s.group + ch) * words;
      const int32_t* qh = five ? s.qweight_hi + (gs * s.group + ch) * hw : nullptr;
      const int64_t e = half * 128 + p;  // (the position in its super-block)
      const int64_t byte = c0 + 32 * (e >> 6) + (e & 31);
      const int hi = (e & 63) >= 32 ? 4 : 0;
      uint8_t* b = out + (r0 * nb + sb) * bb + byte;
      for (int64_t r = 0; r < n; r++, b += nb * bb) {
        const uint8_t code = static_cast<uint8_t>((static_cast<uint32_t>(q[word[r]]) >> shift[r]) & 15);
        *b = hi ? static_cast<uint8_t>((*b & 0x0F) | (code << 4)) : static_cast<uint8_t>((*b & 0xF0) | code);
        if (five) {  // (the fifth bit: bit e / 32 of qh byte e % 32)
          const int64_t c = col[r];
          const uint8_t bit = static_cast<uint8_t>((static_cast<uint32_t>(qh[c >> 5]) >> (c & 31)) & 1);
          b[16 + (e & 31) - byte] |= static_cast<uint8_t>(bit << (e >> 5));
        }
      }
    }
  }
}

}  // namespace

void awq_rows(uint8_t* out, const AwqSource& s, unsigned threads) {
  if (s.group != 128 || s.K % 256 || s.N % 8 || !s.qweight || !s.qzeros || !s.scales)
    throw Error(std::format("awq_rows: K {}, N {}, group {}: groups of 128, K a multiple of 256", s.K, s.N, s.group));
  if (!s.qweight_hi != !s.qzeros_hi || (s.qweight_hi && s.N % 32))
    throw Error(std::format("awq_rows: int5 needs both bit planes and N {} a multiple of 32", s.N));
  const int64_t n = s.rows_of ? s.n_rows : s.N;
  std::memset(out, 0, size_t(n) * (s.K / 256) * (s.qweight_hi ? kAwq5BlockBytes : kAwqBlockBytes));
  const int64_t items = (n + kRowBlock - 1) / kRowBlock;
  const unsigned want = threads ? threads : std::max(1u, std::thread::hardware_concurrency() / 2);
  const unsigned nt = static_cast<unsigned>(std::min<int64_t>(items, want));
  std::vector<std::thread> pool;
  std::atomic<int64_t> next{0};
  std::exception_ptr error;
  std::mutex mutex;
  auto work = [&] {
    try {
      for (int64_t i; (i = next++) < items;) rows(out, s, i * kRowBlock, std::min(n, (i + 1) * kRowBlock));
    } catch (...) {
      std::lock_guard lock(mutex);
      if (!error) error = std::current_exception();
    }
  };
  for (unsigned t = 1; t < nt; t++) pool.emplace_back(work);
  work();
  for (auto& t : pool) t.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace artlite
