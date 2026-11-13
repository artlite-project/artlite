// artlite/memory.h: the host memory operations the shims and the packers share. Host only.
#include <gtest/gtest.h>

#include <sys/mman.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "artlite/memory.h"

TEST(Memory, StreamCopyCopiesEveryAlignmentAndLength) {
  std::vector<uint8_t> src(1000);
  std::iota(src.begin(), src.end(), 1);
  for (size_t offset = 0; offset < 70; offset += 3)
    for (size_t n : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(255), size_t(256), size_t(257),
                     size_t(700)}) {
      std::vector<uint8_t> dst(1000, 0xEE);
      artlite::memory::stream_copy(dst.data() + offset, src.data(), n);
      _mm_sfence();
      EXPECT_EQ(std::memcmp(dst.data() + offset, src.data(), n), 0) << offset << " " << n;
      for (size_t i = 0; i < offset; i++) ASSERT_EQ(dst[i], 0xEE);
      for (size_t i = offset + n; i < dst.size(); i++) ASSERT_EQ(dst[i], 0xEE);
    }
}

TEST(Memory, FlushKeepsTheData) {
  std::vector<uint8_t> a(4096);
  std::iota(a.begin(), a.end(), 0);
  const auto before = a;
  artlite::memory::flush_lines(a.data() + 3, 1000);
  artlite::memory::flush_lines(a.data(), 0);
  const int64_t offsets[] = {0, 64, 128, 4032};
  artlite::memory::flush_line_offsets(a.data(), offsets, 4);
  EXPECT_EQ(a, before);
}

TEST(Memory, PopulateZeroedKeepsFreshMemoryZero) {
  const size_t n = size_t(70) << 20;  // (three pieces of 32 MB)
  void* mem = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mem, MAP_FAILED);
  EXPECT_EQ(artlite::memory::advise_pinned(mem, n), 0);
  artlite::memory::populate_zeroed(mem, n);
  const auto* p = static_cast<const uint64_t*>(mem);
  bool zero = true;
  for (size_t i = 0; i < n / 8; i++) zero = zero && p[i] == 0;
  EXPECT_TRUE(zero);
  ::munmap(mem, n);
}
