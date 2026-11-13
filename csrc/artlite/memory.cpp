#include "artlite/memory.h"

#include <emmintrin.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <atomic>
#include <cerrno>
#include <exception>
#include <thread>
#include <vector>

namespace artlite::memory {

void flush_lines(const void* p, size_t len) {
  if (!len) return;
  constexpr uintptr_t line = 64;
  uintptr_t cur = reinterpret_cast<uintptr_t>(p) & ~(line - 1);
  const uintptr_t end = reinterpret_cast<uintptr_t>(p) + len;
  _mm_mfence();
  for (; cur < end; cur += line) _mm_clflush(reinterpret_cast<const void*>(cur));
  _mm_mfence();
}

void flush_line_offsets(const void* base, const int64_t* offsets, size_t count) {
  _mm_mfence();
  for (size_t i = 0; i < count; i++) _mm_clflush(static_cast<const char*>(base) + offsets[i]);
  _mm_mfence();
}

namespace {

constexpr size_t kPage = 4096;

void fault_in(char* p, size_t n) {
#ifndef _WIN32
  if (!::madvise(p, n, MADV_POPULATE_WRITE)) return;
#endif
  for (size_t o = 0; o < n; o += kPage) p[o] = 0;  // (Windows, Linux before 5.14)
}

}  // namespace

void populate_zeroed(void* mem_, size_t n) {
  constexpr size_t piece = size_t(32) << 20;
  char* const mem = static_cast<char*>(mem_);
  const unsigned threads = static_cast<unsigned>(std::clamp<size_t>(n / (size_t(256) << 20), 1, 8));
  std::atomic<size_t> next{0};
  std::vector<std::exception_ptr> errors(threads);
  auto work = [&](unsigned t) {
    try {
      for (size_t a; (a = next.fetch_add(piece)) < n;) {
        const size_t k = std::min(piece, n - a);
        fault_in(mem + a, k);
        flush_lines(mem + a, k);
      }
    } catch (...) {
      errors[t] = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads; t++) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
}

#ifndef _WIN32
int advise_pinned(void* mem, size_t n) {
  ::madvise(mem, n, MADV_HUGEPAGE);  // (no THP: 4 KB pages)
  return ::madvise(mem, n, MADV_DONTFORK) ? errno : 0;
}
#endif

}  // namespace artlite::memory
