#include "artlite/file_io.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <memory>
#include <system_error>
#include <thread>
#include <vector>

#include "artlite/error.h"
#include "artlite/platform.h"
#include "artlite/shim.h"

namespace artlite {

namespace {

constexpr size_t kBounce = 512 << 10;     // per thread: two hyperthreads share a core's 1 MB L2
constexpr size_t kPerThread = 32 << 20;   // fewer threads for small files
constexpr unsigned kMaxThreads = 16;      // (2 GB from the page cache: 16 threads 71 GB/s, 8 49 GB/s)

// dst gets the bytes of src through streaming stores (the head up to dst's 64-B alignment and
// the tail through memcpy). The calling thread fences its own stores.
ARTLITE_AVX512_FN void stream_copy(char* dst, const char* src, size_t n) {
  size_t head = std::min(n, (64 - reinterpret_cast<uintptr_t>(dst) % 64) % 64);
  std::memcpy(dst, src, head);
  size_t i = head;
  for (; i + 256 <= n; i += 256) {
    __m512i a = _mm512_loadu_si512(src + i), b = _mm512_loadu_si512(src + i + 64);
    __m512i c = _mm512_loadu_si512(src + i + 128), d = _mm512_loadu_si512(src + i + 192);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), a);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i + 64), b);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i + 128), c);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i + 192), d);
  }
  for (; i + 64 <= n; i += 64)
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_loadu_si512(src + i));
  std::memcpy(dst + i, src + i, n - i);
}

void copy_range(const platform::File& file, char* dst, uint64_t file_offset, size_t n) {
  auto bounce = platform::aligned_buffer(64, kBounce);
  if (!bounce) throw std::bad_alloc();
  for (size_t done = 0; done < n;) {
    size_t len = std::min(kBounce, n - done);
    for (size_t got = 0; got < len;) {
      size_t r = file.read_at(bounce.get() + got, len - got, file_offset + done + got);
      if (r == 0) throw Error("read_file: the file ended early");
      got += r;
    }
    stream_copy(dst + done, bounce.get(), len);
    done += len;
  }
  _mm_sfence();
}

}  // namespace

void read_file(shim_xdna::bo& buf, size_t offset, const std::string& path, uint64_t file_offset, size_t nbytes) {
  if (offset > buf.size() || nbytes > buf.size() - offset)
    throw Error(std::format("read_file: {} bytes at offset {} exceed the buffer ({} bytes)", nbytes, offset, buf.size()));
  platform::File file(path);
  file.advise_sequential(file_offset, nbytes);

  char* dst = static_cast<char*>(buf.map()) + offset;
  unsigned threads = static_cast<unsigned>(std::clamp<size_t>(
      nbytes / kPerThread, 1, std::min(kMaxThreads, std::max(1u, std::thread::hardware_concurrency()))));
  size_t step = (nbytes / threads + 4095) & ~size_t(4095);  // (page-aligned slices)
  std::vector<std::exception_ptr> errors(threads);
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads; t++) {
    size_t a = std::min(nbytes, t * step), b = std::min(nbytes, a + step);
    pool.emplace_back([&, t, a, b] {
      try {
        copy_range(file, dst + a, file_offset + a, b - a);
      } catch (...) {
        errors[t] = std::current_exception();
      }
    });
  }
  try {
    copy_range(file, dst, file_offset, std::min(nbytes, step));
  } catch (...) {
    errors[0] = std::current_exception();
  }
  for (auto& th : pool) th.join();
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
}

}  // namespace artlite
