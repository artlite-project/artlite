#include "artlite/file_io.h"

#include <fcntl.h>
#include <immintrin.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <memory>
#include <system_error>
#include <thread>
#include <vector>

#include "artlite/error.h"
#include "shim_xdna/bo.h"

namespace artlite {

namespace {

constexpr size_t kBounce = 512 << 10;     // per thread: two hyperthreads share a core's 1 MB L2
constexpr size_t kPerThread = 32 << 20;   // fewer threads for small files
constexpr unsigned kMaxThreads = 16;      // (2 GB from the page cache: 16 threads 71 GB/s, 8 49 GB/s)

// dst gets the bytes of src through streaming stores (the head up to dst's 64-B alignment and
// the tail through memcpy). The calling thread fences its own stores.
__attribute__((target("avx512f"))) void stream_copy(char* dst, const char* src, size_t n) {
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

void copy_range(int fd, char* dst, uint64_t file_offset, size_t n) {
  std::unique_ptr<char, decltype(&std::free)> bounce(static_cast<char*>(std::aligned_alloc(64, kBounce)), &std::free);
  if (!bounce) throw std::bad_alloc();
  for (size_t done = 0; done < n;) {
    size_t len = std::min(kBounce, n - done);
    for (size_t got = 0; got < len;) {
      ssize_t r = ::pread(fd, bounce.get() + got, len - got, static_cast<off_t>(file_offset + done + got));
      if (r < 0 && errno == EINTR) continue;
      if (r < 0) throw std::system_error(errno, std::generic_category(), "read_file: pread");
      if (r == 0) throw Error("read_file: the file ended early");
      got += static_cast<size_t>(r);
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
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::system_error(errno, std::generic_category(), "read_file: cannot open '" + path + "'");
  ::posix_fadvise(fd, static_cast<off_t>(file_offset), static_cast<off_t>(nbytes), POSIX_FADV_SEQUENTIAL);

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
        copy_range(fd, dst + a, file_offset + a, b - a);
      } catch (...) {
        errors[t] = std::current_exception();
      }
    });
  }
  try {
    copy_range(fd, dst, file_offset, std::min(nbytes, step));
  } catch (...) {
    errors[0] = std::current_exception();
  }
  for (auto& th : pool) th.join();
  ::close(fd);
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
}

}  // namespace artlite
