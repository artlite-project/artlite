#include "artlite/file_io.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <system_error>
#include <thread>
#include <vector>

#include "artlite/error.h"
#include "artlite/memory.h"
#include "artlite/platform.h"
#include "artlite/shim.h"

namespace artlite {

namespace {

constexpr size_t kBounce = 512 << 10;     // per thread: two hyperthreads share a core's 1 MB L2
constexpr size_t kPerThread = 32 << 20;   // fewer threads for small files
constexpr unsigned kMaxThreads = 16;      // (2 GB from the page cache: 16 threads 71 GB/s, 8 49 GB/s)

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
    memory::stream_copy(dst + done, bounce.get(), len);
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

uint64_t file_size(const std::string& path) {
  std::error_code ec;
  const auto n = std::filesystem::file_size(platform::fs_path(path), ec);
  if (ec) throw Error("cannot read '" + path + "': " + ec.message());
  return n;
}

void page_hint(const uint8_t* map, size_t size, PageHint hint, std::span<const int64_t> offsets, size_t nbytes) {
  for (int64_t at : offsets)
    if (at < 0 || static_cast<size_t>(at) > size || nbytes > size - static_cast<size_t>(at))
      throw Error("page_hint: a range outside the mapping");
  const uintptr_t ps = platform::page_size(), base = reinterpret_cast<uintptr_t>(map);
  const bool inside = hint == PageHint::dont_need;  // (a neighbour's bytes stay; else the pages around the range)
  for (int64_t at : offsets) {
    uintptr_t a = base + static_cast<uintptr_t>(at), b = a + nbytes;
    a = inside ? (a + ps - 1) / ps * ps : a / ps * ps;
    b = inside ? b / ps * ps : (b + ps - 1) / ps * ps;
    if (b <= a) continue;
    void* p = reinterpret_cast<void*>(a);
    if (hint == PageHint::dont_need)
      platform::drop_pages(p, b - a);
    else if (hint == PageHint::will_need)
      platform::will_need(p, b - a);
    else
      platform::populate(p, b - a);
  }
}

}  // namespace artlite
