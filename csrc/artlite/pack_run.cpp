// The weight packers' item runner (pack_run.h).

#include "artlite/pack_run.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>

#include "artlite/error.h"
#include "artlite/platform.h"

namespace artlite::pack {

namespace {

constexpr size_t kReadWindow = size_t(512) << 20;
constexpr size_t kReadPiece = size_t(8) << 20;
constexpr unsigned kReadThreads = 16;

/// The whole pages around [p, p + n).
std::pair<uintptr_t, size_t> pages(const uint8_t* p, size_t n) {
  const uintptr_t ps = static_cast<uintptr_t>(platform::page_size()), at = reinterpret_cast<uintptr_t>(p);
  const uintptr_t a = at / ps * ps, b = (at + n + ps - 1) / ps * ps;
  return {a, b - a};
}

}  // namespace

Views::Views(Items& items, const std::vector<MappedFile>& files) : views_(items.tensors.size()), files_(files.size()) {
  const uint64_t unit = platform::view_granularity();
  for (size_t t = 0; t < items.tensors.size(); t++) {
    Tensor& x = items.tensors[t];
    if (x.file < 0 || x.bytes == 0) continue;
    const MappedFile& f = files.at(static_cast<size_t>(x.file));
    if (x.data < f.map || x.bytes > f.size || static_cast<size_t>(x.data - f.map) > f.size - x.bytes)
      throw Error("pack: a tensor outside the mapping of its file " + f.path);
    if (!files_[x.file]) files_[x.file] = std::make_unique<platform::File>(f.path);
    const uint64_t offset = static_cast<uint64_t>(x.data - f.map), start = offset / unit * unit;
    View& v = views_[t];
    v.bytes = static_cast<size_t>(offset - start) + x.bytes;
    v.base = files_[x.file]->map(start, v.bytes);
    v.offset = offset;
    x.data = static_cast<const uint8_t*>(v.base) + (offset - start);
  }
}

Views::~Views() {
  for (View& v : views_)
    if (v.base) platform::File::unmap(v.base, v.bytes);
}

void Views::drop(size_t tensor) {
  View& v = views_[tensor];
  if (!v.base) return;
#ifdef _WIN32
  platform::File::unmap(v.base, v.bytes);  // (what takes a view's pages out of the working set for certain)
  v.base = nullptr;
#else
  // (the pages now, the mapping with the object: an munmap holds mmap_lock exclusively, and a thousand of them between
  // the other threads' populates and page faults made a page-cached 27B pack in 0.60 s instead of 0.41)
  const size_t ps = platform::page_size();
  platform::drop_pages(v.base, (v.bytes + ps - 1) / ps * ps);
#endif
}

void run_items(const Items& items, Views& views, unsigned threads,
               const std::function<std::function<void(size_t)>()>& make) {
  struct State {
    int64_t left = 0;                  // items
    std::atomic<int64_t> pending{0};   // pieces not read yet
  };
  const size_t n_tensors = items.tensors.size();
  auto state = std::make_unique<State[]>(n_tensors);
  for (size_t k = 0; k < items.size(); k++)
    for (size_t i = items.begin[k]; i < items.begin[k + 1]; i++) state[items.tensors_of[i]].left++;

  std::atomic<size_t> next{0}, done_bytes{0};
  std::atomic<size_t> needed{0};  // (tensors 0 .. needed - 1: a claimed item reads them; the readers never hold them
                                  // back)
  std::atomic<bool> failed{false}, finished{false};
  std::mutex mutex;  // (the tensors' counts, the first error)
  std::exception_ptr error;

  auto worker = [&] {
    try {
      auto body = make();
      for (size_t k; !failed && (k = next++) < items.size();) {
        const uint32_t *t0 = items.tensors_of.data() + items.begin[k], *t1 = items.tensors_of.data() + items.begin[k + 1];
        for (const uint32_t* t = t0; t < t1; t++) {
          size_t n = needed;
          while (n < *t + size_t(1) && !needed.compare_exchange_weak(n, *t + size_t(1))) {
          }
        }
        for (const uint32_t* t = t0; t < t1; t++)  // (read by the readers: no page fault reads it in 128 KB)
          while (state[*t].pending > 0 && !failed) std::this_thread::sleep_for(std::chrono::microseconds(100));
        body(k);
        done_bytes += items.bytes[k];
        std::vector<uint32_t> last;  // (the tensors this was the last item of: their pages stay in the page cache)
        {
          std::lock_guard lock(mutex);
          for (const uint32_t* t = t0; t < t1; t++)
            if (--state[*t].left == 0) last.push_back(*t);
        }
        for (uint32_t t : last) views.drop(t);
      }
      _mm_sfence();
    } catch (...) {
      std::lock_guard lock(mutex);
      if (!error) error = std::current_exception();
      failed = true;
    }
  };

  // the readers: the file tensors in pieces, in order of first use, each read into the page cache unless mincore
  // finds it resident, at most a window ahead of the workers but for the tensors of claimed items (the window counts
  // the items' bytes, which can fall behind the tensors': workers waiting on a held-back tensor would never finish
  // one)
  struct Piece {
    const uint8_t* data;
    size_t bytes, tensor;
  };
  std::vector<Piece> pieces;
  size_t window = kReadWindow;  // (>= a tensor: the workers of its first item wait for all of it)
  for (size_t t = 0; t < n_tensors; t++) {
    const Tensor& x = items.tensors[t];
    if (!views.views_[t].base || state[t].left == 0) continue;
    window = std::max(window, x.bytes);
    for (size_t o = 0; o < x.bytes; o += kReadPiece) pieces.push_back({x.data + o, std::min(kReadPiece, x.bytes - o), t});
    state[t].pending = -(-int64_t(x.bytes) / int64_t(kReadPiece));
  }
  std::atomic<size_t> next_piece{0}, issued{0};
  auto reader = [&] {
    const size_t ps = platform::page_size();
    auto buf = platform::aligned_buffer(ps, kReadPiece + ps);
    std::vector<unsigned char> resident(kReadPiece / ps + 2);
    for (size_t k; !finished && !failed && (k = next_piece++) < pieces.size();) {
      const Piece& pc = pieces[k];
      while (!finished && !failed && issued > done_bytes + window && pc.tensor >= needed)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      issued += pc.bytes;
      const Tensor& x = items.tensors[pc.tensor];
      auto [a, len] = pages(pc.data, pc.bytes);
      // (the file offset of the view's address a: the tensor's, less what a lies before its first byte)
      const uint64_t at = views.views_[pc.tensor].offset + a - reinterpret_cast<uintptr_t>(x.data);
      if (buf && platform::needs_reading(reinterpret_cast<void*>(a), len, resident.data()))
        for (size_t got = 0; got < len;) {  // (errors, no buffer: the workers' page faults read it)
          size_t r = 0;
          try {
            r = views.files_[x.file]->read_at(buf.get(), len - got, at + got);
          } catch (const std::system_error&) {
          }
          if (r == 0) break;
          got += r;
        }
      platform::populate(reinterpret_cast<void*>(a), len);  // (mapped: the workers take no page faults)
      state[pc.tensor].pending--;
    }
  };

  threads = threads ? threads : std::max(1u, std::thread::hardware_concurrency() / 2);
  threads = static_cast<unsigned>(std::min<size_t>(threads, std::max<size_t>(1, items.size())));
  std::vector<std::thread> pool, readers;
  for (unsigned i = 0; !pieces.empty() && i < kReadThreads; i++) readers.emplace_back(reader);
  for (unsigned i = 1; i < threads; i++) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  finished = true;
  for (auto& th : readers) th.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace artlite::pack
