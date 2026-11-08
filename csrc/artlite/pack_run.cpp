// The weight packers' item runner (pack_run.h).

#include "artlite/pack_run.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>

#include "artlite/platform.h"

namespace artlite::pack {

namespace {

constexpr size_t kReadWindow = size_t(2) << 30;
constexpr size_t kReadPiece = size_t(8) << 20;
constexpr unsigned kReadThreads = 16;

/// The whole pages inside [p, p + n) (MADV_DONTNEED must not touch a neighbour's) or around it (reading).
std::pair<uintptr_t, size_t> pages(const uint8_t* p, size_t n, bool inside) {
  const uintptr_t ps = static_cast<uintptr_t>(platform::page_size());
  uintptr_t a = reinterpret_cast<uintptr_t>(p), b = a + n;
  a = inside ? (a + ps - 1) / ps * ps : a / ps * ps;
  b = inside ? b / ps * ps : (b + ps - 1) / ps * ps;
  return {a, b > a ? b - a : 0};
}

}  // namespace

void run_items(const Items& items, const std::vector<MappedFile>& files, unsigned threads, bool release,
               const std::function<std::function<void(size_t)>()>& make) {
  struct State {
    int64_t left = 0;                  // items (release)
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
        if (release) {
          std::vector<uint32_t> drop;
          {
            std::lock_guard lock(mutex);
            for (const uint32_t* t = t0; t < t1; t++)
              if (--state[*t].left == 0 && items.tensors[*t].file >= 0) drop.push_back(*t);
          }
          for (uint32_t t : drop) {  // (the pages stay in the page cache)
            auto [a, n] = pages(items.tensors[t].data, items.tensors[t].bytes, true);
            if (n) platform::drop_pages(reinterpret_cast<void*>(a), n);
          }
        }
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
  std::vector<std::unique_ptr<platform::File>> open(files.size());
  struct Piece {
    const uint8_t* data;
    size_t bytes, tensor;
  };
  std::vector<Piece> pieces;
  size_t window = kReadWindow;  // (>= a tensor: the workers of its first item wait for all of it)
  for (size_t t = 0; t < n_tensors; t++) {
    const Tensor& x = items.tensors[t];
    if (x.file < 0 || state[t].left == 0) continue;
    if (!open[x.file]) open[x.file] = std::make_unique<platform::File>(files[x.file].path);
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
      auto [a, len] = pages(pc.data, pc.bytes, false);
      if (buf && platform::needs_reading(reinterpret_cast<void*>(a), len, resident.data()))
        for (size_t got = 0; got < len;) {  // (errors, no buffer: the workers' page faults read it)
          size_t r = 0;
          try {
            r = open[x.file]->read_at(buf.get(), len - got, a - reinterpret_cast<uintptr_t>(files[x.file].map) + got);
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
