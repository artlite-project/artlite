#pragma once

// What the weight packers share (tile_pack.cpp, slot_pack.cpp): the item runner, which packs work items on threads
// with their tensors read into the page cache ahead of them.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace artlite::pack {

/// Bytes the items read: a tensor of a file mapping (``file``: an index into the runner's files) or of memory of its
/// own (-1: never read ahead or released).
struct Tensor {
  const uint8_t* data;
  size_t bytes;
  int file = -1;
};

/// A weight file: its path and the start of its whole mapping.
struct MappedFile {
  std::string path;
  const uint8_t* map;
};

/// The items of a packing: item k reads tensors ``tensors_of[begin[k] .. begin[k + 1])`` (indices into ``tensors``,
/// numbered in order of their first use) and ``bytes[k]`` bytes of them (the read-ahead window).
struct Items {
  std::vector<Tensor> tensors;
  std::vector<size_t> begin{0};
  std::vector<uint32_t> tensors_of;
  std::vector<size_t> bytes;

  size_t size() const { return bytes.size(); }
  void add(const std::vector<uint32_t>& ts, size_t nbytes) {
    tensors_of.insert(tensors_of.end(), ts.begin(), ts.end());
    begin.push_back(tensors_of.size());
    bytes.push_back(nbytes);
  }
};

/// Run the items (claimed in order) on ``threads`` threads (0: one per physical core; the caller's among them), each
/// thread packing them with the callable ``make()`` returns (its scratch with it); every thread ends with an sfence
/// (its non-temporal stores visible). An item starts once its file tensors are in the page cache and mapped:
/// kReadThreads threads read them in 8 MB pieces, in order of first use, at most a window ahead of the items (pread:
/// the device's 128 KB readahead makes page faults and MADV_WILLNEED / POPULATE_READ stream from the disk at 1.2 - 1.8
/// GB/s, 16 threads of 8 MB reads at 3.5 GB/s: a cold 27B packs in 5.3 s), then map them (MADV_POPULATE_READ; the
/// workers' own page faults instead: SD 1.5's page-cached 3.4 GB pack in 0.157 s, not 0.125, Qwen3.6-35B-A3B's 21 GB
/// image in 0.70, not 0.64). release: a file tensor's whole pages given back (MADV_DONTNEED; they stay in the page
/// cache) after its last item. Rethrows the first exception.
void run_items(const Items& items, const std::vector<MappedFile>& files, unsigned threads, bool release,
               const std::function<std::function<void(size_t)>()>& make);

}  // namespace artlite::pack
