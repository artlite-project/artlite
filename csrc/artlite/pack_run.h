#pragma once

// What the weight packers share (tile_pack.cpp, slot_pack.cpp): the item runner, which packs work items on threads
// with their tensors read into the page cache ahead of them, and the views it reads a weight file through.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace artlite::platform {
class File;
}

namespace artlite::pack {

/// Bytes the items read: a tensor of a file (``file``: an index into the packing's files; ``data``: where the caller
/// maps it, and from ``Views`` on where the packer does) or a block of memory outside the files (-1).
struct Tensor {
  const uint8_t* data;
  size_t bytes;
  int file = -1;
};

/// A weight file: its path and the caller's mapping of the whole file (which says where in the file a tensor lies:
/// the packer reads none of its bytes through it).
struct MappedFile {
  std::string path;
  const uint8_t* map;
  size_t size;
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

/// The file tensors of a packing in views of the packer's own, one per tensor (``items.tensors[t].data`` becomes the
/// view's bytes: what the items' bodies read), whose pages leave the process after the tensor's last item
/// (``run_items``: Linux gives them back and unmaps with the object, Windows unmaps the view then). So a packing
/// leaves none of the file's pages in the process, on any system: the caller's mapping is never touched, and Windows,
/// which has no MADV_DONTNEED, needs none. Throws Error for a tensor outside its file's mapping, std::system_error
/// for a file that cannot be opened or mapped.
class Views {
 public:
  Views(Items& items, const std::vector<MappedFile>& files);
  ~Views();
  Views(const Views&) = delete;
  Views& operator=(const Views&) = delete;

 private:
  friend void run_items(const Items&, Views&, unsigned, const std::function<std::function<void(size_t)>()>&);

  struct View {
    void* base = nullptr;  // (the mapping; nullptr: none, or unmapped)
    size_t bytes = 0;
    uint64_t offset = 0;  // (the file offset of the tensor's first byte)
  };
  void drop(size_t tensor);

  std::vector<View> views_;  // (per tensor)
  std::vector<std::unique_ptr<platform::File>> files_;
};

/// Run the items (claimed in order) on ``threads`` threads (0: one per physical core; the caller's among them), each
/// thread packing them with the callable ``make()`` returns (its scratch with it); every thread ends with an sfence
/// (its non-temporal stores visible). An item starts once its file tensors are in the page cache and mapped:
/// kReadThreads threads read them in 8 MB pieces, in order of first use, at most a window ahead of the items (pread:
/// the device's 128 KB readahead makes page faults and MADV_WILLNEED / POPULATE_READ stream from the disk at 1.2 - 1.8
/// GB/s, 16 threads of 8 MB reads at 3.5 GB/s: a cold 27B packs in 5.3 s), then map them (MADV_POPULATE_READ; the
/// workers' own page faults instead: SD 1.5's page-cached 3.4 GB pack in 0.157 s, not 0.125, Qwen3.6-35B-A3B's 21 GB
/// image in 0.70, not 0.64). A file tensor's pages leave the process after its last item (``Views``): it holds the
/// window's pages of the file, never the file. Rethrows the first exception.
void run_items(const Items& items, Views& views, unsigned threads,
               const std::function<std::function<void(size_t)>()>& make);

}  // namespace artlite::pack
