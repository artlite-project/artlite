#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace artlite {

/// The tile formats of ``artlite.operators.formats`` with a native packer (every slot's ``pack``; BFP16 of BF16 and of
/// F16 tensors).
enum class TileFormat { Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, Q4_1, Q2_K, Q3_K, BFP16_BF16, BFP16_F16 };

/// "Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q8_0", "Q4_1", "Q2_K", "Q3_K", "BFP16/BF16", "BFP16/F16" -> the format (throws on
/// others).
TileFormat tile_format(std::string_view name);

/// Bytes of a GGUF row per 256 columns of a format (its blocks; BF16 / F16: 512).
size_t block_bytes(TileFormat f);

/// A run of row tiles in every column block of a weight image's region (``overlay.image.Block``): column c's
/// n rows (indices into the stacked tensors, -1: a zero row) as n / R row tiles of S bytes over n_kc K chunks
/// of 256 nsb columns, row tile rt's tile of chunk kc at ``Region.tile(rt0 + rt, kc)`` of the column block
/// (column c's block at offset + c col_stride).
struct PackBlock {
  TileFormat fmt;
  int R;
  size_t S;
  int nsb;
  int n_kc;
  int kc_group;          // the region's K chunks per group
  int n_rt;              // the region's row tiles
  int rt0;               // the block's first row tile
  size_t offset;         // the region's first byte in the image
  size_t col_stride;     // bytes from a column block to the next (at least the block's)
  std::vector<std::pair<const uint8_t*, int64_t>> tensors;  // stacked (GGUF blocks, rows) of n_kc nsb blocks each
  const int64_t* rows;   // (cols, n)
  int cols;
  int64_t n;
};

struct PackOptions {
  unsigned threads = 0;  // 0: one per physical core
  /// Give back each tensor's pages (MADV_DONTNEED of the whole pages inside it) once its tiles are written: for
  /// tensors of a file mapping only (the pages stay in the page cache), never for anonymous memory.
  bool release = false;
  /// The file the tensors are mapped from and the start of its mapping: the tensors are read into the page cache a
  /// window ahead of the workers (a file not in the page cache streams from the disk while they pack).
  std::string file;
  const uint8_t* map = nullptr;
};

/// Write the tiles of every block into ``out`` (the PAD bytes and other gaps are left as they are), bit-identical
/// to the slots' ``pack_tile``. Threads take items of about 2048 rows of a block's column; the tiles go out with
/// non-temporal stores (the CPU never reads the image back: no flush needed for the device after it).
void pack_tiles(uint8_t* out, size_t out_size, const std::vector<PackBlock>& blocks, const PackOptions& opt = {});

}  // namespace artlite
