#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artlite/pack_run.h"

namespace artlite {

/// The tile formats of ``artlite.formats`` with a native packer (every slot's ``pack``; BFP16 of BF16, F16 and
/// F32 tensors).
enum class TileFormat {
  Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, Q4_1, Q2_K, Q3_K, IQ3_XXS, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ4_NL,
  IQ4_XS, Q6_KH, Q4_1_64, P4_128, P5_128, NVFP4, Q8_0H, BFP16_BF16, BFP16_F16, BFP16_F32
};

/// "Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q8_0", "Q4_1", "Q2_K", "Q3_K", "IQ3_XXS", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS",
/// "IQ2_S", "IQ3_S", "IQ4_NL", "IQ4_XS", "Q6_KH" (Q6_K's high plane, from Q6_K blocks), "Q4_1_64" (from Q4_1 blocks in
/// equal pairs), "P4_128", "P5_128" (from awq_rows' blocks), "NVFP4", "Q8_0H" (Q8_0's high plane, from Q8_0 blocks),
/// "BFP16/BF16", "BFP16/F16", "BFP16/F32" -> the format (throws on
/// others).
TileFormat tile_format(std::string_view name);

/// Bytes of a GGUF row per 256 columns of a format (its blocks; BF16 / F16: 512, F32: 1024).
size_t block_bytes(TileFormat f);

/// Bytes a tile of R rows of nsb super-blocks uses of its format's S (``~size_t(0)``: an R the format has no tile for).
size_t used_bytes(TileFormat f, int R, int nsb);

/// Most super-blocks of a tile.
constexpr int kMaxNsb = 16;

/// A run of row tiles in every column block of a weight image's region (``decode.image.Block``): column c's
/// n rows (indices into the stacked tensors, -1: a zero row) as n / (R G) row tiles of S bytes over n_kc K chunks
/// of 256 nsb columns, row tile rt's tile of chunk kc at ``Region.tile(rt0 + rt, kc)`` of the column block
/// (column c's block at offset + c col_stride). A tensor row holds row_sb super-blocks (at most n_kc nsb): the
/// columns past them are zero super-blocks (a K padded to whole chunks). G > 1 (row groups: one chunk, G row_sb = nsb):
/// a tile row is G tensor rows, consecutive in ``rows``, one after the other along K. BFP16 of F32 tensors: row index
/// 2 i + p is plane p of stacked row i (0: bfp16(w), 1: bfp16(w - plane 0)).
///
/// tile0 >= 0 (a ``Region.half`` region's block, one K group: a row tile's chunks consecutive): the block's tiles from
/// tile tile0 of the column block on, col_tiles tiles a column block; row tile rt's chunks at tile0 + rt n_kc (pair 0),
/// or (pair 1) in pairs of 2 n_kc - 1 tiles: row tile 2 q's full chunks, the pair's tail tile, row tile 2 q + 1's full
/// chunks, the tail tile's tile rows those of 2 q's rows (its first nsb / 2 super-blocks) and 2 q + 1's (the others)
/// over the tensor rows' last nsb / 2 super-blocks (row_sb = (n_kc - 1) nsb + nsb / 2); an odd last row tile's
/// second half zero super-blocks.
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
  int row_sb;            // super-blocks of a tensor row
  std::vector<std::pair<const uint8_t*, int64_t>> tensors;  // stacked (GGUF blocks, rows) of row_sb blocks each
  const int64_t* rows;   // (cols, n)
  int cols;
  int64_t n;
  int G = 1;             // tensor rows per tile row
  int64_t tile0 = -1;    // (>= 0) the block's first tile in a column block: its tiles by index (see above)
  int pair = 0;          // (tile0) row tiles in pairs sharing their tail tile
  int64_t col_tiles = 0; // (tile0) tiles of a column block
};

struct PackOptions {
  unsigned threads = 0;  // 0: one per physical core
  /// The files the tensors lie in and the caller's mappings of them (pack::MappedFile): a tensor inside a mapping is read
  /// through views of the packer's own, read into the page cache a window ahead of the workers (a file not in the page
  /// cache streams from the disk while they pack) and unmapped tensor by tensor (pack::Views); a tensor outside every
  /// mapping is memory of the caller's, read as it is.
  std::vector<pack::MappedFile> files;
  /// IQ1_S tiles' grid masks (``formats.iq1.grid_masks``: 2048 entries A | B << 8): the blocks of IQ1_S need them.
  const uint16_t* iq1_masks = nullptr;
};

/// Write the tiles of every block into ``out`` (the PAD bytes and other gaps are left as they are), bit-identical
/// to the slots' ``pack_tile``. Threads take items of about 2048 rows of a block's column; the tiles go out with
/// non-temporal stores (the CPU never reads the image back: no flush needed for the device after it).
void pack_tiles(uint8_t* out, size_t out_size, const std::vector<PackBlock>& blocks, const PackOptions& opt = {});

}  // namespace artlite
