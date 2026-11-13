#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "artlite/pack_run.h"

namespace artlite {

/// The geometry of a core row's weight image (``vision.layout``): a k-tile is kTileCols columns of a pass's kPassRows
/// output rows, as B slots of bfp16ebs8 blocks (8 values: an exponent byte and 8 mantissa bytes) of kSlotBytes.
constexpr int kPassRows = 80;
constexpr int kTileCols = 64;
constexpr size_t kSlotBytes = size_t(kPassRows) * kTileCols / 8 * 9;
static_assert(kSlotBytes == 5760);  // (the overlay's B slot)

/// The element types of a weight source (``artlite.vision.weights.Source``): a file's F32 / F16 / BF16 / Q8_0 rows, or
/// the identity matrix (no data).
enum class SlotSource { F32, F16, BF16, Q8_0, EYE };

/// Bytes of a job's slots: ``passes`` passes of ``ktiles`` k-tiles (hilo: both halves' tiles counted).
constexpr size_t slot_job_bytes(size_t passes, size_t ktiles) { return passes * ktiles * kSlotBytes; }

/// "F32", "F16", "BF16", "Q8_0", "EYE" -> the source type (throws on others).
SlotSource slot_source(std::string_view name);

/// Bytes of a row of ``cols`` elements of a source type (EYE: 0; Q8_0: cols a multiple of 32).
size_t slot_row_bytes(SlotSource type, int64_t cols);

/// A source matrix of rows x cols elements, row r at data + r row_bytes; file: its mapping in pack_slots' files (-1:
/// memory of its own).
struct SlotTensor {
  SlotSource type;
  const uint8_t* data;
  int64_t rows, cols;
  size_t row_bytes;
  int file = -1;
};

/// A part of a job's matrix: per output row (npo kPassRows of them) its row of tensor ``tensor`` (-1: zeros) times scale,
/// over the runs [k0, n, src0, stride] of output columns k0 .. k0 + n - 1 = the row's elements src0 + i stride (columns
/// of no run: zeros; parts' runs do not overlap).
struct SlotPart {
  int tensor;
  float scale;
  const int32_t* rows;
  const int32_t* runs;
  int n_runs;
};

/// A core row's weight image (``vision.layout``): npo passes of kPassRows output rows, each its nkt k-tiles of kTileCols
/// columns as B slots [16-row block 5][k-step 8][half 2][row 8] of bfp16ebs8 blocks (kSlotBytes a slot; a block 8
/// consecutive columns of one row: [E, m0 .. m7]), at dst. hilo: the matrix has nkt / 2 k-tiles, each pass's slots are those of
/// hi = bfp16(w), then of lo = bfp16(w - hi).
struct SlotJob {
  uint8_t* dst;
  int npo, nkt;
  bool hilo;
  int part0, n_parts;  // (its parts: parts[part0 .. part0 + n_parts))
};

/// Pack every job (bit-identical to the host reference's bfp16 blocks of the same fp32 values: a source element
/// converted to fp32 exactly, Q8_0's d q in fp32, times the part's scale in fp32), on threads over items of a pass's
/// 16 k-tiles, non-temporal stores (no cache flush needed for the device after it); the file tensors read into the
/// page cache ahead of the items, their pages given back after their last item with ``release`` (pack::run_items).
void pack_slots(const std::vector<SlotTensor>& tensors, const std::vector<SlotPart>& parts,
                const std::vector<SlotJob>& jobs, const std::vector<pack::MappedFile>& files, unsigned threads = 0,
                bool release = false);

}  // namespace artlite
