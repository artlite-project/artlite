// The weight packers' tables and geometry checks (artlite/tile_pack.h, slot_pack.h); a few known tiles. Host only: the
// packers' bit-exactness against the Python references is tests/test_gguf_model.py and tests/vision/test_weights.py.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "artlite/error.h"
#include "artlite/platform.h"
#include "artlite/slot_pack.h"
#include "artlite/tile_pack.h"

namespace {

using artlite::TileFormat;

struct Format {
  const char* name;
  TileFormat fmt;
  size_t block_bytes;
};

// Bytes of a GGUF row per 256 columns (BF16 / F16: 512, F32: 1024)
const Format kFormats[] = {
    {"Q4_K", TileFormat::Q4_K, 144},          {"Q5_K", TileFormat::Q5_K, 176},
    {"Q6_K", TileFormat::Q6_K, 210},          {"Q4_0", TileFormat::Q4_0, 144},
    {"Q8_0", TileFormat::Q8_0, 272},          {"Q4_1", TileFormat::Q4_1, 160},
    {"Q2_K", TileFormat::Q2_K, 84},           {"Q3_K", TileFormat::Q3_K, 110},
    {"IQ3_XXS", TileFormat::IQ3_XXS, 98},     {"IQ3_S", TileFormat::IQ3_S, 110},
    {"IQ4_NL", TileFormat::IQ4_NL, 144},      {"IQ4_XS", TileFormat::IQ4_XS, 136},
    {"BFP16/BF16", TileFormat::BFP16_BF16, 512}, {"BFP16/F16", TileFormat::BFP16_F16, 512},
    {"BFP16/F32", TileFormat::BFP16_F32, 1024},
};

// A BFP16 tile of 8 rows of one super-block, one column block: row r's first element 1.0 (bf16 0x3F80)
struct Bfp16Tile {
  static constexpr int R = 8, nsb = 1;
  static constexpr size_t S = 288 * nsb * R;
  std::vector<uint8_t> rows_data = std::vector<uint8_t>(R * 512, 0);
  std::vector<int64_t> rows{0, 1, 2, 3, 4, 5, 6, 7};
  Bfp16Tile() {
    for (int r = 0; r < R; r++) {
      rows_data[r * 512] = 0x80;
      rows_data[r * 512 + 1] = 0x3F;
    }
  }
  artlite::PackBlock block(size_t offset = 0) const {
    artlite::PackBlock b{};
    b.fmt = TileFormat::BFP16_BF16;
    b.R = R;
    b.S = S;
    b.nsb = nsb;
    b.n_kc = 1;
    b.kc_group = 1;
    b.n_rt = 1;
    b.rt0 = 0;
    b.offset = offset;
    b.col_stride = S;
    b.row_sb = 1;
    b.tensors = {{rows_data.data(), R}};
    b.rows = rows.data();
    b.cols = 1;
    b.n = R;
    return b;
  }
};

}  // namespace

TEST(TilePack, FormatNamesAndBlockBytes) {
  for (const Format& f : kFormats) {
    EXPECT_EQ(artlite::tile_format(f.name), f.fmt) << f.name;
    EXPECT_EQ(artlite::block_bytes(f.fmt), f.block_bytes) << f.name;
  }
  EXPECT_THROW(artlite::tile_format("Q9_K"), artlite::Error);
  EXPECT_THROW(artlite::tile_format("BFP16"), artlite::Error);
}

TEST(TilePack, UsedBytes) {
  EXPECT_EQ(artlite::used_bytes(TileFormat::Q4_K, 8, 4), 144u * 4 * 8);
  EXPECT_EQ(artlite::used_bytes(TileFormat::Q6_K, 8, 4), (2u * 4 * 8 + 31) / 32 * 32 + 208u * 4 * 8);
  EXPECT_EQ(artlite::used_bytes(TileFormat::BFP16_F32, 8, 2), 288u * 2 * 8);
  EXPECT_EQ(artlite::used_bytes(TileFormat::IQ4_NL, 8, 2), 144u * 2 * 8);
  EXPECT_EQ(artlite::used_bytes(TileFormat::IQ4_NL, 12, 2), ~size_t(0));  // (R a multiple of 8 only)
  EXPECT_EQ(artlite::used_bytes(TileFormat::IQ3_S, 8, 2), ~size_t(0));    // (R 17 .. 20 only)
  for (const Format& f : kFormats) EXPECT_GT(artlite::used_bytes(f.fmt, 8, 2), 0u) << f.name;
}

TEST(TilePack, PacksABfp16Tile) {
  Bfp16Tile t;
  std::vector<uint8_t> out(Bfp16Tile::S + 128, 0xCC);
  std::vector<artlite::PackBlock> blocks{t.block(64)};
  artlite::PackOptions opt;
  opt.threads = 1;
  artlite::pack_tiles(out.data(), out.size(), blocks, opt);
  for (size_t i = 0; i < 64; i++) ASSERT_EQ(out[i], 0xCC) << i;  // (outside the block: as it was)
  for (size_t i = 64 + Bfp16Tile::S; i < out.size(); i++) ASSERT_EQ(out[i], 0xCC) << i;
  const uint8_t* tile = out.data() + 64;
  for (int r = 0; r < 8; r++) {
    // the block of row r's columns 0 .. 7 at (k 0, r): exponent 127 (1.0), mantissa 1.0 x 2^6, zeros
    EXPECT_EQ(tile[r * 9], 127) << r;
    EXPECT_EQ(tile[r * 9 + 1], 64) << r;
    for (int j = 2; j < 9; j++) EXPECT_EQ(tile[r * 9 + j], 0) << r << " " << j;
  }
  for (size_t i = 8 * 9; i < Bfp16Tile::S; i++) ASSERT_EQ(tile[i], 0) << i;  // (the other blocks: zeros)
}

TEST(TilePack, RefusesInconsistentGeometry) {
  Bfp16Tile t;
  std::vector<uint8_t> out(4 * Bfp16Tile::S, 0);
  auto pack = [&](artlite::PackBlock b, size_t out_size = 0) {
    std::vector<artlite::PackBlock> blocks{std::move(b)};
    artlite::PackOptions opt;
    opt.threads = 1;
    artlite::pack_tiles(out.data(), out_size ? out_size : out.size(), blocks, opt);
  };
  EXPECT_NO_THROW(pack(t.block()));
  artlite::PackBlock b = t.block();
  b.S = Bfp16Tile::S - 64;  // (a tile smaller than its format's bytes)
  b.col_stride = b.S;
  EXPECT_THROW(pack(b), artlite::Error);
  b = t.block();
  b.R = 4;  // (BFP16: 8 rows)
  EXPECT_THROW(pack(b), artlite::Error);
  b = t.block();
  b.nsb = artlite::kMaxNsb + 1;
  EXPECT_THROW(pack(b), artlite::Error);
  b = t.block();
  b.col_stride = Bfp16Tile::S - 64;  // (columns' blocks overlapping)
  EXPECT_THROW(pack(b), artlite::Error);
  b = t.block();
  b.n = 12;  // (rows not whole tiles)
  EXPECT_THROW(pack(b), artlite::Error);
  EXPECT_THROW(pack(t.block(), Bfp16Tile::S - 64), artlite::Error);  // (the block past the image)
  EXPECT_THROW(pack(t.block(3 * Bfp16Tile::S + 64)), artlite::Error);
  b = t.block();
  std::vector<int64_t> bad{0, 1, 2, 3, 4, 5, 6, 8};  // (a row past the tensors')
  b.rows = bad.data();
  EXPECT_THROW(pack(b), artlite::Error);
  b = t.block();
  b.row_sb = 2;  // (more super-blocks than the chunks hold)
  EXPECT_THROW(pack(b), artlite::Error);
}

TEST(SlotPack, GeometryAndSourceTypes) {
  using artlite::SlotSource;
  EXPECT_EQ(artlite::kSlotBytes, 5760u);
  EXPECT_EQ(artlite::kPassRows, 80);
  EXPECT_EQ(artlite::slot_job_bytes(3, 4), 12u * 5760);
  EXPECT_EQ(artlite::slot_source("F32"), SlotSource::F32);
  EXPECT_EQ(artlite::slot_source("F16"), SlotSource::F16);
  EXPECT_EQ(artlite::slot_source("BF16"), SlotSource::BF16);
  EXPECT_EQ(artlite::slot_source("Q8_0"), SlotSource::Q8_0);
  EXPECT_EQ(artlite::slot_source("EYE"), SlotSource::EYE);
  EXPECT_THROW(artlite::slot_source("F64"), artlite::Error);
  EXPECT_EQ(artlite::slot_row_bytes(SlotSource::F32, 64), 256u);
  EXPECT_EQ(artlite::slot_row_bytes(SlotSource::F16, 64), 128u);
  EXPECT_EQ(artlite::slot_row_bytes(SlotSource::BF16, 64), 128u);
  EXPECT_EQ(artlite::slot_row_bytes(SlotSource::Q8_0, 64), 68u);
  EXPECT_EQ(artlite::slot_row_bytes(SlotSource::EYE, 64), 0u);
}

namespace {

// The identity matrix of 80 x 64 as one job of one pass and one k-tile at dst (kSlotBytes), the parts' rows 0 .. 79
struct EyeJob {
  std::vector<artlite::SlotTensor> tensors{{artlite::SlotSource::EYE, nullptr, 80, 64, 0, -1}};
  std::vector<int32_t> rows, runs{0, 64, 0, 1};  // (a run: first column, columns, source column, stride)
  std::vector<artlite::SlotPart> parts;
  std::vector<artlite::SlotJob> jobs;
  EyeJob(uint8_t* dst, int nkt = 1, bool hilo = false) {
    for (int i = 0; i < 80; i++) rows.push_back(i);
    parts.push_back({0, 1.0f, rows.data(), runs.data(), 1});
    jobs.push_back({dst, 1, nkt, hilo, 0, 1});
  }
  void pack() const { artlite::pack_slots(tensors, parts, jobs, {}, 1); }
};

}  // namespace

TEST(SlotPack, PacksTheIdentity) {
  auto buf = artlite::platform::aligned_buffer<uint8_t>(64, artlite::kSlotBytes + 128);
  std::memset(buf.get(), 0xCC, artlite::kSlotBytes + 128);
  uint8_t* dst = buf.get() + 64;
  EyeJob job(dst);
  job.pack();
  for (size_t i = 0; i < 64; i++) ASSERT_EQ(buf.get()[i], 0xCC) << i;
  for (size_t i = 64 + artlite::kSlotBytes; i < artlite::kSlotBytes + 128; i++) ASSERT_EQ(buf.get()[i], 0xCC) << i;
  // row i's block of columns 0 .. 7 at ((i / 16) 16 + (i / 8) % 2) 8 + i % 8, 9 bytes: row 0 holds 1.0 in its first
  // element (exponent 127, mantissa 64), row 1 in its second
  EXPECT_EQ(dst[0], 127);
  EXPECT_EQ(dst[1], 64);
  for (int j = 2; j < 9; j++) EXPECT_EQ(dst[j], 0) << j;
  EXPECT_EQ(dst[9], 127);
  EXPECT_EQ(dst[10], 0);
  EXPECT_EQ(dst[11], 64);
  EXPECT_EQ(dst[72], 0);  // (row 8's block of columns 0 .. 7 is zero: exponent byte 0)
}

TEST(SlotPack, RefusesInconsistentJobs) {
  auto buf = artlite::platform::aligned_buffer<uint8_t>(64, 4 * artlite::kSlotBytes);
  EyeJob ok(buf.get());
  EXPECT_NO_THROW(ok.pack());
  {
    EyeJob j(buf.get(), 3, true);  // (hilo: an even number of k-tiles)
    EXPECT_THROW(j.pack(), artlite::Error);
  }
  {
    EyeJob j(buf.get());
    j.jobs[0].npo = 0;
    EXPECT_THROW(j.pack(), artlite::Error);
  }
  {
    EyeJob j(buf.get());
    j.jobs[0].n_parts = 2;  // (a part past the parts')
    EXPECT_THROW(j.pack(), artlite::Error);
  }
  {
    EyeJob j(buf.get());
    j.parts[0].tensor = 1;
    EXPECT_THROW(j.pack(), artlite::Error);
  }
  {
    EyeJob j(buf.get());
    j.rows[79] = 80;  // (a row past the tensor's)
    EXPECT_THROW(j.pack(), artlite::Error);
  }
  {
    EyeJob j(buf.get());
    j.runs = {0, 65, 0, 1};  // (a run past the job's 64 columns)
    j.parts[0].runs = j.runs.data();
    EXPECT_THROW(j.pack(), artlite::Error);
  }
  {
    EyeJob j(buf.get());
    j.runs = {0, 64, 1, 1};  // (a run reading past the tensor's columns)
    j.parts[0].runs = j.runs.data();
    EXPECT_THROW(j.pack(), artlite::Error);
  }
}
