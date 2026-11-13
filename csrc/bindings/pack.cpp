#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "artlite/error.h"
#include "artlite/slot_pack.h"
#include "artlite/tile_pack.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace artlite::bindings {

void bind_pack(nb::module_& m) {
  m.def(
      "pack_tiles",
      [](nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out, nb::list blocks, unsigned threads,
         bool release, std::optional<std::pair<std::string, nb::ndarray<const uint8_t, nb::ndim<1>, nb::device::cpu>>> file) {
        using Rows = nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        using Raw = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        std::vector<artlite::PackBlock> bs;
        for (nb::handle h : blocks) {
          auto t = nb::cast<nb::tuple>(h);
          if (t.size() != 14) throw artlite::Error("pack_tiles: a block is a tuple of 14");
          artlite::PackBlock b{};
          b.fmt = artlite::tile_format(nb::cast<std::string>(t[0]));
          b.R = nb::cast<int>(t[1]);
          b.S = nb::cast<size_t>(t[2]);
          b.nsb = nb::cast<int>(t[3]);
          b.n_kc = nb::cast<int>(t[4]);
          b.kc_group = nb::cast<int>(t[5]);
          b.n_rt = nb::cast<int>(t[6]);
          b.rt0 = nb::cast<int>(t[7]);
          b.offset = nb::cast<size_t>(t[8]);
          b.col_stride = nb::cast<size_t>(t[9]);
          b.row_sb = nb::cast<int>(t[12]);
          b.G = nb::cast<int>(t[13]);
          const size_t row_bytes = size_t(b.row_sb) * artlite::block_bytes(b.fmt);
          for (nb::handle r : nb::cast<nb::list>(t[10])) {
            auto raw = nb::cast<Raw>(r, false);
            if (raw.shape(1) != row_bytes)
              throw artlite::Error(std::format("pack_tiles: tensor rows of {} bytes, the block's are {}",
                                               raw.shape(1), row_bytes));
            b.tensors.emplace_back(raw.data(), static_cast<int64_t>(raw.shape(0)));
          }
          auto rows = nb::cast<Rows>(t[11], false);
          b.rows = rows.data();
          b.cols = static_cast<int>(rows.shape(0));
          b.n = static_cast<int64_t>(rows.shape(1));
          bs.push_back(std::move(b));
        }
        artlite::PackOptions opt{threads, release};
        if (file) {
          opt.file = file->first;
          opt.map = file->second.data();
        }
        nb::gil_scoped_release release_gil;
        artlite::pack_tiles(out.data(), out.shape(0), bs, opt);
      },
      "out"_a, "blocks"_a, "threads"_a = 0, "release"_a = false, "file"_a = nb::none(),
      nb::sig("def pack_tiles(out: numpy.ndarray[numpy.uint8], blocks: list[tuple], threads: int = 0, "
              "release: bool = False, file: tuple[str, numpy.ndarray[numpy.uint8]] | None = None) -> None"),
      "Pack weight tiles (every tile format: bit-identical to the formats' pack_tile) into out. A block: "
      "(format, R, S, nsb, n_kc, kc_group, n_rt, rt0, offset, col_stride, tensors, rows, row_sb, G) as "
      "decode.image.TileImage.pack builds it (format: a tile format, BFP16 as \"BFP16/BF16\", \"BFP16/F16\" or "
      "\"BFP16/F32\" by its tensors' type; tensors: (rows, row bytes) uint8 arrays of GGUF blocks, stacked; rows: "
      "(columns, n) int64 stacked row indices, -1 a zero row (BFP16/F32: 2 i + p is plane p of row i); row_sb: a "
      "tensor row's super-blocks, at most n_kc nsb: zero super-blocks past them; G: tensor rows per tile row, "
      "consecutive in rows, along K). threads: 0 = one per physical core; release: "
      "MADV_DONTNEED each tensor's pages when done (file mappings only); file: (path, its whole mapping) of the "
      "tensors: read into the page cache ahead of the workers.");

  m.def(
      "pack_slots",
      [](nb::list dsts, nb::list tensors, nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> jobs,
         nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> parts,
         nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> scales,
         nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> rows,
         nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> runs, nb::list files, unsigned threads,
         bool release) {
        using Out = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
        using Raw = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        using Map = nb::ndarray<const uint8_t, nb::ndim<1>, nb::device::cpu>;
        std::vector<std::pair<uint8_t*, size_t>> out;
        for (nb::handle h : dsts) {
          auto a = nb::cast<Out>(h, false);
          out.emplace_back(a.data(), a.shape(0));
        }
        std::vector<artlite::pack::MappedFile> fs;
        for (nb::handle h : files) {
          auto t = nb::cast<nb::tuple>(h);
          fs.push_back({nb::cast<std::string>(t[0]), nb::cast<Map>(t[1], false).data()});
        }
        std::vector<artlite::SlotTensor> ts;
        for (nb::handle h : tensors) {
          auto t = nb::cast<nb::tuple>(h);
          if (t.size() != 5) throw artlite::Error("pack_slots: a tensor is (type, raw, rows, cols, file)");
          const std::string type = nb::cast<std::string>(t[0]);
          artlite::SlotTensor x{};
          x.type = artlite::slot_source(type);
          x.rows = nb::cast<int64_t>(t[2]);
          x.cols = nb::cast<int64_t>(t[3]);
          x.file = nb::cast<int>(t[4]);
          const size_t rb = artlite::slot_row_bytes(x.type, x.cols);
          if (x.rows <= 0 || x.cols <= 0 || x.cols >= (int64_t(1) << 31) ||
              (x.type == artlite::SlotSource::Q8_0 && x.cols % 32) || x.file < -1 || x.file >= int(fs.size()))
            throw artlite::Error(std::format("pack_slots: tensor {}: {} x {} {}, file {}", ts.size(), x.rows, x.cols,
                                             type, x.file));
          if (x.type != artlite::SlotSource::EYE) {
            auto raw = nb::cast<Raw>(t[1], false);
            if (raw.shape(0) != size_t(x.rows) || raw.shape(1) != rb)
              throw artlite::Error(std::format("pack_slots: tensor {}: raw ({}, {}) for {} x {} {}", ts.size(),
                                               raw.shape(0), raw.shape(1), x.rows, x.cols, type));
            x.data = raw.data();
          }
          x.row_bytes = rb;
          ts.push_back(x);
        }
        if (parts.shape(1) != 4 || scales.shape(0) != parts.shape(0) || runs.shape(1) != 4 || jobs.shape(1) != 7)
          throw artlite::Error("pack_slots: jobs (J, 7), parts (P, 4), scales (P,), runs (R, 4)");
        std::vector<artlite::SlotPart> ps;
        for (size_t i = 0; i < parts.shape(0); i++) {
          const int64_t* p = parts.data() + 4 * i;
          if (p[1] < 0 || p[2] < 0 || p[3] < 0 || size_t(p[2] + p[3]) > runs.shape(0))
            throw artlite::Error(std::format("pack_slots: part {}: rows from {}, runs {} + {} of {}", i, p[1], p[2], p[3],
                                             runs.shape(0)));
          ps.push_back({static_cast<int>(p[0]), scales.data()[i], rows.data() + p[1], runs.data() + 4 * p[2],
                        static_cast<int>(p[3])});
        }
        std::vector<artlite::SlotJob> js;
        for (size_t i = 0; i < jobs.shape(0); i++) {
          const int64_t* j = jobs.data() + 7 * i;
          const size_t bytes = artlite::slot_job_bytes(size_t(j[2]), size_t(j[3]));
          if (j[0] < 0 || size_t(j[0]) >= out.size() || j[1] < 0 || size_t(j[1]) + bytes > out[j[0]].second ||
              j[5] < 0 || j[6] < 0 || size_t(j[5] + j[6]) > ps.size())
            throw artlite::Error(std::format("pack_slots: job {}: {} B at {} of buffer {} ({} buffers), parts {} + {}",
                                             i, bytes, j[1], j[0], out.size(), j[5], j[6]));
          for (int64_t k = j[5]; k < j[5] + j[6]; k++)
            if (parts.data()[4 * k + 1] + j[2] * artlite::kPassRows > int64_t(rows.shape(0)))
              throw artlite::Error(std::format("pack_slots: job {}: part {}: rows past the {} given", i, k, rows.shape(0)));
          js.push_back({out[j[0]].first + j[1], static_cast<int>(j[2]), static_cast<int>(j[3]), j[4] != 0,
                        static_cast<int>(j[5]), static_cast<int>(j[6])});
        }
        nb::gil_scoped_release release_gil;
        artlite::pack_slots(ts, ps, js, fs, threads, release);
      },
      "dsts"_a, "tensors"_a, "jobs"_a, "parts"_a, "scales"_a, "rows"_a, "runs"_a, "files"_a = nb::list(),
      "threads"_a = 0, "release"_a = false,
      nb::sig("def pack_slots(dsts: list[numpy.ndarray[numpy.uint8]], tensors: list[tuple], "
              "jobs: numpy.ndarray[numpy.int64], parts: numpy.ndarray[numpy.int64], "
              "scales: numpy.ndarray[numpy.float32], rows: numpy.ndarray[numpy.int32], "
              "runs: numpy.ndarray[numpy.int32], files: list[tuple[str, numpy.ndarray[numpy.uint8]]] = [], "
              "threads: int = 0, release: bool = False) -> None"),
      "Pack the vision overlay's weight images (artlite::pack_slots, artlite.vision.weights): tensors (type \"F32\", "
      "\"F16\", \"BF16\", \"Q8_0\" or \"EYE\", raw (rows, row bytes) uint8 or None for EYE, rows, cols, file: an index "
      "into files or -1), jobs (J, 7) int64 [dst buffer, byte offset, passes, k-tiles, hilo, first part, parts], parts "
      "(P, 4) int64 [tensor, first of its passes x 80 rows in rows (-1: a zero row), first run, runs], scales (P,), "
      "runs (R, 4) int32 [first column, columns, source column, stride]; files: (path, its whole mapping), read "
      "ahead; threads: 0 = one per physical core; release: MADV_DONTNEED each file tensor's pages when done.");
}

}  // namespace artlite::bindings
