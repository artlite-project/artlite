#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "artlite/error.h"
#include "artlite/moe_host.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace artlite::bindings {

void bind_moe(nb::module_& m) {
  // --- The MoE prefill's host step (artlite/moe: plan, ctl) ---
  m.def(
      "moe_route",
      [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> logits, int64_t n, int64_t row,
         int per_col, int64_t pitch, int E, nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> ids,
         nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> weights) {
        const int k = int(ids.shape(1));
        if (ids.shape(0) < size_t(n) || weights.shape(0) < size_t(n) || weights.shape(1) != ids.shape(1) ||
            (n && logits.shape(0) < size_t((n - 1) * row + (E / per_col - 1) * pitch + per_col)))
          throw artlite::Error("moe_route: arrays too small");
        nb::gil_scoped_release release;
        artlite::moe_route(logits.data(), n, row, per_col, pitch, E, k, ids.data(), weights.data());
      },
      "logits"_a, "n"_a, "row"_a, "per_col"_a, "pitch"_a, "E"_a, "ids"_a, "weights"_a,
      nb::sig("def moe_route(logits: numpy.ndarray[numpy.float32], n: int, row: int, per_col: int, pitch: int, E: int, "
              "ids: numpy.ndarray[numpy.int32], weights: numpy.ndarray[numpy.float32]) -> None"),
      "The top k of n tokens' E logits (token t's at logits[t row ..], expert e at (e / per_col) pitch + e % per_col) "
      "into ids and weights (n x k): artlite::moe_route.");
  m.def(
      "moe_plan",
      [](nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> ids, int E, int chunk,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> experts,
         nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> counts,
         nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> sizes,
         nb::ndarray<int32_t, nb::ndim<3>, nb::c_contig, nb::device::cpu> rows,
         nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> pos) {
        const int cols = int(experts.shape(0)), cap = int(experts.shape(1));
        if (counts.shape(0) != size_t(cols) || sizes.shape(0) != size_t(cols) || sizes.shape(1) != size_t(cap) ||
            rows.shape(0) != size_t(cols) || rows.shape(1) != size_t(cap) || rows.shape(2) != size_t(chunk) ||
            pos.shape(0) != ids.shape(0) || pos.shape(1) != ids.shape(1))
          throw artlite::Error("moe_plan: array shapes");
        nb::gil_scoped_release release;
        artlite::moe_plan(ids.data(), int64_t(ids.shape(0)), int(ids.shape(1)), E, cols, cap, chunk, experts.data(),
                          counts.data(), sizes.data(), rows.data(), pos.data());
      },
      "ids"_a, "E"_a, "chunk"_a, "experts"_a, "counts"_a, "sizes"_a, "rows"_a, "pos"_a,
      nb::sig("def moe_plan(ids: numpy.ndarray[numpy.int32], E: int, chunk: int, experts: numpy.ndarray[numpy.int32], "
              "counts: numpy.ndarray[numpy.int32], sizes: numpy.ndarray[numpy.int32], rows: numpy.ndarray[numpy.int32], "
              "pos: numpy.ndarray[numpy.int64]) -> None"),
      "The chunks of a routing (ids n x k) dealt to experts.shape[0] columns of experts.shape[1] chunks: "
      "artlite::moe_plan.");
  m.def(
      "ctl_assemble",
      [](nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> out, uint64_t base,
         nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> block0, nb::list templates,
         nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> blocks,
         nb::ndarray<const uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> bases,
         nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> tails,
         std::optional<nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>> lock_bits) {
        using Words = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
        using Slots = nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
        std::vector<artlite::CtlTemplate> ts;
        for (nb::handle h : templates) {
          auto t = nb::cast<nb::tuple>(h);
          auto w = nb::cast<Words>(t[0], false);
          auto s = nb::cast<Slots>(t[1], false);
          if (s.shape(0) && s.shape(1) != 4) throw artlite::Error("ctl_assemble: slots of 4");
          ts.push_back({{w.data(), w.shape(0)}, {s.data(), s.shape(0) * (s.shape(0) ? 4 : 0)}});
        }
        const size_t n = blocks.shape(0);
        if (bases.shape(0) < n || (lock_bits && lock_bits->shape(0) < n))
          throw artlite::Error("ctl_assemble: a base row (and lock bits) per block");
        nb::gil_scoped_release release;
        return artlite::ctl_assemble(out.data(), out.shape(0), base, {block0.data(), block0.shape(0)}, ts,
                                     {blocks.data(), n}, bases.data(), int(bases.shape(1)),
                                     {tails.data(), tails.shape(0)}, lock_bits ? lock_bits->data() : nullptr);
      },
      "out"_a, "base"_a, "block0"_a, "templates"_a, "blocks"_a, "bases"_a, "tails"_a, "lock_bits"_a = nb::none(),
      nb::sig("def ctl_assemble(out: numpy.ndarray[numpy.uint32], base: int, block0: numpy.ndarray[numpy.uint32], "
              "templates: list[tuple[numpy.ndarray[numpy.uint32], numpy.ndarray[numpy.int64]]], "
              "blocks: numpy.ndarray[numpy.int32], bases: numpy.ndarray[numpy.uint64], tails: numpy.ndarray[numpy.uint32], "
              "lock_bits: numpy.ndarray[numpy.uint32] | None = None) -> int"),
      "A control stream into out (artlite::ctl_assemble): block 0, then block i from template blocks[i] (words, "
      "slots (S, 4): lo, hi (-1: lo alone), base index, offset) patched from bases[i], each with the tail that writes "
      "and pushes the next. Returns its words.");
}

}  // namespace artlite::bindings
