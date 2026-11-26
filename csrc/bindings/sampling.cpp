#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "artlite/error.h"
#include "artlite/sampling.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace {

// Sampling's arguments: a vocabulary's logits (any dtype: checked by logits_arg), candidates' ids and values
using Vector = nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Ints = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Floats = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FloatsOut = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IntsOut = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Mask = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

artlite::Logits logits_arg(const Vector& x) {
  const nb::dlpack::dtype d = x.dtype();
  const bool bf16 = d == nb::dtype<uint16_t>() || d == nb::dtype<int16_t>() ||
                    (d.code == uint8_t(nb::dlpack::dtype_code::Bfloat) && d.bits == 16 && d.lanes == 1);
  if (!bf16 && d != nb::dtype<float>()) throw artlite::Error("logits: bf16 (or their bits as int16 / uint16) or fp32");
  return {x.data(), x.shape(0), bf16};
}

// A mask of n logits (artlite::top_logits' ``mask``): a word per 32 of them, or none.
const uint32_t* mask_arg(const std::optional<Mask>& mask, size_t n) {
  if (!mask) return nullptr;
  if (mask->shape(0) < (n + 31) / 32) throw artlite::Error("mask: a bit per logit (uint32 words)");
  return mask->data();
}

// The first n of an array as a numpy array that owns it.
template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> owned(std::unique_ptr<T[]> a, size_t n) {
  T* p = a.release();
  nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(p, {n}, owner);
}

}  // namespace

namespace artlite::bindings {

void bind_sampling(nb::module_& m) {
  // --- Sampling (models.sampling) ---
  m.def(
      "top_logits",
      [](Vector logits, size_t k, float cap, std::optional<Ints> at, std::optional<Floats> counts, float repeat,
         float presence, float frequency, std::optional<Ints> rows, std::optional<Mask> mask) {
        const artlite::Logits x = logits_arg(logits);
        const uint32_t* bits = mask_arg(mask, x.n);
        const size_t m = at ? at->shape(0) : 0;
        if ((counts ? counts->shape(0) : 0) != m) throw artlite::Error("top_logits: a count per recent logit");
        if (rows && rows->shape(0) != x.n) throw artlite::Error("top_logits: a row per logit");
        for (size_t j = 0; j < m; j++)
          if (size_t(at->data()[j]) >= x.n) throw artlite::Error("top_logits: a recent logit past the logits");
        k = std::min(k, x.n);
        auto values = std::make_unique<float[]>(k);
        auto ids = std::make_unique<int32_t[]>(k);
        size_t n;
        {
          nb::gil_scoped_release release;
          n = artlite::top_logits(x, k, cap, m ? at->data() : nullptr, m ? counts->data() : nullptr, m,
                                  {repeat, presence, frequency}, rows ? rows->data() : nullptr, values.get(), ids.get(),
                                  bits);
        }
        return std::make_pair(owned(std::move(values), n), owned(std::move(ids), n));
      },
      "logits"_a, "k"_a, "cap"_a = 0.0f, "at"_a = nb::none(), "counts"_a = nb::none(), "repeat"_a = 1.0f,
      "presence"_a = 0.0f, "frequency"_a = 0.0f, "rows"_a = nb::none(),
      "mask"_a = nb::none(),
      nb::sig("def top_logits(logits: numpy.ndarray | object, k: int, cap: float = 0.0, "
              "at: numpy.ndarray[numpy.int32] | None = None, counts: numpy.ndarray[numpy.float32] | None = None, "
              "repeat: float = 1.0, presence: float = 0.0, frequency: float = 0.0, "
              "rows: numpy.ndarray[numpy.int32] | None = None, mask: numpy.ndarray[numpy.uint32] | None = None) -> "
              "tuple[numpy.ndarray[numpy.float32], numpy.ndarray[numpy.int32]]"),
      "The k largest of a vocabulary's logits (bf16 - a DLPack bfloat16 tensor or the bits as int16 / uint16 - or "
      "fp32) after a soft cap (cap > 0: tanh(l / cap) cap), the logits at ``at`` penalized (counts: their tokens' "
      "occurrences; repeat, presence, frequency: llama.cpp's penalties) and ranked with the others: their values "
      "(fp32, descending) and ids (the logits' indices, or rows[index]); mask: of the logits whose bit is set only (bit "
      "i % 32 of word i / 32): artlite::top_logits.");
  m.def(
      "mask_ids",
      [](Mask mask, size_t n, IntsOut ids) {
        const uint32_t* bits = mask_arg(mask, n);
        return artlite::mask_ids(bits, n, ids.data(), ids.shape(0));
      },
      "mask"_a, "n"_a, "ids"_a,
      nb::sig("def mask_ids(mask: numpy.ndarray[numpy.uint32], n: int, ids: numpy.ndarray[numpy.int32]) -> int"),
      "The logits a mask of n allows (bit i % 32 of word i / 32), ascending, into ids (the first len(ids)); how many "
      "there are: artlite::mask_ids.");
  m.def(
      "pick",
      [](Floats values, float temp, float top_p, float min_p, double u) {
        if (!values.shape(0)) throw artlite::Error("pick: no candidates");
        return artlite::pick(values.data(), values.shape(0), temp, top_p, min_p, u);
      },
      "values"_a, "temp"_a, "top_p"_a, "min_p"_a, "u"_a,
      nb::sig("def pick(values: numpy.ndarray[numpy.float32], temp: float, top_p: float, min_p: float, u: float) "
              "-> int"),
      "The index of the candidate (values descending) drawn by u in [0, 1) after softmax at temp, top_p, min_p: "
      "artlite::pick.");
  m.def(
      "max_prob",
      [](Vector logits) {
        const artlite::Logits x = logits_arg(logits);
        if (!x.bf16) throw artlite::Error("max_prob: bf16 logits");
        nb::gil_scoped_release release;
        return artlite::max_prob(static_cast<const uint16_t*>(x.data), x.n);
      },
      "logits"_a, nb::sig("def max_prob(logits: numpy.ndarray | object) -> tuple[int, float]"),
      "The largest of bf16 logits (its index, the first of equal ones) and its probability: artlite::max_prob.");
  m.def(
      "logsumexp",
      [](Vector logits, float cap) {
        const artlite::Logits x = logits_arg(logits);
        nb::gil_scoped_release release;
        return artlite::logsumexp(x, cap);
      },
      "logits"_a, "cap"_a = 0.0f, nb::sig("def logsumexp(logits: numpy.ndarray | object, cap: float = 0.0) -> float"),
      "log(sum(exp(l))) of a vocabulary's logits (bf16 or fp32, as top_logits takes them) after a soft cap (cap > 0: "
      "tanh(l / cap) cap): artlite::logsumexp.");
  // --- A screened LM head (artlite.decode.screen) ---
  m.def(
      "screen_stats",
      [](nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> raw, size_t cols, FloatsOut scale,
         FloatsOut bound) {
        const size_t n = scale.shape(0);
        if (cols % 256 || cols / 256 * 210 > raw.shape(1) || n > raw.shape(0) || bound.shape(0) != n)
          throw artlite::Error("screen_stats: Q6_K rows of cols columns, a scale and a bound per row");
        nb::gil_scoped_release release;
        artlite::screen_stats(raw.data(), n, raw.shape(1), cols, scale.data(), bound.data());
      },
      "raw"_a, "cols"_a, "scale"_a, "bound"_a,
      nb::sig("def screen_stats(raw: numpy.ndarray[numpy.uint8], cols: int, scale: numpy.ndarray[numpy.float32], "
              "bound: numpy.ndarray[numpy.float32]) -> None"),
      "Per row of a Q6_K matrix (raw: (rows, row bytes) of its GGUF blocks; the first len(scale) rows) what its high "
      "plane says of its logits: scale = 1 / (1 - rho), bound = the left-over deviation per unit of |g|, scaled: "
      "artlite::screen_stats.");
  m.def(
      "screen_lower",
      [](Vector logits, Floats scale, Floats bound, float gnorm, std::optional<Mask> mask) {
        const artlite::Logits x = logits_arg(logits);
        if (!x.bf16 || bound.shape(0) != x.n || scale.shape(0) != x.n)
          throw artlite::Error("screen_lower: bf16 logits, a scale and a bound per logit");
        nb::gil_scoped_release release;
        return artlite::screen_lower(static_cast<const uint16_t*>(x.data), scale.data(), bound.data(), gnorm, x.n,
                                     mask_arg(mask, x.n));
      },
      "logits"_a, "scale"_a, "bound"_a, "gnorm"_a, "mask"_a = nb::none(),
      nb::sig("def screen_lower(logits: numpy.ndarray | object, scale: numpy.ndarray[numpy.float32], "
              "bound: numpy.ndarray[numpy.float32], gnorm: float, mask: numpy.ndarray[numpy.uint32] | None = None) -> "
              "float"),
      "The largest lower bound of a screened head's logits (bf16), max(scale x - gnorm bound) (mask: of the allowed "
      "ones): artlite::screen_lower.");
  m.def(
      "screen_above",
      [](Vector logits, Floats scale, Floats bound, float gnorm, float thr, IntsOut ids, std::optional<Mask> mask) {
        const artlite::Logits x = logits_arg(logits);
        if (!x.bf16 || bound.shape(0) != x.n || scale.shape(0) != x.n)
          throw artlite::Error("screen_above: bf16 logits, a scale and a bound per logit");
        nb::gil_scoped_release release;
        return artlite::screen_above(static_cast<const uint16_t*>(x.data), scale.data(), bound.data(), gnorm, x.n, thr,
                                     ids.data(), ids.shape(0), mask_arg(mask, x.n));
      },
      "logits"_a, "scale"_a, "bound"_a, "gnorm"_a, "thr"_a, "ids"_a, "mask"_a = nb::none(),
      nb::sig("def screen_above(logits: numpy.ndarray | object, scale: numpy.ndarray[numpy.float32], "
              "bound: numpy.ndarray[numpy.float32], gnorm: float, thr: float, ids: numpy.ndarray[numpy.int32], "
              "mask: numpy.ndarray[numpy.uint32] | None = None) -> int"),
      "The indices of a screened head's logits (bf16) with scale x + gnorm bound >= thr (mask: of the allowed ones), "
      "ascending, into ids (the first len(ids)); how many there are: artlite::screen_above.");
  m.def(
      "screen_dense",
      [](Vector logits, Floats scale, float floor, FloatsOut a) {
        const artlite::Logits x = logits_arg(logits);
        if (!x.bf16 || a.shape(0) != x.n || scale.shape(0) != x.n)
          throw artlite::Error("screen_dense: bf16 logits, a scale and a value per logit");
        nb::gil_scoped_release release;
        artlite::screen_dense(static_cast<const uint16_t*>(x.data), scale.data(), x.n, floor, a.data());
      },
      "logits"_a, "scale"_a, "floor"_a, "a"_a,
      nb::sig("def screen_dense(logits: numpy.ndarray | object, scale: numpy.ndarray[numpy.float32], floor: float, "
              "a: numpy.ndarray[numpy.float32]) -> None"),
      "a = min(scale logits, floor) (bf16 logits as fp32): artlite::screen_dense.");
  m.def(
      "screen_exact",
      [](nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> raw, Ints ids, Floats g) {
        const size_t rows = raw.shape(0), cols = g.shape(0), m = ids.shape(0);
        if (cols % 256 || cols / 256 * 210 > raw.shape(1)) throw artlite::Error("screen_exact: g's columns in Q6_K rows");
        for (size_t j = 0; j < m; j++)
          if (size_t(ids.data()[j]) >= rows) throw artlite::Error("screen_exact: a row past the matrix");
        auto out = std::make_unique<float[]>(m);
        {
          nb::gil_scoped_release release;
          artlite::screen_exact(raw.data(), raw.shape(1), ids.data(), m, g.data(), cols, out.get());
        }
        return owned(std::move(out), m);
      },
      "raw"_a, "ids"_a, "g"_a,
      nb::sig("def screen_exact(raw: numpy.ndarray[numpy.uint8], ids: numpy.ndarray[numpy.int32], "
              "g: numpy.ndarray[numpy.float32]) -> numpy.ndarray[numpy.float32]"),
      "The exact logits of rows ids of a Q6_K matrix (raw: (rows, row bytes) of its GGUF blocks) against g (cols,) "
      "fp32: artlite::screen_exact.");
  m.def(
      "screen_stats_q8",
      [](nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> raw, size_t cols, FloatsOut scale,
         FloatsOut bound) {
        const size_t n = scale.shape(0);
        if (cols % 32 || cols / 32 * 34 > raw.shape(1) || n > raw.shape(0) || bound.shape(0) != n)
          throw artlite::Error("screen_stats_q8: Q8_0 rows of cols columns, a scale and a bound per row");
        nb::gil_scoped_release release;
        artlite::screen_stats_q8(raw.data(), n, raw.shape(1), cols, scale.data(), bound.data());
      },
      "raw"_a, "cols"_a, "scale"_a, "bound"_a,
      nb::sig("def screen_stats_q8(raw: numpy.ndarray[numpy.uint8], cols: int, scale: numpy.ndarray[numpy.float32], "
              "bound: numpy.ndarray[numpy.float32]) -> None"),
      "Per row of a Q8_0 matrix (raw: (rows, row bytes) of its GGUF blocks; the first len(scale) rows) what its high "
      "plane says of its logits (screen_stats'): artlite::screen_stats_q8.");
  m.def(
      "screen_exact_q8",
      [](nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> raw, Ints ids, Floats g) {
        const size_t rows = raw.shape(0), cols = g.shape(0), m = ids.shape(0);
        if (cols % 32 || cols / 32 * 34 > raw.shape(1)) throw artlite::Error("screen_exact_q8: g's columns in Q8_0 rows");
        for (size_t j = 0; j < m; j++)
          if (size_t(ids.data()[j]) >= rows) throw artlite::Error("screen_exact_q8: a row past the matrix");
        auto out = std::make_unique<float[]>(m);
        {
          nb::gil_scoped_release release;
          artlite::screen_exact_q8(raw.data(), raw.shape(1), ids.data(), m, g.data(), cols, out.get());
        }
        return owned(std::move(out), m);
      },
      "raw"_a, "ids"_a, "g"_a,
      nb::sig("def screen_exact_q8(raw: numpy.ndarray[numpy.uint8], ids: numpy.ndarray[numpy.int32], "
              "g: numpy.ndarray[numpy.float32]) -> numpy.ndarray[numpy.float32]"),
      "The exact logits of rows ids of a Q8_0 matrix (raw: (rows, row bytes) of its GGUF blocks) against g (cols,) "
      "fp32: artlite::screen_exact_q8.");
  m.def(
      "q4_0_rows",
      [](nb::ndarray<const uint16_t, nb::ndim<2>, nb::device::cpu> raw, bool bf16, size_t cols,
         nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> out, unsigned threads) {
        const size_t n = out.shape(0);
        if (cols % 32 || cols > raw.shape(1) || raw.stride(1) != 1 || n > raw.shape(0) || out.shape(1) != cols / 32 * 18)
          throw artlite::Error("q4_0_rows: half-precision rows of cols columns, Q4_0 rows of cols / 32 blocks");
        nb::gil_scoped_release release;
        artlite::q4_0_rows(raw.data(), bf16, size_t(raw.stride(0)), n, cols, out.data(), threads);
      },
      "raw"_a, "bf16"_a, "cols"_a, "out"_a, "threads"_a = 0,
      nb::sig("def q4_0_rows(raw: numpy.ndarray[numpy.uint16], bf16: bool, cols: int, out: numpy.ndarray[numpy.uint8], "
              "threads: int = 0) -> None"),
      "The first len(out) rows of a half-precision matrix (bf16 or fp16 words) as Q4_0 GGUF rows (a screened head's "
      "plane): artlite::q4_0_rows.");
  m.def(
      "screen_stats_plane",
      [](nb::ndarray<const uint16_t, nb::ndim<2>, nb::device::cpu> raw, bool bf16,
         nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> q, size_t cols, FloatsOut scale,
         FloatsOut bound) {
        const size_t n = scale.shape(0);
        if (cols % 32 || cols > raw.shape(1) || raw.stride(1) != 1 || n > raw.shape(0) || n > q.shape(0) ||
            q.shape(1) != cols / 32 * 18 || bound.shape(0) != n)
          throw artlite::Error("screen_stats_plane: half-precision rows, their Q4_0 plane, a scale and a bound per row");
        nb::gil_scoped_release release;
        artlite::screen_stats_plane(raw.data(), bf16, size_t(raw.stride(0)), q.data(), n, cols, scale.data(),
                                    bound.data());
      },
      "raw"_a, "bf16"_a, "q"_a, "cols"_a, "scale"_a, "bound"_a,
      nb::sig("def screen_stats_plane(raw: numpy.ndarray[numpy.uint16], bf16: bool, q: numpy.ndarray[numpy.uint8], "
              "cols: int, scale: numpy.ndarray[numpy.float32], bound: numpy.ndarray[numpy.float32]) -> None"),
      "Per half-precision row what its Q4_0 plane says of its logits (screen_stats'): artlite::screen_stats_plane.");
  m.def(
      "screen_exact_half",
      [](nb::ndarray<const uint16_t, nb::ndim<2>, nb::device::cpu> raw, bool bf16, Ints ids, Floats g) {
        const size_t rows = raw.shape(0), cols = g.shape(0), m = ids.shape(0);
        if (cols % 16 || cols > raw.shape(1) || raw.stride(1) != 1)
          throw artlite::Error("screen_exact_half: g's columns in half-precision rows");
        for (size_t j = 0; j < m; j++)
          if (size_t(ids.data()[j]) >= rows) throw artlite::Error("screen_exact_half: a row past the matrix");
        auto out = std::make_unique<float[]>(m);
        {
          nb::gil_scoped_release release;
          artlite::screen_exact_half(raw.data(), bf16, size_t(raw.stride(0)), ids.data(), m, g.data(), cols,
                                     out.get());
        }
        return owned(std::move(out), m);
      },
      "raw"_a, "bf16"_a, "ids"_a, "g"_a,
      nb::sig("def screen_exact_half(raw: numpy.ndarray[numpy.uint16], bf16: bool, ids: numpy.ndarray[numpy.int32], "
              "g: numpy.ndarray[numpy.float32]) -> numpy.ndarray[numpy.float32]"),
      "The exact logits of rows ids of a half-precision matrix (bf16 or fp16 words) against g (cols,) fp32: "
      "artlite::screen_exact_half.");
}

}  // namespace artlite::bindings
