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

artlite::Logits logits_arg(const Vector& x) {
  const nb::dlpack::dtype d = x.dtype();
  const bool bf16 = d == nb::dtype<uint16_t>() || d == nb::dtype<int16_t>() ||
                    (d.code == uint8_t(nb::dlpack::dtype_code::Bfloat) && d.bits == 16 && d.lanes == 1);
  if (!bf16 && d != nb::dtype<float>()) throw artlite::Error("logits: bf16 (or their bits as int16 / uint16) or fp32");
  return {x.data(), x.shape(0), bf16};
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
  // --- Sampling (models.common) ---
  m.def(
      "top_logits",
      [](Vector logits, size_t k, float cap, std::optional<Ints> at, std::optional<Floats> counts, float repeat,
         float presence, float frequency, std::optional<Ints> rows) {
        const artlite::Logits x = logits_arg(logits);
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
                                  {repeat, presence, frequency}, rows ? rows->data() : nullptr, values.get(), ids.get());
        }
        return std::make_pair(owned(std::move(values), n), owned(std::move(ids), n));
      },
      "logits"_a, "k"_a, "cap"_a = 0.0f, "at"_a = nb::none(), "counts"_a = nb::none(), "repeat"_a = 1.0f,
      "presence"_a = 0.0f, "frequency"_a = 0.0f, "rows"_a = nb::none(),
      nb::sig("def top_logits(logits: numpy.ndarray | object, k: int, cap: float = 0.0, "
              "at: numpy.ndarray[numpy.int32] | None = None, counts: numpy.ndarray[numpy.float32] | None = None, "
              "repeat: float = 1.0, presence: float = 0.0, frequency: float = 0.0, "
              "rows: numpy.ndarray[numpy.int32] | None = None) -> tuple[numpy.ndarray[numpy.float32], "
              "numpy.ndarray[numpy.int32]]"),
      "The k largest of a vocabulary's logits (bf16 - a DLPack bfloat16 tensor or the bits as int16 / uint16 - or "
      "fp32) after a soft cap (cap > 0: tanh(l / cap) cap), the logits at ``at`` penalized (counts: their tokens' "
      "occurrences; repeat, presence, frequency: llama.cpp's penalties) and ranked with the others: their values "
      "(fp32, descending) and ids (the logits' indices, or rows[index]): artlite::top_logits.");
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
}

}  // namespace artlite::bindings
