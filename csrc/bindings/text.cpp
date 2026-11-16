#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "artlite/bpe.h"
#include "artlite/error.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace {

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> owned(const std::vector<T>& v) {
  auto a = std::make_unique<T[]>(v.size());
  if (!v.empty()) std::memcpy(a.get(), v.data(), v.size() * sizeof(T));
  T* p = a.release();
  nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(p, {v.size()}, owner);
}

}  // namespace

namespace artlite::bindings {

void bind_text(nb::module_& m) {
  using U32 = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using U8 = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  m.def(
      "pretokenize_spark2_5",
      [](U32 cp, U8 cls) {
        if (cls.shape(0) != cp.shape(0)) throw artlite::Error("pretokenize_spark2_5: a class per codepoint");
        std::vector<uint32_t> v;
        {
          nb::gil_scoped_release nogil;
          v = artlite::pretokenize_spark2_5(cp.data(), cls.data(), cp.shape(0));
        }
        return owned(v);
      },
      "cp"_a, "cls"_a,
      "The pieces' lengths (codepoints, uint32) of llama.cpp's spark2_5 pre-tokenizer over a text's codepoints and "
      "their classes (models.tokenizer.unicode_classes).");
  nb::class_<artlite::BpeEncoder>(m, "BpeEncoder",
                                  "A byte-level BPE vocabulary's merges (the lowest rank first, as llama.cpp).")
      .def(nb::init<const std::vector<std::string>&, const std::vector<std::string>&>(), "tokens"_a, "merges"_a,
           "tokens: the vocabulary's strings (GPT-2's byte-level form), merges: 'a b' in rank order.")
      .def(
          "encode",
          [](const artlite::BpeEncoder& e, U32 cp, U32 lens) {
            std::vector<int32_t> v;
            {
              nb::gil_scoped_release nogil;
              v = e.encode(cp.data(), cp.shape(0), lens.data(), lens.shape(0));
            }
            return owned(v);
          },
          "cp"_a, "lens"_a, "The ids (int32) of a text's pieces: its codepoints and the pieces' lengths in codepoints.");
}

}  // namespace artlite::bindings
