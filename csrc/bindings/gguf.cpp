#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

#include "artlite/dequant.h"
#include "artlite/error.h"
#include "artlite/gguf_json.h"
#include "bindings/bindings.h"

using namespace nb::literals;

namespace artlite::bindings {

void bind_gguf(nb::module_& m) {
  m.def(
      "gguf_strings_json",
      [](nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> data, size_t pos, uint64_t count,
         std::optional<nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>> ids) {
        std::string json;
        {
          nb::gil_scoped_release release;
          json = artlite::gguf_strings_json({data.data(), data.shape(0)}, pos, count,
                                            ids ? std::span<const int64_t>(ids->data(), ids->shape(0))
                                                : std::span<const int64_t>());
        }
        return nb::bytes(json.data(), json.size());
      },
      "data"_a, "pos"_a, "count"_a, "ids"_a = nb::none(),
      nb::sig("def gguf_strings_json(data: numpy.ndarray[numpy.uint8], pos: int, count: int, "
              "ids: numpy.ndarray[numpy.int64] | None = None) -> bytes"),
      "The JSON text of a GGUF metadata string array at byte pos of data (count x [u64 length, UTF-8 bytes]): "
      "without ids the array of the strings, with ids the object {string: id} of those strings.");

  m.def(
      "dequant_rows",
      [](std::string qtype, nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> raw,
         nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> out) {
        if (out.shape(0) != raw.shape(0)) throw artlite::Error("dequant_rows: out's rows are not raw's");
        nb::gil_scoped_release release;  // (a prompt's rows: the worker thread that prepares the next chunk's)
        artlite::dequant_rows(qtype, raw.data(), raw.shape(0), raw.shape(1), out.shape(1), out.data());
      },
      "qtype"_a, "raw"_a, "out"_a,
      nb::sig("def dequant_rows(qtype: str, raw: numpy.ndarray[numpy.uint8], out: numpy.ndarray[numpy.float32]) -> None"),
      "GGUF rows (raw: (n, row bytes) of a tensor's blocks, a type of dequant_supported) dequantized into out (n, cols) fp32, "
      "its first cols columns: the values of gguf_model.QTensor.dequantize (its fp32 operations in their order).");
  m.def(
      "dequant_supported", [](std::string qtype) { return artlite::dequant_supported(qtype); }, "qtype"_a,
      "Whether dequant_rows reads the GGUF type.");
  m.def(
      "quantize_q8_0",
      [](nb::ndarray<const uint16_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> bf16,
         nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> out) {
        if (out.shape(0) != bf16.shape(0) || out.shape(1) != bf16.shape(1) / 32 * 34)
          throw artlite::Error("quantize_q8_0: out is (rows, cols / 32 x 34)");
        nb::gil_scoped_release release;
        artlite::quantize_q8_0(bf16.data(), bf16.shape(0), bf16.shape(1), out.data());
      },
      "bf16"_a, "out"_a,
      nb::sig("def quantize_q8_0(bf16: numpy.ndarray[numpy.uint16], out: numpy.ndarray[numpy.uint8]) -> None"),
      "bf16 rows (their bits, (rows, cols)) as Q8_0 blocks into out (rows, cols / 32 x 34): ggml's "
      "quantize_row_q8_0_ref.");
}

}  // namespace artlite::bindings
