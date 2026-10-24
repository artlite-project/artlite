#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace artlite {

/// Whether ``dequant_rows`` reads a GGUF type ("Q4_0", "Q4_K", "Q6_K", "Q3_K").
bool dequant_supported(std::string_view qtype);

/// n GGUF rows of ``qtype`` (row_bytes apart from raw) dequantized into out (n x cols fp32, its first cols columns: a
/// multiple of the type's block), with the fp32 operations of ``gguf_model.QTensor.dequantize`` in their order (no
/// fused multiply-adds): the same values.
void dequant_rows(std::string_view qtype, const uint8_t* raw, size_t n, size_t row_bytes, size_t cols, float* out);

}  // namespace artlite
