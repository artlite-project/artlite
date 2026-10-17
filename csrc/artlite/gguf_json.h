#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace artlite {

/// The JSON text of a GGUF metadata string array (at byte pos of data: count x [u64 length, UTF-8 bytes]): without
/// ids the array ["s0", "s1", ...], with ids the object {"s_i": i, ...} of those strings (in the order of ids). For a
/// tokenizer.json built from a GGUF vocabulary (``gguf_model.bpe_tokenizer_json``): no Python string per token.
std::string gguf_strings_json(std::span<const uint8_t> data, size_t pos, uint64_t count,
                              std::span<const int64_t> ids = {});

}  // namespace artlite
