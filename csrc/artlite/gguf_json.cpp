#include "artlite/gguf_json.h"

#include <cstring>
#include <format>
#include <vector>

#include "artlite/error.h"

namespace artlite {

namespace {

void put_string(std::string& out, const uint8_t* s, size_t n) {
  static constexpr char hex[] = "0123456789abcdef";
  out += '"';
  for (size_t i = 0; i < n; i++) {
    uint8_t c = s[i];
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      out += "\\u00";
      out += hex[c >> 4];
      out += hex[c & 15];
    } else {
      out += static_cast<char>(c);
    }
  }
  out += '"';
}

}  // namespace

std::string gguf_strings_json(std::span<const uint8_t> data, size_t pos, uint64_t count,
                              std::span<const int64_t> ids) {
  std::vector<size_t> at(count + 1);  // (each string's length field)
  for (uint64_t i = 0; i < count; i++) {
    uint64_t n;
    if (pos + 8 > data.size()) throw Error(std::format("gguf_strings_json: string {} past the end of the data", i));
    std::memcpy(&n, data.data() + pos, 8);
    if (n > data.size() - pos - 8) throw Error(std::format("gguf_strings_json: string {} of {} bytes past the end", i, n));
    at[i] = pos;
    pos += 8 + n;
  }
  at[count] = pos;
  std::string out;
  out.reserve((at[count] - at[0]) * 11 / 10 + 16 * (ids.empty() ? count : ids.size()) + 2);
  auto put = [&](uint64_t i) {
    uint64_t n;
    std::memcpy(&n, data.data() + at[i], 8);
    put_string(out, data.data() + at[i] + 8, n);
  };
  if (ids.empty()) {
    out += '[';
    for (uint64_t i = 0; i < count; i++) {
      if (i) out += ',';
      put(i);
    }
    out += ']';
    return out;
  }
  out += '{';
  for (size_t k = 0; k < ids.size(); k++) {
    if (ids[k] < 0 || static_cast<uint64_t>(ids[k]) >= count)
      throw Error(std::format("gguf_strings_json: id {} of {} strings", ids[k], count));
    if (k) out += ',';
    put(static_cast<uint64_t>(ids[k]));
    out += std::format(":{}", ids[k]);
  }
  out += '}';
  return out;
}

}  // namespace artlite
