// GGUF rows dequantized on the host (``gguf_model.GGUFFile.rows``: a decode step's embedding lookup), the fp32
// operations of ``gguf_model.QTensor.dequantize`` in their order: compiled with -ffp-contract=off (CMakeLists.txt).

#include "artlite/dequant.h"

#include <bit>
#include <cstring>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "artlite/error.h"

namespace artlite {

namespace {

float f16(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
  if (e == 31) return std::bit_cast<float>(s | 0x7F800000u | m << 13);
  if (e) return std::bit_cast<float>(s | (e + 112) << 23 | m << 13);
  const float v = float(m) * 0x1p-24f;  // (subnormal: m 2^-24, exact)
  return s ? -v : v;
}

// Q4_0 (18 B per 32): w = (q - 8) d; values 0 .. 15 the low nibbles, 16 .. 31 the high
void q4_0(const uint8_t* b, float* o) {
  const float d = f16(b);
  for (int j = 0; j < 32; j++) {
    const int q = j < 16 ? b[2 + j] & 15 : b[2 + j - 16] >> 4;
    o[j] = (float(q) - 8.0f) * d;
  }
}

// Q4_1 (20 B per 32): w = d q + m; q as Q4_0's
void q4_1(const uint8_t* b, float* o) {
  const float d = f16(b), m = f16(b + 2);
  for (int j = 0; j < 32; j++) {
    const int q = j < 16 ? b[4 + j] & 15 : b[4 + j - 16] >> 4;
    o[j] = d * float(q) + m;
  }
}

// Q8_0 (34 B per 32): w = d q, q int8
void q8_0(const uint8_t* b, float* o) {
  const float d = f16(b);
  for (int j = 0; j < 32; j++) o[j] = d * float(int8_t(b[2 + j]));
}

// Q4_K (144 B per 256) / Q5_K (176 B: the high bits qh after the scales): w = q (d sc_g) - dmin mn_g, group g = 2
// chunk + half of 32 values; Q5_K: bit g of qh[l] is the high bit of value l of group g
template <bool Q5>
void q45_k(const uint8_t* b, float* o) {
  const float d = f16(b), dmin = f16(b + 2);
  const uint8_t *lo = b + 4, *hi = b + 8, *top = b + 12, *qh = b + 16, *qs = b + (Q5 ? 48 : 16);
  for (int g = 0; g < 8; g++) {
    const int sc = g < 4 ? lo[g] & 63 : (top[g - 4] & 15) | (lo[g - 4] >> 6) << 4;
    const int mn = g < 4 ? hi[g] & 63 : (top[g - 4] >> 4) | (hi[g - 4] >> 6) << 4;
    const float dsc = d * float(sc), dmn = dmin * float(mn);
    for (int l = 0; l < 32; l++) {
      const uint8_t c = qs[32 * (g >> 1) + l];
      const int q = (g & 1 ? c >> 4 : c & 15) | (Q5 ? ((qh[l] >> g) & 1) << 4 : 0);
      const float t = float(q) * dsc;
      o[32 * g + l] = t - dmn;
    }
  }
}

// Q6_K (210 B per 256): w = (q - 32) (sc_j d), value 128 h + 32 jj + l: low nibble (jj < 2) or high of ql[64 h + 32
// (jj & 1) + l], crumb jj of qh[32 h + l]; sc per 16 values
void q6_k(const uint8_t* b, float* o) {
  const uint8_t *ql = b, *qh = b + 128;
  const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
  const float d = f16(b + 208);
  for (int h = 0; h < 2; h++)
    for (int jj = 0; jj < 4; jj++) {
      const uint8_t* c = ql + 64 * h + 32 * (jj & 1);
      const uint8_t* e = qh + 32 * h;
      const int sh = jj < 2 ? 0 : 4;
      float* out = o + 128 * h + 32 * jj;
      const float s0 = float(sc[8 * h + 2 * jj]) * d, s1 = float(sc[8 * h + 2 * jj + 1]) * d;
      for (int l = 0; l < 32; l++) {
        const int q = (((c[l] >> sh) & 15) | ((e[l] >> (2 * jj)) & 3) << 4) - 32;
        out[l] = float(q) * (l < 16 ? s0 : s1);
      }
    }
}

// Q3_K (110 B per 256): w = (d sc_j) (q - 4), j = the 16-value sub-block; value 128 n + 32 j + 16 h + l: crumb j of
// qs[32 n + 16 h + l], high bit 4 n + j of hmask[16 h + l]; the 6-bit scales sc + 32: low nibbles of bytes 0 .. 7
// (scales 0 .. 7) then their high nibbles, the high crumbs of scale 4 k + i in bits 2 k of byte 8 + i
void q3_k(const uint8_t* b, float* o) {
  const uint8_t *hm = b, *qs = b + 32, *s = b + 96;
  const float d = f16(b + 108);
  for (int e = 0; e < 256; e++) {
    const int n = e >> 7, j = (e >> 5) & 3, hl = e & 31, sb = e >> 4;
    const int lo = sb < 8 ? s[sb] & 15 : s[sb - 8] >> 4, hi = (s[8 + (sb & 3)] >> (2 * (sb >> 2))) & 3;
    const int q = ((qs[32 * n + hl] >> (2 * j)) & 3) | (((hm[hl] >> (4 * n + j)) & 1) << 2);
    const float a = d * float((lo | hi << 4) - 32);
    o[e] = a * (float(q) - 4.0f);
  }
}

// The plain types, a value per "block": F32, F16, BF16 (its bits, the high half of the fp32 value)
void f32(const uint8_t* b, float* o) { std::memcpy(o, b, 4); }
void f16_1(const uint8_t* b, float* o) { *o = f16(b); }
void bf16(const uint8_t* b, float* o) {
  uint16_t h;
  std::memcpy(&h, b, 2);
  *o = std::bit_cast<float>(uint32_t(h) << 16);
}

struct Type {
  std::string_view name;
  size_t per, bytes;  // values and bytes of a block
  void (*block)(const uint8_t*, float*);
};
constexpr Type TYPES[] = {{"F32", 1, 4, f32},       {"F16", 1, 2, f16_1},          {"BF16", 1, 2, bf16},
                          {"Q4_0", 32, 18, q4_0},   {"Q4_1", 32, 20, q4_1},        {"Q8_0", 32, 34, q8_0},
                          {"Q3_K", 256, 110, q3_k}, {"Q4_K", 256, 144, q45_k<false>}, {"Q5_K", 256, 176, q45_k<true>},
                          {"Q6_K", 256, 210, q6_k}};

const Type* type_of(std::string_view qtype) {
  for (const Type& t : TYPES)
    if (t.name == qtype) return &t;
  return nullptr;
}

void rows_of(const Type* t, const uint8_t* raw, size_t r0, size_t r1, size_t row_bytes, size_t cols, float* out) {
  for (size_t r = r0; r < r1; r++)
    for (size_t k = 0; k < cols / t->per; k++) t->block(raw + r * row_bytes + k * t->bytes, out + r * cols + k * t->per);
}

}  // namespace

bool dequant_supported(std::string_view qtype) { return type_of(qtype) != nullptr; }

void dequant_rows(std::string_view qtype, const uint8_t* raw, size_t n, size_t row_bytes, size_t cols, float* out) {
  const Type* t = type_of(qtype);
  if (!t) throw Error("dequant_rows: no native " + std::string(qtype));
  if (cols % t->per || cols / t->per * t->bytes > row_bytes)
    throw Error("dequant_rows: " + std::to_string(cols) + " columns of rows of " + std::to_string(row_bytes) + " B");
  // (a prompt's rows on threads of 64 rows at least: a decode step's one row on the caller's)
  const size_t nt = std::min<size_t>(n / 64, std::max(1u, std::thread::hardware_concurrency() / 2));
  if (nt <= 1) {
    rows_of(t, raw, 0, n, row_bytes, cols, out);
    return;
  }
  std::vector<std::thread> ts;
  for (size_t i = 0; i < nt; i++) ts.emplace_back(rows_of, t, raw, n * i / nt, n * (i + 1) / nt, row_bytes, cols, out);
  for (auto& th : ts) th.join();
}

}  // namespace artlite
