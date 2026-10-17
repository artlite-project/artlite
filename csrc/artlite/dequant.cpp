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

// Q4_K (144 B per 256): w = q (d sc_g) - dmin mn_g, group g = 2 chunk + half of 32 values
void q4_k(const uint8_t* b, float* o) {
  const float d = f16(b), dmin = f16(b + 2);
  const uint8_t *lo = b + 4, *hi = b + 8, *top = b + 12, *qs = b + 16;
  for (int g = 0; g < 8; g++) {
    const int sc = g < 4 ? lo[g] & 63 : (top[g - 4] & 15) | (lo[g - 4] >> 6) << 4;
    const int mn = g < 4 ? hi[g] & 63 : (top[g - 4] >> 4) | (hi[g - 4] >> 6) << 4;
    const float dsc = d * float(sc), dmn = dmin * float(mn);
    for (int l = 0; l < 32; l++) {
      const uint8_t c = qs[32 * (g >> 1) + l];
      const float t = float(g & 1 ? c >> 4 : c & 15) * dsc;
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

void rows_of(bool q40, size_t bb, const uint8_t* raw, size_t r0, size_t r1, size_t row_bytes, size_t cols, size_t per,
             float* out) {
  for (size_t r = r0; r < r1; r++)
    for (size_t k = 0; k < cols / per; k++) {
      const uint8_t* b = raw + r * row_bytes + k * bb;
      float* o = out + r * cols + k * per;
      if (q40)
        q4_0(b, o);
      else if (bb == 144)
        q4_k(b, o);
      else
        q6_k(b, o);
    }
}

}  // namespace

bool dequant_supported(std::string_view qtype) { return qtype == "Q4_0" || qtype == "Q4_K" || qtype == "Q6_K"; }

void dequant_rows(std::string_view qtype, const uint8_t* raw, size_t n, size_t row_bytes, size_t cols, float* out) {
  const bool q40 = qtype == "Q4_0";
  if (!dequant_supported(qtype)) throw Error("dequant_rows: no native " + std::string(qtype));
  const size_t per = q40 ? 32 : 256, bb = q40 ? 18 : qtype == "Q4_K" ? 144 : 210;
  if (cols % per || cols / per * bb > row_bytes)
    throw Error("dequant_rows: " + std::to_string(cols) + " columns of rows of " + std::to_string(row_bytes) + " B");
  // (a prompt's rows on threads of 64 rows at least: a decode step's one row on the caller's)
  const size_t nt = std::min<size_t>(n / 64, std::max(1u, std::thread::hardware_concurrency() / 2));
  if (nt <= 1) {
    rows_of(q40, bb, raw, 0, n, row_bytes, cols, per, out);
    return;
  }
  std::vector<std::thread> ts;
  for (size_t t = 0; t < nt; t++)
    ts.emplace_back(rows_of, q40, bb, raw, n * t / nt, n * (t + 1) / nt, row_bytes, cols, per, out);
  for (auto& t : ts) t.join();
}

}  // namespace artlite
