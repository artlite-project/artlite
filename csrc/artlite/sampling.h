#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace artlite {

/// A vocabulary's logits: bf16 (their bits: an overlay's LOGITS) or fp32.
struct Logits {
  const void* data;
  size_t n;
  bool bf16;
};

/// llama.cpp's penalties sampler (``models.sampling.Penalties``): the logit l of a recent token of c occurrences ->
/// (l <= 0 ? l repeat : l / repeat) - presence - c frequency.
struct Penalty {
  float repeat = 1.0f;
  float presence = 0.0f;
  float frequency = 0.0f;
};

/// A mask of allowed logits (``models.sampling.Mask.bits``): bit i % 32 of word i / 32 for logit i (llguidance's
/// token bitmask), nullptr: every logit.

/// The candidates of a choice (``models.sampling.Penalties.top``): the k largest logits after a soft cap (cap > 0:
/// tanh(l / cap) cap) - the m at ``at`` (the recent tokens' logits, counts[i] occurrences each) penalized and ranked
/// with the others -, descending (equal values: the others first, then the lower index), into values (fp32) and ids
/// (their logits' indices; ``rows`` (n): the token of each logit); ``mask``: of the allowed logits only (the recent
/// ones too). Returns how many (min(k, n, the allowed)).
size_t top_logits(Logits x, size_t k, float cap, const int32_t* at, const float* counts, size_t m, Penalty p,
                  const int32_t* rows, float* values, int32_t* ids, const uint32_t* mask = nullptr);

/// The choice among n candidates, values descending (``models.sampling.Sampling.pick``): their softmax at ``temp``,
/// the smallest prefix reaching probability top_p, of those the ones of at least min_p times the first's probability
/// (the first always), one of them drawn by u (uniform in [0, 1)). Returns its index.
size_t pick(const float* values, size_t n, float temp, float top_p, float min_p, double u);

/// The largest of n bf16 logits (its index: the first of equal ones) and its probability, 1 / sum(exp(l - max)).
std::pair<size_t, float> max_prob(const uint16_t* x, size_t n);

/// log(sum(exp(l))) of a vocabulary's logits after a soft cap (cap > 0: tanh(l / cap) cap): a token's log-probability
/// is its (capped) logit minus it (``logprobs``).
float logsumexp(Logits x, float cap);

/// A screened LM head (``artlite.decode.screen``): what the high plane of row i of a Q6_K matrix (raw: n GGUF rows,
/// row_bytes apart, of cols columns) says of its logits. With w = q - 32 = p + r (p the plane's part 16 (q >> 4) - 24,
/// r = (q & 15) - 8) and a = d sc: rho = sum a^2 r w / sum a^2 w^2 (the plane's logit is 1 - rho of the row's, in the
/// mean), nv = mean a^2 (r - rho w)^2 (what is left: its variance per unit of |g|^2). scale[i] = 1 / (1 - rho),
/// bound[i] = sqrt(nv) / (1 - rho).
void screen_stats(const uint8_t* raw, size_t n, size_t row_bytes, size_t cols, float* scale, float* bound);

/// The largest lower bound of the logits of screened ones x (n bf16): max(scale[i] x[i] - gnorm bound[i]) (``mask``:
/// over the allowed ones; none: -inf).
float screen_lower(const uint16_t* x, const float* scale, const float* bound, float gnorm, size_t n,
                   const uint32_t* mask = nullptr);

/// The indices i with scale[i] x[i] + gnorm bound[i] >= thr (``mask``: of the allowed ones), ascending, into ids (the
/// first ``max`` of them). Returns how many there are.
size_t screen_above(const uint16_t* x, const float* scale, const float* bound, float gnorm, size_t n, float thr,
                    int32_t* ids, size_t max, const uint32_t* mask = nullptr);

/// The allowed indices of a mask of n logits, ascending, into ids (the first ``max`` of them). Returns how many there
/// are.
size_t mask_ids(const uint32_t* mask, size_t n, int32_t* ids, size_t max);

/// a[i] = min(scale[i] x[i], floor): the screened logits' estimates as fp32, ranked after the exact ones.
void screen_dense(const uint16_t* x, const float* scale, size_t n, float floor, float* a);

/// The exact logits out[j] = w[ids[j]] . g of m rows of a Q6_K matrix (raw: its GGUF rows, row_bytes apart; cols a
/// multiple of 256, g (cols) fp32): w = d sc (q - 32) summed in fp32, a sub-block's 16 products in lanes of their own,
/// the row's blocks one after the other.
void screen_exact(const uint8_t* raw, size_t row_bytes, const int32_t* ids, size_t m, const float* g, size_t cols,
                  float* out);

/// What the high plane of row i of a Q8_0 matrix (``formats.q8h``; raw: n GGUF rows, row_bytes apart, of cols columns)
/// says of its logits: with w = q = p + r (p = 16 (q >> 4) + 7.5, r = (q & 15) - 7.5) and a = d, rho = sum a^2 r w / sum
/// a^2 w^2 and nv the largest of the blocks' mean a^2 (r - rho w)^2 (a row's mean lets Inkling's head inputs put
/// logits 7.3 sigma off, its largest block 4.0): scale[i] = 1 / (1 - rho), bound[i] = sqrt(nv) / (1 - rho).
void screen_stats_q8(const uint8_t* raw, size_t n, size_t row_bytes, size_t cols, float* scale, float* bound);

/// The exact logits out[j] = w[ids[j]] . g of m rows of a Q8_0 matrix (raw: its GGUF rows, row_bytes apart; cols a
/// multiple of 32): w = d q, fp32 sums of 256 products, the row's in fp64.
void screen_exact_q8(const uint8_t* raw, size_t row_bytes, const int32_t* ids, size_t m, const float* g, size_t cols,
                     float* out);

/// A screened F16 / BF16 head (``artlite.decode.screen``): its plane as Q4_0 rows (out: n rows of cols / 32 GGUF blocks
/// of 18 B, llama.cpp's quantizer: d = the block's extreme / -8, q = min(15, int(x / d + 8.5))) of n rows of cols
/// values (x: half-precision words, ``bf16`` or fp16, rows ``stride`` words apart), on ``threads`` threads (0: half the
/// machine's).
void q4_0_rows(const uint16_t* x, bool bf16, size_t stride, size_t n, size_t cols, uint8_t* out, unsigned threads);

/// What a Q4_0 plane (q: its rows, cols / 32 blocks each) of n half-precision rows w says of their logits: with r = w -
/// the plane's value, rho = sum r w / sum w^2 and nv the largest of the blocks' mean (r - rho w)^2 (the head's input
/// has outlier channels: a row's mean over its blocks puts 7 sigma deviations within reach on Qwen3-0.6B, its largest
/// block 4.1): scale[i] = 1 / (1 - rho), bound[i] = sqrt(nv) / (1 - rho) (``screen_stats``'s).
void screen_stats_plane(const uint16_t* w, bool bf16, size_t stride, const uint8_t* q, size_t n, size_t cols,
                        float* scale, float* bound);

/// The exact logits out[j] = w[ids[j]] . g of m half-precision rows (fp32 products, rows ``stride`` words apart).
void screen_exact_half(const uint16_t* w, bool bf16, size_t stride, const int32_t* ids, size_t m, const float* g,
                       size_t cols, float* out);

}  // namespace artlite
