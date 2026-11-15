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

/// llama.cpp's penalties sampler (``models.common.Penalties``): the logit l of a recent token of c occurrences ->
/// (l <= 0 ? l repeat : l / repeat) - presence - c frequency.
struct Penalty {
  float repeat = 1.0f;
  float presence = 0.0f;
  float frequency = 0.0f;
};

/// The candidates of a choice (``models.common.Penalties.top``): the k largest logits after a soft cap (cap > 0: tanh(l
/// / cap) cap) - the m at ``at`` (the recent tokens' logits, counts[i] occurrences each) penalized and ranked with the
/// others -, descending (equal values: the others first, then the lower index), into values (fp32) and ids (their
/// logits' indices; ``rows`` (n): the token of each logit). Returns how many (min(k, n)).
size_t top_logits(Logits x, size_t k, float cap, const int32_t* at, const float* counts, size_t m, Penalty p,
                  const int32_t* rows, float* values, int32_t* ids);

/// The choice among n candidates, values descending (``models.common.Sampling.pick``): their softmax at ``temp``,
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

/// The largest lower bound of the logits of screened ones x (n bf16): max(scale[i] x[i] - gnorm bound[i]).
float screen_lower(const uint16_t* x, const float* scale, const float* bound, float gnorm, size_t n);

/// The indices i with scale[i] x[i] + gnorm bound[i] >= thr, ascending, into ids (the first ``max`` of them). Returns
/// how many there are.
size_t screen_above(const uint16_t* x, const float* scale, const float* bound, float gnorm, size_t n, float thr,
                    int32_t* ids, size_t max);

/// a[i] = min(scale[i] x[i], floor): the screened logits' estimates as fp32, ranked after the exact ones.
void screen_dense(const uint16_t* x, const float* scale, size_t n, float floor, float* a);

/// The exact logits out[j] = w[ids[j]] . g of m rows of a Q6_K matrix (raw: its GGUF rows, row_bytes apart; cols a
/// multiple of 256, g (cols) fp32): w = d sc (q - 32) summed in fp32, a sub-block's 16 products in lanes of their own,
/// the row's blocks one after the other.
void screen_exact(const uint8_t* raw, size_t row_bytes, const int32_t* ids, size_t m, const float* g, size_t cols,
                  float* out);

}  // namespace artlite
