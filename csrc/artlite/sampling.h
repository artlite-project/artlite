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

}  // namespace artlite
