#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace artlite {

/// The routing of n tokens (``moe.plan.route``): per token t its k largest of E logits (the router GEMM's rows: expert
/// e at logits + t row + (e / per_col) pitch + e % per_col), largest first, ties to the lower id, into ids (n x k) and
/// weights (n x k): e^(l - l_max) over the k in double, normalized (numpy's summation order), rounded to fp32.
void moe_route(const float* logits, int64_t n, int64_t row, int per_col, int64_t pitch, int E, int k, int32_t* ids,
               float* weights);

/// The chunks of n tokens' routing (ids n x k, ``moe.plan.Plan``): an expert's (token, slot) pairs in token order, in
/// chunks of at most ``chunk``, dealt to the cols columns in turn (expert by expert): experts (cols x cap) each
/// column's chunks' experts, counts (cols), sizes (cols x cap) their pairs, rows (cols x cap x chunk) their tokens
/// (padding: the chunk's first), pos (n x k) the row (c cap + j) chunk + i of each pair. Throws if a column needs more
/// than cap chunks.
void moe_plan(const int32_t* ids, int64_t n, int k, int E, int cols, int cap, int chunk, int32_t* experts,
              int32_t* counts, int32_t* sizes, int32_t* rows, int64_t* pos);

/// A control stream's block body (``moe.ctl``): its words and its slots (S x 4: lo word, hi word, base index, offset):
/// word lo = bases[base index] + offset's low 32 bits, word hi |= its high bits (an address; hi < 0: word lo alone).
struct CtlTemplate {
  std::span<const uint32_t> words;
  std::span<const int64_t> slots;
};

/// A control stream from device address ``base`` into out (``moe.ctl.assemble``): block 0 (its words), then per block i
/// the body of template blocks[i] with its slots patched from bases (n_bases per block), and, but for the last block,
/// a tail (tails: 2 x ``NEXT_WORDS``: block i's writes block i + 1's BD tails[i % 2]): word 2 the next block's
/// length in words, 3 / 4 its address, word 11 |= lock_bits[i + 1] (lock_bits: per block, or null). Returns the
/// words written; throws past ``capacity``.
size_t ctl_assemble(uint32_t* out, size_t capacity, uint64_t base, std::span<const uint32_t> block0,
                    std::span<const CtlTemplate> templates, std::span<const int32_t> blocks, const uint64_t* bases,
                    int n_bases, std::span<const uint32_t> tails, const uint32_t* lock_bits);

}  // namespace artlite
