#pragma once

// AWQ linears (a ParoQuant checkpoint's: artlite.paro) as rows of GGUF-like blocks the tile packer takes (formats P4_128,
// P5_128, tile_pack.h): the codes transposed from AWQ's K-major words to rows, in the storage order of the linear's
// rotation.

#include <cstddef>
#include <cstdint>

namespace artlite {

/// An AWQ linear of K inputs and N outputs: ``qweight`` [K][N / 8] (logical column j of a word in nibble
/// [0 4 1 5 2 6 3 7][j]), ``qzeros`` [K / group][N / 8] (the same packing), ``scales`` [K / group][N] fp16; ``chan``
/// (K, or null: as stored): position p of group G of a row takes channel ``chan[G group + p]`` (group-local) of the
/// group; ``rows_of`` (n_rows, or null: N rows): output row n is column rows_of[n]; ``groups_of`` (K / group, or null):
/// output group G is checkpoint group groups_of[G] (its channels and chan entries with it). int5 (``format:
/// int5-bitplane``): ``qweight_hi`` [K][N / 32], ``qzeros_hi`` [K / group][N / 32] the codes' fifth bits (bit n % 32 of
/// word n / 32).
struct AwqSource {
  const int32_t* qweight = nullptr;
  const int32_t* qzeros = nullptr;
  const uint16_t* scales = nullptr;
  int64_t K = 0, N = 0;
  int group = 128;
  const int32_t* chan = nullptr;
  const int64_t* rows_of = nullptr;
  int64_t n_rows = 0;
  const int64_t* groups_of = nullptr;
  const int32_t* qweight_hi = nullptr;
  const int32_t* qzeros_hi = nullptr;
};

/// Bytes of a P4_128 block row per 256 columns (super-block): s0, s1 fp16 (its two groups' scales), z0, z1, 10 bytes of
/// zeros, then the codes in Q4_K's order (code 64 c + l: the low nibble of byte 16 + 32 c + l; 64 c + 32 + l: its high).
constexpr size_t kAwqBlockBytes = 144;

/// Bytes of a P5_128 block (int5): the same 16-byte head, then the codes as a Q5_K block's: qh (32 bytes: bit e / 32 of
/// byte e % 32 the fifth bit of code e), their low nibbles in Q4_K's order from byte 48.
constexpr size_t kAwq5BlockBytes = 176;

/// The rows of an AWQ linear (group 128) as P4_128 blocks (int5: P5_128): out [n_rows][K / 256][144 or 176]. Threads:
/// 0 = one per physical core. Throws Error for a geometry it does not take.
void awq_rows(uint8_t* out, const AwqSource& src, unsigned threads = 0);

}  // namespace artlite
