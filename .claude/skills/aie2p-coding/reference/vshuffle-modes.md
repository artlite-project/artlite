# vshuffle modes (AIE2P)

`shuffle(a, b, mode)` (`__builtin_aie2p_vshuffle`; one-input forms pass `a` twice or an undefined `b`) as measured on
the NPU: every mode with byte labels, the mode a runtime value. Lanes count `a`'s first, then `b`'s (a 512-bit input
holds 16 lanes of 32 bits, 32 of 16, 64 of 8); the last column lists the first 16 source lanes of the result, in the
lane width the mode moves. The headers (`aie2p_enums.h`) name the modes without saying what they do; aie_api uses a
few of them (`T32_1x16_flip`, `T16_1x2_flip`, `T8_1x4_flip` in `reverse.hpp`).

| mode | name | bits | result | source lanes |
|---|---|---|---|---|
| 0 | T8_64x2_lo | 8 | even bytes of a, b | 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 .. |
| 1 | T8_64x2_hi | 8 | odd bytes of a, b | 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 .. |
| 2 | T16_32x2_lo | 16 | even lanes of a, b | 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 .. |
| 3 | T16_32x2_hi | 16 | odd lanes of a, b | 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 .. |
| 4 | T32_16x2_lo | 32 | even lanes of a, b | 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 |
| 5 | T32_16x2_hi | 32 | odd lanes of a, b | 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 |
| 6 | T64_8x2_lo | 32 | even 64-bit pieces of a, b | 0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 |
| 7 | T64_8x2_hi | 32 | odd 64-bit pieces | 2 3 6 7 10 11 14 15 18 19 22 23 26 27 30 31 |
| 8 | T128_4x2_lo | 32 | even 128-bit pieces of a, b | 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 |
| 9 | T128_4x2_hi | 32 | odd 128-bit pieces | 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31 |
| 10 | T256_2x2_lo | 32 | a's low half, b's low half | 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 |
| 11 | T256_2x2_hi | 32 | a's high half, b's high half | 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 |
| 12 | T128_2x4_lo | 32 | 128-bit pieces of a, b interleaved (low halves) | 0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 |
| 13 | T128_2x4_hi | 32 | the same, high halves | 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31 |
| 14 | T64_2x8_lo | 32 | 64-bit pieces interleaved (low halves) | 0 1 16 17 2 3 18 19 4 5 20 21 6 7 22 23 |
| 15 | T64_2x8_hi | 32 | the same, high halves | 8 9 24 25 10 11 26 27 12 13 28 29 14 15 30 31 |
| 16 | T32_2x16_lo | 32 | lanes interleaved (low halves) | 0 16 1 17 2 18 3 19 4 20 5 21 6 22 7 23 |
| 17 | T32_2x16_hi | 32 | the same, high halves | 8 24 9 25 10 26 11 27 12 28 13 29 14 30 15 31 |
| 18 | T16_2x32_lo | 16 | lanes interleaved (low halves) | 0 32 1 33 2 34 3 35 4 36 5 37 6 38 7 39 .. |
| 19 | T16_2x32_hi | 16 | the same, high halves | 16 48 17 49 18 50 19 51 20 52 21 53 22 54 23 55 .. |
| 20 | T8_2x64_lo | 8 | bytes interleaved (low halves) | 0 64 1 65 2 66 3 67 4 68 5 69 6 70 7 71 .. |
| 21 | T8_2x64_hi | 8 | the same, high halves | 32 96 33 97 34 98 35 99 36 100 37 101 38 102 39 103 .. |
| 22 | T512_1x2_lo | 32 | a | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 |
| 23 | T512_1x2_hi | 32 | b | 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 |
| 24 | T16_16x4_lo | 16 | a, b as 16 rows x 4: columns 0, 1 (transpose) | 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 .. |
| 25 | T16_16x4_hi | 16 | columns 2, 3 | 2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62 .. |
| 26 | T16_4x16_lo | 16 | a, b as 4 rows x 16: lane 4 c + r = row r, column c < 8 (transpose) | 0 16 32 48 1 17 33 49 2 18 34 50 3 19 35 51 .. |
| 27 | T16_4x16_hi | 16 | columns 8 .. 15 | 8 24 40 56 9 25 41 57 10 26 42 58 11 27 43 59 .. |
| 28 | T16_8x4 | 16 | one input, 8x4 -> 4x8: lane 8 k + i = in[4 i + k] | 0 4 8 12 16 20 24 28 1 5 9 13 17 21 25 29 .. |
| 29 | T16_4x8 | 16 | one input, 4x8 transpose: lane 4 c + r = in[8 r + c] | 0 8 16 24 1 9 17 25 2 10 18 26 3 11 19 27 .. |
| 30 | T32_8x4_lo | 32 | a, b as 8 rows x 4: lane 8 k + i = in[4 i + k], k = 0, 1 | 0 4 8 12 16 20 24 28 1 5 9 13 17 21 25 29 |
| 31 | T32_8x4_hi | 32 | k = 2, 3 | 2 6 10 14 18 22 26 30 3 7 11 15 19 23 27 31 |
| 32 | T32_4x8_lo | 32 | a, b as 4 rows x 8: lane 4 c + r = in[8 r + c], c < 4 | 0 8 16 24 1 9 17 25 2 10 18 26 3 11 19 27 |
| 33 | T32_4x8_hi | 32 | c = 4 .. 7 | 4 12 20 28 5 13 21 29 6 14 22 30 7 15 23 31 |
| 34 | T32_4x4 | 32 | one input, 4x4 transpose | 0 4 8 12 1 5 9 13 2 6 10 14 3 7 11 15 |
| 35 | T8_8x8 | 8 | one input, bytes 8x8 transpose | 0 8 16 24 32 40 48 56 1 9 17 25 33 41 49 57 .. |
| 36 | T8_16x4 | 8 | one input, bytes 16x4 -> 4x16 | 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 .. |
| 37 | T8_4x16 | 8 | one input, bytes 4x16 -> 16x4 | 0 16 32 48 1 17 33 49 2 18 34 50 3 19 35 51 .. |
| 38 | T16_1x2_flip | 16 | one input, swap of adjacent lanes | 1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 14 .. |
| 39 | T16_4x4 | 16 | one input, 4x4 transpose per half | 0 4 8 12 1 5 9 13 2 6 10 14 3 7 11 15 .. |
| 40 | T16_4x2 | 16 | one input, per 8 lanes: evens, odds | 0 2 4 6 1 3 5 7 8 10 12 14 9 11 13 15 .. |
| 41 | T16_2x4 | 16 | one input, per 8 lanes: halves interleaved | 0 4 1 5 2 6 3 7 8 12 9 13 10 14 11 15 .. |
| 42 | T16_8x2 | 16 | one input, per 16 lanes: evens, odds | 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15 .. |
| 43 | T16_2x8 | 16 | one input, per 16 lanes: halves interleaved | 0 8 1 9 2 10 3 11 4 12 5 13 6 14 7 15 .. |
| 44 | T16_16x2 | 16 | one input: evens, odds | 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 .. |
| 45 | T16_2x16 | 16 | one input: halves interleaved | 0 16 1 17 2 18 3 19 4 20 5 21 6 22 7 23 .. |
| 46 | T8_8x4 | 8 | one input, per 32 bytes: 8x4 -> 4x8 | 0 4 8 12 16 20 24 28 1 5 9 13 17 21 25 29 .. |
| 47 | T8_4x8 | 8 | one input, per 32 bytes: 4x8 transpose | 0 8 16 24 1 9 17 25 2 10 18 26 3 11 19 27 .. |
| 48 | T32_8x2 | 32 | one input: evens, odds | 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15 |
| 49 | T32_2x8 | 32 | one input: halves interleaved | 0 8 1 9 2 10 3 11 4 12 5 13 6 14 7 15 |
| 50 | T64_4x2 | 32 | one input, 64-bit pieces: evens, odds | 0 1 4 5 8 9 12 13 2 3 6 7 10 11 14 15 |
| 51 | T64_2x4 | 32 | one input, 64-bit pieces: halves interleaved | 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15 |
| 52 | T16_8x8_lo | 16 | a, b as 8 rows x 8: lane 8 k + i = in[8 i + k], k < 4 (transpose) | 0 8 16 24 32 40 48 56 1 9 17 25 33 41 49 57 .. |
| 53 | T16_8x8_hi | 16 | k = 4 .. 7 | 4 12 20 28 36 44 52 60 5 13 21 29 37 45 53 61 .. |
| 54 | T32_1x16_flip | 32 | one input, reverse of the 16 lanes | 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 |
| 55 | T8_1x4_flip | 8 | one input, bytes reversed within each 32-bit lane | 3 2 1 0 7 6 5 4 11 10 9 8 15 14 13 12 .. |

- Rows of 8 (`kernel_lib.h` `max8`, `sum8`, `rows4`, `bcast8`): 29 / 32-33 put row r, column c in lane 4 c + r; 28 /
  30-31 are their inverses, and with 4 values repeated (`broadcast_elem_128`) they broadcast each to 8 lanes; 52-53
  (`T16_8x8`) do the same for 16-bit rows of 8 (`al::expand8_of`).
- Only 38 (a pair swap) and 40 / 41 (lane 0 fixed) keep every 16-bit lane in its 8-lane group, and every
  `shift_bytes` rotation crosses groups: no single op moves each lane of a row of 8 to another pair.
