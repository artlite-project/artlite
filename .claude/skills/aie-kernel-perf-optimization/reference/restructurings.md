# Restructurings that worked

The case studies behind SKILL.md section 3: each with its measured effect. The pattern transfers; the numbers show
what to expect.

## Contents
- R1 Accumulators in registers across the reduction
- R2 The reused operand popped once; independent MMUL chains
- R3 Long elementwise chains
- R4 Arithmetic on the MMUL
- R5 Dequantizing on the vector unit
- R6 Per-lane scales of small sub-blocks
- R7 Port-bound loops: fewer shuffles
- R8 Pointer walks instead of index math; constant offsets
- R9 Batched small calls, filled lanes
- R10 Sequential recurrences: token pairs, the reduction on the MMUL
- R11 Several tokens per decode step
- R12 Neighbour cores sharing derived data
- R13 Flash attention with several query heads per core

## R1 Accumulators in registers across the reduction

Symptoms: a loop reads, updates and writes an L1 buffer every group (`y = load; mac; store`), or it spills; on the
NPU it runs 20-30% above its schedule with memory stalls and bank conflicts.

Fix: change the loop order so that fewer outputs are live and they stay in accumulators for the whole reduction;
load them once before it, store them once after.

A Q4_0 GEMM unit (8 weight rows x 256 k against 4 groups of 8 tokens):
- all 4 token groups per weight group, y in L1: II 56 per group, 570 cycles per unit on the NPU, 7.25 ms of memory
  stall per core and chunk, 10^7 conflicts in banks 0 / 1;
- one pair of token groups at a time, y0 / y1 in accumulators across the unit's 8 groups, the weight panels rebuilt
  per pair: II 21 per group and pair, 437 cycles, 0.24 ms of memory stall; the chunk 78.4 -> 68.4 ms.
- Rebuilding the weight panels twice was cheaper than keeping four outputs live. Other orders (token group outermost,
  (group, token group) nests) took 560-800+ cycles; dequantizing to an L1 staging first, then a pure MMUL loop, ~400
  in two loops (the fused form needs 6 `dm`).

## R2 The reused operand popped once; independent MMUL chains

- Order the loops so that the operand reused across the inner loop is popped once into registers and the other
  operand streams exactly once.
- A dependent MMUL on one accumulator issues at best every 2 cycles: split the work over two accumulators on
  alternating blocks.
- A P·V kernel (per row set its 2-4 P panels in registers, the V object streamed once, two chains over alternating dim
  blocks): II 19 per 2 dim blocks, 1344 instead of 2155 cycles per object.
- The gemm MAC loop (`gemm/mac.py`): 4 staggered accumulator chains in k-step order, II 32 = ResMII
  (machine-model.md "Worked example").

## R3 Long elementwise chains

Why they are slow: an fp32 multiply is emulated (3 bf16 MACs plus 2 conversions: `al::mulff`), and a chain of range
reduction, polynomial and Newton steps is latency-bound (II = the chain length): it never software-pipelines, because
each value must be consumed within II cycles and the chain reuses the 5 `dm`.

What took GELU from 622 to 223 cycles per 32 lanes (`al::gelu_lut`; the recipe: numerics.md):
- **A LUT where accurate enough, a short bf16 polynomial where not** (`vtanh` above |w| = 1.25, a fit below it).
- **Identities without cancellation**: for w < 0, 1 + tanh(w) = e^(2w) (1 + |tanh w|).
- **fp32 only where it matters**: x² exact as bf16 x bf16 -> fp32; fp32 x constant by `al::mul_hilo` (3 products);
  2^y by the 1.5·2^23 magic split, a bf16 polynomial and the exponent added as integers; clamps in the integer domain
  (float min / max are expensive).
- **Both branches in full, one select at the end.**
- **x passed twice** (a second load or argument, `xr`) instead of kept live through the chain.
- **`noinline`** when called from several places (one copy in program memory).

**A gather table is often both faster and more accurate**: AIE2P's gather (`load_4x64_lo / hi`, 4 entries of 64 bits
per instruction) with a quadratic per segment: SiLU 0.59 ULP at 873 cycles per 1024 vs the vtanh form's 6.6 ULP at
2682 (`operators/act_lut.py`). It costs 32 bytes of L1 per segment (4 copies).

If it still does not pipeline, split it by **loop fission** through an L1 scratch: loop 1 stores the main result,
loop 2 patches a range in place, each loop walking its own `__restrict` pointers. SiLU with its small-|x| fix: one loop
48-59 cycles per 32, unpipelined; two loops 10 + 10.

## R4 Arithmetic on the MMUL

- **Integer weights, exactly**: unpacked codes (zero point subtracted) in a bfp16 panel whose shared exponent makes the
  value equal the mantissa (exponent byte 0x85 = 133: `formats/q2k.py`, `q3k.py`). The MMUL sums exact products;
  the group scale applies once to the sum (one bf16 MAC per group).
- **A block's values**: an MMUL against an identity panel.
- **Row sums**: A B_j^T, with B_j a panel of ones in row j, sums each row of A into lane j. The panels of a group
  against B_0 .. B_7 give the group sums of every row (a quantized GEMV's offset terms) inside the loop that builds
  the panels, with no vector reductions. l += P J (J a panel of ones) next to P V gives a softmax's row sums
  (`prefill/attn.py`).
- **Exact hi / lo split of fp32 x**: accumulator = x; `mac(bfp16(x), op_transpose(-I))` leaves x - hi exactly; both
  halves converted: about 14 bits per block (`al::split_hilo`, `al::hilo4`).
- **Affine dequant `w = a q + b`**: diag(a) Q^T - diag(|b|) 1^T with unsigned mantissas, exact when the products are.
  A dequant stage went from II 34 per 4 panels to 21 per 8. Diagonal panels assembled in registers cost 2 `mov` per
  panel on the mv slot (SKILL.md section 4; rejected.md "Prefill GEMM dequant").

## R5 Dequantizing on the vector unit

From a quantized GEMV on bf16 MACs; the zips, shifts and slot budget apply to every dequant.

- **Groups on lanes**: the lanes of a 64-lane accumulator are 64 quantization groups of one element index, so a
  group's scale applies once per row, not per 32 weights: 1 shuffle per 32 weights, II 4 per 128 weights (~6 per 32
  with a scale per group).
- **Offsets through the exponent**: an unsigned nibble u zipped with a 0x43 byte (`interleave_zip`) is the bf16
  128 + u (`formats/common.py`). An accumulator started at -136 X_g (the group sums of x) cancels the offset and the
  zero point exactly.
- **Split codes assembled in the integer domain, then one zip**: Q6_K q = ql | (c << 4): 2 zips + 2 MACs per 128
  weights instead of 4 + 4 (row loop II 22 -> 19).
- **An odd row count** pairs its last row with zeros instead of computing a junk row.
- **Shifting bytes costs accumulators**: a uint8 `aie::upshift` is a `vups` + `vsrs` round trip through a `dm`. For
  several shifts of one vector, upshift once (`accum.from_vector(v, 4)`) and read back with each step's downshift
  (`to_vector<uint8>(s)`, store slot): Q5_K's codes loop II 59 -> 41. Each distinct shift amount needs one of the 4
  `s` registers.
- **The mv slot is the dequant budget**: `vshuffle`, `vband` / `vbor`, `vsel` and register moves share it. Bytes ->
  bf16 (one zip) costs 2 per 64 codes, merging a split high bit (and + or) 2 more: Q5_K is mv-bound at twice Q4_K's
  cost per weight.
- **A streaming GEMV competes with its own DMA**: the S2MM fills core L1 at ~4 B / cycle, and the kernel's L1 traffic
  (scratch round trips, stack reloads) in the regions it writes stalls it (memory-module event 39,
  `DMA_S2MM_0_MEMORY_BACKPRESSURE`). Measure a harness with the real design's placement (`MmvOperator` in
  `tests/rtc/mmv.py`: two cores as the decode places them).

## R6 Per-lane scales of small sub-blocks (planar dequant, `formats/k23.py`)

- A scale per row and 16-k sub-block repeats over the row's 8 lanes of a panel. `al::expand8` (zips) costs ~5 mv ops
  per 64 lanes; one `broadcast_elem_128` (`vextbcst`, the 8 values 4 times) and two `shuffle(b, b, T16_8x8_lo / hi)`
  cost 3: Q2_K's dequant II 44 -> 29 per 4 panels.
- Keep the scales in the order the MMUL produced them (lanes [s][r]) and load each 8-value chunk where needed (the
  load slots are idle): an interleave into [sub-block][rows] cost the header pass 448 -> 841 cycles per super-block.
- Scales through the MMUL (column one-hot panels against the nibbles) are exact, but each needs conversions to bf16 on
  the store slot: the loop did not pipeline (II 73 per 4 panels).
- A dequant whose loop holds codes, 4 scale vectors and 11 constants (Q3_K) spills and parks values in accumulator
  registers (`vmov` 17-20, never pipelined, ~85 per 8 panels). Fission through an L1 scratch: loop 1 the code bytes
  and the expanded scales (II 32), loop 2 zips, MACs and pushes (II 34): 750 -> 590 cycles per 64 rows.
- A bfp16 panel push costs a push and a flush on the store slot and the pushes chain: ~3 cycles per panel is the floor
  of any dequant writing panels through one output stream.

## R7 Port-bound loops: fewer shuffles (the prefill GEMMs' dequant)

`vshuffle` (and `vsel`, `vbcst`, `vinsert`, `vshift`) shares `L_RM_PORT` with a bfp16 stream's pushes and flushes (2
per panel): a dequant writing 8 panels per iteration with 30 shuffles ran at II 46 with its mv slot at 30
(variant_search's `port` column, machine-model.md "Shared ports"). The pushes are fixed, so cut shuffles:
- **Expand by a broadcast and two 8x8 transposes** (`al::expand8_of`, `al::expand16_bc`: `vextbcst.128` is not on the
  port, 2 shuffles per 64 lanes instead of the zips' 3-4): GemmQ's tile-major dequant II 46 -> 40, bit-exact. In an
  alu-bound loop the zips schedule better (Q6_K's dequant: II 46 with zips, 88 with transposes): measure.
- **Repeat a 128-bit vector with `grow_replicate<N>()`** (one `vextbcst.128`), not `aie::concat(v, v, ...)`: aie_api
  concatenates 128-bit pieces with one `vshift` each, in sequence. K2 / K3 scales repeated to 64 lanes (`formats/k23.py`
  `_d64`): the GemmQd super-block header's loop II 22 -> 18 (the prefill level: not its bound).
- **Transpose bytes before the arithmetic, not bf16 results after it**: Q4_K's scale header (`formats/q4k.py`
  `_hdr_q4k`) as [row][group] nibbles -> bf16 products -> transposes took 12 shuffles per 128 values; one `T8_8x8`
  shuffle per 8x8 bytes and an even / odd row zip before the math, the crumb words shared by their two super-blocks,
  128 + v built on the integer MAC (lo + 16 c, then OR 0x4300 = the bf16 bits): 2246 -> 920 cycles per K chunk. An
  integer accumulator constant (0x4300 as the MAC's init) took a whole `dm` and spilled.

## R8 Pointer walks instead of index math; constant offsets

- **3-D / 2-D walks instead of decoding a flat loop index**: Q6_K's GemmQd dequant decoding (16-row block, word, half)
  from `it` (37 alu ops: `lshl`, `and`, `or`) ran at II 46, alu-bound. Two pointers advanced by `add_3d_ptr` /
  `add_3d_byte` with `dims_3d_from_steps(2, step_h, 2, step_word, step_block)` after each iteration: alu 0, II 30
  (mv-bound), 1001 -> 639 cycles per k-tile. A per-row scale used for 4 consecutive blocks and a 4-block weight cycled
  per row (bf16 [hi 32 | lo 32] entries): `dims_2d_from_steps(4, 0, 64)` and `(4, 64, 0)` (GdnOut).
- **Small scalar gathers unrolled to constant offsets**: Q6_K's `_hdr_d` gathers 44 (row, super-block) d values per
  tile; with index math in a rolled loop II 17 per value, unrolled (one load per word, its two halves to two places,
  the padding zeroed by vector stores first) 281 instead of 4660 cycles per K chunk.

## R9 Batched small calls, filled lanes

- **Scalar-ish math once for all panels**: the GDN scan's fp32 token math (u, delta, o, e0: ~266 cycles of latency)
  per 8-column panel ran on 16 lanes with 8 used; per token pair the 4 panels' pass 1 stores its parts, the math runs
  once on 32 lanes, then pass 2 per panel: 2548 instead of 4 x 747 cycles per pair, bit-exact (the same per-lane
  arithmetic).
- **One call for independent small reductions**: a pair's 4 l2 norms (reduce -> invsqrt -> scale chains) in one call:
  424 vs 4 x 150 cycles; its 4 dot products: 136 vs 4 x 76.
- **Packed lanes**: the GDN gates (softplus, expm1, sigmoid in fp32: they feed the fp32 state, no bf16 table) on 16
  lanes for 8 tokens of one v head; two heads per call halve the calls.
- **A RecMII-bound loop: two blocks per iteration**: the GDN scan's pass 1 (split + MMUL chain, RecMII 11 per block)
  at II 12-14 per two blocks, written out by hand (`AIE_LOOP_UNROLL` in an inlined helper breaks measure_ii's build:
  aie2p-coding section 9).

## R10 Sequential recurrences: token pairs, the reduction on the MMUL (`qwen35/gdn_prefill.py`)

The Gated DeltaNet scan updates an fp32 state S (128 x 128 per head) token by token: u = S^T k, w = S^T q, then
S + d S + k delta^T. Per token and 8-column panel of S the vector form took 698 cycles (pass 1 mv-bound by the
`tile8` broadcasts of k / q, II 15; pass 2 store-bound by the hi / lo split's conversions and the fp32 stores, II 18).
A generated sweep of ~1300 structures (`variant_search` with factory knobs, a few minutes per sweep) found 413 cycles
per token:

- **Two tokens per pass, exactly.** Token 1's products come from S0 and token 0's result: S1^T k1 = (1 + d0) S0^T k1
  + delta0 (k0 . k1), the same for q1; the update S2 = S0 + D S0 + k0 e0^T + k1 delta1^T with D = d0 + d1 + d0 d1,
  e0 = (1 + d1) delta0. Half the state loads, stores and hi / lo splits, 10.5 instead of 12 MACs per element and
  token. A null second token (d = beta = 0) handles an odd count.
- **The reduction on the MMUL**: S^T [k0 q0 k1 q1] as one 8-row A panel per i-block ([4 rows bfp16 hi | 4 rows lo],
  built once per pair) against the state block's bfp16 hi and lo (`al::split_hilo`, B via `op_transpose`): 3 MMULs
  per block for both tokens, no broadcasts. bfp16 hi + lo (14 bits) was as accurate as bf16 hi / lo products. The loop
  runs at II 11 per block (RecMII 11 whatever the number of accumulators or the A stream: ResMII 4).
- **Pass 2 in two loops**: decay + token 0's rank-1, then token 1's rank-1. Each holds half the loop invariants
  (64-lane hi / lo expansions): one loop at II 22 spilled them, two run at 16 + 5.
- On the NPU the scan went 5.97 -> 5.13 ms per 27B layer (T = 256). The gate math (softplus, expm1, sigmoid: 4 KB)
  does not fit the scan cores' program memory: `GdnPrepOperator` computes it, once per k head instead of once per
  core. Measure a static gain against the whole kernel before porting it.

## R11 Several tokens per decode step (draft verification, short prompts)

A DMA-bound decode streams every weight once per step and leaves the MMUL nearly idle, so a step of S tokens costs
about as much as a step of one (qwen35 27B: 297 ms for one token, 310 for four).

- **Tokens on the MMUL's A rows**: per 8 k one 8-row bfp16 panel [hi of S <= 4 tokens | their lo] (lo = bfp16(x -
  hi): ~14 bits below each block's maximum), against exact-integer weight panels (R4). Round the partial sums once per
  quantization sub-block, with the codes centered (q - 8, q - 32) so the sums stay small, then scale them (the scale
  exact in fp32, applied as bf16 hi + lo). Zero points and mins multiply the tokens' sums of x per sub-block (row
  sums, R4). `overlay/gemv.py`.
- **The tokens' inputs interleaved per 8-element block** ([k block][token][8] bf16): one 64-element load is the
  tokens' rows of a panel, and the producers drain straight into that layout.
- **Recurrent state (SSM, delta-rule layers) without per-token snapshots**: read the state from buffer A, run the S
  tokens, write the final state to buffer B, keep each token's small inputs (k, gates, deltas: a few KB). If all S are
  accepted, the next step reads B; if j < S are, it reads A and first replays the j kept tokens (compute only, no
  extra state traffic). Convolution histories shift by the accepted count; KV records of rejected positions stay
  masked until they are overwritten.
- **Attention with few query rows**: process only the query-row sets in use (S x heads per core, not the maximum),
  and test once per cache block whether any row's running maximum rose: skip the y / l rescale when none did (the
  branch outside the loops).

## R12 Neighbour cores sharing derived data (GemmQ's dequant pairs)

Cores that need the same derived data (the 4 core rows of a GemmQ column dequantize the same weights for their own
tokens: `_dq` 36%, the header 6% of each core's time) can each derive a part and read the rest from a neighbour:
- **Who can share**: a core reads and writes its own memory and its north, south and west neighbours'. Vertical pairs
  (rows 2 / 3, 4 / 5) share both ways; no memory is reachable by all 4 rows of a column (a 4-way split needs DMA, as
  GemmQd's broadcast of one dequantizing core's panels).
- **Mechanism**: one ObjectFifo per core, its partner the consumer (`ObjectFifo(ty, depth=2)`, `.prod()` on one
  Worker, `.cons()` on the other): adjacent ends lower to shared memory (a pool in the producer's memory, locks only,
  no DMA). The producer releases its half right after writing it and **keeps reading it** for its own MAC: the
  acquired buffer is an SSA value, and the FIFO hands that object back only after the partner's release (2 objects
  later). Holding it until after its own use deadlocks the pair (each waits for the other's half).
- **The input split too**: the memtile sends each core only its part of every object (a `split` link). The kernels see
  "a pass of 2 tiles" (`tq` in the format templates): only offsets and trip counts change, the arithmetic stays
  bit-exact.
- **Stream order against bank conflicts**: both cores of a pair read both halves in the MAC. Each reads its own half
  first, then the partner's, so in phase they never stream one memory at the same time (the lock handshake keeps them
  in phase). One loop over [half][band][block] with 3-D walks (B jumps between the two memories by a runtime byte
  offset) matched two loops in cycles (563 vs 569) at half the code (864 vs 1584 B per core).
- **Result** (9B qwen35 prefill, T = 256): per k-tile and core 980 -> ~800 static cycles; gate/up 2.19 -> 1.71 ms,
  O 0.43 -> 0.37 ms; a 2048-token chunk 1803 -> 1671 ms (9B), 5799 -> 5353 ms (27B). The PDI shrank too (209 -> 186
  KB: half the header code, smaller control code).
- **Then the k-tile window**: 880 cycles vs ~780 static; MEMORY_STALL 92 per window = the MAC's C loads and stores in
  one region. Alternating the k-tiles between two C tiles in different regions (`pingpong`, as GemmQd) removed them;
  the second tile doubles as the header collection and the SiLU scratch (free after every even number of k-tiles):
  846 cycles. gate/up then sits near its weight stream (1.24 ms with the kernels compiled out): further kernel gains
  show only in the GEMMs of fewer passes.

## R13 Flash attention with several query heads per core (`prefill/attn.py`)

A core holds QB x qh row blocks (8 queries of one head each) and streams the KV head's records; Llama 3B, 2048 tokens
at p0 = 8K, per layer 88 -> 15.1 ms (cache-bound: 13.1 ms with the compute compiled out). Each step was a measured
ablation (qk / soft / pv compiled out) and a variant_search table:
- **Two panel streams per loop**: the scores of a K unit read Q hi, Q lo, K0, K1 from four streams (II 22, spills);
  the memtile interleaves each unit's halves per d-block (`stream_walk` (nd, 36), (2, nd 36), (36, 1)) so a loop pops
  Q from one stream and K0, K1 from the other into four chains: II 4 = the MMUL bound (2300 -> 571 cycles per call). A
  stream's pops all sit on one load slot. With single bfp16 Q a loop takes a pair of row blocks (2 x 2 blocks: 4 Q and
  4 K pops for 8 MMULs, the accumulators starting at the rows' bias): ~350 cycles per call.
- **P V over pairs of row blocks**: each V panel popped once for two blocks, one chain per block, the O update's load
  and store pointers distinct `__restrict` parameters of a helper: II 8 for 8 MMULs (2339 -> 648).
- **Row sums on the MMUL** (l += P J next to P V): the exp loop holds no accumulators. With the softmax's fp32 max /
  clamp moved to the exp2's bf16 parts (v = max(bf16(A), 129), f = max(A - v, -0.5)), the maxima in bf16 and the rare
  paths (masks, rescale) in functions of their own, the softmax took ~1200 instead of 7161 cycles per chunk (6 row
  blocks x 4 records; its exp loop: `AIE_TRY_INITIATION_INTERVAL(24)`, post-RA II 36 without spills).
- Did not help: unrolling the 16-iteration d loops fully (spills: 2x slower); a single-bfp16 P by itself (P V and its
  exp loop are bound by loads and the exp2's vec ops, not MMULs; it stays, with single Q: smaller buffers, KL within
  the prefill's gate); `minsize` on the cold kernels (-0.5 KB of 7.8; with configuration by control packets program
  memory costs a few µs per design switch).
