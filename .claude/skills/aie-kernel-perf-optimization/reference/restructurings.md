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
- R10 Sequential recurrences: chunkwise on the MMUL
- R11 Several tokens per decode step
- R12 Neighbour cores sharing derived data
- R13 Flash attention with several query heads per core
- R14 Weight panels from gathers

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

What took a GELU (tanh form) from 622 to 223 cycles per 32 lanes (the gather table below does better: 26):
- **A LUT where accurate enough, a short bf16 polynomial where not** (`vtanh` above |w| = 1.25, a fit below it).
- **Identities without cancellation**: for w < 0, 1 + tanh(w) = e^(2w) (1 + |tanh w|).
- **fp32 only where it matters**: x² exact as bf16 x bf16 -> fp32; fp32 x constant by `al::mul_hilo` (3 products);
  2^y by the 1.5·2^23 magic split, a bf16 polynomial and the exponent added as integers; clamps in the integer domain
  (float min / max are expensive).
- **Both branches in full, one select at the end.**
- **x passed twice** (a second load or argument, `xr`) instead of kept live through the chain.
- **`noinline`** when called from several places (one copy in program memory).

**A gather table is often both faster and more accurate**: AIE2P's gather (`load_4x64_lo / hi`, 4 entries of 64 bits
per instruction) with a quadratic per segment: SiLU 0.59 ULP at 822 cycles per 1024 vs the vtanh form's 6.6 ULP at
2682 (`km.act`, `operators/kmath.py`). It costs 32 bytes of L1 per segment (4 copies).

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

- A scale per row and 16-k sub-block repeats over the row's 8 lanes of a panel. Three zips cost ~5 mv ops per 64
  lanes; one `broadcast_elem_128` (`vextbcst`, the 8 values 4 times) and two `shuffle(b, b, T16_8x8_lo / hi)`
  (`al::expand8`) cost 3: Q2_K's dequant II 44 -> 29 per 4 panels.
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
  port, 2 shuffles per 64 lanes instead of the zips' 3-4): GemmQ's tile-major dequant II 46 -> 40, bit-exact; the
  GemmQd dequants 5-20% per call (Q4_0 / Q4_1, Q5_K, Q6_K). In Q4_K's header pass the two are mixed: measure.
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

- **Scalar-ish math once for all panels**: math that needs a few lanes per panel (a recurrence's per-token scalars)
  runs once on 32 lanes for all panels between two passes over them, not on 16 lanes per panel (a token-pair GDN
  scan: 2548 instead of 4 x 747 cycles per pair, bit-exact: the same per-lane arithmetic).
- **One call for independent small reductions**: 4 l2 norms (reduce -> invsqrt -> scale chains) in one call: 424 vs
  4 x 150 cycles; 4 dot products: 136 vs 4 x 76. Several tokens' 1 / rms in one `al::inv_rms<N, K>` (the mean and
  eps on 16 lanes once, then K scalar invsqrt): the prefill norm's 8 tokens 282 cycles less per call. Their sums of
  32 lanes each go through L1 (a rolled loop folding two tokens to 8 lanes by two unzips, then `group8`: 706 -> 191
  cycles): a 64-lane fp32 vector assembled in registers from eight folds lives in the accumulators and spilled them
  whole (1.3-2.1 KB of stack, more than a core's 1 KB).
- **Packed lanes**: the GDN gates (softplus, expm1 in fp32: they feed the fp32 state) of two heads per call
  (`GdnPrepOperator`).
- **A RecMII-bound loop: two blocks per iteration**, two independent chains in one body (a split + MMUL chain of
  RecMII 11 per block: II 12-14 per two blocks), written out by hand (`AIE_LOOP_UNROLL` in an inlined helper breaks
  measure_ii's build: aie2p-coding section 9).

## R10 Sequential recurrences: chunkwise on the MMUL (`qwen35/gdn_prefill.py`)

The Gated DeltaNet scan updates an fp32 state S (128 x 128 per head) token by token (u = S^T k, then S + d S +
k delta^T): every token reads and writes S, and even two tokens per pass (token 1's products from S0 and token 0's
result) took 413 cycles per token and 8-column panel. The chunkwise (WY / UT) form handles C = 16 tokens per state
read and write:

- **The state products on the MMUL**: per chunk X^T = (K S0)^T and Y^T = (Q S0)^T (bfp16 hi + lo operands, three
  products), the update S0 + K^T (Gamma_C / Gamma U). The decays as products E = prod (1 + d) - 1: no cancellation
  for weak decays.
- **The intra-chunk solve in fp32**: U = (I + L)^-1 R by 8-token blocks, T00 and T11 by row substitution on bf16 hi /
  lo MAC chains, U1 = T11 (R1 - L10 U0). T by MMUL products ((I - N)(I + N^2)(I + N^4), hi + lo) doubled the error of
  deep layers.
- **Accuracy first on the CPU**: the chunk form at the kernels' rounding points vs the exact recurrence (numerics.md
  "Precision studies"): o within ~3e-5 rms (token pairs 1.5e-5), ~100x below o's bf16 rounding; the largest site is
  X^T / Y^T (the products of S0 serve all 16 tokens, their rounding correlated).
- **The work around it**: the prep (q, k panels, the gates) on 4 cores per column, a k and a q core per k head (the
  memtile splits each input object between them, their panels joined into one record per chunk); the scan once per
  layer over the whole chunk.
- **Compiler**: block vectors popped in a function's first block and held across a loop exhausted the registers (the
  same pops inside an outer loop compile); `concat` of two half-width broadcasts pushed lane by lane (32 `vpush` per
  step: two full broadcasts and a select instead); program memory forced rolled loops (T inverse 5.4 -> 1.2 KB) and
  `minsize` on cold helpers.
- Per 16-token chunk and core 8.4K cycles (T inverse 719, X / Y 314 per j-block post-RA pipelined, the decay scan
  444). A 2048-token chunk of the 1B (MiniCPM-V's LM): scan 70 -> 24 ms, prefill +18%; 9B / 27B +10%. With the
  chunk math compiled out the scan runs within ~10% of its time: DMA-bound in the program.

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

qwen35 (`qwen35/attn_prefill.py`: Q and K as hi + lo, two row panels per core when a column holds one q head), per
32-key group and core, the 1B (MiniCPM-V's LM) at p0 = 16K:
- **Every pop for as many MMULs as one-chain accumulators allow**: with K objects of two records (the memtile
  interleaves the records per d-block) the score loop pops 4 K and 4 Q panels per d-block for 12 MMULs on 4
  accumulators, one chain each for Q hi K hi + Q lo K hi + Q hi K lo: II 12, the MMUL bound (0.67 pops per MMUL
  instead of 1.33 for one panel and record): 2 x 423 instead of 4 x 358 cycles. P V likewise: a group's 4 V records
  for both panels per pop, II 9 for 8 MMULs.
- **The softmax in two pipelined passes**: the QK chain starts at b = 192 - m (A = s + 192 - m stored), so the exp2
  pass has no per-lane constants (II 26 -> 8 per 64 lanes); 8 steps of 32 lanes, not 4 of 64 (with 4 iterations
  the stages exceed the trip count: no post-RA schedule); the m + 16 bound check folded in as a running bf16 max; P
  to bfp16 straight from the second pass's accumulators (no fp32 round trip): 1156 -> 441 cycles.
- **o in two copies** (P V reads one, writes the other, in different regions) where L1 allows: o loaded and stored in
  place in one region put P V 50% above its schedule (548 vs 364 cycles).
- 9.95 -> 4.95 ms per layer at T = 512 (its DMA floor ~4.1). The 27B (three q heads per core) stays bound by its QK
  pops (each K panel once per head) and its in-place o (no L1 for a second copy).

## R14 Weight panels from gathers (IQ3_XXS, `formats/iq3.py`)

An IQ3_XXS weight is a grid entry (4 values per 8-bit index) with a sign: an 8x8 MMUL panel is 16 entries, 4
`load_4x32` gathers (entry i at byte 8 i, lanes 0, 1 from one table copy, 2, 3 from the other: numerics.md, the
bank-conflicts.md copies), one compaction shuffle (the gather fills the even 32-bit lanes), the signs by `vaddsub8`
(zero - w where a mask bit is set: a row's 8 signs one byte of the 64-bit mask). Per core's half tile (two super-blocks
of 23 rows) 4596 -> ~2300 cycles:
- **The move slot is the budget**: per panel the address lanes out of the accumulator (`vmov`), the tables' bases
  (`vadd`: the gathers keep their memory operand only as add(.., sel(bcast(pointer parameter)))), the shuffle,
  `vaddsub8`, the mask `vextract.64`: 5 ops; the gathers and the panel's store share QEY_RS, the gathers and slot A's
  loads DM_ADA. Peano schedules ~6.5 cycles per panel (2, 3 or 6 panels per iteration; 4: 7.75, extra `vmov`s).
- **Offsets on the load**: `ups_to_v32acc32(*p, 3, 0)` (`vlda.ups`, slot A) loads 32 index bytes as 8 idx into an
  accumulator: no vector register, no MMUL multiply, no shuffle when the indices are stored in panel order.
- **Masks**: two scalar `lda` per panel cost DM_ADA (II 13 instead of 10 per 2 panels) and met the panels' stores in
  the scratch's region (memory stall 254 vs 73 cycles per window with them compiled out); one 64-byte vector load
  per 6 panels and `ext_v2int32`: 101.
- **No FIFO in the gather loop**: a `fifo_ld_*` stream there made a recurrence through its registers (post-RA RecMII
  45); two streams in one loop spill the FIFO state every iteration (136 bytes). Realign in a loop of its own, read
  unaligned words as aligned blocks and `shift_bytes` (`_k23w2`), or store the data aligned (the tile layout is
  ours: the indices in 8-row panels, padded to 24 rows from the tile's spare bytes).
- **The MMUL in a loop of its own**: built panels to the scratch, multiplied by the next loop (II 17 per 12 panels);
  fused with the gathers the loop ran out of registers (post-RA) or took 869 instead of 639 + 164 cycles.
- **Re-encode losslessly**: the 7-bit sign field and its implied parity stored as the prefix XOR c (the 8 signs are
  c ^ (c << 1)): 5 ops per 64 fields instead of a sum of 6 shifts. Write the XOR as (c | t) - (c & t) with the
  subtraction a `vaddsub8` of an all-ones mask: LLVM folds a plain subtraction back into `^`, which AIE2P expands
  into 5 ops.
- Peano: a 128-bit vector load fed to `ext_v2int32` (through `set_v16int32`) fails to legalize (`G_UNMERGE_VALUES`
  <4 x s32> -> <2 x s32>).

