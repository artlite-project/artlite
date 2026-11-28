# Restructurings that worked

The case studies behind SKILL.md section 3, each with its measured effect. The pattern transfers to other kernels; the
numbers show what to expect.

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

Symptoms: a loop reads, updates and writes an L1 buffer in every group (`y = load; mac; store`), or it spills; on the
NPU it runs 20-30% above its schedule, with memory stalls and bank conflicts.

Fix: change the loop order so that fewer outputs are live and they stay in accumulators for the whole reduction. Load
them once before the reduction and store them once after it.

Example: a Q4_0 GEMM unit (8 weight rows x 256 k against 4 groups of 8 tokens).
- With all 4 token groups per weight group and `y` in L1: II 56 per group, 570 cycles per unit on the NPU, 7.25 ms of
  memory stall per core and chunk, and 10^7 conflicts in banks 0 / 1.
- With one pair of token groups at a time, `y0` / `y1` held in accumulators across the unit's 8 groups, and the weight
  panels rebuilt per pair: II 21 per group and pair, 437 cycles, and 0.24 ms of memory stall. The chunk went from 78.4
  to 68.4 ms.
- Rebuilding the weight panels twice was cheaper than keeping four outputs live. Other orders (token group outermost,
  or (group, token group) nests) took 560-800+ cycles. Dequantizing into an L1 staging buffer first and then running a
  pure MMUL loop took ~400 cycles in two loops (the fused form needs 6 `dm` registers).

## R2 The reused operand popped once; independent MMUL chains

- Order the loops so that the operand reused across the inner loop is popped into registers once and the other operand
  streams exactly once.
- Dependent MMULs on one accumulator issue at best every 3 cycles, so split the work over two or more accumulators on
  alternating blocks.
- A P·V kernel (each row set keeps its 2-4 P panels in registers, the V object streams once, and two chains run over
  alternating dim blocks) reaches II 19 per 2 dim blocks: 1344 instead of 2155 cycles per object.
- The gemm MAC loop (`gemm/mac.py`) runs 4 staggered accumulator chains in k-step order at II 32 = ResMII
  (`machine-model.md` "Worked example").

## R3 Long elementwise chains

They are slow for two reasons. An fp32 multiply is emulated (3 bf16 MACs plus 2 conversions: `al::mulff`), and a chain
of range reduction, polynomial and Newton steps is latency-bound (II = the chain length). Such a chain never
software-pipelines, because each value must be consumed within II cycles and the chain reuses the 5 `dm` registers.

These changes took a GELU (tanh form) from 622 to 223 cycles per 32 lanes (the gather table below does better: 26):
- **Use a LUT where it is accurate enough, and a short bf16 polynomial where it is not** (`vtanh` above `|w| = 1.25`,
  a fit below it).
- **Use identities without cancellation**: for `w < 0`, `1 + tanh(w) = e^(2w) (1 + |tanh w|)`.
- **Use fp32 only where it matters**: `x²` is exact as bf16 x bf16 -> fp32; multiply fp32 by a constant with
  `al::mul_hilo` (3 products); compute `2^y` with the `1.5 * 2^23` magic split, a bf16 polynomial and the exponent added
  as integers; clamp in the integer domain (float min / max are expensive).
- **Compute both branches in full and select once at the end.**
- **Pass `x` twice** (a second load or argument, `xr`) instead of keeping it live through the chain.
- **Mark the function `noinline`** when it is called from several places, so program memory holds one copy.

**A gather table is often both faster and more accurate**: AIE2P's gather (`load_4x64_lo / hi`, 4 entries of 64 bits
per instruction) with a quadratic per segment computes SiLU to 0.59 ULP at 822 cycles per 1024 elements vs. 6.6 ULP at
2682 cycles for the `vtanh` form (`km.act`, `operators/kmath/`). It costs 32 bytes of L1 per segment (4 copies).

If the chain still does not pipeline, split it by **loop fission** through an L1 scratch: loop 1 stores the main
result, and loop 2 patches a range in place, each loop walking separate `__restrict` pointers. SiLU with its small-|x|
fix took 48-59 cycles per 32 lanes as one unpipelined loop, and 10 + 10 as two loops.

## R4 Arithmetic on the MMUL

- **Exact integer weights**: put the unpacked codes (zero point subtracted) in a bfp16 panel whose shared exponent makes
  each value equal its mantissa (exponent byte 0x85 = 133: `formats/q2k.py`, `q3k.py`). The MMUL sums exact products,
  and the group scale applies once to the sum (one bf16 MAC per group).
- **A block's values**: an MMUL against an identity panel.
- **Row sums**: `A B_j^T`, where `B_j` is a panel of ones in row `j`, sums each row of `A` into lane `j`. Multiplying a
  group's panels against `B_0 .. B_7` gives the group sums of every row (a quantized GEMV's offset terms) inside the
  loop that builds the panels, with no vector reductions. Likewise, `l += P J` (`J` a panel of ones) next to `P V`
  gives a softmax's row sums (`prefill/attn/`).
- **Exact hi / lo split of fp32 `x`**: set the accumulator to `x`; `mac(bfp16(x), op_transpose(-I))` then leaves
  `x - hi` exactly. With both halves converted, this keeps about 14 bits per block (`al::split_hilo`, `al::hilo4`).
- **Affine dequant `w = a q + b`**: compute `diag(a) Q^T - diag(|b|) 1^T` with unsigned mantissas, which is exact when
  the products are. This took a dequant stage from II 34 per 4 panels to II 21 per 8. Diagonal panels assembled in
  registers cost 2 `mov` per panel on the mv slot (SKILL.md section 4; `rejected.md` "Prefill GEMM dequant").

## R5 Dequantizing on the vector unit

These lessons come from a quantized GEMV on bf16 MACs; the zips, shifts and slot budget apply to every dequant.

- **Groups on lanes**: the 64 lanes of an accumulator hold 64 quantization groups of one element index, so a group's
  scale applies once per row rather than once per 32 weights. This costs 1 shuffle per 32 weights, at II 4 per 128
  weights (~6 per 32 with a scale per group).
- **Offsets through the exponent**: an unsigned nibble `u` zipped with a 0x43 byte (`interleave_zip`) is the bf16 value
  `128 + u` (`formats/common.py`). An accumulator initialized to `-136 X_g` (the group sums of `x`) cancels both the
  offset and the zero point exactly.
- **Assemble split codes in the integer domain, then zip once**: for Q6_K, `q = ql | (c << 4)` costs 2 zips + 2 MACs
  per 128 weights instead of 4 + 4 (row loop II 22 -> 19).
- **An odd row count** pairs its last row with zeros instead of computing a junk row.
- **Shifting bytes costs accumulators**: a uint8 `aie::upshift` is a `vups` + `vsrs` round trip through a `dm`
  register. For several shifts of one vector, upshift once (`accum.from_vector(v, 4)`) and read it back with each
  step's downshift (`to_vector<uint8>(s)`, on the store slot): this took Q5_K's code loop from II 59 to 41. Each
  distinct shift amount needs one of the 4 `s` registers.
- **The mv slot is the dequant budget**: `vshuffle`, `vband` / `vbor`, `vsel` and register moves all share it.
  Converting bytes to bf16 (one zip) costs 2 ops per 64 codes, and merging a split high bit (and + or) costs 2 more, so
  Q5_K is mv-bound at twice Q4_K's cost per weight.
- **A streaming GEMV competes with its own DMA**: the S2MM fills core L1 at ~4 B/cycle, and the kernel's L1 traffic
  (scratch round trips, stack reloads) in the regions the DMA writes stalls it (memory-module event 39,
  `DMA_S2MM_0_MEMORY_BACKPRESSURE`). Measure with a harness that uses the real design's placement (`MmvOperator` in
  `tests/decode/mmv.py`: two cores placed as the decode places them).

## R6 Per-lane scales of small sub-blocks (planar dequant, `formats/k23.py`)

- A scale per row and 16-k sub-block repeats over that row's 8 lanes of a panel. Three zips cost ~5 mv ops per 64
  lanes; one `broadcast_elem_128` (`vextbcst`: the 8 values repeated 4 times) plus two `shuffle(b, b, T16_8x8_lo / hi)`
  (`al::expand8`) cost 3. This took Q2_K's dequant from II 44 to 29 per 4 panels.
- Keep the scales in the order the MMUL produced them (lanes `[s][r]`) and load each 8-value chunk where it is needed
  (the load slots are idle). Interleaving them into `[sub-block][rows]` raised the header pass from 448 to 841 cycles
  per super-block.
- Scales computed on the MMUL (column one-hot panels against the nibbles) are exact, but each needs a conversion to
  bf16 on the store slot, and the loop did not pipeline (II 73 per 4 panels).
- A dequant loop that holds codes, 4 scale vectors and 11 constants (Q3_K) spills and parks values in accumulator
  registers (17-20 `vmov`s, never pipelined, ~85 cycles per 8 panels). Fission through an L1 scratch fixes it: loop 1
  produces the code bytes and the expanded scales (II 32), and loop 2 zips, MACs and pushes (II 34), taking 750 -> 590
  cycles per 64 rows.
- A bfp16 panel push costs a push and a flush on the store slot, and the pushes form a chain, so ~3 cycles per panel is
  the floor for any dequant that writes panels through one output stream.

## R7 Port-bound loops: fewer shuffles (the prefill GEMMs' dequant)

`vshuffle` (as well as `vsel`, `vbcst`, `vinsert` and `vshift`) shares `L_RM_PORT` with a bfp16 stream's pushes and
flushes (2 per panel). A dequant writing 8 panels per iteration with 30 shuffles ran at II 46 while its mv slot count
was only 30 (variant_search's `port` column, `machine-model.md` "Shared ports"). The pushes are fixed, so cut shuffles:
- **Expand with a broadcast and two 8x8 transposes** (`al::expand8_of`, `al::expand16_bc`): `vextbcst.128` is not on
  the port, so this costs 2 shuffles per 64 lanes instead of 3-4 for the zips. GemmQ's tile-major dequant went from II
  46 to 40, bit-exact, and the GemmQd dequants gained 5-20% per call (Q4_0 / Q4_1, Q5_K, Q6_K). In Q4_K's header pass
  the two approaches give mixed results, so measure.
- **Repeat a 128-bit vector with `grow_replicate<N>()`** (one `vextbcst.128`) rather than `aie::concat(v, v, ...)`,
  which concatenates 128-bit pieces with one `vshift` each, in sequence. Repeating the K2 / K3 scales to 64 lanes this
  way (`formats/k23.py` `_d64`) took the GemmQd super-block header loop from II 22 to 18 (at the prefill level this
  loop is not the bound).
- **Build a vector from distinct 128-bit pieces with `al::concat` / `al::gather`** (8 bf16 rows of a panel: 6
  `vshuffle`s) rather than `aie::concat` (a `vshift` per piece, plus `vmov`s joining 256-bit halves) or `insert` (a
  `vshift` + `vsel` per piece). Per call, this took the residual norm's sums pass from 1114 to 851 cycles, the prefill
  norm's pass B from 1031 to 848, the prefill attention's V panels from 1109 to 821, and the decode helper's x strip
  from 1490 to 974. Avoid it on a memory recurrence: GemmQd's pack loads its carry from the C row that the previous
  iteration stored (which may alias), and the shuffle tree lengthened that circuit (RecMII 30 -> 34, II 35 -> 39).
  Keeping aie_api's shift chain there and using `al::concat` for the other rows gave II 32.
- **Transpose bytes before the arithmetic rather than bf16 results after it**: Q4_K's scale header (`formats/q4k.py`
  `_hdr_q4k`), computed as `[row][group]` nibbles -> bf16 products -> transposes, took 12 shuffles per 128 values.
  Doing one `T8_8x8` shuffle per 8x8 bytes and an even / odd row zip before the math, sharing the crumb words between
  their two super-blocks, and building `128 + v` on the integer MAC (`lo + 16 c`, then OR 0x4300 to get the bf16 bits)
  took it from 2246 to 920 cycles per K chunk. An integer accumulator constant (0x4300 as the MAC's initial value)
  occupied a whole `dm` register and spilled.

## R8 Pointer walks instead of index math; constant offsets

- **Use 3-D / 2-D walks instead of decoding a flat loop index**: Q6_K's GemmQd dequant decoded (16-row block, word,
  half) from `it` with 37 alu ops (`lshl`, `and`, `or`) and ran at II 46, alu-bound. Two pointers advanced by
  `add_3d_ptr` / `add_3d_byte` with `dims_3d_from_steps(2, step_h, 2, step_word, step_block)` after each iteration need
  no alu ops and reach II 30 (mv-bound): 1001 -> 639 cycles per k-tile. In GdnOut, a per-row scale used for 4
  consecutive blocks and a 4-block weight cycled per row (bf16 `[hi 32 | lo 32]` entries) use
  `dims_2d_from_steps(4, 0, 64)` and `(4, 64, 0)`.
- **Unroll small scalar gathers to constant offsets**: Q6_K's `_hdr_d` gathers 44 (row, super-block) `d` values per
  tile. With index math in a rolled loop it ran at II 17 per value; unrolled (one load per word, its two halves stored
  to two places, the padding zeroed by vector stores first) it takes 281 instead of 4660 cycles per K chunk.

## R9 Batched small calls, filled lanes

- **Run scalar-like math once for all panels**: math that needs only a few lanes per panel (a recurrence's per-token
  scalars) runs once on 32 lanes for all panels between two passes over them, rather than on 16 lanes per panel. A
  token-pair GDN scan took 2548 instead of 4 x 747 cycles per pair, bit-exact because the per-lane arithmetic is the
  same.
- **Make one call for independent small reductions**: 4 l2 norms (reduce -> invsqrt -> scale chains) in one call take
  424 vs. 4 x 150 cycles; 4 dot products take 136 vs. 4 x 76. Computing several tokens' `1 / rms` in one
  `al::inv_rms<N, K>` (the mean and eps once on 16 lanes, then K scalar invsqrt) saves 282 cycles per call for the
  prefill norm's 8 tokens. Their 32-lane sums go through L1 (a rolled loop folds two tokens to 8 lanes with two unzips,
  then `group8`: 706 -> 191 cycles). A 64-lane fp32 vector assembled in registers from eight folds lives in the
  accumulators and spilled all of them (1.3-2.1 KB of stack, more than a core's 1 KB).
- **Pack lanes**: compute the GDN gates (softplus and expm1 in fp32, because they feed the fp32 state) for two heads
  per call (`GdnPrepOperator`).
- **Size epilogues to the tile**: a decode tile's row epilogue (the main core's `collect`) ran after the slower core's
  half on every tile. In the MoE frame of 32-row blocks it scaled all four blocks of `yt` (`al::mulff`, II 27 per 32
  floats: 440 cycles) even for a one-block tile, and it added an expert down's 4 row groups into `acc` with two `_slot`
  calls (16 + 7 rows) per group and slot (~130 cycles each). Scaling only the tile's blocks and making one `_accum` call
  per tile (a group's 32 rows onto one 48-lane window) took Q3_K downs from 3503 to 2965 cycles per tile at S 1 and from
  5984 to 3626 at S 4, and cut the whole step by 2.4 ms of 44 at S 1, bit-exact. Time a tile's period (from its command
  to the end of its epilogue) against the DMA's time per tile (`measuring.md`), not only the kernel.
- **For a RecMII-bound loop, process two blocks per iteration**, with two independent chains in one body (a split +
  MMUL chain of RecMII 11 per block reaches II 12-14 per two blocks). Write the copies out by hand: `AIE_LOOP_UNROLL`
  in an inlined helper breaks measure_ii's build (aie2p-coding section 9).

## R10 Sequential recurrences: chunkwise on the MMUL (`qwen35/gdn_prefill.py`)

The Gated DeltaNet scan updates an fp32 state `S` (128 x 128 per head) token by token (`u = S^T k`, then
`S + d S + k delta^T`). Every token reads and writes `S`, and even with two tokens per pass (token 1's products computed
from `S0` and token 0's result) it took 413 cycles per token and 8-column panel. The chunkwise (WY / UT) form handles
C = 16 tokens per state read and write:

- **The state products run on the MMUL**: per chunk, `X^T = (K S0)^T` and `Y^T = (Q S0)^T` (bfp16 hi + lo operands,
  three products), and the update `S0 + K^T (Gamma_C / Gamma U)`. The decays are computed as products
  `E = prod (1 + d) - 1`, which avoids cancellation for weak decays.
- **The intra-chunk solve runs in fp32**: `U = (I + L)^-1 R` in 8-token blocks, with `T00` and `T11` computed by row
  substitution on bf16 hi / lo MAC chains and `U1 = T11 (R1 - L10 U0)`. Computing T with MMUL products
  (`(I - N)(I + N^2)(I + N^4)`, hi + lo) doubled the error in deep layers.
- **Check accuracy on the CPU first**: the chunk form at the kernels' rounding points vs. the exact recurrence
  (`numerics.md` "Precision studies") keeps `o` within ~3e-5 rms (1.5e-5 for token pairs), ~100x below `o`'s bf16
  rounding. The largest error site is `X^T` / `Y^T`: the products of `S0` serve all 16 tokens, so their rounding errors
  are correlated.
- **The surrounding work**: the prep (q and k panels, the gates) runs on 4 cores per column, with a k core and a q core
  per k head (the memtile splits each input object between them, and their panels are joined into one record per
  chunk). The scan runs once per layer over the whole chunk.
- **Compiler issues**: block vectors popped in a function's first block and held across a loop exhausted the registers
  (the same pops inside an outer loop compile). `concat` of two half-width broadcasts was pushed lane by lane (32
  `vpush` per step); two full broadcasts and a select replace it. Program memory forced rolled loops (T inverse 5.4 ->
  1.2 KB) and `minsize` on cold helpers.
- Per 16-token chunk and core the scan takes 8.4K cycles (T inverse 719, X / Y 314 per j-block post-RA pipelined, the
  decay scan 444). On a 2048-token chunk of the 1B (MiniCPM-V's LM) the scan went from 70 to 24 ms and prefill got 18%
  faster; 9B / 27B gained 10%. With the chunk math compiled out the scan runs within ~10% of this time, so it is
  DMA-bound in the program.

## R11 Several tokens per decode step (draft verification, short prompts)

A DMA-bound decode streams every weight once per step and leaves the MMUL nearly idle, so a step of S tokens costs about
as much as a step of one (qwen35 27B: 297 ms for one token, 310 ms for four).

- **Put the tokens on the MMUL's A rows**: per 8 k, one 8-row bfp16 panel `[hi of S <= 4 tokens | their lo]`
  (`lo = bfp16(x - hi)`: ~14 bits below each block's maximum) multiplies exact-integer weight panels (R4). Round the
  partial sums once per quantization sub-block, with the codes centered (`q - 8`, `q - 32`) so that the sums stay small,
  then scale them (the scale is exact in fp32 and applied as bf16 hi + lo). Zero points and mins multiply the tokens'
  sums of `x` per sub-block (row sums, R4). See `decode/gemv.py`.
- **Interleave the tokens' inputs per 8-element block** (`[k block][token][8]` bf16): one 64-element load then holds
  the tokens' rows of a panel, and the producers drain straight into that layout.
- **Recurrent state (SSM, delta-rule layers) without per-token snapshots**: read the state from buffer A, run the S
  tokens, write the final state to buffer B, and keep each token's small inputs (k, gates, deltas: a few KB). If all S
  tokens are accepted, the next step reads B; if only j < S are, it reads A and first replays the j kept tokens
  (compute only, no extra state traffic). Convolution histories shift by the accepted count; KV records of rejected
  positions stay masked until they are overwritten.
- **Attention with few query rows**: process only the query-row sets in use (S x heads per core, not the maximum), and
  test once per cache block whether any row's running maximum rose; when none did, skip the `y` / `l` rescale (the
  branch sits outside the loops).

## R12 Neighbour cores sharing derived data (GemmQ's dequant pairs)

Cores that need the same derived data can each derive a part and read the rest from a neighbour. For example, the 4
core rows of a GemmQ column dequantize the same weights for different tokens (`_dq` takes 36% and the header 6% of each
core's time).
- **Who can share**: a core reads and writes its own memory and that of its north, south and west neighbours. Vertical
  pairs (rows 2 / 3, 4 / 5) share in both directions; no memory is reachable by all 4 rows of a column, so a 4-way split
  needs DMA (as in GemmQd's broadcast of one dequantizing core's panels).
- **Mechanism**: one ObjectFifo per core, with its partner as the consumer (`ObjectFifo(ty, depth=2)`, `.prod()` on one
  Worker, `.cons()` on the other). Adjacent ends lower to shared memory (a pool in the producer's memory, locks only, no
  DMA). The producer releases its half right after writing it and **keeps reading it** for its own MAC: the acquired
  buffer is an SSA value, and the FIFO hands that object back only after the partner releases it (2 objects later).
  Holding the object until after its own use deadlocks the pair, because each core waits for the other's half.
- **Split the input too**: the memtile sends each core only its part of every object (a `split` link). The kernels see
  "a pass of 2 tiles" (`tq` in the format templates): only offsets and trip counts change, and the arithmetic stays
  bit-exact.
- **Order the streams against bank conflicts**: both cores of a pair read both halves in the MAC. Each reads its own
  half first and then the partner's, so while in phase they never stream from the same memory at the same time (the lock
  handshake keeps them in phase). One loop over `[half][band][block]` with 3-D walks (B jumps between the two memories by
  a runtime byte offset) matched two separate loops in cycles (563 vs. 569) at half the code size (864 vs. 1584 B per
  core).
- **Result** (9B qwen35 prefill, T = 256): per k-tile and core, 980 -> ~800 static cycles; gate/up 2.19 -> 1.71 ms,
  O 0.43 -> 0.37 ms; a 2048-token chunk 1803 -> 1671 ms (9B) and 5799 -> 5353 ms (27B). The PDI also shrank (209 -> 186
  KB: half the header code and smaller control code).
- **Next, the k-tile window**: 880 cycles vs. ~780 static, with MEMORY_STALL 92 per window from the MAC's C loads and
  stores landing in one region. Alternating the k-tiles between two C tiles in different regions (`pingpong`, as in
  GemmQd) removed the stalls; the second tile doubles as the header collection buffer and the SiLU scratch (it is free
  after every even number of k-tiles). Result: 846 cycles. gate/up then sits close to its weight stream time (1.24 ms
  with the kernels compiled out), so further kernel gains show only in the GEMMs with fewer passes.

## R13 Flash attention with several query heads per core (`prefill/attn/`)

A core holds QB x qh row blocks (8 queries of one head each) and streams the KV head's records. On Llama 3B, 2048 tokens
at p0 = 8K, this took a layer from 88 to 15.1 ms (cache-bound: 13.1 ms with the compute compiled out). Each step below
was backed by a measured ablation (qk / soft / pv compiled out) and a variant_search table:
- **Two panel streams per loop**: the scores of a K unit read Q hi, Q lo, K0 and K1 from four streams (II 22, spills).
  The memtile now interleaves each unit's halves per d-block (`stream_walk` (nd, 36), (2, nd 36), (36, 1)), so a loop
  pops Q from one stream and K0, K1 from the other into four chains: II 4 = the MMUL bound (2300 -> 571 cycles per
  call). All of a stream's pops sit on one load slot. With single bfp16 Q, a loop takes a pair of row blocks (2 x 2
  blocks: 4 Q and 4 K pops for 8 MMULs, with the accumulators starting at the rows' bias): ~350 cycles per call.
- **P V over pairs of row blocks**: each V panel is popped once for two blocks, with one chain per block, and the O
  update's load and store pointers are distinct `__restrict` parameters of a helper: II 8 for 8 MMULs (2339 -> 648).
- **One pass for twice the rows** (`decode/attn256.py` "One pass"): a step of 8 tokens has 48 query rows per KV head.
  Position splits leave all 48 rows on each core (two passes over the cache: +38 ms at 32K). Instead, both cores of a
  chain read one broadcast stream, each taking its own heads' 24 rows, and Q and P become single bfp16 (one panel of 8
  rows, half the MMULs, no KL cost at model level). What binds then is not arithmetic: ~2600 of a group's ~6500 cycles
  went to the object loop and the maxima. Comparing the group's maximum against the rows' maxima skips the maxima's 270
  cycles per set, and computing P V over a whole V object takes `y` once for four records (S 5 at 32K: 377.8 ->
  342.7 ms).
- **Row sums on the MMUL** (`l += P J` next to `P V`): the exp loop then holds no accumulators. With the softmax's
  fp32 max / clamp moved to the exp2's bf16 parts (`v = max(bf16(A), 129)`, `f = max(A - v, -0.5)`), the maxima
  computed in bf16, and the rare paths (masks, rescale) moved into separate functions, the softmax took ~1200 instead of
  7161 cycles per chunk (6 row blocks x 4 records; its exp loop uses `AIE_TRY_INITIATION_INTERVAL(24)` and reaches
  post-RA II 36 without spills).
- What did not help: fully unrolling the 16-iteration d loops (spills, 2x slower); single-bfp16 P by itself (P V and its
  exp loop are bound by loads and the exp2's vec ops, not by MMULs; it stays together with single Q because it gives
  smaller buffers and KL within the prefill's gate); `minsize` on the cold kernels (-0.5 KB of 7.8 KB; with
  configuration by control packets, program memory costs a few µs per design switch).
- **Rare paths that are not rare** (the packed variant on real rerank data): the raise (a row's reference maximum
  rising, so `o` and `l` are rescaled) runs at the first computed chunk of every pass, costing 7,280 cycles compiled
  with `minsize`. Since `o` and `l` are still zero there, it only needs to move the bias: 2,540 cycles. The masks below
  the rows' segments were built bit by bit (a scalar loop per row block, record and row); qwen35's form replaced them:
  per query block, a 64-lane int16 vector of its rows' segment starts (built once per pass) is compared against the
  lanes' key offsets. The softmax went from 1,804 to 885 cycles per chunk and the layer from 3.73 to 3.13 ms
  (Qwen3-Reranker-0.6B, 4096 tokens).

- **LFM3's broadcast design** (two columns per KV head and one cache stream, so it is core-bound where Llama's and
  Qwen3's per-column streams are cache-bound), 2048 tokens at p0 = 8K with hi + lo P: per chunk of 4 records and core
  2150 -> 1790 cycles, the layer 11.4 -> 9.6 ms (single P: 10.6 -> 8.2; at p0 = 32K: 41.3 -> 34.5 ms):
  - **Keep `o` in two copies in different regions** (P V reads one and writes the other). In place, its loads and
    stores met in one region's banks: 1.16 M conflict cycles per core and layer, ~60 cycles per P V call. The P panels
    also moved out of the workspace's region (the exp loop's loads and pushes met there), and every hot loop's streams
    now sit in distinct regions.
  - **Put both P V panel streams on load slot B** (one bank annotation), leaving slot A for `o`'s accumulator loads:
    II 38 -> 32, the MMUL bound (single P: 25 -> 20). Read the `lda` / `ldb` columns before blaming register pressure.
  - **Move the exp's range reduction into the score loop's epilogue**, where the vector slot is idle
    (`v = max(bf16(A), 129)` and `f = bf16(A - v)`, stored as bf16 in place of the fp32 scores): the exp loop's vec
    count drops from 21 to 15 and its II from 26 to 22, and the softmax's raise test becomes a flag set by that epilogue
    (569 -> 440 cycles per chunk together with the rest).
  - **Process both K units of a chunk in one call**, with the (pair, unit) loop outer-loop-pipelined (`loop-hints.md`):
    532 -> 494 cycles.
- **The vision overlay's encoder attention** (role a: 16 heads of d 80, dense joint keys; Qwen3-VL, LFM3-VL, d3,
  MiniCPM-V), bit-identical: per core, unit (image, head) and 64-key group 4123 -> 3436 cycles; an op of 4096 keys
  1166 -> 972 µs. The changes:
  - The exp2 runs in two passes over all of a pair's key panels (reduce: II 8 per 32 lanes; poly: II 14 per 64, with its
    two 32-lane halves joined in registers into the bfp16 P panels). This costs 34 cycles per key panel vs. 48 with P
    stored as fp32 for a panel loop; a single loop ran at II 23, and with passes per 4 key panels the ramps ate the gain.
  - QK uses constant trip counts and its block loop is outer-loop-pipelined (520 -> 252 cycles per pair; unrolling the
    d-blocks spills).
  - P V of both pairs runs in one MAC call, alternating block by block between their O buffers in two regions (region
    0's conflicts: 33 -> 1 µs per op).

qwen35 (`prefill/attn256.py`: Q and K as hi + lo, two row panels per core when a column holds one q head), per 32-key
group and core, on the 1B (MiniCPM-V's LM) at p0 = 16K:
- **Get as many MMULs per pop as single-chain accumulators allow**: with K objects of two records (the memtile
  interleaves the records per d-block) the score loop pops 4 K and 4 Q panels per d-block for 12 MMULs on 4
  accumulators, one chain each for Q hi K hi + Q lo K hi + Q hi K lo. This reaches II 12, the MMUL bound (0.67 pops per
  MMUL instead of 1.33 for one panel and record): 2 x 423 instead of 4 x 358 cycles. P V works the same way: each pop
  serves a group's 4 V records for both panels, at II 9 for 8 MMULs.
- **Run the softmax in two pipelined passes**: the QK chain starts at `b = 192 - m` (storing `A = s + 192 - m`), so the
  exp2 pass needs no per-lane constants (II 26 -> 8 per 64 lanes). Use 8 steps of 32 lanes, not 4 of 64 (with 4
  iterations the stages exceed the trip count and there is no post-RA schedule). Fold the `m + 16` bound check in as a
  running bf16 max, and convert P to bfp16 straight from the second pass's accumulators (no fp32 round trip): 1156 ->
  441 cycles.
- **Keep `o` in two copies** (P V reads one and writes the other, in different regions) where L1 allows. Loading and
  storing `o` in place in one region put P V 50% above its schedule (548 vs. 364 cycles).
- The layer went from 9.95 to 4.95 ms at T = 512 (its DMA floor is ~4.1 ms). The 27B (three q heads per core) stays
  bound by its QK pops (each K panel is popped once per head) and its in-place `o` (no L1 for a second copy).
- **One pass regardless of a step's S** (`artlite/decode/attn256.py`, designs of 8 slots): rows are head-major (octet h
  holds head h's 8 slots), so a step's compute does not depend on S. K hi alone, read as pairs of records, needs 2
  objects per group instead of 3, cutting a pass by 1/3. Row maxima come from the first group and hold for the whole
  pass (no rescale; P clamps), then QK runs from the baseline `192 - m` and the group's P is computed in two pipelined
  passes (756 vs. 1068 cycles per group at 3 octets; a single loop spills the exp's constants). A new K is written from
  the popped panels' lanes (900 cycles per record vs. 43,000 for a byte loop). For a step that crosses a group, the
  first group's open record keeps its scores in the slot that the second group does not write. 27B at 32K: V(8) / V(1)
  1.071 -> 1.016.
- **A chain of 4 on two streams** (`artlite/decode/attn256.py` `SPLIT_ROUND`, `AttnShape.rounds`; `decode.design`
  "Shared A objects"): 32 query rows per KV head vs. a core's 16. Splitting positions alone took two passes, and one
  pass on one stream reads at half the port's rate. Instead, two halves, each of two cores holding its heads with the
  groups split between them, read two streams, each read once and delivered to both halves: 35B-A3B at 32K, S 3 / 4
  83.8 / 97.0 -> 77.2 / 89.2 ms together with the changes below. Two streams read ~46 GB/s vs. the chain's 56, so a
  shorter sub-step stays on the whole chain; the split is chosen per fill. A stream shared by two cores also removes
  what hid their compute: a group of two octets ran at ~4360 cycles vs. ~2900 cycles of stream, 30% over its schedule.
  So the one-pass core's single bfp16 Q and P were adopted (QK 624 -> 316 cycles per group, P V 582 -> 365, the Q
  panels 5100 -> 594 per sub-step), with the rows' maxima still rising group by group (`numerics.md`: a held `m` broke a
  chat at 32K), the gates applied through a table, and V encoded once. The one-pass core's K path (1.4 KB of code) and
  its head-major rows (a chain's S 1 would run two octets) were not adopted. Count a port's bytes first. This core had
  176 B of program memory left, and a helper moved out of line to save code (`e192`, 64 B) spilled around its call on
  the hottest path (`psoft` +37%): read the frame in measure_ii.
- **The same in decode** (`artlite/decode/attn256.py`: up to 24 query rows in octets of two sets of
  `[4 rows hi | 4 rows lo]`, vs. a pass over the whole cache):
  - The cache is d-block major (`decode.kv`: K records `[d-block][hi | lo]`, V records in pairs
    `[d-block][record 2]`, the order the prefill's stream reads). Per d-block, K hi, K lo and an octet's two Q panels are
    each popped once, with a chain per set and K half: II 4 for 4 MMULs, 158 cycles per record and octet (K hi then
    K lo with each Q panel popped twice: 288; on two chains: II 6).
  - P V runs over octets (two sets' 8 rows, with P hi and P lo as two MMULs into one 8-row accumulator; this avoids
    folding hi into lo rows, which bound the mv slot). A pair of records per `y` round trip on one V stream gives II 8
    for 8 MMULs.
  - The softmax's exp2 uses the bias-192 form, and the softmax runs while the V object streams in (`data-movement.md`
    "Streams").
  - Result: ~2420 -> under 700 cycles per set and group; a pass at 32K takes 2.04 / 2.07 ms per layer (2 / 3 sets), the
    DMA floor. 24 rows (the 27B's 4 slots: 6 sets) fit in one pass with the A stream at depth 2, the records read from
    the two held last objects, the gates applied by the main core, and the Q panels of octets 1-2 placed in the main
    core's memory (`artlite/decode/attn256.py` docstring): 2.13 ms vs. 4.15 for two passes, near the DMA floor of 2.04
    (with K hi then K lo per record and each Q panel popped twice: 2.64 ms, ~9.3K cycles per group, ~3.9K of it in QK).
  - Llama's cores (`decode/attn.py`, one head dim: 16 rows, A depth 2) with the octet softmax and P V: on the 1B at
    16K, a step of 4 tokens went from 32.8 to 26.4 ms (its `pexp`'s per-pair exp2 calls were the largest part).

## R14 Weight panels from gathers (IQ3_XXS, `formats/iq3.py`)

An IQ3_XXS weight is a grid entry (4 values per 8-bit index) with a sign. An 8x8 MMUL panel is 16 entries, read by 4
`load_4x32` gathers (entry i at byte 8 i; lanes 0, 1 read one table copy and lanes 2, 3 the other: see `numerics.md`
and the copies in `bank-conflicts.md`), followed by one compaction shuffle (the gather fills the even 32-bit lanes) and
the signs applied by `vaddsub8` (zero - w where a mask bit is set; a row's 8 signs are one byte of the 64-bit mask).
Per core's half tile (two super-blocks of 23 rows) this went from 4596 to ~2300 cycles:
- **The move slot is the budget**: per panel it takes the address lanes out of the accumulator (`vmov`), the tables'
  bases (`vadd`: the gathers keep their memory operand only as `add(.., sel(bcast(pointer parameter)))`), the shuffle,
  `vaddsub8`, and the mask's `vextract.64`: 5 ops. The gathers and the panel's store share QEY_RS, and the gathers and
  slot A's loads share DM_ADA. Peano schedules ~6.5 cycles per panel with 2, 3 or 6 panels per iteration (with 4: 7.75,
  due to extra `vmov`s).
- **Load the offsets directly**: `ups_to_v32acc32(*p, 3, 0)` (`vlda.ups`, slot A) loads 32 index bytes as 8 idx into
  an accumulator, with no vector register, no MMUL multiply, and no shuffle when the indices are stored in panel order.
- **Masks**: two scalar `lda` per panel cost DM_ADA slots (II 13 instead of 10 per 2 panels) and collided with the
  panels' stores in the scratch's region (memory stall 254 vs. 73 cycles per window with them compiled out). One
  64-byte vector load per 6 panels plus `ext_v2int32` brings it to 101.
- **No FIFO in the gather loop**: a `fifo_ld_*` stream there created a recurrence through its registers (post-RA RecMII
  45), and two streams in one loop spill the FIFO state every iteration (136 bytes). Instead, realign in a separate
  loop, read unaligned words as aligned blocks with `shift_bytes` (`_k23w2`), or store the data aligned (the tile layout
  is ours: the indices sit in 8-row panels, padded to 24 rows using the tile's spare bytes).
- **Run the MMUL in a separate loop** when the panels come from gathers: write the built panels to the scratch and
  multiply them in the next loop (II 17 per 12 panels). Fused with the gathers, the loop ran out of registers (post-RA)
  or took 869 instead of 639 + 164 cycles (IQ2_XXS's value gathers: 270 vs. 171 + 87, with spills). Panels built from
  masks (no address vectors live) go straight into the MMULs: IQ1_S builds them from the tile at II 32 per n (8 panels,
  each taking 2 `vextract.64`, a `vsel8` and a `vaddsub8` on the move slot), 283 vs. 266 + 98..144 cycles per 64 panels;
  IQ1_M builds them from its gather loop's masks at II 48 with the deltas. Without the panels in L1, their stores and
  reloads (in one region) also disappear.
- **Store the values in the table when they fit** (IQ2_XXS, `formats/iq2.py` `value_words`): 256 entries of 8 values
  take 2 KB as bytes, and a 4x64 table (each 32-byte group stored twice) takes 4 KB. Its halves, selected by index bit
  7, go into the two 2 KB slots, every lane reads the first copy, and the offsets are `16 (lo & 127) + D b7`. A panel is
  then its two gathers (row r's 8 values in lane r) plus the signs' `vaddsub8`, instead of masks through a byte
  transpose and two `vsel8`: 430 -> 171 cycles per 48 panels, despite the halves' bank conflicts (a gather's lanes 0, 2
  and 1, 3 share a region in 3 of 4 gathers).
- **Re-encode losslessly**: store the 7-bit sign field and its implied parity as the prefix XOR `c` (the 8 signs are
  `c ^ (c << 1)`), which costs 5 ops per 64 fields instead of a sum of 6 shifts. Write the XOR as `(c | t) - (c & t)`,
  with the subtraction done as a `vaddsub8` of an all-ones mask: LLVM folds a plain subtraction back into `^`, which
  AIE2P expands into 5 ops.
- **16-bit entries** (`load_4x16`: 2 B per entry, 4 B per entry and copy; IQ4's value pairs and IQ3_S's grid as
  nibbles, `formats/iq4.py`, `iq3s.py`): an instruction's entries land in halfword 0 of its 64-bit lanes, and
  `shuffle(c0, c1, T16_16x4_lo)` puts those of two concatenated pairs in order (16 entries per shuffle, `mmv/g16`).
  Using two shuffles instead (`T32_16x2_lo`, then `T16_32x2_lo`) left a 4-panel loop unallocatable at every II. An index
  byte covers at most 2 values of a 4-bit code, so IQ4 needs 8 gathers per panel (II 20-21 per 2 panels, bound 17 by
  DM_ADA).
- **A 9th index bit** (IQ3_S's 512-entry grid): add it on the MMUL together with the offsets,
  `mac_elem_64_2(bits, 8, mul_elem_64_2(idx, 4))`, from bits stored as bytes 0 / 128 by a separate loop. A `sel` from a
  scalar 64-bit mask inside the gather loop left it unpipelined (RecMII 48 in the physical mode). For the prefill's
  lists, the index bytes are zipped with their bits (bytes 0 / 1) into 16-bit indices, with one
  `ups_to_v32acc32(v32int16, 2)` per list (v32acc32 has no add).
- **A table too big to hold twice** (IQ1_M's 2048 16-bit entries, `formats/iq1.py`): a lane reads only the 32-byte half
  matching its parity, so a 4x16 table needs each half stored twice (4 B per entry and copy). With only 4 KB of room,
  each entry is stored once and a row takes an even and an odd lane at the same address (4 gathers per panel), with an
  index bit picking the half. That costs one select per two panels after the byte transpose (`T8_16x4` gives
  `[Ae | Be | Ao | Bo]` of 16 rows, and the select picks between the halves 32 bytes apart). Fused with the panels, the
  loop ran at II 24 per pair (post-RA 30, from the chain through the gathers, shuffles and extracts). Split into a
  gather loop (II 10, bound by the gathers' DM_ADA) and a panel loop, it takes 685 vs. 781 cycles per 64 panels, ~10.7
  per panel vs. 8.5 for IQ2 with its doubled tables. A gather's lanes 0, 2 read bank 0 and lanes 1, 3 bank 1; with the
  copies split by an index bit, the gather's two rows share a region in half of the gathers, which then wait a cycle
  (~512 cycles per IQ1 tile and core).
- **Skip the table when the tile has room** (IQ1_S): its 9280-B tile held 6400 B of indices, scales and `d`. The packer
  instead stores each 8-k block's looked-up `[A | B]` masks (2240 B per super-block, with the tile's size unchanged;
  `iq1.grid_masks` is passed to the native packer), and a panel takes two `vextract.64`, a `vsel8` and a `vaddsub8` from
  one 64-byte load per 4 panels (II 16): no offsets, gathers, half pick or table page. The decode tile went from 6309
  to 3411 cycles and the UD-IQ1_M step got 20% faster. Check a format's tile bytes against its unit before building a
  lookup.
- **Signs after an unpack** (IQ3_S): with `AIE_TRY_INITIATION_INTERVAL(24)` the loop reaches II 29 per 4 panels (the
  heuristic search stopped at 40), against its QEY_RS bound of 24 (`machine-model.md` "Shared ports"). Per super-block
  it went from 976 to 736 cycles, and the decode step got 3% faster.
- **Compacted rows from constant windows** (the prefill's IQ3_S passes: tiles of 20 rows packed into MMUL rows): with
  the weight object, its records and the units 64-byte aligned, a window's block and shift depend only on the record's
  phase (48 B records: 4 per 192 B; 24 B records: 8). Processing 4 lists (8 records) per iteration therefore makes every
  `shift_bytes` and select mask a constant: the index lists went from 968 to 587 cycles per k-tile (in two loops: low
  bytes, then 9th bits from registers), and the signs from 1271 cycles per unit to 174 per k-tile. Copying a unit's
  scalar slots straight to their compacted places (constant offsets) instead of a strided gather plus a compaction took
  2223 -> 659 cycles per unit.
