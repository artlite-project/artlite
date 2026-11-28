# Ideas measured and rejected

Check here before trying something. Retry only with a new reason: a toolchain update, a different shape, or a fixed
tool.

## Loop structure and pipelining

| idea | result |
|---|---|
| Flatten a 2-level loop nest with div / mod or shift / mask index math | not pipelined, or RecMII 41-50 with spills |
| Walks for cheap index math in an mv-bound loop (sd15 `exp40`: `i & 1`, `i >> 1`; the GDN scan's V^T blocks) | II 15 -> 17, 21 -> 20: no gain |
| Flatten the GDN conv's (column block, token) nest, the 8 tap vectors reloaded per token | II 13 (bound 8): 444 vs. 431 cycles |
| Flatten with `add_2d_ptr/byte` iterators for separate stream bases | 8 spills (the pointers plus iterator state exceed the `p` / `d` registers). What works instead: persistent streams rewound by `stream.seek(dims_2d_t&)`, or two plain pointers walking a 3-D pattern instead of index math (`restructurings.md` R8) |
| Reorder a 4-accumulator MMUL loop "two accumulators at a time", or skew the chains by k-steps | ResMII 36, II 42-61; the k-step-major order reaches II 32 once the post-RA rules hold |
| 4-k-step groups holding B panels in registers | spills (12 `ex`) |
| Interleave independent phases inside one loop (dequant chunks in the MAC loop) | 95-158 spills and 11-13 KB of stack; post-pipeliner flags did not help. Shrink the live state across MAC blocks before retrying |
| One long fp32 chain per iteration (accurate SiLU: exp2 + Newton, 16-32 lanes) | never pipelines (RecMII 57-77 vs. ResMII 12-17); splitting it into two loops through L1 did not help either (5.3 cycles per element) |
| Fix SiLU's small-\|x\| error inside the pipelined loop (odd polynomial + second select) | II 11 -> 48-59, unpipelined. Fission solved it (a second in-place loop) |
| Accurate negative tail for the LUT SiLU (exp from native ops) | ~3 more fissioned loops, ~40 cycles per 32 lanes |
| `AIE_TRY_INITIATION_INTERVAL` below the register bound; `-aie-postpipeliner-maxii` on register-bound loops | no effect |
| WAW-rewrite / GPR-realloc hints; ignore-mem-deps alone on the MAC loop | ±0-3 cycles |
| Pre-RA pipeliner with a forced small II (`AIE_TRY_INITIATION_INTERVAL(4)`) | reports a small II, then produces 10+ spills (NS 10, a 20-bundle body) |
| Raising `-aie-postpipeliner-maxii` / `-maxtry-ii` for one operator in a fused ELF | flags apply to the whole ELF, not one operator; maxii 140 / maxtry 80 **miscompiled** a spilling loop (it was pipelined with stack accesses in its body and gave wrong results) |

## Dequant GEMV loops (Q5_K)

| idea | result |
|---|---|
| Q5_K codes, scaling and the row reduction in one loop (2 rows, y in accumulators) | II 60 (MII 44), with ramps every 4 super-blocks |
| Flattened (row, super-block) loop, with the row's accumulator zeroed by `mac_elem_64_conf(zero_acc)` | the post-RA pipeliner fails at every II from 27 to 46 ("Latency not met" on a loop-carried node) |
| Codes of super-block sb fused with the scaling of sb - 1 (independent work in one loop) | not pipelined |
| Two upshifts so that 4 distinct shift amounts fit in the `s` registers | II 53 vs. 41 |
| Loop-invariant x reloaded through an opaque pointer instead of being hoisted and spilled | same II and NPU time |

## Decode GEMV tile kernels (`formats/*.mmv`)

Cycles per core and half tile, measured in NPU event windows (`tests.decode.mmv.MmvOperator`,
`variant_search --npu --region`).

| idea | result |
|---|---|
| Inline (`always_inline`) a format's helpers into its tile kernel (each helper has that single caller) | Q8_0 -10%, Q4_K -1%, Q5_K +1%, Q6_K +4%, NVFP4 +5% (`iq4_w` stays a call). The calls that remain keep the caller's values in callee-saved registers, and the inlined loops lose II (NVFP4's `nv_x`: 14 -> 20) or spill (Q5_K: 21 registers) |
| ... and inline the shared helpers too (`_sc`, `_sc1`, `iq4_w`, so the tile kernel calls nothing) | Q8_0 -14%, Q4_K -5%, Q6_K -2%, NVFP4 -2.4% at 4 slots but +5% at 8 (`nv_sc` merges into `nv_x`'s exit block: 226 cycles), Q5_K 0% and "ran out of registers" with `_trq` also inlined. The formats that gain most are DMA-bound in decode, and copying a shared helper into each format costs program memory on full cores |

## Program memory (decode overlay main core)

| idea | result |
|---|---|
| Roll or pipeline the Q3_K header group loop (any loop hint) | saves 160-240 B, but raises cycles per super-block from 95 to 176-318 (the tile is already at its DMA time) |
| The header's halves as a rolled loop around one group body | clang crashes with most hints; with the outer loop unrolled: 191 cycles, +656 B |
| Move SiLU's Newton steps or `mulff` out of line (noinline helper, rolled loop) | -96..+368 B at 1.0-1.4x the cycles; a single 64-lane pass won instead (352 cycles, +384 B) |
| `minsize` on the per-tile wrappers (`mv_qkv` / `mv_acc` / `mv_gu`, `qkv_in`, `resid`, ...) | 0 B saved, and `mv_qkv` grows by 96 B |
| The tile dispatch as one noinline function per format, tail-called | +0..+16 B |
| Cold helpers moved out of line, taking 64-lane vectors or accumulators by value (`mulff<64>`, a panel push) | +1.5 KB on the GDN core: the call sites cost more than the inlined bodies |
| Roll a 4-token solve's identity and substitution | +320 B vs. unrolled (from `x[t]`'s scalar stores and the loop itself) |

## GDN core of 4 slots at S 1

| idea | result |
|---|---|
| Fill a head's group and panels as one BD chain (one push, one wait) | GDN 0.94 -> 0.98 ms on the 1B: two chains fill the channel's 4 BDs, leaving nothing queued ahead |
| Apply SiLU to the 4 rows before reordering them into panel order (512 elements instead of 1024) | no change in the GDN window (2.82 -> 2.81 ms), because that compute overlaps the DMA; kept for its smaller code size |

## Planar dequant (Q2_K / Q3_K GemmQd)

The kept forms and the expansion / MMUL-scale variants are in `restructurings.md` R6.

| idea | result |
|---|---|
| Q3_K in a single loop (any of: code bytes or `add` for the 3-bit combine, C shared per scale, `0x41`-zip with bit 7 for 16 + q, 10 hint sets) | mv 60-71 plus register parking: II 80-87, never pipelined. Two loops reach II 32 + 34 |
| Q3_K loop 2 with `S / 8` precomputed in loop 1 | loop 2 reaches II 32, but loop 1 rises to 40: 633 vs. 590 cycles per 64 rows |

## GDN delta-rule panel

Measured per token and 8-column state panel; the kept form (`restructurings.md` R10) takes 413 cycles.

| idea | result |
|---|---|
| Precompute the k / q broadcasts once per token and load them in the loop | 1 token: 698 -> 523-532 cycles (load-bound instead of mv-bound), at 8 KB of L1 per token; unnecessary with the MMUL pass (shuffled: 413.5 vs. 408.5) |
| Two accumulators per dot product (even / odd blocks) | II 31-34 per two blocks: worse |
| Pass 1 stores the bf16 hi / lo split and pass 2 loads it | 1 token: 532 cycles (the best vector form); cannot be combined with the MMUL pass, which uses a bfp16 split |
| Two tokens on the vector unit (4 dot products per pass) | 561 cycles: pass 1 is load- or mv-bound (II 26-34 per two tokens) and spills |
| Load pass 2's loop invariants from L1 in each block | the compiler hoists the loads: no change |
| Bank qualifiers (state on bank a, panels on bank b) | identical schedules |
| `AIE_PREPARE_FOR_POSTPIPELINING` / `AIE_LOOP_AGGRESSIVE_REALLOC` | ±2% |

## GEMM kernels

The loop orders of a GEMM unit (y in L1, other nests, an L1 dequant staging) are in `restructurings.md` R1.

| idea | result |
|---|---|
| MAC loop with bf16 B staging converted in registers (32 x 8 blocks instead of 16 x 16) | II 45 vs. 43 per 32 MMULs (load-slot bound either way) |
| Upstream mlir-aie's bf16-staging MAC loop over k = 128 per call | II 75 per 128 k vs. 2 x 43 (-13% on gemm_q gate / up); its ResMII lies above the post-RA II window |
| GEMM on weight tiles of 14 / 11 rows in 16-row MMUL slots | the last pass overwrites the next column's outputs; this would need a second C FIFO |
| Prefill GEMM on 1 core row at T = 64 (for a smaller PDI) | the weights stream twice: 56 vs. 47 ms |
| A bf16 x bfp16 GEMM on 1-2 core rows at M = 64 (for a cheaper `ConfigureOp`) | 95 / 63 ms vs. 54 ms on 4 rows: compute-bound |
| Shave cycles off a DMA-floor Q4_K GEMM's kernel (memory stalls at 11% of busy time) | no end-to-end gain: every shape is at its DMA floor (within 1-3% of the empty kernel) |
| Planar GemmQd with a pinned L1 layout (A, D and the C tiles each in a separate 16 KB region, cold buffers in the gaps, the weight objects in one or two regions) | 1B Q5_K at T = 256: QKV / O / down -7 / -12 / -4%; 3B at T = 2048: down +1.4% and gate/up +3-4% with the same MAC and dq windows (so the loss is on the DMA side). Per-window conflict classes applied only on the MAC cores (`PLANAR_PLANS`) do better: 3B Q2_K down -10%, gate/up -4%; Q5_K unchanged |
| A smaller C depth so that a pinned planar layout fits (a single C object) | inconsistent: 3B Q2_K QKV -8% and 1B Q3_K_L O -20% at T = 256, but +8% at T = 2048; splitting the SiLU scratch instead gave 3B Q5_K gate/up +2% |
| Passes of up to 128 rows (`n_max`) for every planar down / O | Q5_K down 5.55 -> 6.04 ms, Q4_1 (14 rows) 8.06 -> 9.09 ms; adopted only where `planar_seconds` estimates them 15% faster (Q3_K with 21 rows, Q4_0) |
| A Q5_K_M layer's V GEMM (Q6_K, a separate op) on 1 or 2 MAC rows at T = 256 (for a smaller `ConfigureOp`) | 147.8 / 147.2 vs. 146.8 ms on 3 rows |
| GemmQ pairs: the split MAC loop ([half][band][block], 3-D walks) at II 32 | II 33 with every pragma (ii32, post, rewrite / realloc / OLP / WAW hints, two seek orders) |
| GemmQ pairs: unroll the 4-iteration dequant (fully / by 2) to avoid its 76-cycle ramp | 212 / 227 vs. 196 cycles |
| GemmQ pairs on gate/up: memtile weight FIFO depth 3 / 4, L1 A depth 3 (`cons(depth=3)`) | 1693 / 1711 / 1741 vs. 1728 us device busy: noise (the GEMM runs close to its weight stream's rate) |
| Llama prefill gate/up at T = 128 on pair GemmQ instead of a 2-row GemmQd | 119.9 vs. 117.9 ms (3B, 128 tokens) |
| Plain planar GemmQd (Gemma prefill qkv / o / down) with a C ping-pong: one C object plus a C tile `acc`, accumulating the first k-tile in `acc` alone, then `acc` -> C -> `acc` | the MAC window drops from 974 to 927 cycles (memory stall 159 -> 112) and qkv alone from 0.242 to 0.232 ms, but the 1152-token chunk is unchanged (391 ms; +1% at 288 tokens), because waits on the single C object's drain offset the gain. Its "separate" placement (both A objects in one region, both D objects in another): 0.275 ms |
| Gemma prefill GEMMs with 128 k per C round trip (simulated) or an fp32 C tile | 128 k: no measurable accuracy gain; fp32 C does not fit in L1 at 96 tokens per MAC row (24 KB per C tile) |
| Dequant pair (GemmQ) with one A object instead of two, so that the DMA refills it only during the next k-tile's dequant (avoiding A conflicts in the MAC loop's region) | +29% (Llama 3B gate/up 6.69 -> 8.62 ms): A is broadcast to the 8 columns, so a single object puts the columns in lockstep |
| Move the dequant pair's stack out of region 0 (`stack_address`), where its reloads collide with the X writes | aiecc refuses a stack outside bank A when kernels are compiled separately (as in Artlite's `kernel.o`) |
| Load the dequant pair's last spilled constant (-128) from a bank-1 table walked by two copies | reloads 2 -> 0 but II 40 -> 42: the saved conflicts and the 2 extra cycles cancel out |
| Place SD 1.5's activation table copies in two regions | its fixed L1 layout has no free 1.6 KB in another region; the table stalls are only ~0.1-0.2% of a UNet step |

## Gather loops (IQ formats: IQ3_S, IQ3_XXS, IQ4)

| idea | result |
|---|---|
| IQ3_S prefill: gather per tile in its row groups and compact by 4-row halves on the store (no index or sign compaction) | 168 tile-panels per k-tile instead of 128, and 32-byte half stores on QEY_RS: II 35-37 per 4 panels (12 panels per iteration for constant offsets do not pipeline) |
| IQ3_S prefill: 32-bit grid entries (no unpack, 4 KB copies) | `_iq3sg` 532 vs. 535 cycles (II 31 either way), and no pinned layout fits the dequantizing core |
| IQ3_S prefill: gathers, signs and the bfp16 conversion in a single loop | II 59 per 4 panels with spills (mv 47, including 17 `vmov`): 981 vs. 538 + 233 cycles for two loops |
| IQ3_S decode: a word of 9th bits per 4 panels in the records (a re-layout), with a `vsel` inside the loop instead of `_iq3s_hb` | II 33 instead of 29: 841 vs. 736 + 122 cycles per super-block |
| Unroll gather loops to 8 panels per iteration (any target II) | not pipelined, spills |
| IQ3_XXS / IQ4 decode panel loops at target IIs | II 39 / 21 unchanged (their QEY_RS bounds are 30 / 18) |
| IQ1 decode: store the panels' vsel8 / vaddsub8 masks in the gather loop and load them as scalars (no `vextract.64`) | each mask needs a `mov` into the mask registers: II 13 vs. 9 per two panels |
| Do the 8-slot GEMV's scaling inside the MMUL loop (two row groups' sums in accumulators, c = bf16(C) in registers: `_iq1_xs`) | II 16 per n, bound by the qwa port (the conversions can no longer use the store slot's `vst.conv`): 376-406 cycles per half super-block vs. 98 + 220 in two loops |

## Data movement

| idea | result |
|---|---|
| Weights over 2 shim channels per column | 50.0 vs. 49.9 GB/s; 56 GB/s with both 8 and 16 streams: one channel per column already reaches the ceiling |
| Deeper L1 / L2 FIFOs, pinned layouts, AxCACHE hints on a DMA-bound operator | no end-to-end change |
| Run the vision encoders' GEMMs on two 64-token tiles per weight pass (each B k-tile serves 128 tokens per core) | not built: the ops are core-bound (Qwen3-VL's up projection, K 1152 -> 4480 + GELU, one band: 203 us = MAC 252 x 1006 cycles + ~55 us of epilogues; 155 us with the MAC compiled out), and a 128 x 80 fp32 C needs 40 KB of an already full 64 KB L1. bf16 C between k-tiles was rejected for precision (sd15_v2: +16% error at K 1280, +52% at K 5120) |
| Gemma decode overlay: A memtile depth 20 / 22 (channels pinned to both parities), X depth 2 in the main core's L1 or 4 in the memtile | 55.6-56.4 ms vs. 55.6-56.2 ms at depth 12, in alternated processes: its boundaries are not prefetch-bound |
| qwen35 decode overlay (35B-A3B): A memtile depth 12 / 16 / 20 instead of 8 | S = 1: 41.4-41.7 ms, S = 3: 66.1-66.6 ms at every depth |
| An MoE layer's router as a separate phase before the shared expert's gate / up (the logits' sync and the route object next to its tiles; 35B-A3B decode) | +0.2-0.6 ms per step at S 1-4: the extra phase and the round trip of the shared down's input after routing cost more than the overlap saves |
| Gemma decode overlay: wait for each column's X fills at a sync (one wait per run of columns) instead of before each fill | 57.06 vs. 57.02 ms (A / B in one process) |
| Prefetch the FFN down-projection weights behind gate / up on the same channel | no end-to-end change |
| Stream a residual update's v and r from two regions at once | a core must hold both, but aiecc rejects a release that does not match an acquire in the same loop body; interleave pairs in one stream instead |
| Broadcast a KV head's cache with hi + lo rows to the two decode attention cores that read it (each still consumes all of it), up to 4 tokens | no change at any context; splitting the positions between the two cores and merging (m, l, y) gave -8% at 32K. With single bfp16 rows and each core handling its own heads, this becomes the 8-slot designs' single pass (`restructurings.md` R13) |
| A chain of 4 on one stream (the one-pass form on MiniCPM-V), or on two streams at every S | vs. one stream per KV head: S 1 at 32K 15.6 -> 20.3 ms (1B); two streams at S 1, 2 on the 35B-A3B: +1.7 ms at 32K. Sub-steps with more rows than a chain holds run in two rounds of its quarters, one KV head each, on 4 shared streams (qwen35 `attn.py`) |
| An MTP drafter's attention over a window of its cache (the last 2K / 8K positions at 32K) | a draft takes 10.8 / 11.3 ms instead of 13.9 ms, but accepted drafts drop from 196 / 120 / 66 to 167 / 76 / 30 and 169 / 80 / 32 at depths 1-3: the block uses the far context |
| Split a DMA-bound attention block's query rows over a 4th core row (Llama 3B prefill: 96 rows as 4 x 24) | not built: 12.09 vs. 12.16 ms with the compute compiled out; 64-query blocks (half the KV reads) gave -25% |
| Two shim channels per KV head joined in one memtile (Gemma prefill attention) | 7 memtile inputs do not build (aie2p-coding section 8); two streams through two memtiles work, each fed from two shims once Q rides the KV stream (qwen35: `data-movement.md`) |
| A second o copy for the qwen35 27B's attention (three row panels per core) | no L1 left (61 of 64 KB used): its P V stays in place, with ~2.8 ms of memory stall per layer at 16K |

## Numerics

| idea | result |
|---|---|
| 3-bit weights (MSE-searched) as an LM head; a 2-bit screening head refined on the NPU by blocks of rows | KL x2-3; a screening error of 0.8 logits leaves 14-76% of the blocks as candidates. What works: the head's own high plane on the NPU plus the candidate rows computed exactly on the host (`data-movement.md`, "An LM head's high plane") |
| Outlier channels / FFN units in dedicated bfp16 blocks (pad slots, with the largest units' down weights as hi + lo there) | Qwen3-VL's ViT: median per-token error 4.1 -> 3.2-3.9%, with the fragile tokens unchanged (`artlite.vision.vit`) |

## Upstream mlir-aie kernels

- **Adopted**:
  - the softmax max taken over the raw inputs and scaled by the same multiply as the elements;
  - no scalar soft-float math (`al::inv_rms`);
  - the idea of warning about library arithmetic (`checks.py`).
- **Kept Artlite's**:
  - exp2 (`km.exp2`, 0.4 bf16 ULP; upstream's polynomial exp2 costs ~15 cycles per element, and its softmax uses the
    chord `vexp2`);
  - SiLU (upstream uses plain `vtanh`, with the same small-|x| error).
- **Rejected**: the bf16-staging MAC loop (above).
- **Global pre-RA stage cap** (`-aie-pipeliner-max-stagecount=5`): helps some loops and slows others. The per-loop
  equivalent is `AIE_TRY_INITIATION_INTERVAL(N)`: -2.4% on gemm_q's `_hdr` / `_hcp` gate / up (Q4_0), bit-identical.
- `AIE_PREPARE_FOR_POSTPIPELINING` beat `AIE_TRY_INITIATION_INTERVAL(12)` on the LUT SiLU loop (II 11 vs. 13).
  Re-check pragma choices after a toolchain update.

## Prefill GEMM dequant and qwen35 prefill kernels

| idea | result |
|---|---|
| Q4_K dequant on the MMUL: diag(a) Q^T - diag(\|b\|) 1^T with unsigned-mantissa diagonal panels built in registers | a bfp16 operand assembled in registers moves its 64-bit exponent into `el` / `eh` (2 `mov`s on mv) per panel, and so do the codes (SKILL.md section 4): mv ~40 per 8 panels vs. 28 for the vector form; pre-RA II 53 with spills |
| Convert codes to bf16 with `vunpack` u8 -> u16 + `vbor 0x4300` (avoiding the shuffle port) | `crunpacksize` switches between the 4 -> 8 bit loads and the 8 -> 16 bit `vunpack`: mv 37-58, II 40-59 vs. 40 for the zips |
| Convert codes to bf16 on the integer MAC (acc32 = 0x4300 + q, SRS to int16) | 2 `vsrs` per panel alongside the panel pushes: store slot 32-40, not pipelined (RecMII 58-96) |
| Loop pragmas on the GemmQ MAC loop (post, ii32, rewrite / realloc hints) | 555-573 cycles per k-tile; none gets below the baseline's II 33 |
| fp32-C MAC loop (the vision overlay: II 41 per 32 MMULs with the C-load annotation from `bank-conflicts.md`, vs. 33 with bf16 C, because of four 2048-bit C hand-overs per iteration) with `AIE_TRY_INITIATION_INTERVAL(34-40)`, `AIE_PREPARE_FOR_POSTPIPELINING` (II 55-57), or rewrite / realloc / WAW hints | II 41-57: the hand-over is the bound (C loads use slot A only: 34 ops on slot A vs. 19 on slot B) |
| Two k-tiles per fp32-C MAC call (for an m-64 A object, with the B locks inside; the loop still pipelines) | busy time -3% per 16x16 GEMM, but +400 B on both core roles |
| The same loop as 16 x 8 blocks (two accumulators: II 19 per 16 MMULs, B packed as [8-column block][k-step]) | on the NPU 59 -> 67 us per 1280 x 1280 GEMM: NS 3 places a block's C stores next to the loads two blocks later (breaking the region alternation), and A is popped twice as often next to the DMA's writes (memory stall 13.5K -> 33K cycles per op) |
| `_hdr_q4k` keeping every crumb / d word of a tile live at once | 3.3 KB of stack (54 spills), over the worker's 2 KB; processing per (plane, tile, super-block pair) with the d words reloaded instead: 920 cycles, 1 KB |
| GDN scan pass 2a as two blocks per iteration, or with its MAC chain split across two accumulators | 2729-2941 vs. 2548 cycles per pair (RecMII 16 from register reuse either way) |
| Attention `_pexp` split into two loops (P, then l += P) | 312 vs. 380 cycles, but only when fully unrolled, with a 2.3 KB frame |
| sd15's exp pass (a cubic with every clamp: pre-RA II 23, register allocation fails with 9 constant vectors): a quadratic 2^f (`numerics.md`) with the reference vectors walked by `add_3d_ptr` reaches II 19, and II 15 without the f clamps (overflowing above y = 320); the cubic with that walk gets II 55; fission (v, f, then P) takes 397-590 vs. 424 cycles per call (two ramps; f stays live for II 22 until it is loaded twice) | ~30 us per UNet section: not taken |
| A reciprocal by Newton steps from a seed built from the bits, instead of `aie::inv`'s 32 scalar inversions, 32 lanes (the 27B's one-pass gates) | fp32 products are emulated: 180 vs. 133 bundles, plus 1024 B of spills |
| `AIE_LOOP_AGGRESSIVE_REALLOC` on the prefill attention's exp loop (`prefill.attn` `ppan`) | II 22 -> 20 with 4 reloads, but wrong results (inf) on the NPU |
| Compute the prefill attention's 2^(v - 128) from v's bits (`vsub.16`, `vups`, `vsrs`: no `vexp2`, one fewer vector op) | II 23-29 vs. 22: bound by the store slot |
| Keep the chunk's maximum in the prefill attention's exp loop (a loop-carried vector `max`) | II 32-39 vs. 22; the maximum is computed in the score loop's epilogue instead |
| P V over both V halves of a chunk in one loop (`add_3d_ptr` walks, 3-D stream seeks) | 4 spills, II 38: 640 vs. 2 x 298 cycles |
| Chunks of 8 records in the prefill attention (CR 8) | P V's body of 64 MMULs exceeds the post-pipeliner's II window (60): not pipelined, 810-906 vs. 2 x 298 cycles; L1 fits it only without o's second copy |
