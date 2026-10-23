# Ideas measured and rejected

Check here before trying something. Retry only with a new reason: a toolchain update, a
different shape, or a fixed tool.

## Loop structure and pipelining

| idea | result |
|---|---|
| Flatten a 2-level loop nest with div / mod or shift / mask index math | not pipelined, or RecMII 41-50 with spills |
| Flatten with `add_2d_ptr/byte` iterators for separate stream bases | 8 spills (pointers + iterator state exceed `p` / `d`). What works: persistent streams rewound by `stream.seek(dims_2d_t&)`; two plain pointers walking a 3-D pattern instead of index math (restructurings.md R8) |
| Reorder a 4-accumulator MMUL loop "two accumulators at a time", or skew the chains by k-steps | ResMII 36, II 42-61; the k-step-major order reaches II 32 once the post-RA rules hold |
| 4-k-step groups holding B panels in registers | spills (12 `ex`) |
| Interleave independent phases inside one loop (dequant chunks in the MAC loop) | 95-158 spills, 11-13 KB of stack; post-pipeliner flags did not help. Shrink the live state across MAC blocks before retrying |
| One long fp32 chain per iteration (accurate SiLU: exp2 + Newton, 16-32 lanes) | never pipelines: RecMII 57-77 vs ResMII 12-17; two loops through L1 did not help either (5.3 cycles per element) |
| Fix SiLU's small-|x| error inside the pipelined loop (odd polynomial + second select) | II 11 -> 48-59 unpipelined. Solved by fission (a second in-place loop) |
| Accurate negative tail for the LUT SiLU (exp from native ops) | ~3 more fissioned loops, ~40 cycles per 32 |
| `AIE_TRY_INITIATION_INTERVAL` below the register bound; `-aie-postpipeliner-maxii` on register-bound loops | no effect |
| WAW-rewrite / GPR-realloc hints; ignore-mem-deps alone on the MAC loop | ±0-3 cycles |
| Pre-RA pipeliner with a forced small II (`AIE_TRY_INITIATION_INTERVAL(4)`) | reports a small II, then 10+ spills (NS 10, a 20-bundle body) |
| Raising `-aie-postpipeliner-maxii` / `-maxtry-ii` for one operator in a fused ELF | one flag set per ELF; maxii 140 / maxtry 80 **miscompiled** a spilling loop (pipelined with stack slots in its body: wrong results) |

## Dequant GEMV loops (Q5_K)

| idea | result |
|---|---|
| Q5_K codes, scaling and the row reduction in one loop (2 rows, y in accumulators) | II 60 (MII 44), ramps every 4 super-blocks |
| Flattened (row, super-block) loop, the row's accumulator zeroed by `mac_elem_64_conf(zero_acc)` | post-RA pipeliner fails at every II 27-46 ("Latency not met" on a loop-carried node) |
| Codes of super-block sb fused with the scaling of sb - 1 (independent work in one loop) | not pipelined |
| Two upshifts so that 4 distinct shift amounts fit the `s` registers | II 53 vs 41 |
| Loop-invariant x reloaded through an opaque pointer instead of hoisted and spilled | same II and NPU time |

## Program memory (decode overlay main core)

| idea | result |
|---|---|
| Q3_K header group loop rolled / pipelined (any loop hint) | -160..-240 B, but 95 -> 176-318 cycles per super-block (the tile is at its DMA time) |
| The header's halves as a rolled loop around one group body | clang crash for most hints; outer unrolled: 191 cycles, +656 B |
| SiLU's Newton steps or `mulff` out of line (noinline helper, rolled loop) | -96..+368 B and 1.0-1.4x the cycles; one 64-lane pass won instead (352 cycles, +384 B) |
| `minsize` on the per-tile wrappers (mv_qkv / mv_acc / mv_gu, qkv_in, resid, ...) | 0 B, mv_qkv +96 B |
| The tile dispatch as one noinline function per format, tail-called | +0..+16 B |

## Planar dequant (Q2_K / Q3_K GemmQd)

The kept forms and the expansion / MMUL-scale variants: restructurings.md R6.

| idea | result |
|---|---|
| Q3_K in one loop (any of: code bytes or `add` for the 3-bit combine, C shared per scale, `0x41`-zip with bit 7 for 16 + q, 10 hint sets) | mv 60-71 and register parking: II 80-87, never pipelined. Two loops: 32 + 34 |
| Q3_K loop 2 with S / 8 precomputed in loop 1 | loop 2 II 32, loop 1 40: 633 vs 590 cycles per 64 rows |

## GDN delta-rule panel

Per token and 8-column state panel; the kept form (restructurings.md R10): 413 cycles.

| idea | result |
|---|---|
| k / q broadcasts precomputed once per token, loaded in the loop | 1 token: 698 -> 523-532 (load-bound instead of mv-bound), 8 KB L1 per token; with the MMUL pass not needed (shuffled: 413.5 vs 408.5) |
| Two accumulators per dot product (even / odd blocks) | II 31-34 per two blocks: worse |
| Pass 1 stores the bf16 hi / lo split, pass 2 loads it | 1 token: 532 (best vector form); not combinable with the MMUL pass (bfp16 split) |
| Two tokens on the vector unit (4 dot products per pass) | 561: pass 1 load- or mv-bound (II 26-34 per two tokens), spills |
| Pass 2's loop invariants loaded from L1 each block | the compiler hoists the loads: no change |
| Bank qualifiers (state on a, panels on b) | identical schedules |
| `AIE_PREPARE_FOR_POSTPIPELINING` / `AIE_LOOP_AGGRESSIVE_REALLOC` | +-2% |

## GEMM kernels

The loop orders of a GEMM unit (y in L1, other nests, an L1 dequant staging): restructurings.md R1.

| idea | result |
|---|---|
| MAC with bf16 B staging converted in registers (32 x 8 blocks instead of 16 x 16) | II 45 vs 43 per 32 MMULs (load-slot bound either way) |
| Upstream mlir-aie bf16-staging MAC loop over k = 128 per call | II 75 per 128 k vs 2 x 43 (-13% on gemm_q gate / up); its ResMII lies above the post-RA II window |
| GEMM on weight tiles of 14 / 11 rows in 16-row MMUL slots | the last pass overwrites the next column's outputs; needs a second C FIFO |
| Prefill GEMM on 1 core row at T = 64 (smaller PDI) | the weights stream twice: 56 vs 47 ms |
| A bf16 x bfp16 GEMM on 1-2 core rows at M = 64 (cheaper ConfigureOp) | 95 / 63 ms vs 54 ms on 4 rows: compute-bound |
| Shaving a DMA-floor Q4_K GEMM's kernel (memory stalls 11% of busy) | no e2e gain: every shape is at its DMA floor (within 1-3% of the empty kernel) |
| Planar GemmQd with a pinned L1 layout (A, D and the C tiles each in its own 16 KB region, cold buffers in the gaps; the weight objects in one or two regions) | 1B Q5_K at T = 256: QKV / O / down -7 / -12 / -4%; 3B at T = 2048: down +1.4%, gate/up +3-4% with the same MAC and dq windows (the DMA side). Per-window conflict classes on the MAC cores only (`PLANAR_PLANS`) do better: 3B Q2_K down -10%, gate/up -4%; Q5_K unchanged |
| A smaller C depth to make a pinned planar layout fit (one C object) | inconsistent: 3B Q2_K QKV -8%, 1B Q3_K_L O -20% at T = 256 but +8% at T = 2048; splitting the SiLU scratch instead: 3B Q5_K gate/up +2% |
| Passes of up to 128 rows (n_max) for every planar down / O | Q5_K down 5.55 -> 6.04 ms, Q4_1 (14 rows) 8.06 -> 9.09; adopted only where `planar_seconds` estimates 15% (Q3_K of 21 rows, Q4_0) |
| A Q5_K_M layer's V GEMM (Q6_K, its own op) on 1 or 2 MAC rows at T = 256 (a smaller ConfigureOp) | 147.8 / 147.2 vs 146.8 ms on 3 rows |
| GemmQ pairs: the split MAC loop ([half][band][block], 3-D walks) at II 32 | II 33 with every pragma (ii32, post, rewrite / realloc / OLP / WAW hints, two seek orders) |
| GemmQ pairs: the 4-iteration dequant unrolled (full / by 2) against its 76-cycle ramp | 212 / 227 vs 196 cycles |
| GemmQ pairs on gate/up: memtile weight FIFO depth 3 / 4, L1 A depth 3 (`cons(depth=3)`) | 1693 / 1711 / 1741 vs 1728 us device busy: noise (the GEMM sits near its weight stream) |
| Llama prefill gate/up at T = 128 on pair GemmQ instead of a 2-row GemmQd | 119.9 vs 117.9 ms (3B, 128 tokens) |
| Plain planar GemmQd (Gemma prefill qkv / o / down) with a C pingpong: one C object and a C tile `acc`, the first k-tile in acc alone, then acc -> C -> acc | the MAC window 974 -> 927 cycles (memory stall 159 -> 112), qkv alone 0.242 -> 0.232 ms, the 1152-token chunk unchanged (391 ms; 288 tokens +1%): the single C object's drain waits offset it. Its "separate" placement (both A, both D in one region each): 0.275 ms |
| Gemma prefill GEMMs with 128 k per C round trip (simulated) or the C tile in fp32 | 128 k: no measurable accuracy gain; fp32 C: does not fit L1 at 96 tokens per MAC row (24 KB per C tile) |
| Dequant pair (GemmQ): one A object instead of two, so the DMA refills it only during the next k-tile's dequant (the A conflicts in the MAC loop's region) | +29% (Llama 3B gate/up 6.69 -> 8.62 ms): A is broadcast to the 8 columns, one object puts the columns in lockstep |
| The dequant pair's stack out of region 0 (`stack_address`: its reloads meet the X writes there) | aiecc refuses a stack outside bank A when kernels are compiled separately (Artlite's kernel.o) |
| The dequant pair's last spilled constant (-128) as a load from a bank-1 table walked by two copies | reloads 2 -> 0, II 40 -> 42: the saved conflicts and the 2 cycles cancel |
| SD 1.5's activation table copies in two regions | its fixed L1 layout has no 1.6 KB in another region; the tables' stalls are ~0.1-0.2% of a UNet step |

## Data movement

| idea | result |
|---|---|
| Weights over 2 shim channels per column | 50.0 vs 49.9 GB/s; 56 GB/s for 8 and for 16 streams: one channel per column already reaches the ceiling |
| Deeper L1 / L2 FIFOs, pinned layouts, AxCACHE hints on a DMA-bound operator | no e2e change |
| Gemma decode overlay: A memtile depth 20 / 22 (channels pinned to both parities), X depth 2 in the main core's L1 or 4 in the memtile | 55.6-56.4 ms vs 55.6-56.2 at depth 12, alternated processes: its boundaries are not prefetch-bound |
| qwen35 decode overlay (35B-A3B): A memtile depth 12 / 16 / 20 instead of 8 | S = 1 41.4-41.7 ms, S = 3 66.1-66.6 at every depth |
| An MoE layer's router as a phase before the shared expert's gate / up (the logits' sync and the route object beside its tiles; 35B-A3B decode) | +0.2-0.6 ms per step at S 1-4: the extra phase and the shared down's input round trip after the routing cost more than the overlap |
| Gemma decode overlay: each column's X fills waited at a sync (one wait per run of columns) instead of before each fill | 57.06 vs 57.02 ms (A / B in one process) |
| Prefetching the FFN down-projection weights behind gate / up on the same channel | no e2e change |
| Streaming a residual update's v and r from two regions at once | a core must hold both: aiecc rejects a release that does not match an acquire of the same loop body; interleave pairs in one stream |
| Broadcasting a KV head's cache to the two decode attention cores that read it (each still consumes all of it) | no change at any context; splitting the positions between them and merging (m, l, y) gave -8% at 32K |
| Splitting a DMA-bound attention block's query rows over the 4th core row (Llama 3B prefill: 96 rows as 4 x 24) | not built: 12.09 vs 12.16 ms with the compute compiled out; 64-query blocks (half the KV reads) gave -25% |
| Two shim channels per KV head joined in one memtile (Gemma prefill attention) | 7 memtile inputs: does not build (aie2p-coding section 8); two streams through two memtiles work, and each from two shims once Q rides the KV stream (qwen35: data-movement.md) |
| A second o copy for the qwen35 27B's attention (three row panels per core) | no L1 left (61 of 64 KB): its P V stays in place, ~2.8 ms of memory stall per layer at 16K |

## Numerics

| idea | result |
|---|---|
| 3-bit weights (MSE-searched); 2-bit LM-head screening + exact refinement | KL x2-3; screening error 0.8 logits -> 14-76% candidate blocks |

## Upstream mlir-aie kernels

- **Adopted**:
  - the softmax max of the raw inputs, scaled by the element's own multiply;
  - no scalar soft-float math (`al::inv_rms`);
  - the idea of warning about library arithmetic (`checks.py`).
- **Kept Artlite's**:
  - exp2 (`km.exp2`, 0.4 bf16 ULP; upstream's polynomial exp2 is ~15 cycles per element,
    its softmax uses the chord vexp2);
  - SiLU (upstream: plain vtanh, same small-|x| error).
- **Rejected**: the bf16-staging MAC loop (above).
- **Global pre-RA stage cap** (`-aie-pipeliner-max-stagecount=5`): helps some loops, slows others.
  The per-loop equivalent is `AIE_TRY_INITIATION_INTERVAL(N)`: gemm_q `_hdr` / `_hcp` gate / up
  -2.4% (Q4_0), bit-identical.
- `AIE_PREPARE_FOR_POSTPIPELINING` beat `AIE_TRY_INITIATION_INTERVAL(12)` on the LUT SiLU loop
  (II 11 vs 13). Re-check pragma choices after a toolchain update.

## Prefill GEMM dequant and qwen35 prefill kernels

| idea | result |
|---|---|
| Q4_K dequant on the MMUL: diag(a) Q^T - diag(\|b\|) 1^T with unsigned-mantissa diagonal panels built in registers | a bfp16 operand assembled in registers moves its 64-bit exponent into `el` / `eh` (2 `mov` on mv) per panel, the codes' too (SKILL.md section 4): mv ~40 per 8 panels vs the vector form's 28; pre-RA II 53 with spills |
| Codes -> bf16 by `vunpack` u8 -> u16 + `vbor 0x4300` (no shuffle port) | `crunpacksize` switches between the 4 -> 8 bit loads and the 8 -> 16 bit vunpack: mv 37-58, II 40-59 vs the zips' 40 |
| Codes -> bf16 on the integer MAC (acc32 = 0x4300 + q, SRS to int16) | 2 `vsrs` per panel next to the panel pushes: store slot 32-40, not pipelined (RecMII 58-96) |
| Loop pragmas on the GemmQ MAC loop (post, ii32, rewrite / realloc hints) | 555-573 cycles per k-tile, none below the base's II 33 |
| fp32-C MAC loop (sd15, II 41 per 32 MMULs with the C-load annotation (bank-conflicts.md) vs 33 with bf16 C: four 2048-bit C hand-overs per iteration): `AIE_TRY_INITIATION_INTERVAL(34-40)`, `AIE_PREPARE_FOR_POSTPIPELINING` (II 55-57), rewrite / realloc / WAW hints | II 41-57: the hand-over binds (C loads are slot A only: slot A 34 ops vs slot B 19) |
| Two k-tiles per fp32-C MAC call (an m-64 A object's, the B locks inside; the loop still pipelines) | busy -3% per 16x16 GEMM, +400 B on both core roles |
| The same loop as 16 x 8 blocks (two accumulators: II 19 per 16 MMULs, B packed [8-column block][k-step]) | on the NPU 59 -> 67 us per 1280 x 1280 GEMM: NS 3 puts a block's C stores next to the loads two blocks later (the region alternation breaks), and A is popped twice as often next to the DMA's writes (memory stall 13.5K -> 33K cycles per op) |
| `_hdr_q4k` with every crumb / d word of a tile live at once | 3.3 KB stack (54 spills): over the worker's 2 KB; per (plane, tile, super-block pair) with the d words reloaded: 920 cycles, 1 KB |
| GDN scan pass 2a as two blocks per iteration, or its MAC chain in two accumulators | 2729-2941 vs 2548 cycles per pair (RecMII 16 from register reuse either way) |
| Attention `_pexp` in two loops (P, then l += P) | 312 vs 380 cycles only fully unrolled, with a 2.3 KB frame |
| sd15's exp pass (cubic, every clamp: pre-RA II 23, register allocation fails with 9 constant vectors): a quadratic 2^f (numerics.md) with the reference vectors walked by `add_3d_ptr` II 19; also without the f clamps II 15 (overflow above y = 320); the cubic with that walk II 55; fission (v, f, then P) 397-590 vs 424 cycles per call (two ramps; f live for II 22 until loaded twice) | ~30 us per UNet section: not taken |
