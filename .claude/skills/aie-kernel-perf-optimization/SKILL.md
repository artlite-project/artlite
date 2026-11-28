---
name: aie-kernel-perf-optimization
description: Make AIE2P (NPU2) kernels and designs fast in Artlite (Peano / llvm-aie). Use before optimizing an operator, when writing a hot loop, when a loop's II is above its bound, spills or does not pipeline, when a kernel runs slower on the NPU than its schedule (memory stalls, bank conflicts), or when a core's program memory overflows. Covers finding where the time goes on the NPU (ablations, per-core event windows and stall counters), bounding a loop, the measure_ii / variant_search loop, the source rules that decide pipelining, proven restructurings (accumulators in registers, independent MMUL chains, compile-time lanes, loop fission, LUT math, arithmetic on the MMUL, neighbour cores sharing work), L1 bank placement, data-movement levers, program memory and rejected ideas.
---

# AIE2P kernel performance (Peano)

Reference files (read the one each step points to):

| file | read when |
|---|---|
| [reference/measuring.md](reference/measuring.md) | timing a region on the NPU: hygiene, ablations, event windows, core / memory-module counters (also in host-built transactions), reading L1 |
| [reference/machine-model.md](reference/machine-model.md) | computing a loop's bound: slots, shared ports, registers, latencies, load-slot binding, a worked example |
| [reference/compiler-feedback.md](reference/compiler-feedback.md) | reading remarks and assembly; symptom -> fix table |
| [reference/restructurings.md](reference/restructurings.md) | applying a section-3 restructuring: the case studies and their numbers |
| [reference/loop-hints.md](reference/loop-hints.md) | choosing `AIE_*` pragmas, per-loop hints, global `-mllvm` flags |
| [reference/numerics.md](reference/numerics.md) | fp32 via bf16 splits, exp2 / SiLU / GELU recipes, bfp16 precision, rounding, accuracy studies |
| [reference/bank-conflicts.md](reference/bank-conflicts.md) | memory stalls: the DM regions and banks, lookup-table copies, finding the conflicting region, placement, case studies |
| [reference/data-movement.md](reference/data-movement.md) | a DMA-bound operator: port and DDR rates, layouts, broadcasts, BD limits, task queues, overlays, libraries |
| [reference/rejected.md](reference/rejected.md) | before trying an idea: what was measured and did not help |

Related skills: **measure-ii** (`measure_ii` and `variant_search` reports), **profiling** (counters of compiled
operators), **aie2p-coding** (architecture facts and DMA limits; section 9 covers compiler crashes, miscompiles and
`aie_api` traps).

## 1. Workflow

1. **Find where the time goes on the NPU** (`measuring.md`). Kernel II is irrelevant until you know the operator is
   compute-bound.
   - Ablate: build variants with one piece compiled out (a loop's compute, a phase, or the whole kernel, which gives
     the DMA floor) inside the real design (library program, overlay), not standalone. Full minus ablated is that
     piece's cost in context. Ablate through a parameter or a replaced method of the operator's class: patching a
     module global (a slot's `formats.SOURCES` entry, a template) keeps the render manifest's key, so the build reruns
     unpatched (`render_cache.py`).
   - Count per core: put `event0()` / `event1()` around one region, then read the cycles in the region, the window
     count, lock stalls, memory stalls and the memory module's bank conflicts. Comparing cycles per window with the
     static schedule reveals memory stalls; lock stalls show time spent waiting for data.
   - Classify:
     - DMA-bound (an empty kernel takes about as long as the full one, and lock stalls dominate): `data-movement.md`.
     - Compute-bound: steps 2-4.
     - Slower than its schedule: memory stalls, section 5.
     - Pieces that add up (full ≈ floor + each piece's ablation): the stages do not overlap, so shrinking one saves only
       that stage's share (`measuring.md` "Ablations").
   - In a model's program, rank by share before optimizing anything: per operator kind (drop its items from a library
     program) and per design switch (`data-movement.md` "Overlays, libraries and fused ELFs").
     `python -m tests.schedule_ledger rank APP` (no NPU) ranks a prefill chunk's kinds by the static cycles of their
     busiest cores, and its loops by weight and by slack (II above the resource bound). A loop already at its bound
     only gets faster with fewer operations; a loop with no weight does not move the model. Attention cores show upper
     bounds there (both paths of their branches are counted); the decode overlays have no static weights
     (`measuring.md`, calls).
2. **Bound the hot loop** (`machine-model.md`): per-slot and per-port op counts give ResMII, and dependence chains give
   RecMII. Count live values against the registers (5 `dm` accumulators, 12 `x` / `ex`, 8 `p`) and write the bound
   down. If the achieved II is already near it, change the algorithm (sections 3 and 4), not the syntax.
3. **Iterate on the compiler's output** (`compiler-feedback.md`, the **measure-ii** skill):
   - Use `ARTLITE_MEASURE_II=1` for an operator and `artlite.measure_ii.measure(path)` for a standalone `.cc`.
   - For variants of an operator's functions, use `python -m artlite.variant_search` (loop pragmas, text
     substitutions, replaced functions, operator parameters) instead of a throwaway script. It prints one table per
     loop with II, cycles, spills, stack, ops per slot and shared port, the post-pipeliner's failure reasons (`--why`)
     and NPU time (`--npu`).
   - For a loop far above its slot bound, don't try fixes one at a time. Generate the cross product in one
     `variant_search --variants FILE.py` run (dozens of variants compile in parallel): statement orders (e.g.
     interleaving accumulator chains), operand templates, the `operators.h` loop hints (`AIE_LOOP_REG_REWRITE_*`,
     `*_REALLOC`), outer-loop unrolls, pragmas, and `__restrict` on each pointer (Peano is finicky about it). Rank by
     cycles per call. One such sweep took the Q3_K decode loops from II 27 to 14 after single attempts had failed. Of
     the hints, `AIE_LOOP_REG_REWRITE_BASIC` helps when applied to individual loops, never file-wide (`loop-hints.md`).
4. **Verify on hardware**, in this order:
   1. bit-exact against the previous kernel when the arithmetic order is unchanged (otherwise the tolerance test plus
      a model-level metric such as KL);
   2. the region's cycles per window;
   3. end to end (`measuring.md` "Timing hygiene"): one process per variant, processes alternated (A B A B ...),
      medians and p10 over several reps on an idle CPU. Compare host-side variants (transactions) A / B within one
      process.

**Clock: 1.81 GHz** in turbo, under any load: seconds = cycles / 1.81e9. A window's counted cycles are its schedule
plus stall cycles (memory, lock, stream), and the stall counters account for the difference exactly. When end-to-end
time exceeds busy cycles / 1.81 GHz, look for the stalls, the waits outside the windows or the DMA floor; never fit a
multiplier. In the driver's default power mode a stalled core's counters stop, so count in turbo (`measuring.md`
"Which counters").

## 2. Rules for hot loops

Violating any of these usually costs more than any micro-tuning gains.

- **Compile-time shapes**: render trip counts, strides and offsets as constants (Jinja). A runtime trip count is fine
  on an outer call loop, but not on the loop you want pipelined ("The loop structure is not supported"). Pad to a fixed
  trip count and mask the tail. A `const int` initialized from a runtime argument is still a runtime value. So is a
  constant argument of a `noinline` helper when other callers pass other values: a design's single `kernel.cc` holds
  every caller (in a decode overlay, every tile format's pages), whereas a harness with one caller propagates its
  constants and runs faster. Make such constants template arguments (`_k23sc8<NJ, CS, SS>`: 156 -> 76 cycles per call
  in the qwen35 8-slot design, making the UD-IQ1_M decode step 11% faster).
- **One function per hot loop, with its buffers as distinct `__restrict` parameters.** Pointers derived from one
  parameter or workspace (`ws + a`, `ws + b`, `ps = buf; po = buf;`) are not noalias even as `__restrict` locals: the
  pipeliner sees a loop-carried store -> load dependence (RecMII = the whole chain). GdnOut's loop over its tables and
  `y` in one workspace had RecMII equal to its 48-bundle body; as an `always_inline` helper with distinct parameters it
  reached II 14 (7713 -> 4565 cycles per call). An in-place pass got II 23 vs. 5. An in-place pass may take the same
  buffer twice if every element is read before it is written.
- **DM-bank qualifiers** (`__aie_dm_resource_a/b/c/d`) go on every load pointer and must survive casts. They model bank
  conflicts and pick the load slot (`machine-model.md`). A qualifier is an address space, so an inline helper with a
  plain `T *` parameter casts it away and the loads become generic. Give helpers the qualified type or a deduced
  pointee type (`template <typename T> f(const T *p)`, as in `al::copy`), or hand the plain pointer to a block stream
  templated on the resource.
- **Width**: use 512-bit vectors or wider, and stores of at least 32 bits (sub-word stores are read-modify-write).
- **Nothing that splits the body**: no division or modulo (`__divsi3`, which signed power-of-two divides also call; use
  unsigned operands, shifts or `al::div_small`), no branches, no calls, and no scalar float math (soft-float calls).
  Use selects and loop fission instead. Keep fp32 values in vectors and extract a lane at the end.
- **Compile-time lane indices**: `extract<8>(j)`, `insert(j, ...)` or a shuffle with a runtime `j` compiles to a switch
  over all cases, and the loop stops pipelining. Unroll such loops fully (`AIE_LOOP_UNROLL_FULL`) or render the index.
  A Q extraction loop went from 2.83 to 1.29 ms with the same arithmetic, and a V-record builder's rolled
  `insert(r, ..)` went from ~26K to 478 cycles per 8-token group (II 14).
- **Walk pointers**: advance each stream pointer by a constant. Recomputing `base + i * stride` costs `mv` / scalar
  slots, and decoding a flat index costs `alu` slots (`restructurings.md` R8). Use separate load and store pointers,
  since a pointer used at both ends of an iteration stays live for the whole iteration. Keep at most ~6 live pointers
  (there are 8 `p` registers, including `sp`).
- **Register budget**: keep at most 4 accumulators live across a loop; the fifth `dm` holds transients for conversions
  and MMUL results. An `aie::accum` constant (e.g. a MAC's init value) also takes a `dm`, so prefer a `vector`
  constant with `aie::mac(acc, v, c)` where you can. A 64-lane bf16 vector takes two of the 12 `x` registers: an
  elementwise loop with four replicated coefficient vectors spilled at 64 lanes per step (II 26-30 per 64) and runs at
  II 8 per 32 lanes instead (sd15's GroupNorm rows).
- **Short conversion chains**: convert directly (`aie::to_float`, `accum<accfloat>(v).to_vector<bfloat16>()`), fold
  power-of-two scales into the SRS shift (`to_vector<T>(shift)`), and reduce in the native type before converting the
  scalar result.
- **Tables as worker `Buffer`s**: pass a LUT's coefficients as `Buffer(initial_value=...)` (`km.act(table_arg=True)`),
  and scratch space likewise (`docs/development.md`'s NPU pitfalls: no kernel statics). A static array also collides
  with L1 buffers pinned at fixed addresses (`pin_l1_buffers`). A gather table needs two copies in two 16 KB regions
  (`bank-conflicts.md`).
- **Every name at namespace scope is `{{ p }}_...`** (`p = kernel_prefix(stem)`, AGENTS.md "Rules"), helpers included.
  The build names each definition by its code, so the operators of a library or fused ELF (one translation unit) share
  identical helpers and never collide. A helper renderer that several operators use renders under its own stem
  (`kmath`: `km`, `gemm.mac.MAC`), and a helper that differs only in constants can take them as template parameters
  (`mac<M, N>`), giving one definition for all callers. Keep templates and helpers out of `extern "C"` blocks.
- **`noinline` boundaries** cost a call, a return and argument moves (~10-20 cycles). Use one when a loop pipelines
  standalone but not inside a big function (II 64 / 111 inlined vs. 11 / 16), for a helper called from several places
  (one copy in program memory), when the inlined frame would blow the stack (an exact fp32 GELU inlined takes 7.3 KB),
  or to get a fresh noalias scope. Inlining a chain of big helpers back into the caller can be slower, as in the decode
  GEMV (`rejected.md`).

## 3. Restructurings that worked

Pick one based on the symptom; `restructurings.md` has each case study with its numbers.

- **R1 Accumulators in registers across the reduction**: a loop that loads, updates and stores an L1 output every
  group, or spills, runs 20-30% above its schedule with bank conflicts. Reorder it so that fewer outputs are live and
  they stay in accumulators for the whole reduction, even if an operand has to be rebuilt twice (II 56 -> 21, memory
  stall 7.25 -> 0.24 ms).
- **R2 Pop the reused operand once and run two independent MMUL chains**: one accumulator chain issues at best every 3
  cycles, so alternate two or more accumulators over blocks (P·V 2155 -> 1344 cycles per object; the gemm MAC loop's 4
  staggered chains reach II 32 = ResMII).
- **R3 Long elementwise chains** (fp32 is emulated and latency-bound, so these never pipeline): use a gather LUT or a
  short bf16 polynomial, identities without cancellation, fp32 only where it matters, both branches followed by one
  select, `x` passed twice, and loop fission (GELU 622 -> 223 cycles per 32 lanes; recipes in `numerics.md`).
- **R4 Arithmetic on the MMUL**: exact-integer weight panels with the scale applied once per group, identity panels to
  read values, panels of ones for row sums, the exact hi / lo split against `-I`, and affine dequantization as diagonal
  panels.
- **R5 Dequantizing on the vector unit**: put quantization groups on lanes, apply offsets through the bf16 exponent
  (`0x43` zips), assemble split codes in the integer domain before a single zip, read back one upshift per shift,
  budget by the `mv` slot, and count the GEMV's own DMA as a source of stalls.
- **R6 Per-lane scales of small sub-blocks**: use `broadcast_elem_128` + two 8x8 transposes instead of zips
  (II 44 -> 29), keep scales in the MMUL's lane order, and split a spilling dequantization by fission (750 -> 590
  cycles per 64 rows).
- **R7 Port-bound loops** (shuffles and panel pushes share `L_RM_PORT`): reduce rows of 8 in the transposed layout
  (`al::max8` / `sum8`: 14 -> 8 mv ops), use broadcast + transpose expansions, build
  128-bit repeats with `grow_replicate` (not `concat` chains, which cost a `vshift` each), assemble vectors from
  128-bit pieces with `al::concat` / `al::gather` (not `aie::concat` / `insert`), and transpose bytes before the
  arithmetic (II 46 -> 40; a Q4_K header 2246 -> 920 cycles).
- **R8 Pointer walks instead of index math** (`add_3d_ptr` walks: II 46 -> 30 with no `alu` ops), and small scalar
  gathers unrolled to constant offsets (4660 -> 281 cycles).
- **R9 Batch small calls and fill the lanes**: do scalar-like math once for all panels, run independent small
  reductions in one call, handle two heads per call, process two blocks per iteration of a RecMII-bound loop, and run
  a tile's epilogue on its own rows in one call.
- **R10 Sequential recurrences** (the GDN scan): process them chunkwise (WY / UT form) with 16 tokens per state read
  and write, run the state products and the update on the MMUL, and solve within a chunk in fp32 by 8-token blocks (a
  1B chunk's scan 70 -> 24 ms). A decode step's tokens work the same way: one pass per state panel at any S (~690
  cycles vs. ~6000 token by token at S 8); the grouping of a step's tokens then shows only at the rounding level.
- **R11 Several tokens per decode step**: a DMA-bound step of S tokens costs about as much as one (tokens on the
  MMUL's A rows, inputs interleaved per 8-element block, recurrent state replayed instead of snapshotted). Past 4
  tokens, the A rows hold single bfp16 `x`, Q and P instead of hi + lo (`decode.gemv`, `decode.attn256`'s "One
  pass"), and a core's per-slot arrays outgrow L1: put what only cold code touches into a neighbour's memory
  (`decode.design`'s "Designs of 8 slots": slots 4-7's residual goes to the helper core's memory, and the helper's
  scratch goes to the GDN core's).
- **R12 Neighbour cores sharing derived data** through shared memory (GemmQ's dequantization pairs: 980 -> ~800 cycles
  per k-tile, making the 9B prefill chunk 7% faster).
- **R13 Flash attention with several query heads per core**: use at most two panel streams per loop (the memtile
  interleaves the rest), and use every pop for as many MMULs as the accumulators allow (two row panels x two records
  reaches the MMUL bound). Run P V over pairs of row blocks with both of its streams on one load slot (the other slot
  takes `o`'s loads). Run the softmax in pipelined passes with no per-lane constants, with its range reduction in the
  score loop's epilogue. Keep `o` in two copies one region apart, and move rare paths out of line (88 -> 15.1 ms per
  layer; qwen35 1B 9.95 -> 4.95 ms; LFM3 11.4 -> 9.6 ms).
- **R14 Weight panels from gathers** (IQ3_XXS's grid, IQ3_S's grid, IQ4's value pairs): upshift indices into an
  accumulator on the load, load the masks for several panels with one vector load, keep FIFOs out of the gather loop,
  use a lossless re-encoding that removes decode steps, and move the MMUL into a separate loop (4596 -> ~2300 cycles
  per half tile). Order 16-bit entries with one `T16_16x4_lo` shuffle, apply a 9th index bit on the MMUL from
  precomputed bytes, count the stores against the `QEY_RS` port they share with the gathers, and build compacted rows
  from constant windows (enough records per iteration that every shift is a constant).

## 4. Pipelining playbook

- **Two pipeliners.** Pre-RA (MachinePipeliner) plans an II that register allocation may then break with spills and
  moves; post-RA schedules real registers. `AIE_PREPARE_FOR_POSTPIPELINING` skips pre-RA. Try with and without it, and
  in two statement orders (the same values gave II 11 vs. 13; one GEMM MAC loop got II 41 without it and 60 with it).
- **Post-RA register rules.** Every value must be consumed within II cycles of its definition; the pipeliner never
  rotates a value through two registers and re-allocates only iteration-local values. Therefore:
  - carry nothing in vector registers across iterations (no software prefetch into `x`);
  - avoid variable pointer offsets (selects, index math): they need `m` / `r` registers that re-allocation cannot add;
  - make load / store independence provable (distinct `__restrict` parameters, section 2).
- **II above ResMII without spills?** Look for a shared port first: the post-pipeliner's ResMII counts slots only, so
  "out of resources" at every II up to the final one means a port is the bottleneck (`variant_search`'s `port` column,
  R7). Then look for a loop-carried value, a pointer used at both ends of the iteration, or an unprovable memory
  dependence. Only then blame the register file (`machine-model.md` "Worked example": II 42 -> 32).
- **Pop constant bfp16 operands instead of assembling them.** A panel built in the loop from a mantissa load and a
  constant exponent (`b.mantissa = load; b.exponent = e8(0x85)`) pins its `ex` register pair, and the next
  iteration's loads into the same `x` register wait for the MMUL that reads it (RecMII 33). The same panels stored as
  bfp16 blocks and popped from a `BIn` stream reach II 18, the floor set by the panel pushes (`decode/design.py`
  `_ohp`).
- **Load loop-invariant constants that spill.** Masks held in registers across a loop compete with the iteration's
  values: the compiler parks them in accumulator registers and moves them back every iteration (`vmov x, bmlh`).
  Stored as a small L1 table (two copies, walked with `add_2d_ptr` so that the loads cannot be hoisted), they become
  iteration-local loads on idle load slots: the Q3_K decode MMUL loop went from II 20 with 3 spills to II 13 with none
  (`formats/q3k.py` `_q3k_mm`).
- **At most two bfp16 panel streams per loop** (there are 2 `lf` registers). A third `BIn` stream's load state is
  spilled and reloaded every iteration: three streams (Q, K hi, K lo) run at II 12 per d-block, whereas two loops of
  two streams each run at II 4. With K stored as [d-block][hi | lo] (one stream), a single loop pops each Q panel once
  for both halves: II 4 for 4 MMULs (R13). All of a stream's pops sit on one load slot, each with a `vlda.fill.512`
  that unrolling does not amortize (`machine-model.md`). That gives one panel per 2 cycles, so a loop whose MMULs
  each pop an operand is bound at 2 cycles per MMUL; reuse every popped panel for two or more MMULs (R13). Output
  streams behave the same way: the push / flush chain of one `BOut` bounds a loop that pushes two panels per
  iteration at II ~16-17 (slot bound 7), and two `BOut`s spilled their state (II 22).
- **Hold panels across a loop in 1-D arrays**: `BV P[2][4]` popped before a P V loop spilled a panel every iteration
  (II 13), whereas the same 8 panels held as four `BV x[2]` arrays reached II 8, the MMUL bound.
- **Chains**: a single long dependence chain (dequant -> scale -> MAC) has an II about equal to its latency. Unroll it
  2x into independent chains and pipeline post-RA (a fused dequant GEMV went from 14 cycles per group to 13 per two
  groups). 4x bodies exceed what the pipeliner handles (> ~70 bundles).
- **Output fusion**: compute several outputs that share a loaded operand in one loop (4 GEMV rows: II 4 vs. 3 per row)
  until accumulator pressure causes spills (8 rows did).
- **`AIE_LOOP_IGNORE_MEM_DEPS` also drops the ordering of spills and reloads**: a spilling loop computed garbage with
  it. Use distinct `__restrict` parameters instead.
- **`AIE_TRY_INITIATION_INTERVAL(N)`** runs an exact search at N. It finds schedules the heuristics miss (`gemm_q`
  header copies II 9 -> 6, gather loops 40 -> 29 and 28 -> 21: `loop-hints.md`) and is useless below the bound.
- **Short trips**: a pipelined loop peels NS - 1 iterations into its prologue and epilogue, so for a few iterations
  the ramp can cost as much as the steady state.
- **Global flags** (`-aie-postpipeliner-maxii` etc.) apply to every loop in the kernel file and have miscompiled a
  spilling loop. Tune per loop instead (`loop-hints.md`).

## 5. Memory stalls and L1 placement

- **Signature**: cycles per window above the static schedule, MEMORY_STALL (event 23) inside the window, and
  `CONFLICT_DM_BANK_*` (events 77-84) high on one pair.
- **Fixes, in this order**:
  1. keep values in registers instead of L1 (R1);
  2. put the hot loop's streams in different 16 KB regions (`pin_l1_buffers`, `Buffer(mem_bank=)`), including a lookup
     table's two copies;
  3. place DMA-written buffers next to buffers that only a cold loop reads.
- `reference/bank-conflicts.md` covers the region model, the gather rule, finding the conflicting region, the placement
  tools and the case studies (C tiles, objects read in two parts, annotations as hints, more streams than regions).

## 6. Data-movement levers (rates and details: `data-movement.md`)

- **Re-reads cost DDR bandwidth** (the NPU path has no system cache): broadcast what every column needs, process more
  tokens per weight pass (GemmQ 32 -> 64 tokens per core row made GEMMs 15-26% faster; several tokens per decode step,
  R11), and run more queries per pass over a KV cache (Llama 3B prefill attention 32 -> 64: 25% faster).
- **When two cores need the same stream, split it rather than broadcast it** once a core's stream or compute becomes
  the bottleneck (decode attention at long context): each core takes half the positions, and the two merge partial
  softmax results. Once a core can no longer hold all the rows in one pass, broadcasting wins instead: both cores read
  one broadcast stream, each taking half the rows (R13's "One pass"). Four cores can combine both approaches: two
  halves, each splitting the positions, on two streams (R13's "chain's halves").
- **A broadcast is bounded by each consumer's input port** (8 B / cycle): feed each core on both of its input channels
  (two streams, each carrying half of every object), which made Gemma's full attention 41% faster.
- **Read every byte once**: fixed-size gather units waste bytes when a part is smaller (a 1 KB tile header in a 4 KB
  unit reads 1.33x the bytes), so group the small parts (`GemmQOperator.hg`). In long-K GEMMs, K parts keep the A
  strip resident in the memtile (`plan_phase(k_part=)`).
- **Queue depth, not BD count, limits a shim channel**: at most 4 tasks can be queued. Rolling retirement
  (`Sequence`) keeps the queue full. Issue the fills the cores need to make progress before the bulk of a phase's
  weight fills, or the queue deadlocks.
- **Deep memtile FIFOs run out of BD ids** (24 per channel parity). Deeper FIFOs do not help a DMA-bound operator; they
  help only a weight stream whose consumer pauses (an MoE step: `data-movement.md`).
- **Design switches**: every change of design in a library program costs a reset and a reconfiguration (~80-90 µs).
  Count them per layer, and where it fits, prefer one design whose workers run every phase.

## 7. Program memory

Each core has 16 KB for the startup code (144 B), the Worker's control code (`core_<col>_<row>`, which aiecc generates
from the IRON program: ~2-3 KB in an overlay), and every kernel function its entry kernels reach. An overflow shows up
as "section '.text' will not fit in region 'program'" at link time, or `_XAie_LoadProgMemSection: Overflow of program
memory` at load time. Count a core's bytes with `llvm-nm -S --size-sort` on its `design.mlir.prj/elfs_*/*.elf`
(`llvm-size -A` for its `.text`), or, before linking, walk the kernel object's call graph from the core's entry
kernels (calls are relocations to function symbols in `llvm-objdump -dr`; sizes come from `llvm-nm -S`). Rank variants
by the core's bytes and cycles in one `variant_search` run.

Levers that worked (1.1 KB saved on the decode overlay's main core):
- **Worker loops**: use IRON's `range_` (aiecc narrows its counter to i32 when the bound is a constant or read from a
  buffer; an index counter costs two registers and a carry per loop), `design.range32` for a body that computes with
  the i32 induction variable, and `forever_()` with `Worker(while_true=False)` for a core's infinite loop
  (`scf.while(true)` with no counter saves 48-192 B per core, and 816 B on a small core whose counted loop LLVM had
  unrolled). Together, i32 loops and `while_true=False` took 624 B off the decode main core. opt fully unrolls a
  constant-trip worker loop to drop the rotation of the FIFO objects' pointers (+800 B for the 32 panels of the Qwen
  decode GDN core); `range32(n, unroll=False)` keeps such a loop rolled. A global cap on unrolling cost the Gemma
  prefill 22% instead, because other cores' unrolled loops are hot. The cores' llc runs with
  `--enable-aie-zol-without-minitercount=false` (`operators/toolchain.py` `CORE_LLC_FLAGS`, aiecc `-Xllc`); otherwise
  loops without a known minimum trip count (FIFO skip loops) would become zero-overhead loops padded with NOP bundles
  (~180 B each). aiecc's `-Xopt=<arg>` reaches the cores' opt (`-opt-bisect-limit=<n>`, `-opt-disable=<passes>`) to
  find which pass grows a core.
- **Code per core**: render a shared function per core with only that core's cases; use one kernel entry per operation
  and fold small end-of-phase work into the last call of an existing kernel (every kernel call also adds Worker
  control code). A Python helper called twice in the IRON program inlines its loops twice. Split the work between
  neighbour cores by kind (one core builds the input panels, the other computes and keeps the outputs).
- **Smaller functions**: move shared helpers out of line (`noinline`). Apply `minsize` to cold dispatchers, rolled
  (`AIE_LOOP_NO_UNROLL`) call loops and copies, deciding per function by compiling the kernels alone without linking.
  On a GDN core's constants and transposes it saved 130 B each; on its gate helpers and the scan's copy kernels it saved
  nothing or grew them, and an unrolled solve grew by 32 B, so split such code across ops or cores instead. `minsize`
  turns loop rotation off: a `for` loop stays two blocks (its test at the top, with the delay slots of its `jnz` and of
  the jump into it left empty) and gets no ZOL, costing ~7 more cycles per iteration. When a loop's ZOL is too big
  (post-pipelined, or a short body padded to the ZOL's 112 B from setup to loop end), use
  `AIE_LOOP_HINT(enable-aie-hardware-loops, 0)` without `minsize` instead (`gout`, `norm_rope`, `gmax`). A minimum trip
  count no longer keeps a loop out of hardware loops, but `AIE_LOOP_MIN_ITERATION_COUNT(1)` on a runtime count still
  drops its zero-trip guard (32 B in `gmax`). The result has no hardware loop, a single block, and `jnz` delay slots
  filled from the body, so it is both smaller and faster than `minsize`. Give a constant loop of 2-4 trips that Peano
  left rolled (a pipelined ZOL: `loop-hints.md`) `AIE_LOOP_UNROLL_FULL`; this saved 160 B and 21 cycles on the MoE
  `collect`'s partial sums. Use vector compares against a lane-index table instead of 64-bit mask arithmetic. Use
  runtime parameters instead of template instances, since each instance is a copy: one dequantization routine with a
  runtime sub-block shift (`shuffle_down` of a short vector by a runtime amount, then `filter_even`) replaces one
  instance per sub-block. A table activation (`km.act`) is both smaller and faster than a polynomial chain.
- Avoid runtime-library arithmetic (`checks.py` warns when a kernel calls it).
- **Pages: code the cores load themselves** (`operators/paging.py`, whose docstring lists the hazards and their guards;
  used by the vision overlay). A core keeps resident code (aiecc's link of the core) plus pages holding what only a
  page's entries reach. Each page is linked at the slot from a copy of `kernel.o` with the page's sections renamed; the
  slot starts at the resident end (`_main_init`, the crt's startup code, is linked last). On every core, `paging.link`
  checks that the resident code and data match aiecc's, that each page fits within 16 KB, that pages hold no data, that
  no page reaches the pager, and that the stack covers the entries. Every core's kernels must lie at the same
  addresses behind its control code (`core_<col>_<row>`, which is linked first), so a per-core constant that one core
  can fold (column 0's index 0) changes that core's control-code size (decode overlays pass COLS + the column instead).
  The core triggers a page and sleeps on its wake lock in one resident function, and enters page code through the
  pager's state words. With self-DMA (the vision overlay places the page in two B slots of its input stream, and its
  MM2S 1 writes into its own TileControl), loading runs at ~2.4 B / cycle of slots, i.e. 2.7 µs for two 5760-B slots
  regardless of the code; the host knows each row's role and places the page before the op that needs it. With a
  memtile (the core's request packet pushes a page BD on a dedicated memtile MM2S channel), loading takes 86 cycles +
  ~0.39 µs per KB of code, with at most 4 cores per channel. The memtile's store and BDs must be reloaded after another
  hardware context has run and after a pause in which the NPU may have powered down (`OverlayDevice.set_prologue`:
  0.3 ms). Anything else a design keeps on the array across dispatches needs the same treatment: Linux suspends the
  idle device after 5 s, and a resume restores only its configuration. Decode overlays page every tile format except
  the one with the most bytes when four or more formats do not fit (`DecodeOverlayOperator(paged=)`: Llama 3B Q3_K_M,
  a few loads per step, 2.4 KB freed on the main core). Each page also takes one memtile BD per core half, next to the
  A FIFO's BDs in the same half. When a slot runs short of BDs or room, formats can share a page (`PAGE_GROUPS`:
  Inkling's BFP16 + Q8_0, with its A FIFO 2 objects deeper and no loads between them), and compact kernels can fill a
  short slot (`gemv.render(compact=)`: for each format, the unpipelined loops with the fewest cycles per byte; Q5_K
  takes 608 B for ~20% of its half tile, still under the tile's DMA time). Put a rarely used epilogue in the page of the
  role that runs before it (sd15's GEGLU in the LayerNorm page took 1.7 KB off the resident code). The vision overlay's
  pages hold 7232 B (16 KB minus its 9152-B resident code; read sizes with
  `llvm-readelf -S design.mlir.prj/page_0_2_<role>.elf`, `.ptxt`). A feature for a full role goes into a separate role
  (a template instance of the role's code with the feature on and its unused paths off, as in role g), not into a
  helper the roles share, which grows every page that reaches it (+192 B in four pages). Per-element math that a
  consumer needs (LFM3's attention gate, 1 / sigmoid) goes into the epilogue of its producer GEMM (a page with room),
  leaving the consumer a multiply. A page's constant tables must be resident, kept alive by a read in resident code
  that only the paging cores run (`gemv`'s `_keep`): `retain` would keep the section in the link of every core of the
  design, and an 8-slot qwen35 design's attention core had no 1.6 KB to spare.
- After a toolchain update, compare each core's `core_<col>_<row>` size with the previous build's.

Rejected size levers: `rejected.md` "Program memory".

## 8. Review checklist

1. Bottleneck identified on the NPU (ablation, window counters) before kernel work.
2. Bound computed; achieved II compared against it; cycles per window compared against the schedule.
3. Constant trip counts; a unique kernel entry name encoding all template parameters; every namespace-scope helper
   prefixed too.
4. Each hot loop in a separate function, with its buffers as distinct `__restrict` parameters; bank qualifiers on every
   load pointer.
5. No spills in hot loops; stack within the Worker's `stack_size`; no switches, calls or branches in loop bodies; no
   index math that a pointer walk can replace.
6. ≤ 4 live accumulators; no runtime lane indices; ≤ 2 panel streams per loop; stores ≥ 32 bits; 512-bit vectors;
   aligned buffers.
7. Pre-RA vs. post-RA compared; variants generated and compiled in parallel.
8. Data arrives in pop order; DMA-written buffers are not in the hot loop's banks.
9. Output bit-exact against the previous kernel (same arithmetic order), or tolerance + KL.
10. No kernel statics (tables included: use worker `Buffer`s); rounding set in every entry point that rounds; no
    runtime-library arithmetic; each core within its program memory.
