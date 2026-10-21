---
name: aie-kernel-perf-optimization
description: Make AIE2P (NPU2) kernels and designs fast in Artlite (Peano / llvm-aie). Use before optimizing an operator, when writing a hot loop, when a loop's II is above its bound, spills or does not pipeline, when a kernel runs slower on the NPU than its schedule (memory stalls, bank conflicts), or when a core's program memory overflows. Covers finding where the time goes on the NPU (ablations, per-core event windows and stall counters), bounding a loop, the measure_ii / variant_search loop, the source rules that decide pipelining, proven restructurings (accumulators in registers, independent MMUL chains, compile-time lanes, loop fission, LUT math, arithmetic on the MMUL, neighbour cores sharing work), L1 bank placement, data-movement levers, program memory and rejected ideas.
---

# AIE2P kernel performance (Peano)

Reference files (read the one the step points to):

| file | read when |
|---|---|
| [reference/measuring.md](reference/measuring.md) | timing a region on the NPU: hygiene, ablations, event windows, core / memory-module counters (also in host-built transactions), reading L1 |
| [reference/machine-model.md](reference/machine-model.md) | computing a loop's bound: slots, shared ports, registers, latencies, load-slot binding, a worked example |
| [reference/compiler-feedback.md](reference/compiler-feedback.md) | reading remarks and assembly; symptom -> fix table |
| [reference/restructurings.md](reference/restructurings.md) | applying a section-3 restructuring: the case studies and their numbers |
| [reference/loop-hints.md](reference/loop-hints.md) | choosing `AIE_*` pragmas, per-loop hints, global `-mllvm` flags |
| [reference/numerics.md](reference/numerics.md) | fp32 via bf16 splits, exp2 / SiLU / GELU recipes, bfp16 precision, rounding, accuracy studies |
| [reference/data-movement.md](reference/data-movement.md) | a DMA-bound operator: port and DDR rates, layouts, broadcasts, BD limits, task queues, overlays, libraries |
| [reference/rejected.md](reference/rejected.md) | before trying an idea: what was measured and did not help |

Related skills: **measure-ii** (measure_ii and variant_search reports), **profiling** (counters of compiled
operators), **aie2p-coding** (architecture facts, DMA limits; section 9: compiler crashes, miscompiles and aie_api
traps).

## 1. Workflow

1. **Find where the time goes, on the NPU** (measuring.md). Kernel II is irrelevant until you know the operator is
   compute-bound.
   - Ablate: build variants with one piece compiled out (a loop's compute, a phase, the whole kernel = the DMA
     floor), inside the real design (library program, overlay), not standalone. Full minus ablated is that piece's
     cost in context.
   - Count per core: `event0()` / `event1()` around one region, then read cycles in the region, window count, lock
     stall, memory stall and the memory module's bank conflicts. Cycles per window vs the static schedule shows
     memory stalls; lock stall shows waiting for data.
   - Classify:
     - DMA-bound (empty kernel ≈ full, lock stall dominates): data-movement.md.
     - Compute-bound: steps 2-4.
     - Slower than its schedule: memory stalls, section 5.
     - Pieces that add up (full ≈ floor + each piece's ablation): the stages do not overlap, and shrinking one buys
       only its share (measuring.md "Ablations").
   - In a model's program, rank by share before optimizing anything: per operator kind (its items dropped from a
     library program) and per design switch (data-movement.md "Overlays, libraries and fused ELFs").
2. **Bound the hot loop** (machine-model.md): per-slot and per-port op counts give ResMII, dependence chains give
   RecMII; count live values against the registers (5 `dm` accumulators, 12 `x` / `ex`, 8 `p`). Write the bound down.
   If the achieved II is already near it, change the algorithm (sections 3 and 4), not the syntax.
3. **Iterate on the compiler's output** (compiler-feedback.md, the **measure-ii** skill):
   - `ARTLITE_MEASURE_II=1` for an operator; `artlite.measure_ii.measure(path)` for a standalone `.cc`.
   - Variants of an operator's functions: `python -m artlite.variant_search` (loop pragmas, text substitutions,
     replaced functions, operator parameters), not a throwaway script: one table per loop with II, cycles, spills,
     stack, ops per slot and shared port, the post-pipeliner's failure reasons (`--why`) and NPU time (`--npu`).
   - A loop far above its slot bound: don't try fixes one at a time. Generate the cross product in one
     `variant_search --variants FILE.py` run (dozens compile in parallel): statement orders (e.g. interleaving
     accumulator chains), operand templates, the `operators.h` loop hints (`AIE_LOOP_REG_REWRITE_*`, `*_REALLOC`),
     outer-loop unrolls, pragmas, `__restrict` on each pointer (Peano is finicky about it). Rank by cycles per call.
     One such sweep took the Q3_K decode loops from II 27 to 14, where single attempts had not. Of the hints,
     `AIE_LOOP_REG_REWRITE_BASIC` pays off on single loops, never file-wide (loop-hints.md).
4. **Verify on hardware**, in this order:
   1. bit-exact against the previous kernel when the arithmetic order is unchanged (otherwise the tolerance test plus
      a model-level metric such as KL);
   2. the region's cycles per window;
   3. end to end (measuring.md "Timing hygiene"): one process per variant, processes alternated (A B A B ...),
      medians and p10 over several reps on an idle CPU; host-side variants (transactions) A / B within one process.

**Clock: 1.81 GHz** in turbo, under any load: seconds = cycles / 1.81e9. A window's counted cycles are its schedule
plus stall cycles (memory, lock, stream), and the stall counters account for the difference exactly. When e2e
exceeds busy / 1.81 GHz, find the stalls, the waits outside the windows or the DMA floor; never fit a multiplier.
In the driver's default power mode a stalled core's counters stop: count in turbo (measuring.md "Which counters").

## 2. Rules for hot loops

Violating any of these usually costs more than any micro-tuning gains.

- **Compile-time shapes**: trip counts, strides and offsets rendered as constants (Jinja). A runtime trip count is
  fine on an outer call loop, not on the loop you want pipelined ("The loop structure is not supported"). Pad to a
  fixed trip count and mask the tail. A `const int` initialized from a runtime argument is still a runtime value.
- **One function per hot loop, its buffers distinct `__restrict` parameters.** Pointers derived from one parameter or
  workspace (`ws + a`, `ws + b`, `ps = buf; po = buf;`) are not noalias even as `__restrict` locals: the pipeliner sees
  a loop-carried store -> load dependence (RecMII = the whole chain). GdnOut's loop over its tables and y in one
  workspace: RecMII = its 48-bundle body; as an `always_inline` helper with distinct parameters II 14 (7713 -> 4565
  cycles per call); an in-place pass: II 23 vs 5. An in-place pass may take the same buffer twice if every element is
  read before it is written.
- **DM-bank qualifiers** (`__aie_dm_resource_a/b/c/d`) on every load pointer, kept through casts. They model bank
  conflicts and pick the load slot (machine-model.md). A qualifier is an address space: an inline helper with a plain
  `T *` parameter casts it away (the loads become generic). Give helpers the qualified type or a deduced pointee type
  (`template <typename T> f(const T *p)`: `al::copy`), or hand the plain pointer to a block stream templated on the
  resource.
- **Width**: 512-bit vectors or wider; stores of at least 32 bits (sub-word stores are read-modify-write).
- **Nothing that splits the body**: no division / modulo (`__divsi3`; also signed power-of-two divides: use unsigned
  operands, shifts, `al::div_small`), no branches, no calls, no scalar float math (soft-float calls). Use selects and
  loop fission. Keep fp32 values in vectors and extract a lane at the end.
- **Compile-time lane indices**: `extract<8>(j)`, `insert(j, ...)` or a shuffle with a runtime `j` compiles to a switch
  over all cases, and the loop stops pipelining. Unroll such loops fully (`AIE_LOOP_UNROLL_FULL`) or render the index:
  a Q extraction loop 2.83 -> 1.29 ms with the same arithmetic; a V-record builder's rolled `insert(r, ..)` ~26K ->
  478 cycles per 8-token group (II 14).
- **Walk pointers**: advance each stream pointer by a constant (a recomputed `base + i * stride` costs mv / scalar
  slots; decoding a flat index costs alu: restructurings.md R8); separate load and store pointers (one pointer used at
  both ends of an iteration lives the whole iteration); ≤ ~6 live pointers (8 `p` registers including `sp`).
- **Register budget**: at most 4 accumulators live across a loop; the fifth `dm` is the transient for conversions and
  MMUL results. An `aie::accum` constant (e.g. a MAC's init value) takes a `dm` too; prefer a `vector` constant with
  `aie::mac(acc, v, c)` where you can. A 64-lane bf16 vector takes two of the 12 `x`: an elementwise loop with four
  replicated coefficient vectors spilled at 64 lanes a step (II 26-30 per 64) and runs at II 8 per 32 lanes
  (sd15's GroupNorm rows).
- **Short conversion chains**: convert directly (`aie::to_float`, `accum<accfloat>(v).to_vector<bfloat16>()`); fold
  power-of-two scales into the SRS shift (`to_vector<T>(shift)`); reduce in the native type and convert the scalar
  result.
- **Tables as worker `Buffer`s** (a LUT's coefficients: `Buffer(initial_value=...)`, `act_lut`
  `render(table_arg=True)`), scratch too (AGENTS.md: no kernel statics); a static array also collides with L1 buffers
  pinned at fixed addresses (`pin_l1_buffers`).
- **Every name at namespace scope is `{{ p }}_...`** (`p = kernel_prefix(stem)`, AGENTS.md "Rules"), helpers too:
  the build names each definition by its code, so the operators of a library or fused ELF (one translation unit)
  share equal helpers and never collide. A helper renderer several operators use renders under its own stem
  (`act_lut.LUT`, `gemm.mac.MAC`) and a helper that differs only in constants can take them as template parameters
  (`mac<M, N>`): one definition for all of them. Keep templates and helpers out of `extern "C"` blocks.
- **`noinline` boundaries** cost a call, a return and argument moves (~10-20 cycles). Use one when a loop pipelines
  standalone but not inside a big function (II 64 / 111 inlined vs 11 / 16); for a helper called from several places
  (one copy in program memory); when the inlined frame would blow the stack (an exact fp32 GELU inlined: 7.3 KB); for
  a fresh noalias scope.

## 3. Restructurings that worked

Pick by the symptom; restructurings.md has each case study with its numbers.

- **R1 Accumulators in registers across the reduction**: a loop that loads, updates and stores an L1 output every
  group, or spills, runs 20-30% above its schedule with bank conflicts. Reorder so fewer outputs are live and stay in
  accumulators for the whole reduction, even if an operand is rebuilt twice (II 56 -> 21, memory stall 7.25 -> 0.24 ms).
- **R2 The reused operand popped once, two independent MMUL chains**: one accumulator chain issues every 2 cycles at
  best; alternate two accumulators over blocks (P·V 2155 -> 1344 cycles per object; the gemm MAC loop's 4 staggered
  chains: II 32 = ResMII).
- **R3 Long elementwise chains** (fp32 emulated, latency-bound, never pipelined): a gather LUT or a short bf16
  polynomial, identities without cancellation, fp32 only where it matters, both branches then one select, x passed
  twice, loop fission (GELU 622 -> 223 cycles per 32; recipes: numerics.md).
- **R4 Arithmetic on the MMUL**: exact-integer weight panels with the scale applied once per group, identity panels to
  read values, panels of ones for row sums, the exact hi / lo split against -I, affine dequant as diagonal panels.
- **R5 Dequantizing on the vector unit**: quantization groups on lanes, offsets through the bf16 exponent (0x43 zips),
  split codes assembled in the integer domain before one zip, one upshift read back per shift, the mv slot as the
  budget, the GEMV's own DMA as a stall source.
- **R6 Per-lane scales of small sub-blocks**: `broadcast_elem_128` + two 8x8 transposes instead of zips (II 44 -> 29),
  scales kept in the MMUL's lane order, fission of a spilling dequant (750 -> 590 cycles per 64 rows).
- **R7 Port-bound loops** (shuffles and panel pushes share `L_RM_PORT`): broadcast + transpose expansions, 128-bit
  repeats by `grow_replicate` (not `concat` chains: a `vshift` each), transposes of bytes before the arithmetic (II
  46 -> 40; a Q4_K header 2246 -> 920 cycles).
- **R8 Pointer walks instead of index math** (`add_3d_ptr` walks: II 46 -> 30, alu 0) and small scalar gathers
  unrolled to constant offsets (4660 -> 281 cycles).
- **R9 Batched small calls, filled lanes**: scalar-ish math once for all panels, independent small reductions in one
  call, two heads per call, two blocks per iteration of a RecMII-bound loop.
- **R10 Sequential recurrences** (the GDN scan): chunkwise (WY / UT form), 16 tokens per state read and write, the
  state products and the update on the MMUL, the intra-chunk solve in fp32 by 8-token blocks (a 1B chunk's scan
  70 -> 24 ms).
- **R11 Several tokens per decode step**: a DMA-bound step of S tokens costs about one (tokens on the MMUL's A rows,
  inputs interleaved per 8-element block, recurrent state replayed instead of snapshotted).
- **R12 Neighbour cores sharing derived data** through shared memory (GemmQ's dequant pairs: per k-tile 980 -> ~800
  cycles, the 9B prefill chunk -7%).
- **R13 Flash attention with several query heads per core**: at most two panel streams per loop (the memtile
  interleaves the rest), every pop for as many MMULs as the accumulators allow (two row panels x two records: the
  MMUL bound), P V over pairs of row blocks, the softmax in pipelined passes with no per-lane constants, o in two
  copies, rare paths out of line (88 -> 15.1 ms per layer; qwen35 1B 9.95 -> 4.95).

## 4. Pipelining playbook

- **Two pipeliners.** Pre-RA (MachinePipeliner) plans an II that register allocation may then break with spills and
  moves; post-RA schedules real registers. `AIE_PREPARE_FOR_POSTPIPELINING` skips pre-RA. Try with and without it,
  and in two statement orders (the same values gave II 11 vs 13; one GEMM MAC loop II 41 without, 60 with).
- **Post-RA register rules.** Every value must be consumed within II cycles of its definition; the pipeliner never
  rotates a value through two registers and re-allocates only iteration-local values. So:
  - carry nothing in vector registers across iterations (no software prefetch into `x`);
  - avoid variable pointer offsets (selects, index math): they need `m` / `r` registers the re-allocation cannot add;
  - make load / store independence provable (distinct `__restrict` parameters, section 2).
- **II above ResMII without spills?** Look for a shared port (the post-pipeliner's ResMII counts slots only, and "out
  of resources" at every II up to the final one means a port binds: variant_search's `port` column, R7), a value
  carried across iterations, a pointer used at both ends of the iteration, an unproven memory dependence. Only then
  blame the register file (machine-model.md "Worked example": II 42 -> 32).
- **Constant bfp16 operands: pop them, do not assemble them.** A panel built in the loop from a mantissa load and a
  constant exponent (`b.mantissa = load; b.exponent = e8(0x85)`) pins its `ex` register pair, and the next
  iteration's loads into the same `x` register wait for the MMUL that reads it: RecMII 33. The same panels stored as
  bfp16 blocks and popped from a `BIn` stream: II 18, the panel pushes' floor (`overlay/decode.py` `_ohp`).
- **Loop-invariant constants that spill: load them.** Masks held in registers across a loop compete with the
  iteration's values (the compiler parks them in accumulator registers and moves them back: `vmov x, bmlh` per
  iteration). Stored as a small L1 table (two copies, walked with `add_2d_ptr` so the loads cannot be hoisted), they
  become iteration-local loads on idle load slots: the Q3_K decode MMUL loop II 20 (3 spills) -> 13, none
  (`formats/q3k.py` `_q3k_mm`).
- **At most two bfp16 panel streams per loop** (2 `lf` registers). A third `BIn` stream's load state is spilled and
  reloaded every iteration: three streams (Q, K hi, K lo) II 12 per d-block, as two loops of two streams II 4 each.
  A stream's pops all sit on one load slot. Output streams likewise: the push / flush chain of one `BOut` bounds a loop
  pushing two panels per iteration at II ~16-17 (slot bound 7); two `BOut`s spilled their state (II 22).
- **Chains**: a single long dependence chain (dequant -> scale -> MAC) has II ≈ its latency. Unroll it 2x into
  independent chains and pipeline post-RA (a fused dequant GEMV: 14 per group -> 13 per two groups). 4x bodies exceed
  the pipeliner (> ~70 bundles).
- **Output fusion**: compute several outputs that share a loaded operand in one loop (4 GEMV rows: II 4 vs 3 per row),
  until accumulator pressure spills (8 rows did).
- **`AIE_LOOP_IGNORE_MEM_DEPS` also drops spill / reload ordering**: a spilling loop computed garbage with it. Use
  distinct `__restrict` parameters instead.
- **`AIE_TRY_INITIATION_INTERVAL(N)`**: an exact search at N. It finds schedules the heuristics miss (gemm_q header
  copies II 9 -> 6) and is useless below the bound.
- **Short trips**: a pipelined loop peels NS - 1 iterations into its prologue and epilogue; for a few iterations the
  ramp can cost as much as the steady state.
- **Global flags** (`-aie-postpipeliner-maxii` etc.) apply to every loop of the kernel file and have miscompiled a
  spilling loop. Tune per loop (loop-hints.md).

## 5. Memory stalls and L1 placement

- **Signature**: cycles per window above the static schedule, MEMORY_STALL (event 23) inside the window,
  `CONFLICT_DM_BANK_*` (events 77-84) high on one pair.
- **Causes, most expensive first**: two streamed loads from one bank in a bundle; read-modify-write of an L1
  accumulator buffer, or spills / reloads (stack bank), next to streamed loads; DMA writes into a bank the loop streams
  from; occasional loads and stores against a stream.
- **Fixes, in this order**:
  1. registers instead of L1 (R1);
  2. the hot loop's streams in different 16 KB regions with `pin_l1_buffers` (`operators/l1_layout.py`);
  3. DMA-written buffers next to buffers that only a cold loop reads.
  Examples: a Q4_K GEMM, memory stalls halved, -8%; the GemmQ MAC 900 -> 783 cycles, GEMM -9%; planar GemmQd MAC cores
  (3B Q2_K down), where aiecc put a k-tile's A and D objects in one region (32% of the MAC window in memory stalls),
  pinned (`gemm_qd.PLANAR_PLANS`) 5.74 -> 5.15 ms.
- **Where aiecc put things**: `design.mlir.prj/checked_bank_placement.mlir` in the cache dir (when aiecc dumps
  intermediates); otherwise `llvm-nm -n` on the core's ELF lists the buffer symbols at the core's addresses, its own
  memory at 0x70000. `pin_l1_buffers` gives each core-tile ObjectFifo pool its own `aie.buffer`s with an `address` (it
  runs `aie-objectfifo-split` in-process); Worker `Buffer`s take an `address` attribute. Derive the addresses from the
  same buffer list as the L1 budget (`GemmQOperator._l1_layout`).
- The bank counters come in near-equal pairs: AIE2P's 8 banks are 32 B wide and pairs interleave at 32 B (AM027), so
  every 64-byte access uses both banks of its 16 KB region. Any two accesses to one region in a cycle conflict: place
  at 16 KB granularity.
- **The MAC loop's C tile** (`gemm/mac.py`: loaded and stored every 16x16 block) puts ~16% memory stall into a GemmQd
  MAC window when the C loads and stores hit one region (974 cycles per k-tile window vs 792 scheduled). A second C
  tile in another region (`pingpong`, alternating k-tiles) removes it: free where the operator has one anyway
  (gate/up's epilogue scratch), not worth a C object's double buffering where it does not (rejected.md). Or one C tile
  across two regions (its halves at 0x4000 -/+ half) walked so that consecutive blocks alternate halves (`pats` walks:
  band pairs (0, 2), (1, 3) block by block): sd15's fp32 C at m = 64, conflicts in the C regions 15.9K -> 0.4K
  cycles per 1280 x 1280 GEMM op, the op -8%. It needs a two-stage loop (NS 2): with NS 3 a block's stores meet the
  loads two blocks later.
- **An object read in two parts, each part paired with another stream** (an m-64 A object's two k-tiles, each against
  its own B slot): split it over two regions (two chained BDs: aie2p-coding section 8) so that part j sits in region
  2 + j, and keep the other stream's parity (the op's command in B slot 0, so that k-tile i is in B slot (i + 1) % 2).
  sd15's 16x16 GEMM: memory stall 7.6 -> 3.6 us, -12% per op; with the parity off by one every k-tile conflicts.
  Where one object feeds several calls (a conv's rows, in two regions), order the calls so that their regions
  alternate with the B slots (sd15 `Gemm.ORDER2`, rotated by one in odd chunks of 9 taps: memory stall -37%).
- **A bank annotation is also a scheduling hint**: the fp32-C MAC with its second 8-row panel's C loads annotated as
  bank b (the same addresses, still on slot A: machine-model.md) pipelines at II 41 instead of 43 (`render_mac(c2_bank=)`;
  the stores annotated too, or every load: not pipelined or II 43-44). Sweep annotations like pragmas.
- **When the streams outnumber the regions** (double-buffered A and D, a C tile, DMA writes of the next objects), list
  per window which buffers the core streams (hard: never two in one region) and which the DMA writes (soft: keep them
  out of the streamed regions), then pick the assignment with the fewest soft pairs that fits. Crossing the pairs (A0
  with D1, A1 with D0) keeps a window's two streams apart when a region cannot hold both D objects. Measure each
  candidate (pin by role and address in a throwaway patch of the design), one process each, alternating.

## 6. Data-movement levers (rates and details: data-movement.md)

- **Re-reads cost DDR bandwidth** (no system cache on the NPU path): broadcast what every column needs; more tokens per
  weight pass (GemmQ 32 -> 64 tokens per core row: GEMMs -15..26%; several tokens per decode step, R11); more queries
  per pass over a KV cache (Llama 3B prefill attention 32 -> 64: -25%).
- **Two cores that need the same stream: split it, don't broadcast it** once a core's stream or compute binds (decode
  attention at long context): each core takes half the positions and they merge partial softmax results.
- **A broadcast is bounded by each consumer's input port** (8 B / cycle): feed each core on both its input channels
  (two streams, each with half of every object): Gemma's full attention -41%.
- **Read every byte once**: fixed-size gather units waste bytes when a part is smaller (a 1 KB tile header in a 4 KB
  unit: 1.33x); group the small parts (`GemmQOperator.hg`). Long-K GEMMs: K parts keep the A strip resident in the
  memtile (`plan_phase(k_part=)`).
- **Queue depth, not BD count, limits a shim channel**: at most 4 queued tasks. Rolling retirement
  (`Sequence(rolling=True)`) keeps it full. Issue the fills the cores need to make progress before the bulk of a
  phase's weight fills, or the queue deadlocks.
- **Deep memtile FIFOs run out of BD ids** (24 per channel parity); deeper FIFOs do not help a DMA-bound operator.
- **Design switches**: every change of design in a library program is a reset and a reconfiguration (~80-90 µs); count
  them per layer, and prefer one design whose workers run every phase where it fits.

## 7. Program memory

16 KB per core for the startup (144 B), the Worker's control code (`core_<col>_<row>`, what aiecc generates from the
IRON program: ~2-3 KB in an overlay) and every kernel function its entry kernels reach. Overflow: "section '.text'
will not fit in region 'program'" at link time, `_XAie_LoadProgMemSection: Overflow of program memory` at load.
Count a core's bytes with `llvm-nm -S --size-sort` on its `design.mlir.prj/elfs_*/*.elf` (`llvm-size -A` for its
`.text`), or before linking by walking the kernel object's call graph from the core's entry kernels (calls are
relocations to function symbols: `llvm-objdump -dr`; sizes from `llvm-nm -S`). Rank variants by the core's bytes and
cycles in one variant_search run.

Levers that worked (the decode overlay's main core: -1.1 KB):
- **Worker loops**: IRON's `range_` (aiecc narrows its counter to i32 when the bound is a constant or read from a
  buffer; an index counter is two registers and a carry per loop), `design.range32` for a body that computes with the
  i32 induction variable, a core's infinite loop as
  `forever_()` with `Worker(while_true=False)` (`scf.while(true)`, no counter: 48-192 B per core; 816 B on a small core
  whose counted loop LLVM unrolled); i32 loops and `while_true=False` took 624 B off the decode main core. opt fully
  unrolls a constant-trip worker loop to drop the rotation of the FIFO objects' pointers (the Qwen decode GDN core's
  32 panels: +800 B): `range32(n, unroll=False)` keeps it rolled. A global cap on unrolling instead cost the Gemma
  prefill 22% (other cores' unrolled loops are hot). The cores' llc
  runs with `--enable-aie-zol-without-minitercount=false` (`operators/base.py` `CORE_LLC_FLAGS`, aiecc `-Xllc`): loops
  without a known minimum trip count (FIFO skip loops) would become zero-overhead loops padded with NOP bundles (~180 B
  each). aiecc's `-Xopt=<arg>` reaches the cores' opt (`-opt-bisect-limit=<n>`, `-opt-disable=<passes>`: which pass
  grows a core).
- **Code per core**: a shared function rendered per core with only that core's cases; one kernel entry per operation,
  and small end-of-phase work folded into the last call of an existing kernel (every kernel call is Worker control
  code too); a Python helper called twice in the IRON program inlines its loops twice; the work split between
  neighbour cores by kind (one core builds the input panels, the other computes and keeps the outputs).
- **Smaller functions**: shared helpers out of line (`noinline`); `minsize` on cold dispatchers, rolled
  (`AIE_LOOP_NO_UNROLL`) call loops and copies (it shrank nothing on the GDN gate helpers and the scan's copy kernels:
  split such code across ops or cores); vector compares against a lane-index table instead of 64-bit mask arithmetic;
  runtime parameters instead of template instances (each instance is a copy: one dequant routine with a runtime
  sub-block shift, `shuffle_down` of a short vector by a runtime amount then `filter_even`, instead of one instance per
  sub-block); a table activation (`act_lut`) instead of a polynomial chain: smaller and faster.
- No runtime-library arithmetic (`checks.py` warns when a kernel calls it).
- **Pages: code the cores load themselves** (`operators/paging.py`, its docstring the hazards and their guards; sd15):
  the resident code (aiecc's link of the core) and pages of what only a page's entries reach, each linked at the slot
  (the resident end: `_main_init`, the crt's startup code, is linked last and the slot starts there) from a copy of
  `kernel.o` with the page's sections renamed; `paging.link` checks the resident code and data against aiecc's on every
  core, the page within 16 KB, no page data, no page reaching the pager, the stack with the entries. The core triggers a
  page and sleeps on its wake lock in one resident function and enters page code through the pager's state words.
  Self-DMA (sd15: the page in two B slots of its input stream, its MM2S 1 into its own TileControl): ~2.4 B / cycle of
  slots, 2.7 µs for two 5760-B slots whatever the code; the host knows each row's role and puts the page before the op
  that needs it. Memtile (the core's request packet pushes a page BD on a memtile MM2S channel of its own): 86 cycles
  + ~0.39 µs per KB of code, at most 4 cores per channel; its store and BDs reloaded after another hardware context ran
  (`OverlayDevice.set_prologue`: 0.3 ms). Decode overlays page the tile formats beyond the one of the most bytes when
  four or more do not fit (`DecodeOverlayOperator(paged=)`: Llama 3B Q3_K_M, a few loads per step, the main core 2.4 KB
  freer). Put a rarely used epilogue in the page of the role that runs before it (sd15's GEGLU in the LayerNorm page:
  1.7 KB off the resident code). A page's constant tables must be resident (`retain`: `gemv`'s `_nz`).
- After a toolchain update, compare each core's `core_<col>_<row>` size with the previous build's.

Rejected size levers: rejected.md "Program memory".

## 8. Review checklist

1. Bottleneck identified on the NPU (ablation, window counters) before kernel work.
2. Bound computed; achieved II compared against it; cycles per window compared against the schedule.
3. Constant trip counts; a unique kernel entry name encoding all template parameters; every namespace-scope helper
   prefixed too.
4. Each hot loop in its own function; its buffers distinct `__restrict` parameters; bank qualifiers on every load
   pointer.
5. No spills in hot loops; stack within the Worker `stack_size`; no switches, calls or branches in loop bodies; no
   index math that a pointer walk can replace.
6. ≤ 4 live accumulators; no runtime lane indices; ≤ 2 panel streams per loop; stores ≥ 32 bits; 512-bit vectors;
   aligned buffers.
7. Pre-RA vs post-RA compared; variants generated and compiled in parallel.
8. Data arrives in pop order; DMA-written buffers are not in the hot loop's banks.
9. Output bit-exact against the previous kernel (same arithmetic order), or tolerance + KL.
10. No kernel statics (tables included: worker `Buffer`s); rounding set in every rounding entry point; no
    runtime-library arithmetic; each core within its program memory.
