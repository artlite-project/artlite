---
name: aie2p-coding
description: AIE2P (NPU2 / XDNA2) facts and constraints for writing Artlite kernels and array designs - tile, memtile and register-file architecture, clocks and bandwidth, vector widths and MMUL shapes, L1 budget and GEMM tiling, DMA / BD / ObjectFifo limits (what builds, what hangs), Peano and aie_api traps (crashes, miscompiles, silent garbage), stack sizing, numerical verification, the build and debugging checklist and performance reference points. Use when writing or debugging a new kernel or design, choosing tile sizes or data layouts, checking a hardware limit, or when a compile crashes or a kernel returns wrong values; use aie-kernel-perf-optimization to make one fast.
---

# AIE2P coding reference

What the AIE2P array is and what breaks on it. Each point names its source: AMD manuals by number (AM027: AIE-ML v2
architecture, AM029: its register reference), mlir-aie / llvm-aie paths relative to their checkouts (AGENTS.md
"References"), papers by title, or "measured" (on this machine).

Related skills: **aie-kernel-perf-optimization** (speed; its section 2 holds the source rules every kernel follows:
widths, compile-time shapes and lanes, `__restrict` parameters, bank qualifiers, no division / branches / calls in
loop bodies; its reference/compiler-feedback.md: reading schedules, spills and assembly), **measure-ii** (loop
schedules), **profiling** (counters).

The two GEMM papers cited: [ATB] "Can Asymmetric Tile Buffering Be Beneficial?" (BFP16 GEMM on XDNA2: microkernel,
tiling model) and [GEMM-gen] "Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen AI
NPUs" (GEMM design and evaluation on XDNA / XDNA2).

## 1. Architecture identity

- Internal name `aie2p` == NPU2 == XDNA2. Chips: Strix Point, Strix Halo, Krackan. [mlir-aie `skills/aie-kernel-opt`]
- Peano target: `--target=aie2p-none-unknown-elf`. [`src/artlite/operators/base.py` (`AIE_TARGET`)]
- Physical array (mlir-aie `NPU2TargetModel`): 1 shim row + 1 memory-tile row + 4 compute rows = 6 rows, 8 columns:
  32 compute cores, 8 memory tiles, 8 shim tiles. [`include/aie/Dialect/AIE/IR/AIETargetModel.h`, ATB]
- The same model has a `VirtualizedNPU2TargetModel` that runs on fewer columns; check the device name against the
  physical board. [`AIETargetModel.h`]

## 2. Compute tile micro-architecture

- VLIW / SIMD vector processor; 7-way VLIW instruction schedule. [AM027, ATB]
- Data memory per tile: **64 KB** (`getLocalMemorySize = 0x00010000`), shared by ObjectFifos, BSS and stack; leave
  headroom. [AM027, `AIETargetModel.h`]
- Banks: AM027 describes 8 physical 8 KB banks interleaved into 4 logical 512-bit wide banks (16 KB regions); the
  mlir-aie target model reports `getNumBanks(...) = 4` for core tiles and 8 for memory tiles. [AM027,
  `AIETargetModel.h`]
- Program memory per tile: **16 KB**. [AM027]
- Core load / store: two 512-bit load units + one 512-bit store unit, each with an AGU (2D / 3D addressing). [AM027]
- Streams: one 32-bit input and one 32-bit output AXI4-Stream, plus 512-bit cascade streams (accumulator forwarding)
  with a two-deep 512-bit FIFO on each side. [AM027]
- DMA channels per tile: compute and shim tiles 2 S2MM + 2 MM2S; memory tiles 6 S2MM + 6 MM2S. [AM027, GEMM-gen]
- A core reads and writes its own data memory and its north, south and west neighbours' (shared data memory).
  [AM027, measured]
- Register files (the `aie2p` TableGen target model):
  - Scalar: `r0-r31` (32b), `m0-m7` (20b modifier), `p0-p7` (20b pointer).
    [llvm-aie `llvm/lib/Target/AIE/aie2p/AIE2PRegisterInfo.td`]
  - Vector: 24 x 256b `wl/wh`, aliasable as 12 x 512b `x` and 6 x 1024b `y`; 12 x 576b `ex` block-float registers
    overlap the `x`. [`AIE2PRegisterInfo.td`]
  - Mask: 4 x 128b `Q` registers for sparsity. [AM027]
  - Accumulators: 20 x 512b `bm`, aliased as 10 x 1024b `cm` and 5 x 2048b `dm`. Some summaries claim 12 / 6 / 3
    and AM027's table is generic; ATB uses five 8x8 fp32 (2048b) accumulators for bfp16. The compiler / assembly
    view is the ground truth. [`AIE2PRegisterInfo.td`, ATB, AM027]
- Vector unit: 32 x 32b multipliers; shuffle / shift / add / compare and SRS / UPS conversion [AM027]. AM027's
  wide fp8 and MX6 / MX9 multipliers belong to AIE-ML v2 "Telluride" (llvm-aie `aie2ps`), not to Strix (`aie2p`):
  the only wide block-float multiplier here is the bfp16ebs8 / ebs16 one. [measured: instruction set, mode fuzzing]

## 3. Memory tiles (L2)

- 512 KB per memory tile (`getMemTileSize = 0x00080000`), 8 banks of 64 KB. [`AIETargetModel.h`, AM027]
- MemTile DMA channels 0-3 can access the east / west neighbours' banks and locks; channels 4-5 are local only.
  [AM027]
- MemTile BD ids are split by channel parity: even channels BD 0-23, odd channels BD 24-47. [`AIETargetModel.h`]
- Locks: 16 per core / shim tile, 64 per memory tile. [`AIETargetModel.h`]

## 4. Clock and bandwidth

- NPU2 parts expose `h_clock` via `DRM_AMDXDNA_QUERY_CLOCK_METADATA`; the DPM levels' `.hclk` is the NPU clock (PMF
  `npuclk_freq`). `h_clock` DPM levels in MHz: 792, 1056, 1152, 1267, 1267, 1408, 1584, 1800; `turbo` selects the
  max (1800 MHz) and `powersaver` the min (792 MHz).
- Power modes for `xrt-smi configure --pmode <mode>`: `default` = dynamic DPM; `powersaver` = 792 MHz; `balanced` =
  medium DPM (1267 MHz); `performance` = 1800 MHz with clock gating on; `turbo` = 1800 MHz with clock gating off
  (cannot be set while an NPU context is active). `sudo xrt-smi configure --pmode performance --device
  0000:c5:00.1` is safe at any time.
- Measured with the cores' counters: 1.81 GHz in turbo under any load; the reported `h_clock` is a lagging average.
- Off-chip bandwidth: ~65 GB/s effective on Strix Point, ~50 GB/s of effective DRAM bandwidth in GEMM workloads
  [ATB, GEMM-gen]. Measured: ~57 GB/s read for the whole NPU (one 32 B / cycle fabric port shared by all shims),
  ~13.8 GB/s per shim MM2S channel, ~27 GB/s per shim; DMA channels and stream-switch ports carry 8 B per cycle
  (14.4 GB/s). The full table: aie-kernel-perf-optimization `reference/data-movement.md`.

## 5. Vector widths and data types

- The optimal vector width is **512 bits** (`v32` for bf16 / int16, `v64` for int8, `v16` for float / int32):
  `optimal_elements = 512 / element_bits`. 256-bit (`v16` bf16) loads / stores lower to half-register `vldb wl` and
  leave the 512-bit port half used. [measured: v16 vs v32 assembly of the same kernels]
- The `tanh` / `exp2` intrinsics consume 16-element float vectors (512b of float): for bf16, split into 16-element
  halves, compute, then `concat`. AIE2P has hardware `vtanh` and `vexp2`, both approximate (their errors and the
  LUT-based replacements: `operators/act_lut.py`, aie-kernel-perf-optimization `reference/numerics.md`).
- BFP16 (`bfp16ebs8`): 8 mantissas share one 8-bit exponent, 9 bytes per 8 elements. [ATB]
- Block vector types of the NPU2 target model: `v8bfp16ebs8` and `v16bfp16ebs16`.
  [mlir-aie `lib/Dialect/AIE/IR/AIETargetModel.cpp`]

## 6. Matrix multiplication / MMUL

- There is no native bf16 mmul: `aie::mmul<*,8,8,bfloat16>` lowers to elementwise-emulated intrinsics (II 34 for a
  4-MAC k-step loop). The hardware 8x8x8 path is bfp16ebs8 (`mac_8x8_8x8T`), as in the GEMMs
  (`operators/gemm/mac.py`). [measured, `artlite.measure_ii`, ATB]
- IRON's bf16 `kernels.mm(...)` uses the `(4, 8, 8)` mmul geometry on AIE2P; with
  `emulate_bf16_mmul_with_bfp16=True` it switches to `(8, 8, 8)`. [mlir-aie `python/iron/kernels/linalg.py`]
- Use `kernels.mm(...).mac_dims` instead of hardcoding tiling: the returned `ExternalFunction` knows the layout the
  compiled kernel expects. [mlir-aie `programming_guide/iron_configuration.md`]
- An operand assembled for an `mmul` must be written in the exact layout the `mmul` expects, else the consumer falls
  back to scalar byte copies. [mlir-aie `skills/aie-kernel-opt`]
- GEMM tiling has four levels: the AIE API intrinsic tile `r x s x t` (8x8x8 for `bfp16ebs8`), the L1 kernel tile
  `m_ct x k_ct x n_ct`, the whole-array native tile, and the outer `M x K x N`. The L1 dimensions are integer
  multiples of `r, s, t`. [GEMM-gen, mlir-aie `programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering`]

## 7. L1 capacity and tiling

- In a GEMM each output tile is fully accumulated in L1 before write-back: memory tiles have no accumulators. [ATB]
- Arithmetic intensity favours small `T_K` and a large asymmetry ratio `rho`; compute efficiency favours large `T_K`
  and small `rho`. [ATB]
- Asymmetric tile buffering: keep the buffered `M` dimension of A (`T_MA`) smaller than the C tile (`T_MC`);
  `rho = T_MC / T_MA`. ATB's whole-array L1 constraint: `2a/rho*T_MC*T_K + 2b*T_K*T_N + c*T_MC*T_N <= 63 KB`. [ATB]
- Whole-array tiling for NPU2 (4 rows, 8 columns): `T^L2_M = 4*T^L1_MC`, `T^L2_N = 8*T^L1_N`, `T^L2_K = T^L1_K`.
  [ATB]
- GEMM mapping is output-stationary: the C tile stays in L1 for the whole K reduction while A and B stream in. One
  L1 C buffer and double-buffered A / B frees L1 for larger `m_ct x k_ct x n_ct`: up to ~18% faster on XDNA2 than a
  double-buffered C. [GEMM-gen]
- On XDNA2 store B column-major when possible: A and B both traverse `k_mt` contiguous bytes, ~19% higher int8-int8
  throughput, ~25% int8-int16, ~8.7% bf16-bf16 than row-major B, with less variability. [GEMM-gen]
- L1 overflow (ObjectFifo buffers + BSS + stack over ~64 KB) is undefined behaviour (no stack protector): split work
  across tiles or cascade partial results. [mlir-aie `skills/aie-hw-bringup`]

## 8. DMA, ObjectFifo and BD limits

Walks:
- AIE2P DMA transformations (ObjectFifo `to_stream` / `from_stream`) work at **4-byte (i32)** granularity: no
  sub-word unpacking, zero-stride replication or transposes. Pure data movement (transposes, deinterleaves, gathers)
  fits ObjectFifo forwards when the element is >= 4 bytes; smaller types need DMA + kernel (`VSHUFFLE`).
  [GEMM-gen, mlir-aie `skills/aie-kernel-opt`]
- ObjectFifo walks are `TensorAccessPattern`s over one transfer (offset 0; `operators.design.stream_walk` builds one
  from (size, stride) pairs), dimensions **outermost first**: sizes `[8, 3]`, strides `[1, 8]` mean
  `for i in range(8): for j in range(3): idx = i*1 + j*8`.
  [mlir-aie `programming_guide/section-2/section-2c/to_stream_transformations`]
- An explicit 1-dim `to_stream` (sizes `[N]`, strides `[1]`) on an ObjectFifo forward is NOT the same as `None`: it
  bakes per-object strides that desynchronize a stream (objects replayed). Use `to_stream=None` for a linear split.
- IRON zeroes the stride of an outer tap dimension of size 1: a parameterized length over a [1, 1, d1, d0] tap re-reads
  its first pass. Give that dimension size 2 and the unit explicitly (`ChainPart(length_unit=d1 d0)`).
- A small-shape bring-up (K == mtk, one strip per fill) can be correct while the multi-window path (K > mtk)
  misdelivers: verify layout changes at a shape with window refills, on the host first (`tests/rtc/dma_model.py`,
  `tests/rtc/test_gemm_q_dma.py`).

Links and FIFOs:
- `cons().split` and `prod().join` offsets are in **elements**, not bytes.
- `split` cannot de-interleave: the link lowering cuts the memtile pool into consecutive segments. Deliver
  `[K rows | V rows]` from the shim and split at the boundary. A `split` link writes its input per segment: the input
  FIFO's `from_stream` walks one segment (`offsets[i + 1] - offsets[i]` elements), each output's `to_stream` its own.
- A join's segments must match what each input sends per joined object. A mismatch (a core object grown to two query
  blocks, the join's offsets still for one) does not fail at build time: the dispatch hangs (TIMEOUT, state 8). An
  input may send it as smaller objects (a core's 288-byte chunks fill a 46 KB segment), and a one-input join's
  `from_stream` may interleave them (`prefill.norm` hold design: two groups' half panels into whole panel rows).
- A join streams each core's segment whole: the core's BD length is its object size, and the joined FIFO's
  `to_stream` applies per segment. To send part of a larger core buffer, make the object that part and pin the rest
  right after it (`pin_l1_buffers`; prefill attention `qx`).
- A memtile has 6 S2MM and 6 MM2S channels: a design that joins a seventh stream into one memtile does not build.
- A release must match an acquire of the same loop body (aiecc): two regions a core must hold at once cannot stream
  separately; interleave them into one stream of pairs.
- Pool sizes: a pool is sized from the acquires (largest + 1): `acquire(2)` deepens a depth-2 pool by one (also with
  `cons(depth=2)`); `forward(depth=3)` with `acquire(1)` still gives 2 objects; `cons(depth=3)` (a depth per endpoint)
  is taken at its word. Budget L1 for it, or take the objects one at a time.
- A FIFO between vertical neighbours lowers to shared memory: a pool in the producer's memory and locks, no DMA, no
  BDs (aie-kernel-perf-optimization `reference/restructurings.md` R12).
- FIFO depth: depth 2 on weights enables pipelining; depth > 1 on B / C can explode program memory through unrolled
  worker bodies.
- IRON `Runtime(seq, [...])` takes the FIFO endpoints of a sequence argument as a list (one per column or row), not
  a dict.
- Runtime `TaskGroup`s express cross-phase dependencies. A single `TaskGroup` defers all awaits to the end and can
  race.

BDs:
- Counts: core / shim tiles 16 BDs, memory tiles 48 (24 per channel parity); core / shim BDs address 3-D, memory
  tiles 4-D with compression and constant padding. [`AIETargetModel.h`, AM027]
- Fields: core `Buffer_Length` 14 bits, memory tile 17 bits (one memtile BD moves < 512 KB: a 442 KB A strip resent
  per pass runs), shim 32 bits; wrap counts 8 bits (core) and 10 bits (memory tile / shim). A C FIFO of
  `memref<64128xf32>` fails with `Size may not exceed 1023`: tile or stream large outputs. A BD chain cannot carry a
  repeat count. [`AIETargetModel.h`]
- Memtile BD ids run out with deep FIFOs: an A FIFO of depth 8-12 next to an X FIFO of depth 8 failed (6 and 4 work),
  depth 16 failed elsewhere. A link's two directions on one channel number draw on one parity (the decode overlay's
  A FIFO tops out at depth 12); pinning the channels (`cons(channel=)`: the memtile S2MM, `forward(channel=)`: its
  MM2S) puts the directions on different parities (depth 20-22 compiled).
- Shim BDs: 3 dims plus the outer repeat; the innermost dimension at most 1023 words (AGENTS.md): write long rows as
  `[n][512 words]`. An inner run of 1280 words fails lowering ("Do not include the highest dimension size in transfer
  length"); a longer one can make aiecc run for 25+ minutes instead. A gathered unit longer than 1023 words (32-row
  slices of 128 B per row = 1024 words) goes as sub-units of 16 rows: the memtile then receives [sub-unit][tile][row
  16], and the consumer reads its rows in that order (`GemmQOperator.us`, the Q4 slot's dequant row permutation).
- A core-tile BD takes no lock or exactly one acquire and one release (aiecc). A buffer filled by two BDs in turn (an
  object split over two L1 regions: sd15_v2's `design.R23`): the first acquires the free lock and releases a lock only
  the DMA uses, the second acquires that one and releases the full lock.
- Core BD sizes are at most 255 per dimension, and IRON coalesces contiguous tap dimensions first (a [2, 9, 128] walk
  of 1152-word runs becomes [2, 1152]: "Size may not exceed 255"). A long contiguous run is a length-only BD (rank 1,
  stride 1): split a buffer into one BD per run instead.
- A memtile walk that reads past the 512 KB memtile hangs the dispatch (TDR, no error): e.g. a tensor at the heap's
  top whose consumer reads whole fixed-size objects past its end (sd15_v2's GEMM residual reads; `Gemm` asserts it).
- A chained fill of more BDs costs shim BD ids: keep a column's chains within 14 BDs (`b_chain` asserts it).
- Shim BD iteration (word 6): `Iteration_Current` 31:26, `Iteration_Wrap` 25:20 = count - 1, `Iteration_Stepsize`
  19:0 = words - 1; a repeated task advances its base address per execution: one BD writes nh x nph consecutive
  objects (`txn.shim_bd(..., iteration=(step_words, count))`).
- A memtile BD's word 6 changes under the DMA: `Iteration_Current` (bits 28:23) advances each time the BD loads
  (AM029 DMA_BD0_6); writing the BD resets it.
- Runtime-variable lengths (a KV fill up to the position): in compiled designs a scratchpad `length_parameter`
  (`fill(tap, length_parameter=)`: n passes of `tap`; `ChainPart(length_parameter=)`), applied by the firmware at
  dispatch at no measurable cost, in whole steps of the walk's third dimension; in host transactions the BD built
  with the length; the core derives the same trip count from its own data.
- Scratchpad offset parameters are range-checked against the argument's declared size (a value that would move a
  transfer past its buffer raises at write time): declare the argument with the padding the largest offset needs
  (the Gemma overlay's `KV_PAD`).
- With 16 shim BDs, large GEMM K / output dimensions reuse BDs in flight, synchronized by task-completion tokens of the
  command processor's per-channel task queues (at most 4 queued tasks per channel). [GEMM-gen, AM027]
- A GEMM's DDR -> NPU transfers need a long enough contiguous `k_mt` (and matching `k_ct`) to reach effective DRAM
  bandwidth: choose the smallest `k_mt` at the saturation point (larger ones inflate L2 padding). The A tile's L2 -> L1
  transform splits the 5-D iteration into a 4-D MemTile transform and a 3-D CompTile S2MM transform, linearizing the
  inner `r x s` tile into one dimension. [GEMM-gen]

## 9. Peano and aie_api traps

Crashes:
- **`MachineBlockPlacement` "Not a current successor!"**: a flag test inside certain loops (`if (flags & F)` in a
  loop body). Choose between two functions outside the loop, or always do the work.
- **PreLegalizerCombiner segfault**: a rolled loop around bfp16 operands whose exponent or mask depends on the
  iteration (`e8(h ? a : b)`, `broadcast(h ? 12 : 3)`); unroll that loop.
- **`AIE_LOOP_UNROLL` in an always_inline helper inlined into a loop** makes llvm-aie emit a debug location in the
  wrong subprogram: "Broken module found" (`!dbg attachment points at wrong subprogram`) only with line tables, so
  measure_ii's compile fails while the plain build works. Write the unrolled body out by hand.
- Peano cannot lower inline `asm`.

Silent wrong results:
- **A load followed by a store to the same address can be reordered** (`yv = load(y); store(y, zeros); ... use(yv)`:
  the store issued one cycle after the load, which saw it). Store after the last use.
- **A rolled `extract<8>(h)` loop stored wrong lanes** (besides compiling to a switch): unroll lane loops.
- **`aie::shuffle_up / down(_fill)` of a 2048-bit vector with a runtime shift** returns an uninitialized vector
  (aie_api implements only a shift by the native length, `FIXME: CRVO-12835`; no diagnostic): shift 1024-bit halves.
- **A bfp16 block-vector stream (`al::BIn`) constructed at an address not 64-B aligned** (a 72-B panel of a packed
  array) reads garbage (NaN state). Open one stream at the array's start and pop in order.
- **Sub-byte and block-float pointer arithmetic**: loop-carried or large-offset `uint4*` advancement can return zeros
  or garbage, and `sizeof(bfp16ebs8)` is 9, `sizeof(uint4)` 1. Offset a `uint8_t *` and cast at the load or the stream
  constructor (`uint8_t[N*9]` for blocks). `aie::load_v<N>(uint4*)` + `.unpack()` yields linear nibble order (lane
  `2i` = low nibble of byte `i`); 1024-bit `load_v<256>(uint4*)` + unpack + `to_float<bfloat16>` is bit-exact.
- `alignas()` does not work for `bfloat16` on AIE: use `__attribute__((aligned(64)))` (AGENTS.md: 64-byte alignment of
  128-byte loads).
- `AIE_LOOP_IGNORE_MEM_DEPS` also drops the compiler's spill / reload ordering: a spilling loop computes garbage.
- **Jinja `trim_blocks`** eats the newline after a block tag: `{% endif %}` at the end of a line glues the next line
  onto it (e.g. `inline` onto a return type). Put block tags on their own lines and read the rendered `kernel.cc` in
  the cache dir.

Address spaces and layout:
- Bank annotations are address spaces `a=5`, `b=6`, `c=7`, `d=8` (`__aie_dm_resource_a` etc.); aie_api's
  `fifo_buffer_stream` propagates them.
- **Function placement in `.text` follows Clang's order of first reference**, not the source order: an extern
  function called from an earlier one is emitted after its caller; templates and internal-linkage helpers are
  deferred to the end of the translation unit. Code that must sit at the same address in two builds (patched variants)
  needs plain external functions and a table referencing them in the wanted order. A call into an empty function of
  the same translation unit is deleted (inferred side-effect free); weak definitions block that inference (and
  inlining).
- The firmware's PDI-load reset between configurations leaves program and data memory as they were (AM027's zeroing
  "upon boot and reset" is not it; measured: a word written before the reset reads back after it): a buffer that must
  start at 0 needs its `initial_value`.

## 10. Stack

- Stack size is per `Worker` (`stack_size=<bytes>`), the linker-script allocation for that core
  (`_sp_start_value_DM_stack` + `. += <size>`). Measure a kernel's per-function usage with `ARTLITE_MEASURE_II=1`
  (`stack N B` per function), `-fstack-size-section` + `llvm-readelf <o> --stack-sizes`, or the `paddxm [sp],
  #<offset>` prolog in the assembly. The static maximum is a lower bound; every compile's `checks.py` compares the
  deepest call chain plus the core's own frame (64-128 B) with `stack_size` (a kernel needing exactly `stack_size`
  zeroed parts of a workspace).
- **Stack overflow is silent and has a parity signature.** A spill area past the reservation clobbers the adjacent
  ObjectFifo buffer: the corruption hits alternating k-tiles / buffer parity ("even k-tile, first panels wrong +
  NaN"), looking like a DMA / layout bug. Unrolling or straight-lining pipelined code can triple stack usage:
  re-measure after restructuring.

## 11. Quantization formats

- The GGUF-derived tile formats (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, BF16, BFP16): geometry, lossless packing,
  reference numerics and each format's kernel fragments are in its slot in `src/artlite/operators/formats/`; the
  decode GEMV kernels and the prefill GEMM (`gemm/gemm_q.py`) document their dequant arithmetic (unsigned codes
  zipped with 0x43 into bf16 128 + q, offsets folded into the accumulator). bfp16ebs8 host helpers:
  `src/artlite/operators/bfp16.py`.

## 12. A design to copy

- **The GEMM MAC loop** (`src/artlite/operators/gemm/mac.py`, GemmQ / GemmQd): `aie::mmul<8,8,8,bfp16ebs8>` over
  aie_api block-vector streams with explicit bank annotations (`__aie_dm_resource_*`), one flat II 32 loop per
  stripe (persistent streams rewound by 2-D seeks, separate `__restrict` C load / store pointers).

## 13. Numerical verification

- Compare with absolute and relative tolerances (`tests/rtc/common.py` `verify_close`; ATB's GEMM: `abs_tol=0.5`,
  `rel_tol=0.10` for 2x bf16), the reference rounded to bf16 first; `torch.equal` where the arithmetic order is the
  kernel's. [ATB]
- For large outputs a stochastic 1000-sample check is acceptable.
- One-hot A / B inputs catch layout regressions.

## 14. Build and debugging checklist

- The compile cache key (AGENTS.md) covers the rendered sources, flags, design MLIR, `operators.h` and the toolchain
  versions; in-place edits of installed toolchain headers are not detected: `ARTLITE_IGNORE_CACHE=1`.
- `ARTLITE_LOG_COMMANDS=1` with `ARTLITE_IGNORE_CACHE=1` shows the exact `clang++` / `aiecc` invocations.
- `dmesg` shows `amdxdna ... IO_PAGE_FAULT` when DMA addresses or buffer layouts are wrong.
- Bring designs up incrementally: one block / layer at a time, then the host handoff between dispatches, then the
  whole chain. [mlir-aie `skills/aie-hw-bringup`]
- Before blaming the compiler, rule out the stack (section 10), the layout (`tests/rtc/dma_model.py`) and the DMA
  patterns. When a stored value reads back as the buffer's initial value, check the store's address in the `.s`.
- Program memory overflow (`_XAie_LoadProgMemSection: Overflow of program memory`, or "section '.text' will not fit
  in region 'program'"): aie-kernel-perf-optimization section 7.

## 15. Performance reference points

- BFP16 8x8x8 MAC: 512 MACs per cycle per core; at ~1.8 GHz 1.84 TFLOPS per core, 58.8 TFLOPS peak for 32 cores.
  [ATB]
- Whole-array BFP16-BF16 mixed GEMM on Strix Point: up to 24.6 TFLOPS with the `128x64x128` asymmetric tile. [ATB]
- Single-core balanced kernels on XDNA2: bf16-bf16 `112x48x96` (137.2 MACs per cycle, 60 KB L1, 61% L2), int8-int8
  `144x72x144` (343 MACs per cycle). XDNA2 `k_mt`: int8-int8 432, int8-int32 384, bf16-bf16 384. [GEMM-gen]
- Whole-array GEMM peak TOPS on XDNA2 (column-major B): int8-int8 38.05, int8-int16 31.52, int8-int32 25.31,
  bf16-bf16 14.71. [GEMM-gen]
- A full xclbin reconfiguration costs ~4.9 ms on XDNA2, about a 4K GEMM; overlapping reconfiguration with DMA is worth
  ~28% int8-int16 in GEMM-gen's design. [GEMM-gen] (An Artlite library's design switch: ~80-90 µs, data-movement.md.)

## 16. Sources of truth

- AM027 (AIE-ML v2 architecture: tiles, memory, registers, vector unit, DMA) and AM029 (AIE-ML v2 register
  reference); UG1079 (AI Engine kernel / graph programming, AIE1-oriented but the concepts transfer).
- mlir-aie `include/aie/Dialect/AIE/IR/AIETargetModel.h` (rows, columns, BD / lock / bank counts) and
  `lib/Dialect/AIE/IR/AIETargetModel.cpp` (block formats, DMA details).
- llvm-aie `llvm/lib/Target/AIE/aie2p/AIE2PRegisterInfo.td` (the register file).
- mlir-aie `skills/aie-kernel-opt` (kernel optimization catalog) and `skills/aie-hw-bringup` (bring-up and
  bisection).
