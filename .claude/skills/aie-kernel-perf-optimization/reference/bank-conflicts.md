# DM banks and bank conflicts

A core's data memory: how accesses collide, how to find the collisions on the NPU, and how placement removes them.
SKILL.md section 5 has the workflow; machine-model.md the load-slot binding the bank annotations drive.

## Contents

- The model
- Gathers (lookup tables)
- Finding conflicts
- Planning a layout
- Case studies

## The model

- 64 KB in four 16 KB regions (0x0000, 0x4000, 0x8000, 0xC000), each two banks 32 B wide interleaved at 32 B (AM027):
  every 64-byte access uses both banks of its region. Two accesses to one region in one cycle conflict, and one of them
  waits (MEMORY_STALL). Place at 16 KB granularity.
- `__aie_dm_resource_a/b/c/d` qualifiers move nothing. They tell the scheduler which accesses may share a cycle and
  bind loads to slots (machine-model.md "Load-slot binding"). A wrong one costs schedule quality, not correctness;
  where the buffers are decides the conflicts.
- Causes, most expensive first: two streamed loads from one region in a bundle; read-modify-write of an L1
  accumulator buffer, or spills / reloads (the stack's region), next to streamed loads; gathers whose table copies
  share a region; DMA writes into a region the loop streams from; occasional loads and stores against a stream.

## Gathers (lookup tables)

- `load_4x64_lo / hi` (`vldb.4x64`, kmath's tables) read two copies of a table, ab and cd (the lane mapping: kmath.py
  "Activations through the gather LUT"). The copies belong in two different regions. In one region every gather waits
  a cycle: the SiLU table activation 4227 vs 3287 cycles per 4096 values (memory stall 1099 vs 159 per call), a
  GemmQ gate/up epilogue through the SiLU table 2290 cycles per call against 1753 scheduled (512 gathers).
- mlir-aie's own LUT kernels place their pairs the same way (`AIE_BANK_A` / `AIE_BANK_B`, `aie_bank_placement.h`).
- `kmath.table_pass` takes the copies as two pointers (static: the bank sections below; `table_arg`: two buffers of the
  core, `ActTable.table_words` each).

## Finding conflicts

1. **Window counters** (the reliable check): event0 / event1 around the region (an operator switch such as GemmQ's
   `ablate=("ev_act",)`, or a kernel's own markers), `ARTLITE_PROFILE_COUNTERS=33:34,33:33,23:23` (window cycles,
   windows, memory stall: the profiling skill). Compare cycles per window with the region's static cycles
   (`measure_ii.call_cycles` on the same kernel). Equal: no conflicts. Above with memory stall: conflicts or spills.
   Count in turbo (measuring.md "Which counters").
2. **Which region**: the memory module's `m:77:77` .. `m:84:84` (CONFLICT_DM_BANK_0 .. 7, two per run). They come in
   near-equal pairs (the two banks of a region) and count the whole dispatch, not the window: compare builds that
   differ only in the region of interest.
3. **The compiler remark** (`-Rpass-missed=aie-multi-slot-pseudo`, "missing-memory-bank") covers only loads that can
   issue on either slot. Loads into accumulators, gathers and 2D / 3D walks are bound to a slot and never reported:
   no remark means neither "annotated" nor "conflict-free".
4. **Where aiecc put things**: `design.mlir.prj/checked_bank_placement.mlir` in the cache dir (when aiecc dumps
   intermediates); otherwise `llvm-nm -n` on the core's ELF lists the buffer symbols at the core's addresses, its own
   memory at 0x70000.

## Planning a layout

1. **List each loop's accessors**: every pointer the core loads or stores through (a tile loaded and stored every
   block counts twice), each copy of a gathered table, the DMA writing the next object of a FIFO or reading a finished
   one, a neighbour core writing into this core (GemmQd's dequantizing core fills the next D object), the stack when
   the loop spills or calls.
2. **Give them regions**: any two accessors that can meet in a cycle need different regions. With at most four per
   loop there is an assignment with zero conflicts: one region each. With more, some must share; keep apart, in this
   order: the core's own streams (they meet in every bundle), the core's streams and the DMA or neighbour writes, then
   the two sides of a tile loaded and stored every block (its halves in two regions, the blocks alternating: case
   studies). Attribute before moving anything: compile pieces out (GemmQ's ``ablate``: "mm", "dq", "act") and count
   per region; a count that is the same in every run is a loop's own pattern, one that varies involves a DMA.
3. **Pin every buffer by address**, from the same buffer list as the L1 budget, so that the plan is the layout. A plan
   that does not fit must fail loudly or fall back measured: GemmQd's planar plan returned "no layout" when a table's
   second copy needed a region of its own (64 bytes short at Gemma's T = 1024), aiecc then placed the whole core, and
   a `mem_bank` on the two copies made it put the MAC loop's streams together (memory stall 0.99 -> 1.41 M cycles per
   core and dispatch, the MAC cores slower than the dequantizing core, Gemma's 8K TTFT +2%). A 1 KB MAC stack (the
   deepest call ~320 B) made the plan fit: 0.87 M, the TTFT back.
4. **Verify** with the window counters: memory stall about zero inside the window, then the per-region counters for
   what is left.

Tools:

- **Pinned layouts**: `pin_l1_buffers` (`operators/l1_layout.py`) gives each core-tile ObjectFifo pool its own
  `aie.buffer`s with an `address` (it runs `aie-objectfifo-split` in-process); worker `Buffer`s take `address=`
  (`GemmQOperator._l1_layout`, `GemmQdOperator._place_plan`).
- **`Buffer(..., mem_bank=k)`**: aiecc places the buffer in region k and the others around it. In a design aiecc
  places, one such hint reshuffles the rest (step 3): pin everything, or measure the hot loops after it.
- **Statics**: the linker's bank sections, the attribute after the declarator, one section per array:
  `alignas(64) static const uint32_t t_ab[N] __attribute__((section(".aie.bank0.t_ab"))) = {...};` (before the
  declarator clang rejects the attribute list). mlir-aie's linker script gives `.aie.bankN` and `.aie.bankN.*` the free
  space of region N (aiecc reserves it before placing buffers) and fails by name when there is none. A library or fused
  ELF links its whole translation unit on every core and drops the sections a core does not reference: in one shared
  `.aie.bank0` every table stays on every core (a qwen35 prefill core reserved 3712 bytes per bank instead of 1600, and
  a buffer no longer fit).

## Case studies

- **A Q4_K GEMM, the GemmQ MAC, planar GemmQd**: memory stalls halved, -8%; the GemmQ MAC 900 -> 783 cycles, GEMM -9%;
  planar GemmQd MAC cores (3B Q2_K down), where aiecc put a k-tile's A and D objects in one region (32% of the MAC
  window in memory stalls), pinned (`gemm_qd.PLANAR_PLANS`) 5.74 -> 5.15 ms.
- **GemmQd's C tiles in halves** (`csplit`, plan g1s): a two-halves gate/up pass (m 96) loaded and stored each C
  tile in one region, 220800 conflicts per tile, core and dispatch, the same in every run. Each tile's halves 16 KB
  apart (regions 1 and 2) and the `pats` walks pairing bands b and b + m / 32 block by block (the MAC loop has two
  stages: a block's stores meet only the next block's loads): those regions 0, MAC memory stall 0.87 -> 0.52 M cycles,
  Gemma E4B 8K TTFT -1.35%, Llama Q5_K_M / Q6_K gate/up -2%. The scratch in two parts where that is what makes the
  pinned plan fit (`_l1_bytes`); not the C depth (one C object: -20% / +8% at two sizes).
- **A spilled constant reloaded next to a stream**: the dequant pair's dequant loop reloaded two spilled 64-lane
  constants from the stack (region 0) while pushing its output into X (region 0): 114K conflicts per core and dispatch.
  `mac(accum(M), D, k128)` instead of `mac(mul(D, k128), M, ones)` needs no ones vector (4 reloads -> 2): 31K, the pair
  GEMM -1% (Llama 3B / Qwen 9B gate/up). The stack itself cannot move (rejected.md).
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
