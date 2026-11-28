# DM banks and bank conflicts

How accesses to a core's data memory collide, how to find the collisions on the NPU, and how buffer placement removes
them. `SKILL.md` section 5 has the workflow; `machine-model.md` describes the load-slot binding that the bank
annotations drive.

## Contents

- The model
- Gathers (lookup tables)
- Finding conflicts
- Planning a layout
- Case studies

## The model

- Data memory is 64 KB in four 16 KB regions (`0x0000`, `0x4000`, `0x8000`, `0xC000`). Each region has two 32 B wide
  banks interleaved at 32 B (AM027), so every 64-byte access uses both banks of its region. Two accesses to the same
  region in the same cycle conflict, and one of them waits (`MEMORY_STALL`). Place buffers at 16 KB granularity.
- The `__aie_dm_resource_a/b/c/d` qualifiers do not move anything. They tell the scheduler which accesses may share a
  cycle and bind loads to slots (`machine-model.md` "Load-slot binding"). A wrong qualifier costs schedule quality, not
  correctness; the buffers' actual addresses decide the conflicts.
- Causes, most expensive first: two streamed loads from the same region in one bundle; read-modify-write of an L1
  accumulator buffer, or spills / reloads (in the stack's region), next to streamed loads; gathers whose table copies
  share a region; DMA writes into a region the loop streams from; occasional loads and stores against a stream.

## Gathers (lookup tables)

- `load_4x64_lo / hi` (`vldb.4x64`, used by kmath's tables) read two copies of a table, `ab` and `cd` (lane mapping:
  `kmath/gather.py` "Activations through the gather LUT"). The two copies belong in different regions. When they share a
  region, every gather waits a cycle: the SiLU table activation of 4096 values stalls on memory for 1099 cycles per call
  instead of 159, and a GemmQ gate/up epilogue through the SiLU table takes 2290 cycles per call vs. 1753 scheduled (512
  gathers).
- mlir-aie's LUT kernels place their table pairs the same way (`AIE_BANK_A` / `AIE_BANK_B`, `aie_bank_placement.h`).
- aiecc's placement puts the two copies wherever they fit, often in the same region. When a GemmQd format uses a gather
  table and its MAC cores do not fit a planar plan (m 96), aiecc placed both copies in one region, causing 1177 of 3170
  cycles per call in memory stalls on the IQ4_XS down dequant, and 360 of 1461 on IQ3_S. In that case
  `GemmQdOperator._dq_layout` pins only the dequantizing core (weight objects in regions 2 / 3, or, when they are wider
  than a region, the two copies at the two ends of the core's memory): the stalls drop to 127 and 47 cycles, and the
  GEMMs get 28% and 24% faster.
- `kmath.table_pass` takes the two copies as two pointers (for static tables: the bank sections below; with
  `table_arg`: two buffers of the core, each `ActTable.table_words` long).

## Finding conflicts

1. **Window counters** (the reliable check): put event0 / event1 around the region (with an operator switch such as
   GemmQ's `ablate=("ev_act",)`, or with explicit markers in the kernel) and set
   `ARTLITE_PROFILE_COUNTERS=33:34,33:33,23:23` (window cycles, window count, memory stall: see the profiling skill).
   Compare cycles per window with the region's static cycles (`measure_ii.call_cycles` on the same kernel). If they are
   equal, there are no conflicts. If the window takes longer and shows memory stalls, there are conflicts or spills.
   Count in turbo mode (`measuring.md` "Which counters").
2. **Which region**: the memory module's counters `m:77:77` .. `m:84:84` (`CONFLICT_DM_BANK_0` .. `7`, two per run).
   They come in near-equal pairs (the two banks of a region) and count the whole dispatch, not the window, so compare
   builds that differ only in the region of interest.
3. **The compiler remark** (`-Rpass-missed=aie-multi-slot-pseudo`, "missing-memory-bank") covers only loads that can
   issue on either slot. Loads into accumulators, gathers, and 2D / 3D walks are bound to a slot and never reported, so
   the absence of a remark means neither that loads are annotated nor that they are conflict-free.
4. **Where aiecc put things**: `design.mlir.prj/checked_bank_placement.mlir` in the cache dir (when aiecc dumps
   intermediates); otherwise `llvm-nm -n` on the core's ELF lists the buffer symbols at the core's addresses, where local
   memory starts at `0x70000`.

## Planning a layout

1. **List each loop's accessors**: every pointer the core loads or stores through (a tile that is loaded and stored
   every block counts twice), each copy of a gathered table, the DMA writing the next object of a FIFO or reading a
   finished one, a neighbour core writing into this core (GemmQd's dequantizing core fills the next D object), and the
   stack when the loop spills or calls.
2. **Assign regions**: any two accessors that can meet in the same cycle need different regions. With at most four
   accessors per loop, a zero-conflict assignment exists: one region each. With more, some must share; keep these
   apart, in this order: the core's own streams (they meet in every bundle), the core's streams and the DMA or neighbour
   writes, then the two sides of a tile that is loaded and stored every block (its halves in two regions, with blocks
   alternating between them: see the case studies). Attribute conflicts before moving anything: compile pieces out
   (GemmQ's `ablate`: `"mm"`, `"dq"`, `"act"`) and count per region. A count that is the same in every run is inherent
   to a loop, whereas a varying count points to a DMA interaction.
3. **Pin every buffer by address**, from the same buffer list as the L1 budget, so that the plan is the layout. A plan
   that does not fit must fail loudly or fall back to a measured alternative. GemmQd's planar plan returned "no layout"
   when the second table copy needed a separate region (64 bytes short at Gemma's T = 1024), and aiecc fell back to
   placing the whole core; adding a `mem_bank` to the two copies forced the MAC loop's streams into the same region
   (memory stalls rose from 0.99 to 1.41 M cycles per core and dispatch, making the MAC cores slower than the
   dequantizing core and adding 2% to Gemma's 8K TTFT). Shrinking the MAC stack to 1 KB (the deepest call uses
   ~320 B) let the plan fit, dropping stalls to 0.87 M cycles and recovering the TTFT.
4. **Verify** with the window counters: memory stalls inside the window should be near zero; then use the per-region
   counters for what remains.

Tools:

- **Pinned layouts**: `pin_l1_buffers` (`operators/l1_layout.py`) gives each core-tile ObjectFifo pool dedicated
  `aie.buffer`s with an `address` (it runs `aie-objectfifo-split` in-process); worker `Buffer`s take `address=`
  (`GemmQOperator._l1_layout`, `GemmQdOperator._place_plan`).
- **`Buffer(..., mem_bank=k)`**: aiecc places the buffer in region k and the other buffers around it. In a design that
  aiecc places, a single such hint reshuffles the rest (step 3): either pin everything, or measure the hot loops after
  adding it.
- **Statics**: use the linker's bank sections, with the attribute after the declarator and one section per array:
  `alignas(64) static const uint32_t t_ab[N] __attribute__((section(".aie.bank0.t_ab"))) = {...};` (clang rejects the
  attribute list before the declarator). mlir-aie's linker script gives `.aie.bankN` and `.aie.bankN.*` the free space
  of region N (aiecc reserves it before placing buffers) and fails with the section name when there is none. A library
  or fused ELF compiles the whole translation unit for every core and strips unreferenced sections; if tables share a
  single `.aie.bank0`, none get stripped and every table stays on every core (a qwen35 prefill core reserved 3712 bytes
  per bank instead of 1600, and a buffer no longer fit).

## Case studies

- **A Q4_K GEMM, the GemmQ MAC, planar GemmQd**: memory stalls halved, 8% faster; the GemmQ MAC went from 900 to 783
  cycles, 9% faster GEMM. In planar GemmQd MAC cores (3B Q2_K down), aiecc put a k-tile's A and D objects in the same
  region (32% of the MAC window in memory stalls); pinning them (`gemm_qd.PLANAR_PLANS`) took the GEMM from 5.74 to
  5.15 ms.
- **GemmQd's C tiles in halves** (`csplit`, plan g1s): a two-halves gate/up pass (m 96) loaded and stored each C tile
  in the same region, causing 220800 conflicts per tile, core and dispatch, identical in every run. Placing each tile's
  halves 16 KB apart (regions 1 and 2), with the `pats` walks pairing bands b and b + m / 32 block by block (the MAC
  loop has two stages, so a block's stores meet only the next block's loads), brought the conflicts in those regions to
  0 and the MAC memory stalls from 0.87 to 0.52 M cycles: Gemma E4B 8K TTFT 1.35% faster, Llama Q5_K_M / Q6_K gate/up
  2% faster. Split the scratch into two parts where that is what makes the pinned plan fit (`_l1_bytes`); do not change
  the C depth instead (one C object: -20% / +8% at two sizes).
- **A spilled constant reloaded next to a stream**: the dequant pair's dequant loop reloaded two spilled 64-lane
  constants from the stack (region 0) while pushing its output into X (also region 0): 114K conflicts per core and
  dispatch. `mac(accum(M), D, k128)` instead of `mac(mul(D, k128), M, ones)` needs no ones vector (4 reloads -> 2),
  which cut the conflicts to 31K and made the pair GEMM 1% faster (Llama 3B / Qwen 9B gate/up). The stack itself cannot
  move (`rejected.md`).
- **The MAC loop's C tile** (`gemm/mac.py`: loaded and stored every 16x16 block) adds ~16% memory-stall overhead to a
  GemmQd MAC window when the C loads and stores hit the same region (974 cycles per k-tile window vs. 792 scheduled). A
  second C tile in another region (`pingpong`, alternating k-tiles) removes it. This is free when the buffer already
  exists (such as gate/up's epilogue scratch), but not worth double-buffering a C object just for this (`rejected.md`).
  Alternatively, spread one C tile across two regions (its halves at `0x4000` -/+ half) and walk it so that consecutive
  blocks alternate halves (`pats` walks: band pairs (0, 2), (1, 3) block by block). On sd15's fp32 C at m = 64, this
  cut conflicts in the C regions from 15.9K to 0.4K cycles per 1280 x 1280 GEMM op and made the op 8% faster. It needs a
  two-stage loop (NS 2): with NS 3, a block's stores meet the loads two blocks later.
- **An object read in two parts, each part paired with another stream** (an m-64 A object's two k-tiles, each paired
  with a different B slot): split the object over two regions (two chained BDs: `aie2p-coding` section 8) so that part
  j sits in region 2 + j, and keep the other stream's parity (the op's command in B slot 0, so that k-tile i is in B
  slot (i + 1) % 2). On sd15's 16x16 GEMM, memory stalls dropped from 7.6 to 3.6 us, 12% faster per op; with the parity
  off by one, every k-tile conflicts. Where one object feeds several calls (a conv's rows, in two regions), order the
  calls so that their regions alternate with the B slots (the vision overlay's `ops.conv.ORDER2`, rotated by one in odd
  chunks of 9 taps: 37% fewer memory stalls).
- **A bank annotation is also a scheduling hint**: annotating the C loads of the fp32-C MAC's second 8-row panel as
  bank b (same addresses, still on slot A: `machine-model.md`) makes it pipeline at II 41 instead of 43
  (`render_mac(c2_bank=)`; annotating the stores too, or every load, gives no pipelining or II 43-44). Sweep annotations
  like pragmas.
- **When the streams outnumber the regions** (double-buffered A and D, a C tile, DMA writes of the next objects), list
  per window which buffers the core streams (a hard constraint: never two in one region) and which ones the DMA writes
  (a soft constraint: keep them out of the streamed regions), then pick the fitting assignment with the fewest soft
  pairs. Crossing the pairs (A0 with D1, A1 with D0) keeps a window's two streams apart when a region cannot hold both
  D objects. Measure each candidate (pin by role and address in a throwaway patch of the design), each in a separate
  process, alternating.
