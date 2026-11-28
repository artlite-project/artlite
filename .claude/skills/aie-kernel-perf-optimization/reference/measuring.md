# Measuring on the NPU

Where a design's time goes, measured on the hardware rather than inferred from schedules.

## Contents
- Timing hygiene
- Comparing against another commit
- Ablations
- Per-core event windows and counters
  - compiled operators (profiling skill)
  - designs driven by host-built transactions (overlays)
  - the firmware's timeline and stream stalls
  - which counters to program
  - reading the numbers
- Reading core L1 from the host
- The core's timer in a kernel
- What does not work

## Timing hygiene

- Run one NPU job at a time, and each variant in a separate process: switching to a HW context reloads its PDI
  (~2.2 ms for a decode overlay's 0.56 MB PDI, ~0 for a library's empty one). For variants that differ by less than
  run-to-run noise (~±1 ms on a 30 ms chunk), run two instances in one process and alternate them.
- Time an operator in a library program or as `FusedOperator(expand_load_pdis=False)`. By default, a single-op
  `FusedOperator` reloads its PDIs on every dispatch (+0.8 ms for a 32-core attention design), which costs more than a
  `ConfigureOp` does in a library.
- Report the median and p10 over several reps. Desktop DRAM traffic shifts end-to-end time by ±15%, and a memory-heavy
  CPU job slows DMA-bound NPU work by up to 40%. Check `top` for a busy CPU first.
- The AIE clock runs at 1.81 GHz in turbo regardless of what the kernel does, so seconds = cycles / 1.81e9. (The H
  clock the driver reports is a lagging average, not the actual clock.)

## Comparing against another commit

Run the other commit's code from a git worktree in the same session, in processes alternated with this tree's
(absolute numbers drift by ±5% between sessions). A small launcher removes the editable install's import finder, puts
the worktree's `src` first on the path, and links the current native module into it. Keep the worktree in the
scratchpad and remove it afterwards (`git worktree remove --force`).

## Ablations

An ablation compiles one piece of work out and runs the real design. full - ablated gives that piece's cost including
its interaction with the DMA (a standalone kernel can hide behind a stream that the full design keeps busy).

- Implement ablations as operator parameters that change the rendered kernel or the core program. Examples:
  - `GemmQOperator(ablate=("ev_kt",))`;
  - the decode overlay: `DecodeOverlayOperator.ablate = ("strip",)` (drops the helper's input strips);
  - in a host-built transaction, leave work out of the transaction, such as the syncs between phases or the attention
    phases (the Gemma decode step takes 57.0 ms in full and 55.4 ms without both, which equals its bytes at 56 GB/s;
    the syncs and attention phases account for the difference).
- The empty kernel gives the DMA floor. If empty ≈ full, the operator is DMA-bound: work on bytes and streams
  (`data-movement.md`), not on II.
- Ablated builds produce wrong results: never compare their outputs.
- **A library program** (`fused.Library.run`): drop every item of one operator kind and time the rest. The difference
  is that kind's share, including its `ConfigureOp`s (e.g., a Gemma prefill chunk broken down by operator kind).
- **When the ablations add up** (full ≈ floor + piece A + piece B, e.g., gate/up's 1.18 ms ≈ 0.81 ms DMA floor +
  0.29 ms MAC + 0.12 ms epilogue), the stages run one after another instead of overlapping. Reducing any single stage
  then saves only that stage's time: cutting the gate/up weight bytes by a quarter sped up the GEMM by 5% and the chunk
  by only 0.6%. Find what serializes the stages (lock stalls in the windows, buffer depths, an epilogue that blocks
  the next pass) before cutting bytes or cycles.
- Typical harnesses: a decode step with transaction pieces left out; one prefill GEMM with `"dq"` / `"mm"` / `"act"`
  compiled out, plus its cores' window cycles and stalls from the profiling hook.
- **Break a step down by phase kind before any ablation**: build once with a main core that closes a window at every
  new object and opens one wherever a metadata word requests it (the host marks the phase kind of each run). That single
  build gives a table of phase kinds vs. S (in ms, summing to the step time). On the 27B at 1K, going from S 1 to S 8
  adds 25 ms; the GDN step and the attention account for +8.7 ms and +8.5 ms of it, in windows where the weight stream
  waits, and `down` adds +3.9 ms from its input's bytes. Earlier ablations had pointed at the wrong pieces. A window
  that ends at the next dispatch (the last phase) includes the host's time between dispatches: measure the host's time
  separately.

## Per-core event windows and counters

The kernel marks a region with `event0()` at its start and `event1()` at its end. The core's performance counters
count the cycles between the two events, the occurrences of an event, or the cycles during which a level event is
high. The readout is a 32-bit delta between two reads; it wraps, so take it modulo 2^32.

### Compiled operators (runtime sequence from MLIR)

Use the **profiling** skill: `ARTLITE_PROFILE=1 ARTLITE_PROFILE_COUNTERS=33:34,...`. It injects the counter setup into
the runtime sequence and reads the counters after sampled dispatches.

The hook does not apply in a fused ELF. Instead, emit `npu_write32(reg, value, column=c, row=r)` in the operator's
sequence (`prefill.attn.AttnOperator(ablate=("ev_<region>",))`) and read `fused._ctx.read_aie_reg` around a dispatch,
with `FusedOperator(expand_load_pdis=False)`: an expanded PDI load ends the ELF with an array reset that zeroes the
counters.

To read each core's counters for every item of a library program (e.g., a prefill chunk's items), run the items one
per dispatch (`Library.run([item])`; each piece starts with an array reset), using an `Assembler.assemble` that writes
the counter zeroing and control words right after the item's configuration and zeroes the control words after the
item's run (which ends by waiting for its outputs). The counters stop there and can be read at leisure.
ACTIVE - LOCK_STALL - STREAM_STALL per core and item gives each item's standalone compute time (with no overlap with
the next item); for a 1024-token Qwen3.5 9B chunk, the 1272 items take 4 s.

### Designs driven by host-built transactions (the decode overlay: Gemma, Qwen3.5, Llama)

These have no MLIR runtime sequence, so the profiling hook does not apply. Do it manually:

1. **One region per build.** Wrap each candidate region in an RAII marker that fires only for the selected region, so
   every build runs the same code:

   ```cpp
   template <int K> struct Win {
       [[clang::always_inline]] Win() { if constexpr (K == WIN) event0(); }
       [[clang::always_inline]] ~Win() { if constexpr (K == WIN) event1(); }
   };
   #define CLK(k) Win<(k)> win_   // at the top of a region's scope
   ```

   Example: `DecodeOverlayOperator(win="tile")` windows the main core's tile; the regions are listed in
   `decode.protocol.WINS`, and those of the attention cores (Gemma, Llama) in `decode.attn.WINS`. Spans across kernel
   calls use explicit `event0()` / `event1()` at their two ends, e.g., `"gap"` spans from a phase's last tile to the
   next X object.
2. **Program the counters in the transaction**, once per dispatch, for every core: set `overlay.Sequence.TILE_WRITES`
   (the (column, row, register, value) words that every transaction starts with) before the model builds its step
   transactions. To change them later, drop the cached transactions (`overlay.Transactions.all`, `drop()`); they are
   rebuilt.
3. **Read them** with `ctx.read_aie_reg(col, row, addr)` before and after one dispatch (`ctx = op._program._ctx`). Each
   read is one firmware round trip (~20 µs), so read outside the timed region.

Useful harness switches: the region and position to window, memory stalls instead of lock stalls in counter 1, the
step without its syncs, and conflicts on two banks (memory-module counters).

### The firmware's timeline and stream stalls

A step's transaction can time itself: after a chosen operation, `write32` a user event (124 .. 127) into a core
tile's `Event_Generate` (0x34008), and count cycles between the events with that tile's counters (0 -> 1, 0 -> 2, ...
gives the cycles from one operation to another; any tile's counters work, including those of the design's cores).
With such an event after every `Sequence.sync` and one tile per segment type (a segment is the set of transfers
between two syncs of one phase kind), a single dispatch breaks the step down by phase kind: each segment's time, the
firmware's time spent issuing its operations, and the time spent in the next sync's waits. Compare these against each
segment's bytes from the transaction: tally fills and drains per stream and buffer on the fake device of
`tests/txn_dump.py`, and take the array's own pushes (such as an MoE's expert slabs) from the image, per layer.

The shim tile's two counters (control 0x31000, values 0x31020 / 0x31024) on a channel's level events show what is
stalling the channel. For an MM2S channel: `MEMORY_STARVATION` (36 / 37: it waits for DDR, so the port is the bound)
and `STREAM_BACKPRESSURE` (32 / 33: the array does not accept its words). For an S2MM channel: `STREAM_STARVATION`
(30 / 31) and `MEMORY_BACKPRESSURE` (34 / 35). Together with the transfer time (bytes at 8 B per cycle), the two add up
to the time the channel has a task (a decode step's LM head: 3018 us = 1474 + 1530 + 14). To split by phase kind,
write the events into the control register at the start of a kind's segments and 0 at their end, using one column per
kind (the columns carry similar loads). Backpressure on one column does not stall the DDR port, which serves the other
columns in the meantime; the actual loss is a segment's time minus its bytes at the port's rate.

Time the same thing on both sides of a comparison: `artlite-cli`'s "dispatches of N ms" measures `overlay.step`,
including the host's work inside it, while a `Transaction.durations` entry measures the dispatch alone (35B-A3B:
37.1 vs. 36.2 ms).

### Which counters to program

Core module (AM029 `Performance_Control0/1`):

| register | value | meaning |
|---|---|---|
| `0x31500` | `33 \| 34 << 8 \| 26 << 16 \| 26 << 24` | counter 0: cycles from event0 to event1 (INSTR_EVENT_0 = 33, INSTR_EVENT_1 = 34); counter 1: LOCK_STALL (26) cycles |
| `0x31504` | `1 \| 1 << 8 \| 33 << 16 \| 33 << 24` | counter 2: all cycles (TRUE = 1, the normalizer); counter 3: event0 occurrences, i.e., the number of windows |
| `0x31520 + 4 k` | | counter k's value |

- For memory stalls, use `23 | 23` (MEMORY_STALL) in counter 1.
- STREAM_STALL is 24 and CASCADE_STALL is 25.
- **A core's compute time with free memory** (what an operator would take if the DMA never kept it waiting) is ACTIVE
  (28) minus LOCK_STALL (26), memory stalls included; the busiest core sets the operator's floor. The two counters are
  read one at a time, so over a single dispatch their difference drifts by tens of us (a 12 us GEMM item read as
  12-37 us). Sample every 10 or more dispatches (`ARTLITE_PROFILE_SAMPLE_EVERY`); the result then repeats within ~1%.
  Measure in turbo.
- **Power mode**: in the driver's default mode (clock gating), a stalled core's counters stop. A core waiting 10 ms on
  a lock counted a 0.02 M-cycle window and 0.2 M cycles of TRUE; in turbo, the window was 18.5 M cycles (including the
  lock stall), and the stall counters account for it. Count in turbo (`xrt-smi configure --pmode turbo` with no context
  open, `default` afterwards), especially for light loads (a few cores, long waits).
- A start == stop pair counts occurrences for pulse events and cycles for level events (stalls).

For compiled operators, use the profiling skill's `m:` counters, e.g.,
`ARTLITE_PROFILE_COUNTERS=33:34,23:23,m:77:77,m:79:79` (window cycles, memory stall, and conflicts on banks 0 and 2,
i.e., regions 0 and 1; this also works on a prefill GEMM's cores). Map buffers to regions with `llvm-nm -n` on the
core's ELF.

The memory module of a core tile has 2 counters: control `0x11000` (start / stop bytes as above), values `0x11020` and
`0x11024`. `CONFLICT_DM_BANK_0 .. 7` are events 77 .. 84. The counts come in near-equal pairs, because banks 2i and
2i + 1 share 16 KB region i.

### Reading the numbers

- **Cycles per call** = counter 0 / counter 3. Compare this with the region's static cost (measure_ii's cycles
  including calls).
  - 20-30% above the static cost with memory stalls inside the window: bank conflicts or spills (SKILL.md section 5).
  - Equal: the schedule is accurate, and further gains must come from the schedule.
- **Lock stall** over the whole dispatch means waiting for data or for output space. When it dominates the GEMM cores,
  the stream (or the producer core) is pacing them.
- The stall counters (start == stop) cover the whole dispatch, not just the window, so they give per-window numbers
  only where the windows cover the dispatch (as in a harness). For a window over only some of a design's calls (e.g.,
  one format's tiles in an overlay), compare its cycles with the static cost, and compare the step time with the region
  ablated or doubled.
- **Share of the dispatch**: counter 0 / counter 2 per core. Sum the regions of one core type to find the unaccounted
  time (control code, waits).
- **A tile loop** (the decode overlays' main core): window a tile's period, from its command to the end of its epilogue
  (with the wait for the A object outside the window), per format and phase kind, and compare it against the DMA's time
  per tile (9280 B per column at 3.87 B / cycle: ~2400 cycles). Above that, the tiles are compute-bound even when both
  cores' kernels are below it (`restructurings.md` R9: the MoE frame's epilogue). To get one format's share of the
  step, ablate its compute while keeping its page loads. Window a paged format inside its page entries, not in the
  resident dispatcher: an event in one of the dispatcher's cases breaks its switch table and grows the resident code
  past the room left by the pages. Compare its cycles with the GEMV harness's: if they exceed the harness's without
  any stalls to explain it, the shared code is picking up other formats' calls (SKILL.md section 2, compile-time
  shapes).
- **Inside page code**: PC_RANGE_0_1 (event 20; set PC_Event0 / 1 at 0x32020 / 0x32024 with bit 31 set) over the page
  slot, combined through COMBO_EVENT_0 (event 9: Combo_event_inputs 0x34400, control 0x34404, 0 = A and B) with
  MEMORY_STALL, counts a core's cycles and memory stalls in its pages. With every other format's page entries emptied,
  the range can cover a single format's functions. Get their addresses with `llvm-nm -n` on the build's
  `page_<col>_<row>_<format>.elf` of the first core of a core kind (row 4 for the main cores, row 3 for the helpers).
- **Calls of a function** (in an overlay, whose loops run whatever the commands say): set PC_Event0..3
  (`npu_regs.PC_EVENTS`, the address | `PC_VALID`) to four functions' entry addresses, and count `EVENT_PC` (16 .. 19)
  with start == stop counters. This gives calls per step for four functions per core per pass. Get the addresses with
  `llvm-nm` on the core's ELF; for a page's function, use its page's ELF, and only where no other function of the core
  starts at that address (every page's entry shares the slot). The counts follow the call structure (one call of each
  of a tile's kernels per command). Multiplying call counts by measure_ii's cycle estimates did not match an overlay
  core's busy time (36% off on a Qwen 9B decode step, because trip counts come from arguments and branches are counted
  on every path).
- Example: `restructurings.md` R1 (570 -> 437 cycles per unit; memory stall 7.25 -> 0.24 ms per core and chunk).

## Reading core L1 from the host

`read_aie_reg(col, row, addr)` also reads data memory. The tile address is the core's address minus 0x70000: a
workspace at core address 0x71000 (see the linker script in the cache dir) is at tile address 0x1000. Use it to read
state words or debug values that the kernel stored after a dispatch. Each call reads one word, so keep it to a few
words.

## The core's timer in a kernel

`get_cycles()` (aie_api's `aie::tile::current().cycles()`) reads the tile's 64-bit timer (Timer_Low / Timer_High) in a
single `mov r1:r0, cntr`, with no setup in the runtime sequence (used by `probe.py`'s clock probe and GemmQ's pacing).
A `read_tm` of Timer_Low (0x340F8) stalls the core unless the runtime sequence has enabled its processor bus (bit 0 of
0x32038).

## What does not work

- Memtile counters: see the profiling skill's KMD limits (reading registers ≥ 0x91030 hangs the context).
- Each core tile has only 4 core counters and 2 memory-module counters: measure more events over several dispatches.
