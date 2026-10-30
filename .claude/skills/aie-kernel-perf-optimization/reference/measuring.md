# Measuring on the NPU

Where a design's time goes, measured on the hardware rather than inferred from schedules.

## Contents
- Timing hygiene
- Comparing against another commit
- Ablations
- Per-core event windows and counters
  - compiled operators (profiling skill)
  - designs driven by host-built transactions (overlays)
  - which counters to program
  - reading the numbers
- Reading core L1 from the host
- What does not work

## Timing hygiene

- One NPU job at a time. Each variant in its own process: a switch to a HW context reloads its PDI (~2.2 ms for a
  decode overlay's 0.56 MB, ~0 for a library's empty one). Variants that differ by less than run-to-run noise (~±1 ms
  on a 30 ms chunk): two instances in one process, alternated.
- Time an operator in a library program or as `FusedOperator(expand_load_pdis=False)`: a single-op FusedOperator's
  default reloads its PDIs per dispatch (+0.8 ms for a 32-core attention design), more than a ConfigureOp costs in a
  library.
- Report median and p10 over several reps. Desktop DRAM traffic moves e2e by ±15%, and a
  memory-heavy CPU job slows DMA-bound NPU work by up to 40%. Check `top` for a busy CPU first.
- The AIE clock is 1.81 GHz in turbo, whatever the kernel does: cycles / 1.81e9 = seconds. (The
  driver's reported H clock is a lagging average, not the clock.)

## Comparing against another commit

Run the other commit's code from a git worktree in the same session, in processes alternated with this tree's
(absolute numbers drift by +-5% between sessions): a small launcher removes the editable install's import finder, puts
the worktree's `src` first on the path and links the current native module into it.
Keep the worktree in the scratchpad and remove it afterwards (`git worktree remove --force`).

## Ablations

A variant with one piece of work compiled out, run in the real design. full - ablated = the
piece's cost including its interaction with the DMA (a standalone kernel can hide behind a
stream that the full design keeps busy).

- Implement ablations as operator parameters that change the rendered kernel or the core
  program. Examples:
  - `GemmQOperator(ablate=("ev_kt",))`;
  - the decode overlay: `DecodeOverlayOperator.ablate = ("strip",)` (the helper's input strips);
  - in a host-built transaction, leave work out of the transaction: no syncs between phases, no attention phases
    (the Gemma decode step: 57.0 ms full, 55.4 without both = its bytes at 56 GB/s; the syncs and attentions cost
    the difference).
- The empty kernel is the DMA floor. If empty ≈ full, the operator is DMA-bound: work on bytes and
  streams (data-movement.md), not on II.
- Results are wrong in ablated builds: never compare their outputs.
- **A library program** (`fused.Library.run`): drop every item of one operator kind from the items and time
  the rest. The difference is that kind's share including its ConfigureOps (e.g. a Gemma prefill chunk per
  operator kind).
- **When the ablations add up** (full ≈ floor + piece A + piece B: gate/up's 1.18 ms ≈ 0.81 DMA floor + 0.29 MAC +
  0.12 epilogue), the stages run one after the other instead of overlapping. Shrinking one of them then buys only
  its own share: the gate/up weight bytes cut by a quarter moved the GEMM by 5% and the chunk by 0.6%. Look for what
  serializes them (lock stalls in the windows, buffer depths, an epilogue that blocks the next pass) before cutting
  bytes or cycles.
- Typical harnesses: a decode step with transaction pieces left out; one prefill GEMM with "dq" / "mm" / "act"
  compiled out, plus its cores' window cycles and stalls through the profiling hook.
- **A step per phase kind, before any ablation**: one build whose main core closes a window at every new object and opens
  one where a metadata word asks (the host marks a phase kind per run), so one build gives a table of phase kinds against
  S (ms, summing to the step). The 27B's S 1 -> 8 at 1K: the GDN step and the attention +8.7 and +8.5 of +25 ms, windows
  in which the weights' stream waits; `down` +3.9 its input's bytes. Ablations had blamed other pieces. A window that
  ends at the next dispatch (the last phase) holds the host's time between dispatches: time the host apart.

## Per-core event windows and counters

The kernel marks a region with `event0()` at its start and `event1()` at its end. The core's
performance counters count cycles between the two events, or occurrences of an event, or
cycles a level event is high. Readout is a 32-bit delta between two reads (wraps; take it modulo
2^32).

### Compiled operators (runtime sequence from MLIR)

Use the **profiling** skill: `ARTLITE_PROFILE=1 ARTLITE_PROFILE_COUNTERS=33:34,...`. It injects
the counter setup into the runtime sequence and reads the counters after sampled dispatches.

In a fused ELF the hook does not apply: emit `npu_write32(reg, value, column=c, row=r)` in the
operator's sequence (`prefill.attn.AttnOperator(ablate=("ev_<region>",))`) and read
`fused._ctx.read_aie_reg` around a dispatch, with `FusedOperator(expand_load_pdis=False)`: an
expanded PDI load ends the ELF with an array reset that zeroes the counters.

### Designs driven by host-built transactions (the decode overlay: Gemma, Qwen3.5, Llama)

These have no MLIR runtime sequence, so the profiling hook does not apply. Do it by hand:

1. **One region per build.** Wrap each candidate region in an RAII marker that fires only for the
   selected region, so every build runs the same code:

   ```cpp
   template <int K> struct Win {
       [[clang::always_inline]] Win() { if constexpr (K == WIN) event0(); }
       [[clang::always_inline]] ~Win() { if constexpr (K == WIN) event1(); }
   };
   #define CLK(k) Win<(k)> win_   // at the top of a region's scope
   ```

   Example: `DecodeOverlayOperator(win="tile")` (the main core's tile; regions in `decode.design.WINS`, the Gemma
   attention cores' (Gemma, Llama) in `decode.attn.WINS`). Spans across kernel calls are explicit `event0()` / `event1()` at
   their two ends, e.g. `"gap"`: a phase's last tile to the next X object.
2. **Program the counters in the transaction**, once per dispatch, for every core:
   `seq.txn.write32(txn.tile_address(col, row, reg), value)` (a `Sequence` subclass that writes them first).
3. **Read them** with `ctx.read_aie_reg(col, row, addr)` before and after one dispatch
   (`ctx = op._program._ctx`). Each read is one firmware round trip (~20 µs), so read outside
   the timed region.

Useful harness switches: a region and position to window, memory stalls instead of lock stalls in counter 1, the
step without its syncs, and two banks' conflicts (memory-module counters).

### Which counters to program

Core module (AM029 `Performance_Control0/1`):

| register | value | meaning |
|---|---|---|
| `0x31500` | `33 \| 34 << 8 \| 26 << 16 \| 26 << 24` | counter 0: cycles from event0 to event1 (INSTR_EVENT_0 = 33, INSTR_EVENT_1 = 34); counter 1: LOCK_STALL (26) cycles |
| `0x31504` | `1 \| 1 << 8 \| 33 << 16 \| 33 << 24` | counter 2: all cycles (TRUE = 1: the normalizer); counter 3: event0 occurrences = windows |
| `0x31520 + 4 k` | | counter k's value |

- For memory stalls use `23 | 23` (MEMORY_STALL) in counter 1.
- STREAM_STALL is 24 and CASCADE_STALL 25.
- **A core's compute with free memory** (what an operator would take if DMA never kept it waiting): ACTIVE (28)
  minus LOCK_STALL (26), memory stalls included, its busiest core the operator's floor. The two are read one at a
  time and over one dispatch their difference drifts by tens of us (a 12 us GEMM item read 12-37): sample every 10
  dispatches or more (`ARTLITE_PROFILE_SAMPLE_EVERY`), then it repeats within ~1%. In turbo.
- **Power mode**: in the driver's default mode (clock gating) a stalled core's counters stop: a core waiting 10 ms on
  a lock counted a 0.02 M-cycle window and 0.2 M cycles of TRUE; in turbo the window was 18.5 M cycles (the lock
  stall inside it) and the stall counters account for it. Count in turbo (`xrt-smi configure --pmode turbo` with no
  context open, `default` after), above all for light loads (a few cores, long waits).
- A start == stop pair counts occurrences for pulse events, and cycles for level events
  (stalls).

Compiled operators: the profiling skill's `m:` counters, e.g. `ARTLITE_PROFILE_COUNTERS=33:34,23:23,m:77:77,m:79:79`
(window cycles, memory stall, conflicts of banks 0 and 2 = regions 0 and 1; also usable on a prefill GEMM's cores). Map the buffers to regions with `llvm-nm -n` on the core's ELF.

Memory module of a core tile (2 counters): control `0x11000` (start / stop bytes as above),
values `0x11020` and `0x11024`. `CONFLICT_DM_BANK_0 .. 7` are events 77 .. 84. The counters come
in near-equal pairs (banks 2i and 2i + 1 share 16 KB region i).

### Reading the numbers

- **Cycles per call** = counter 0 / counter 3. Compare with the static cost of the region
  (measure_ii's cycles including calls).
  - 20-30% above it with memory stall inside the window: bank conflicts or spills (SKILL.md
    section 5).
  - Equal: the schedule is the truth, and further gains come from the schedule.
- **Lock stall** over the whole dispatch = waiting for data or for output space. When it dominates
  the GEMM cores, the stream (or the producer core) paces them.
- The stall counters (start == stop) count the whole dispatch, not the window: per window only where the windows
  cover the dispatch (a harness). For a window on some of a design's calls (one format's tiles in an overlay),
  compare its cycles with the static cost, and the step with the region ablated or doubled.
- **Share of the dispatch**: counter 0 / counter 2 per core. Sum the regions of one core type to
  see what is left unaccounted (control code, waits).
- **A tile loop** (the decode overlays' main core): window a tile's period, from its command to the end of its
  epilogue (the wait for the A object outside), per format and phase kind, against the DMA's time per tile (9280 B
  per column at 3.87 B / cycle: ~2400 cycles). Above it the tiles are compute-bound even when both cores' kernels
  are below it (restructurings.md R9: the MoE frame's epilogue). Ablate one format's compute (its page loads kept)
  for its share of the step.
- Example: restructurings.md R1 (570 -> 437 cycles per unit, memory stall 7.25 -> 0.24 ms per core and chunk).

## Reading core L1 from the host

`read_aie_reg(col, row, addr)` also reads data memory. The tile address is the core's address
minus 0x70000: a workspace at core address 0x71000 (see the linker script in the cache dir) is at
tile address 0x1000. Use it to read state words or debug values the kernel stored after a
dispatch. One word per call: keep it to a few words.

## What does not work

- Reading the core's timer from the kernel (`read_tm` of Timer_Low, 0x340F8) hangs the core.
- `aie::tile::cycles()` needs a `get_cycles` that Peano does not provide (link error).
- Memtile counters: the profiling skill's KMD limits (reading registers ≥ 0x91030 hangs the context).
- There are 4 core counters and 2 memory-module counters per core tile: measure more events over
  several dispatches.
