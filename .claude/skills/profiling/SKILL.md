---
name: profiling
description: NPU-side profiling of Artlite RTC operators — hardware performance counters around kernel event0()/event1() markers and stream-traced event timelines, dumped to Chromium JSON (Perfetto) with minimal runtime overhead. Use when measuring on-tile cycles of AIE kernels, DMA stalls, or event-level timelines.
---

# NPU Profiling in Artlite

There are two mechanisms, both controlled by environment variables. Both dump **Chromium JSON**, which you can load
in https://ui.perfetto.dev or `perfetto`, or grep.

Fused operators and libraries (`operators/fused/`) and the decode overlays (host-built transactions with no MLIR
runtime sequence) bypass the hook. For them, program the same counters by hand (aie-kernel-perf-optimization
`reference/measuring.md` "Per-core event windows and counters"): an overlay through `overlay.Sequence.TILE_WRITES`,
and a library with one item per dispatch.

## Modes

### 1. `counters` (default) — event0()/event1() cycle counters

Kernels emit pulses on the core events `INSTR_EVENT_0` (0x21 = 33) / `INSTR_EVENT_1` (0x22 = 34) through the
`event0()` / `event1()` intrinsics. At compile time Artlite places the design with aie-opt, then injects
`aiex.npu.write32` ops into the runtime sequence that program each core tile's 4 performance counters
(`Performance_Control0/1` at 0x31500/0x31504, values at 0x31520..0x3152C). After every sampled dispatch the C++
runtime (`artlite._native.Profiler`) reads the counters through the context-scoped `AMDXDNA_AIE_TILE_READ` ioctl and
appends Chromium "C" counter events to `$ARTLITE_PROFILE_DIR/<OpClass>_<key>_counters.json`. The ioctl needs no root,
but only xdna-driver's module provides it: with Linux's in-tree driver and with the XRT backend, counters raise
`NotImplementedError` (traces work everywhere).

Values are **per-dispatch deltas** of cumulative counters. For example, the sum of all `event0→event1` window cycles
since the previous sample, divided by the event0 count, gives the average kernel-call cycles within the window.

The same file has a `host` process with three tracks per sampled dispatch: `dispatch wall us`, `device busy us` (the
driver's fdinfo busy time from mailbox send to response; wall - busy is the host's submit and wake-up time; Linux's
in-tree driver reports no busy time, so this track reads 0 there) and `H clock MHz` (as reported by the driver).

Use `trace` instead when you need per-call waveforms or DMA-stream timing detail.

### 2. `trace` — stream-switched event timeline

This mode uses mlir-aie's trace flow (`aie.iron.Program.enable_trace`): every traced core streams timestamped events
over the stream switches into a DDR buffer, which is appended as an extra runtime-sequence argument. The C++ side
(`_native.dump_buffer`) dumps the buffer per dispatch to `.ttbin`, and at process exit
`aie.utils.trace.parse.parse_trace` converts it to Chromium JSON using `design.mlir.prj/input_with_addresses.mlir`
(profiling builds add aiecc `--dump-intermediates` automatically).

Limitations (validated on hardware):
- **Routing**: each traced core needs a trace packet flow to a shim DMA, and an inter-tile link carries at most 4
  southbound streams. A design that already saturates a hop (a memtile join of 4 core rows uses 4/4 southbound
  channels) fails routing in trace mode ("Unable to find a legal routing", `aie-create-pathfinder-flows`), and
  multi-column data + trace mixes conflict too. In those cases, use counters mode or trace a reduced design: one
  column (`num_columns=1`) or fewer core rows (`rows=`; 2 or 1 rows per column leave ≥ 2 southbound channels free).
  This works for up to ~8 traced workers, after which the 4-rule packet slots on some slave ports run out
  (`ARTLITE_PROFILE_TILES="first:N"` with N ≤ 8 and a trace buffer of ≥ 1 MiB). Alternatively, use
  `ARTLITE_PROFILE_LATERAL=1`.
- At most ≈ 28 tiles can be traced (5-bit packet ids); select them with `ARTLITE_PROFILE_TILES`.
- The shim writes the 32-byte trace packets in 64-byte bursts (BD burst length 64, the minimum). When the stream ends
  on an odd packet, the last packet never reaches DDR, so the last dispatch's final events are missing in about 1 of
  2 runs.
- The buffer accumulates across dispatches (same BD), so the single `.ttbin` dump per program contains all runs; each
  run's events carry that run's timestamps (the timer resets every run).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ARTLITE_PROFILE` | unset | `1`/`counters`/`trace`; unset/`0`/`off` disables profiling (zero cost, no hook installed) |
| `ARTLITE_PROFILE_DIR` | `./artlite_profile` | output directory |
| `ARTLITE_PROFILE_COUNTERS` | `33:34,34:33,33:33,34:34` | perf-counter `start:stop` pairs. Core module: max 4. The prefix `m:` selects the core tile's **memory module** (max 2) for DM bank conflicts / DMA events, `mt:` the **memory tile** (row 1, max 4), and `sh:` the **shim/PL tile** (row 0, max 2); each class dumps to a separate file (`_mt_counters.json` / `_sh_counters.json`). Event **names** from the matching enum (CoreEvent / MemEvent / MemTileEvent / ShimTileEvent) are accepted. Fewer pairs mean less readback overhead |
| `ARTLITE_PROFILE_SAMPLE_EVERY` | `1` | sample counters every Nth dispatch; the deltas cover the N preceding dispatches |
| `ARTLITE_PROFILE_TRACE_SIZE` | `4194304` | trace buffer bytes per program (trace mode) |
| `ARTLITE_PROFILE_EVENTS` | mlir-aie defaults | comma-separated list for `coretile_events`: names from `aie.utils.trace.events.CoreEvent` or hex. Defaults: INSTR_EVENT_0/1, INSTR_VECTOR, MEMORY_STALL, STREAM_STALL, LOCK_STALL, PORT_RUNNING_0/1 |
| `ARTLITE_PROFILE_TILES` | `all` | traced workers: `all`, `first:N`, or a list of pinned tiles `col,row;col,row` (trace mode; `col,row` needs operators that pin their worker tiles) |
| `ARTLITE_PROFILE_LATERAL` | `0` | `1`: pre-lower the trace with `aie-insert-trace-flows{lateral-routing=true}`, which routes it to a spare shim column (requires an unused column) |
| `ARTLITE_PROFILE_DISTRIBUTE` | `0` | `1`: split traces across the shim's 2 S2MM channels (halves the bandwidth per channel; the trace buffer must hold 2x) |
| `ARTLITE_PROFILE_CONVERT` | `1` | `0`: skip the automatic .ttbin→.json conversion at exit |

Manual conversion: `python -m artlite.profiling dump.ttbin <rtc_dir>/design.mlir.prj/input_with_addresses.mlir`

Counter limits come from aie-codegen's AIE2P map (mlir-aie's
`third_party/aie-codegen/aie-regdb/globalparams/xaie2pgbl_params.h`): memtiles have 4 counters and shims 2 (AM029's
6 are Versal's).
- **Reading an address that is not a register hangs the KMD context permanently** (ETIME, then ENODEV; the NPU needs a
  driver reload or a reboot). Reading memtile counters at ≥ 0x91030 did this, and so did reading core 0x38004 (AM029's
  Core_Status; AIE2P's is at 0x32004), after which the driver reload failed to start the NPU (SMU) and only a reboot
  helped. Look an address up in the AIE2P map before reading it.
- Memtile per-task START->FINISHED window pairs count 0; use GROUP_DMA_ACTIVITY dwell and stall levels instead.

## Performance overhead (measured on Strix Point, NPU2)

Counters mode, per sampled dispatch:

```
cost ≈ 20 us x num_core_tiles x num_counter_pairs   (one firmware round trip per read; serialized,
                                                     multi-threading does NOT help)
```

Measured on a 32-core bf16 x bfp16 GEMM (2048x512x1024), per-dispatch median:

| config | dispatch time | overhead |
|---|---|---|
| profiling off | 274 us | — |
| counters, 4 pairs, every dispatch | 2925 us | +2.65 ms |
| counters, 2 pairs (`33:34,34:33`), every dispatch | 1560 us | +1.29 ms |
| counters, 1 pair (`33:34`), every dispatch | 941 us | +0.67 ms |
| counters, 4 pairs, `SAMPLE_EVERY=10` | 268 us (unsampled) | spikes when sampled |

- The injected write32 preamble itself is **negligible** (268 vs. 274 us when sampling every 10th dispatch): counter
  configuration does not slow the NPU side.
- The same GEMM at 4096x4096x2048 (DMA-bound, ~24.5 TFLOPS unprofiled) drops to 13 TFLOPS with 4 pairs every
  dispatch and to a 22 TFLOPS mean with 4 pairs at SAMPLE_EVERY=5; **2 pairs at SAMPLE_EVERY=10 reach 23.1 TFLOPS
  (within 6%)**.
- For small, µs-scale operators, always set `ARTLITE_PROFILE_SAMPLE_EVERY` ≥ the op count, or the readback dominates.
- **Observer effect**: the readback (~20 us x tiles x pairs) happens between dispatches, so it shows up in the
  inter-window (34:33) counter and in LOCK_STALL totals. For idle / stall attribution, aggregate over the whole loop
  (`SAMPLE_EVERY = loop length`) or subtract the readback before calling the remainder "overhead" or "waiting for
  data". Window-internal metrics (busy cycles, counts, vector instructions) are unaffected.

Trace mode costs one C++ buffer dump of `TRACE_SIZE` bytes per dispatch (~1 ms per 4 MB), and the trace streams share
stream-switch and NoC bandwidth with the data, so do not trust bandwidth timings measured in trace mode.

## Usage examples

```bash
source .venv/bin/activate  # in the repo root

# Counter profile a kernel (event0/event1 windows), ~95% less overhead than
# per-dispatch sampling:
ARTLITE_PROFILE=1 ARTLITE_PROFILE_COUNTERS=33:34,34:33 \
  ARTLITE_PROFILE_SAMPLE_EVERY=10 \
  pytest -s "tests/operators/test_fused.py::TestFusedOperator::test_silu_add"

# Full waveform trace of a single-column GELU kernel (then open the .json in
# Perfetto — synthesized X slices = named kernel windows):
ARTLITE_PROFILE=trace ARTLITE_PROFILE_TRACE_SIZE=262144 \
  python -c "import numpy as np; from ml_dtypes import bfloat16; from tests.operators.elementwise import GeluOperator; \
             GeluOperator(size=65536, tile_size=1024).compile().run(np.zeros(65536, bfloat16))"
```

Kernel side: put `event0();` / `event1();` around the code of interest. The GELU / SiLU harnesses in
`tests/operators/elementwise.py` already have them, and operators put them behind switches (the prefill attention
under `ablate=("ev_...",)`, the decode overlay under `win=`). In an operator with several kernel calls per step, put
`event0()` at the start of one kernel and `event1()` at the end of another to time the whole step (GemmQ:
`GemmQOperator(ablate=("ev_kt",))`, also `"ev_dq"` / `"ev_mm"`); the per-window average is then 33:34 / 33:33.
Counter labels in the JSON come from `aie.utils.trace.events.CoreEvent` names for known ids.

## Interpreting the output

### Counter semantics (AIE2P, from aie-codegen's xaie_perfcnt + measurements)

- `start != stop`: the counter accumulates **cycles between the start and stop events** (e.g. 33:34 = cycles inside
  the event0()→event1() markers, accumulated over all windows in the dispatch).
- `start == stop` on a PULSE event (INSTR_EVENT_0/1, INSTR_LOCK_ACQUIRE_REQ, INSTR_VECTOR...): the counter counts
  **occurrences**.
- `start == stop` on a LEVEL event (LOCK_STALL, STREAM_STALL, MEMORY_STALL, CASCADE_STALL): the counter counts **cycles
  while the level is high** (e.g. 26:26; on a GEMM, the LOCK_STALL dwell ≈ 100% of the measured inter-window idle
  time, and STREAM_STALL = 0).
- The readout is a 32-bit modulo delta vs. the previous sample. Each JSON row is one dispatch window with args =
  `{"value": <delta>}` (a single key, so that Perfetto renders exactly one track per counter label per tile).
  Counters keep counting between dispatches, so the idle counter captures host gaps.
- In Perfetto, each tile shows as a process (`tile(c,r)`) and each counter as a track (e.g.
  `INSTR_EVENT_0->INSTR_EVENT_1 cycles`); trace mode adds "X" kernel-window slices named after the kernel function.
- Column indices in the counter output are design-absolute; the runtime translates them through the context's
  `partition_start_col()`.

### Event catalog (aie2p / NPU2)

Get the full list (402 events) from `aie.utils.trace.events.get_events_for_device("npu2p")`, in the classes
CoreEvent, MemEvent, ShimTileEvent and MemTileEvent. Only a fraction are useful day to day:

**CoreEvent (core tile, core module)** — what runs on the core:
| id | name | note |
|---|---|---|
| 33 / 34 | INSTR_EVENT_0/1 | kernel markers, `event0()`/`event1()` |
| 35 / 36 | INSTR_CALL / INSTR_RETURN | function call/return |
| 37 | INSTR_VECTOR | vector instruction count (0.57/cycle inside GEMM windows) |
| 38 / 39 | INSTR_LOAD / INSTR_STORE | LSU activity |
| 23 | MEMORY_STALL | core waits on data-memory bank conflict arbitration (aggregate) |
| 24 | STREAM_STALL | a core stream get/put has no data / no room |
| 26 | LOCK_STALL | core waits on a lock (ObjectFifo acquire) |
| 25 | CASCADE_STALL | the cascade (tile-to-tile accumulator) has no data |
| 44 / 45 | LOCK_ACQUIRE/RELEASE_REQ | occurrence counts |
| 28 / 29 | ACTIVE / DISABLED | core state dwell |
| 40 / 41 / 42 / 43 | INSTR_STREAM/CASCADE_GET/PUT | instruction activity per class |
| 74-105 | PORT_IDLE/RUNNING/STALLED/TLAST_0..7 | per stream-switch port; needs the port-select register mapping (trace mode emits it automatically through PortEvent) |
| 107-122 | BROADCAST_0..15 | broadcast channels (the trace start / stop sync uses 14/15) |
| 124-127 | USER_EVENT_0..3 | user events |

**MemEvent (core tile MEMORY module; only 2 counters)** — the tile's local 64 KB DM plus its DMAs and locks:
| id | name | use for |
|---|---|---|
| 77-84 | CONFLICT_DM_BANK_0..7 | **per-bank conflict counts**, the metric for DM bank placement. The heavy hitters show which bank your buffers fight over (a GEMM whose stack, A FIFOs and A staging shared banks 0-1: ≈ 8.4M conflicts over 23 runs there vs. 0 on banks 2 and 3) |
| 19-30 | DMA_*_START/FINISHED_TASK/BD | DMA channel task rates (4 channels = 2x S2MM + 2x MM2S) |
| 31-34 | DMA_*_STALLED_LOCK | DMA waits on locks (FIFO underrun) |
| 35-38 | DMA_S2MM_*_STREAM_STARVATION / MM2S_*_STREAM_BACKPRESSURE | stream handshakes inside the tile |
| 39-42 | DMA_S2MM_*_MEMORY_BACKPRESSURE / MM2S_*_MEMORY_STARVATION | memory ports |
| 44-75 | LOCK_SELx_ACQ_EQ / _ACQ_GE / _REL / _EQUAL_TO_VALUE | per-lock transitions (the lock is selected through the lock-select registers) |
| 16/17 | WATCHPOINT_0/1 | address watchpoints (debug) |

**ShimTileEvent** (per column): DMA S2MM_0/1 + MM2S_0 start/finish/stall/starvation (14-37), i.e. the DDR-facing DMAs.
`DMA_mm2s_START_TASK→FINISHED_TASK` pairs measure fill latency from host DDR.

**MemTileEvent** (row 1): the same DMA family SEL0/1 (21-44), **CONFLICT_DM_BANK_0..15** (the memtile has 16 banks),
and PORT_*; this is where B fan-out and C-join conflicts show up.

UG1076 sections worth reading: "Memory Stall Analysis" (bank arbitration is round-robin; a conflict means 2+ accesses
to the same bank in the same cycle), "DMA Stall Analysis" (a table of lock-starved DMA patterns), "Stream Stall
Analysis", and the "Metric Sets" tables (functions / partial_stalls / all_stalls / s2mm_channels etc.). AM029 has the
register layouts; AM027 describes the event-logic architecture.

### Trace JSON: use the synthesized slices

Raw INSTR_EVENT_0/1 events are 1-cycle B/E pulses, which look empty in Perfetto. `convert_trace_dump` (and therefore
the atexit conversion) appends **"X" complete slices** from each INSTR_EVENT_0 to the following INSTR_EVENT_1, named
after the tile's kernel functions (parsed from the placed MLIR), with the args `cycles`, `tile` and `window_class`.
When a tile runs two kernels with clearly separated durations (e.g. a long matmul and a short zero-fill), the slices
get `window_class = "zero/init (short)" / "compute (long)"`.

Analysis helpers (`import artlite.profile_analysis`):
- `synthesize_kernel_slices(events, kernel_names_by_tile(mlir))`
- `summarize_counters(events)` / `print_counter_summary(events)` — per-tile totals, per-dispatch means and cross-tile
  imbalance.

`parse_trace` timestamps are in **AIE cycles** (not us); the cores run at 1.81 GHz in turbo (the reported `h_clock`
is a lagging average).

## Files

- `src/artlite/profiling.py` — the configuration (`ProfileConfig.from_env`, parsed once per process, so set the
  environment before the first compile; a malformed variable raises an error that names it), MLIR injection, the IRON
  hook, and exit-time trace conversion (`convert_trace_dump`).
- `src/artlite/npu_regs.py` — the performance-counter registers of the four tile modules (core, memory, memtile, shim)
  and their control words; a design that programs counters itself (`window_counter_writes`) takes them from there.
- `csrc/artlite/profiler.{h,cpp}` — `artlite::Profiler` (context-scoped counter reads + Chromium JSON writer) and
  `dump_bo`.
- `csrc/shim_xdna/hwctx.{h,cpp}` — `hw_ctx::read_aie_reg` (`AMDXDNA_AIE_TILE_READ`) and `partition_start_col` (the
  device-wide READ/WRITE_AIE_REG ioctls return EOPNOTSUPP, and SET_STATE is root-only).
- `src/artlite/operators/build.py` — compile-time integration; the cache key includes all profiling parameters.
- `src/artlite/profile_analysis.py` — slice synthesis + counter summaries.
- Tests: `tests/test_profiling.py`.

Reference docs: AM027 §"Profiling (Performance Counters)", AM029 CORE_MODULE `Performance_Control0..2` /
`Performance_Counter0..3`, mlir-aie `programming_guide/section-4/section-4b/README.md`,
`mlir_aie/python/aie/utils/trace/` (parse.py, setup.py, events/).
