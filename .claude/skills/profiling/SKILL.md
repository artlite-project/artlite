---
name: profiling
description: NPU-side profiling of Artlite RTC operators — hardware performance counters around kernel event0()/event1() markers and stream-traced event timelines, dumped to Chromium JSON (Perfetto) with minimal runtime overhead. Use when measuring on-tile cycles of AIE kernels, DMA stalls, or event-level timelines.
---

# NPU Profiling in Artlite

Two mechanisms, controlled by environment variables. Both dump **Chromium JSON** (load in https://ui.perfetto.dev
or `perfetto`; grep-able).

Fused operators and libraries (`operators/fused/`) and the decode overlays (host-built transactions, no MLIR runtime
sequence) are outside the hook: program the same counters by hand (aie-kernel-perf-optimization
`reference/measuring.md` "Per-core event windows and counters").

## Modes

### 1. `counters` (default) — event0()/event1() cycle counters

Kernels emit pulses on core events `INSTR_EVENT_0` (0x21=33) / `INSTR_EVENT_1` (0x22=34) via the `event0()` /
`event1()` intrinsics. At compile time Artlite places the design with aie-opt, then injects `aiex.npu.write32` ops
into the runtime sequence that program each core tile's 4 performance counters (AM029 `Performance_Control0/1` at
0x31500/0x31504, values at 0x31520..0x3152C). After every sampled dispatch the C++ runtime
(`artlite._native.Profiler`) reads the counters through the context-scoped `AMDXDNA_AIE_TILE_READ` ioctl (no root
needed) and appends Chromium "C" counter events to `$ARTLITE_PROFILE_DIR/<OpClass>_<key>_counters.json`.

Values are **per-dispatch deltas** of cumulative counters: e.g. the sum of all `event0→event1` window cycles since
the previous sample, divided by the event0 count = average kernel-call cycles within the window.

The same file has a `host` process with three tracks per sampled dispatch: `dispatch wall us`, `device busy us` (the
driver's fdinfo busy time, mailbox send to response: wall - busy is the host's submit and wake-up) and `H clock MHz`
(as the driver reports it).

Use `trace` instead when you need per-call waveforms or DMA-stream timing detail.

### 2. `trace` — stream-switched event timeline

Uses mlir-aie's trace flow (`aie.iron.Program.enable_trace`): every traced core streams timestamped events over the
stream switches to a DDR buffer (appended as an extra runtime-sequence argument). The buffer is dumped per dispatch
from C++ (`_native.dump_buffer`) to `.ttbin` and converted to Chromium JSON at process exit via
`aie.utils.trace.parse.parse_trace`, using `design.mlir.prj/input_with_addresses.mlir` (aiecc `--dump-intermediates`
is added automatically in profiling builds).

Limitations (validated on hardware):
- **Routing**: each traced core needs a trace packet flow to a shim DMA, and an inter-tile link carries at most 4
  southbound streams. A design that already saturates a hop (a memtile join of 4 core rows: 4/4 southbound channels)
  fails routing in trace mode ("Unable to find a legal routing", `aie-create-pathfinder-flows`); multi-column
  data + trace mixes conflict too. Use counters mode there, or trace a reduced design: one column (`num_columns=1`)
  or fewer core rows (`rows=`: 2 or 1 rows per column leave ≥ 2 southbound channels), up to ~8 traced workers before
  the 4-rule packet slots on some slave ports run out (`ARTLITE_PROFILE_TILES="first:N"`, N ≤ 8, a ≥ 1 MiB trace
  buffer), or `ARTLITE_PROFILE_LATERAL=1`.
- At most ≈ 28 traced tiles (5-bit packet ids): `ARTLITE_PROFILE_TILES`.
- The shim writes the 32-byte trace packets in 64-byte bursts (BD burst length 64, the minimum): when the stream ends
  on an odd packet, the last one never reaches DDR, so the last dispatch's final events (~1 in 2 runs) are missing.
- The buffer is cumulative across dispatches (same BD): the single `.ttbin` dump per program contains all runs;
  per-run events have per-run timestamps (the timer resets each run).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ARTLITE_PROFILE` | unset | `1`/`counters`/`trace`; unset/`0`/`off` disables (zero cost, no hook installed) |
| `ARTLITE_PROFILE_DIR` | `./artlite_profile` | output directory |
| `ARTLITE_PROFILE_COUNTERS` | `33:34,34:33,33:33,34:34` | perf-counter `start:stop` pairs. Core module: max 4. Prefix `m:` selects the core-tile **memory module** (max 2) for DM bank conflicts / DMA events; `mt:` the **memory tile** (row 1, max 4), `sh:` the **shim/PL tile** (row 0, max 4); each class dumps to its own `_mt_counters.json` / `_sh_counters.json`. Event **names** accepted from the matching enum (CoreEvent / MemEvent / MemTileEvent / ShimTileEvent). Fewer pairs = less readback overhead |
| `ARTLITE_PROFILE_SAMPLE_EVERY` | `1` | sample counters every Nth dispatch; deltas cover the N preceding dispatches |
| `ARTLITE_PROFILE_TRACE_SIZE` | `4194304` | trace buffer bytes per program (trace mode) |
| `ARTLITE_PROFILE_EVENTS` | mlir-aie defaults | comma list for `coretile_events`: names from `aie.utils.trace.events.CoreEvent` or hex. Defaults: INSTR_EVENT_0/1, INSTR_VECTOR, MEMORY_STALL, STREAM_STALL, LOCK_STALL, PORT_RUNNING_0/1 |
| `ARTLITE_PROFILE_TILES` | `all` | traced workers: `all`, `first:N`, or pinned-tile list `col,row;col,row` (trace mode; `col,row` needs operators that pin worker tiles) |
| `ARTLITE_PROFILE_LATERAL` | `0` | `1`: pre-lower trace with `aie-insert-trace-flows{lateral-routing=true}` (routes trace to a spare shim column; requires an unused column) |
| `ARTLITE_PROFILE_DISTRIBUTE` | `0` | `1`: split traces across the shim's 2 S2MM channels (halves per-channel bandwidth; trace buffer must hold 2x) |
| `ARTLITE_PROFILE_CONVERT` | `1` | `0`: skip automatic .ttbin→.json conversion at exit |

Manual conversion: `python -m artlite.profiling dump.ttbin <rtc_dir>/design.mlir.prj/input_with_addresses.mlir`

Counter limits of the KMD (hence the 4-counter cap):
- **Reading memtile counter registers ≥ 0x91030 hangs the KMD context permanently** (ETIME, then ENODEV; the NPU
  needs a driver reload or a reboot). Reading core register 0x38004 (AM029's Core_Status offset) did the same, and the
  driver reload then failed to start the NPU (SMU): a reboot. Read only registers the tools already read (core and
  memory-module counters, lock values from 0x1F000, 0x32004, data memory).
- Memtile per-task START->FINISHED window pairs count 0: use GROUP_DMA_ACTIVITY dwell and stall levels instead.
- The 3rd start/stop control register (counters 4/5) silently ignores write32.

## Performance overhead (measured on Strix Point, NPU2)

Counters mode, per sampled dispatch:

```
cost ≈ 20 us x num_core_tiles x num_counter_pairs   (one firmware round trip per read; serialized,
                                                     multi-threading does NOT help)
```

Measured (a 32-core bf16 x bfp16 GEMM, 2048x512x1024, per-dispatch median):

| config | dispatch time | overhead |
|---|---|---|
| profiling off | 274 us | — |
| counters, 4 pairs, every dispatch | 2925 us | +2.65 ms |
| counters, 2 pairs (`33:34,34:33`), every dispatch | 1560 us | +1.29 ms |
| counters, 1 pair (`33:34`), every dispatch | 941 us | +0.67 ms |
| counters, 4 pairs, `SAMPLE_EVERY=10` | 268 us (unsampled) | spikes when sampled |

- The injected write32 preamble itself is **negligible** (268 vs 274 us when sampling every 10th): counter
  configuration does not slow the NPU side.
- The same GEMM at 4096x4096x2048 (DMA-bound, ~24.5 TFLOPS unprofiled): 4 pairs every dispatch → 13 TF; 4 pairs
  SAMPLE_EVERY=5 → 22 TF mean; **2 pairs SAMPLE_EVERY=10 → 23.1 TF (within 6%)**.
- Small / µs-scale operators: always use `ARTLITE_PROFILE_SAMPLE_EVERY` ≥ op count, or the readback dominates.
- **Observer effect**: the readback (~20 us x tiles x pairs) lands between dispatches, in the inter-window (34:33)
  counter and in LOCK_STALL totals. For idle / stall attribution aggregate over the whole loop
  (`SAMPLE_EVERY = loop length`) or subtract it before calling the remainder "overhead" or "waiting for data".
  Window-internal metrics (busy, counts, vector instructions) are unaffected.

Trace mode: per dispatch one C++ buffer dump of `TRACE_SIZE` bytes (~1 ms per 4 MB), and the trace streams share
stream-switch and NoC bandwidth with the data: do not trust bandwidth timings measured in trace mode.

## Usage examples

```bash
source .venv/bin/activate  # in the repo root

# Counter profile a kernel (event0/event1 windows), ~95% less overhead than
# per-dispatch sampling:
ARTLITE_PROFILE=1 ARTLITE_PROFILE_COUNTERS=33:34,34:33 \
  ARTLITE_PROFILE_SAMPLE_EVERY=10 \
  pytest -s "tests/rtc/test_gelu.py::TestRtcGELU::test_gelu_lut"

# Full waveform trace of a single-column GELU kernel (then open the .json in
# Perfetto — synthesized X slices = named kernel windows):
ARTLITE_PROFILE=trace ARTLITE_PROFILE_TRACE_SIZE=262144 \
  python -c "import torch; from tests.rtc.elementwise import GeluOperator; \
             GeluOperator(size=65536, tile_size=1024).compile().run(torch.zeros(65536, dtype=torch.bfloat16))"
```

Kernel side: `event0();` / `event1();` around the code of interest (the GELU / SiLU harnesses of
`tests/rtc/elementwise.py` have them; operators put them behind switches: the prefill attention under
`ablate=("ev_...",)`, the decode overlay under `win=`). In an operator with several kernel calls per step, put
`event0()` at the start of one kernel and `event1()` at the end of another to time the whole step (GemmQ:
`GemmQOperator(ablate=("ev_kt",))`, also "ev_dq" / "ev_mm"); per-window averages are 33:34 / 33:33. Counter labels in
JSON come from `aie.utils.trace.events.CoreEvent` names for known ids.

## Interpreting the output

### Counter semantics (AIE2P, from aie-rt xaie_perfcnt + measurements)

- `start != stop`: the counter accumulates **cycles between the start and stop events** (e.g. 33:34 = cycles inside
  event0()→event1() markers, accumulated over all windows in the dispatch).
- `start == stop` on a PULSE event (INSTR_EVENT_0/1, INSTR_LOCK_ACQUIRE_REQ, INSTR_VECTOR...): counts
  **occurrences**.
- `start == stop` on a LEVEL event (LOCK_STALL, STREAM_STALL, MEMORY_STALL, CASCADE_STALL): counts **cycles the level
  is high** (e.g. 26:26; a GEMM's LOCK_STALL dwell ≈ 100% of the measured inter-window idle; STREAM_STALL = 0).
- Counter readout is a 32-bit modulo delta vs. the previous sample; each JSON row is one dispatch window, args =
  `{"value": <delta>}` (a single key so Perfetto renders exactly one track per counter label per tile). Counters keep
  counting between dispatches (the idle counter captures host gaps).
- In Perfetto: each tile shows as a process (`tile(c,r)`), each counter as a track (e.g.
  `INSTR_EVENT_0->INSTR_EVENT_1 cycles`); trace mode adds "X" kernel-window slices named with the kernel function.
- Column indices in counter output are design-absolute; the runtime translates via the context's
  `partition_start_col()`.

### Event catalog (aie2p / NPU2)

Get the full list (402 events) via `aie.utils.trace.events.get_events_for_device("npu2p")` — classes CoreEvent,
MemEvent, ShimTileEvent, MemTileEvent. Only a fraction is useful day-to-day:

**CoreEvent (core tile, core module)** — what runs on the core:
| id | name | note |
|---|---|---|
| 33 / 34 | INSTR_EVENT_0/1 | kernel markers, `event0()`/`event1()` |
| 35 / 36 | INSTR_CALL / INSTR_RETURN | function call/return |
| 37 | INSTR_VECTOR | vector instr count (0.57/cycle inside gemm windows) |
| 38 / 39 | INSTR_LOAD / INSTR_STORE | LSU activity |
| 23 | MEMORY_STALL | core waits on data-mem bank conflict arbitration (aggregate) |
| 24 | STREAM_STALL | core stream get/put has no data / no room |
| 26 | LOCK_STALL | core waits on a lock (ObjectFifo acquire) |
| 25 | CASCADE_STALL | cascade (tile-to-tile accumulator) no data |
| 44 / 45 | LOCK_ACQUIRE/RELEASE_REQ | occurrence counts |
| 28 / 29 | ACTIVE / DISABLED | core state dwell |
| 40 / 41 / 42 / 43 | INSTR_STREAM/CASCADE_GET/PUT | instruction activity per class |
| 74-105 | PORT_IDLE/RUNNING/STALLED/TLAST_0..7 | per stream-switch port; need port-select register mapping (trace mode emits it automatically via PortEvent) |
| 107-122 | BROADCAST_0..15 | broadcast channels (the trace start / stop sync uses 14/15) |
| 124-127 | USER_EVENT_0..3 | user events |

**MemEvent (core tile MEMORY module; only 2 counters)** — the tile's own 64KB DM + its DMAs/locks:
| id | name | use for |
|---|---|---|
| 77-84 | CONFLICT_DM_BANK_0..7 | **per-bank conflict counts** — the DM-bank-placement metric. Heavy hitters show which bank your buffers fight over (a GEMM whose stack, A FIFOs and A staging shared banks 0-1: ≈ 8.4M conflicts/23 runs there vs 0 on banks 2,3) |
| 19-30 | DMA_*_START/FINISHED_TASK/BD | DMA channel task rates (4 ch = 2xS2MM + 2xMM2S) |
| 31-34 | DMA_*_STALLED_LOCK | DMA waits on locks (fifo underrun) |
| 35-38 | DMA_S2MM_*_STREAM_STARVATION / MM2S_*_STREAM_BACKPRESSURE | stream handshakes inside the tile |
| 39-42 | DMA_S2MM_*_MEMORY_BACKPRESSURE / MM2S_*_MEMORY_STARVATION | memory ports |
| 44-75 | LOCK_SELx_ACQ_EQ / _ACQ_GE / _REL / _EQUAL_TO_VALUE | per-lock transitions (lock selected via lock-select registers) |
| 16/17 | WATCHPOINT_0/1 | address watchpoints (debug) |

**ShimTileEvent** (col): DMA S2MM_0/1 + MM2S_0 start/finish/stall/starvation (14-37) — the DDR-facing DMAs;
`DMA_mm2s_START_TASK→FINISHED_TASK` pairs measure fill latency from host DDR.

**MemTileEvent** (row 1): same DMA family SEL0/1 (21-44), **CONFLICT_DM_BANK_0..15** (memtile has 16 banks), PORT_*
— where B fan-out/C-join conflicts show up.

UG1076 reference sections worth reading: "Memory Stall Analysis" (bank arbitration is round-robin; conflict = 2+
accesses same bank same cycle), "DMA Stall Analysis" (lock-starved DMA pattern table), "Stream Stall Analysis", plus
the "Metric Sets" tables (functions / partial_stalls / all_stalls / s2mm_channels etc.). AM029 has register layouts;
AM027 has the event-logic architecture.

### Trace JSON: use the synthesized slices

Raw INSTR_EVENT_0/1 events are 1-cycle B/E pulses (look empty in Perfetto). `convert_trace_dump` (and therefore the
atexit conversion) appends **"X" complete slices** from each INSTR_EVENT_0 to the following INSTR_EVENT_1, named
after the kernel functions of that tile (parsed from the placed MLIR), with args `cycles`, `tile`, `window_class`.
When a tile runs two kernels with clearly separated durations (e.g. a long matmul and a short zero-fill), slices get
`window_class = "zero/init (short)" / "compute (long)"`.

Analysis helpers (`import artlite.profile_analysis`):
- `synthesize_kernel_slices(events, kernel_names_by_tile(mlir))`
- `summarize_counters(events)` / `print_counter_summary(events)` — per-tile totals, per-dispatch means, cross-tile
  imbalance.

parse_trace timestamps are in **AIE cycles** (not us); the cores run at 1.81 GHz in turbo (the reported h_clock is a
lagging average).

## Files

- `src/artlite/profiling.py` — env parsing, MLIR injection, iron hook, exit-time trace conversion
  (`convert_trace_dump`). Config is parsed once per process (set the environment before the first compile).
- `csrc/artlite/profiler.{h,cpp}` — `artlite::Profiler` (context-scoped counter reads + Chromium JSON writer) and
  `dump_bo`.
- `csrc/shim_xdna/hwctx.{h,cpp}` — `hw_ctx::read_aie_reg` (`AMDXDNA_AIE_TILE_READ`), `partition_start_col` (the
  device-wide READ/WRITE_AIE_REG ioctls return EOPNOTSUPP, and SET_STATE is root-only).
- `src/artlite/operators/base.py` — compile-time integration; the cache key includes all profiling parameters.
- `src/artlite/profile_analysis.py` — slice synthesis + counter summaries.
- Tests: `tests/rtc/test_profiling.py`.

Reference docs: AM027 §"Profiling (Performance Counters)", AM029 CORE_MODULE `Performance_Control0..2` /
`Performance_Counter0..3`, mlir-aie `programming_guide/section-4/section-4b/README.md`,
`mlir_aie/python/aie/utils/trace/` (parse.py, setup.py, events/).
