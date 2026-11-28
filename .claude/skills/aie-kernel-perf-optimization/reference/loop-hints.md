# Loop pragmas and per-loop hints (Peano)

All macros live in `src/artlite/include/operators.h`, which is auto-included into every RTC kernel. Place them
immediately before the `for` statement; they apply to that loop only. Never write raw `#pragma clang loop`.

## Contents
- Pragma macros
- Per-loop backend hints (`AIE_LOOP_HINT`)
- Global backend options (clang_flags)

## Pragma macros

| macro | effect |
|---|---|
| `AIE_LOOP_RANGE(min,)` / `AIE_LOOP_RANGE(min, max)` | Trip-count bounds; they gate pipelining, ZOL, and guard removal. Not needed when the bound is a Jinja-rendered constant: the hardware-loop pass attaches a constant trip count as the same metadata, giving the same asm. |
| `AIE_LOOP_MIN_ITERATION_COUNT(N)` / `AIE_LOOP_MAX_ITERATION_COUNT(N)` | The two halves of `AIE_LOOP_RANGE`. The minimum also caps the post-RA pipeliner's stage count ("trip count too small for the stages" in `variant_search --why`), so specify the actual minimum trip count rather than a conservative lower bound (raising the vision overlay's epilogue loops from 4 to 16 took II 4 -> 2, 3 -> 2, and 2 -> 1). Entering the loop with fewer iterations than the specified minimum produces wrong results. |
| `AIE_PREPARE_FOR_POSTPIPELINING` | `pipeline(disable)`: skips the pre-RA pipeliner so that the post-RA pipeliner schedules the loop. |
| `AIE_TRY_INITIATION_INTERVAL(N)` | Pre-RA pipeliner: searches II from the MII up to N (Peano turns on `-pipeliner-pragma-as-max-ii`), with no stage-count cap and no hand-off to the post-RA pipeliner. Post-RA: sets the exact solver's target II. This is the per-loop form of `-aie-pipeliner-max-stagecount` (below). Combining it with `AIE_PREPARE_FOR_POSTPIPELINING` is a compile error. Try it on every post-RA loop above its bound: on gather loops the post-pipeliner's heuristic search stops early (IQ3_S's decode panels went from II 40 to 29, the IQ4 prefill dequant from 28 to 21), and any target from the bound to a few above it gives the same result. |
| `AIE_LOOP_UNROLL(N)` / `AIE_LOOP_UNROLL_FULL` / `AIE_LOOP_NO_UNROLL` | Unrolling control. `UNROLL_FULL` on a short loop whose index selects vector lanes (`extract<8>(j)`, `insert(j, ...)`) turns a switch into constant lanes, matching the schedule of manually unrolled code (IQ3_XXS's gather and MMUL loops). `NO_UNROLL` on outer loops keeps program memory and stack in check. Peano leaves short constant loops rolled even at -O2 and in `minsize` code, emitting a ~180 B ZOL with its setup instead of straight-line loads and stores (~8 B per load, and fewer cycles); `al::copy` unrolls itself, and its thresholds show where. |
| `AIE_LOOP_VERSIONING` | An IR pass clones an innermost loop into a pipelined high-trip copy and a plain low-trip copy behind a runtime guard (for runtime trip counts only). |

## Per-loop backend hints

`AIE_LOOP_HINT(key, value)` attaches `llvm.loop.hint.<key>` metadata, which selected backend passes read as a per-loop
override of their `cl::opt`. Only the keys below have consumers; any other key is silently dead metadata. Enum options
take integers (use the macros).

| macro | key | effect |
|---|---|---|
| `AIE_LOOP_GPR_REALLOC` | `aie-gpr-realloc` | the WAW rewriter also re-allocates scalar / pointer registers |
| `AIE_LOOP_REALLOC_LOOPAWARE` | `aie-realloc-loopaware` | the rewriter's LRU wraps around the loop back-edge |
| `AIE_LOOP_AGGRESSIVE_REALLOC` | `aie-aggressive-realloc` | also de-allocates live-through registers |
| `AIE_LOOP_WAW_MIN_LAT(N)` | `aie-waw-reg-rewrite-min-lat` | minimum latency considered for WAW rewriting (default 3) |
| `AIE_LOOP_REALLOC_II_BIAS(N)` | `aie-realloc-ii-bias` | MinII bias in swpaware mode |
| `AIE_LOOP_REG_REWRITE_{BASIC,LATENCYAWARE,SWPAWARE,SWPAWARE_AUTO}` | `aie-reg-rewrite-mode` | rewriting mode (0/2/3/4; default auto = 1) |
| `AIE_LOOP_IGNORE_MEM_DEPS` | `aie-safe-to-ignore-memory-deps` | The scheduler ignores ALL memory dependences in this loop, including the compiler's own spill / reload pairs. Use it only for spill-free loops whose dependences you know are false (a spilling k=32 MAC loop miscompiled); prefer distinct `__restrict` helper parameters. |
| `AIE_LOOP_OLP_WAR_RENAME` | `aie-olp-war-rename` | WAR renaming in the epilogues of outer-loop-pipelined loops |
| `AIE_LOOP_OUTER_PIPELINING` / `_NO_OUTER_PIPELINING` | `aie-enable-outer-loop-pipelining` | the IR outer-loop pipeliner (put it on the OUTER loop) |
| `AIE_LOOP_OUTER_MIN_TRIP_COUNT(N)`, `_NO_SPLIT_PROLOGUE`, `_LEAN_STAGE0`, `_SPECULATIVE` | `aie-outer-loop-pipelining-*` | outer-loop pipeliner tuning |

The outer-loop pipeliner takes a counted outer loop of three or more trips around one pipelined inner loop (a 2-trip
loop is lowered to a flag loop, reported as "Cannot adjust loop bound" under `-mllvm -debug-only=aie-outer-loop-pipeliner`
in a debug llvm-aie); compute index math with selects, not division. With `_SPECULATIVE` (the last iteration's stage 0
loads anyway, so no last-iteration copy is needed), the prefill attention's scores over (pair of row blocks, K unit),
four 8-iteration MMUL loops dominated by their ramps and stores, took 494 cycles per chunk instead of 2 x 266. Without
`_SPECULATIVE` it was slower than no hint (647 static cycles vs. 584).

These hints are cheap to sweep. On the 4-accumulator MMUL loop none of them improved II alone (the fix was structural:
`machine-model.md` "Worked example"); on loops bound by false dependences or renameable WAW chains, they can.

`AIE_LOOP_REG_REWRITE_BASIC` is the most useful hint, but only when applied selectively per loop. Applied to every loop
of every Artlite kernel (1358 loops), it made 23 faster, 50 slower, and left the rest unchanged:

- wins: a post-RA pipelined loop that the default rewrite left above its slot bound (the Q3_K dequant's code-byte loop:
  II 34 -> 28, the mv bound) and a pre-RA pipelined transpose loop (the decode GEMV's `_dtab`: II 13 -> 11);
- losses when applied to a whole file: up to 39% slower per kernel (the GemmQ SiLU and SD 1.5 row epilogues,
  `silu_lut`).

`AIE_LOOP_REG_REWRITE_LATENCYAWARE` matched it on the Q3_K loop only. The default rewrite's choices shift between
llvm-aie builds: when a toolchain update moves a loop's II, try this hint first.

Debug switch: `-mllvm --aie-ignore-loop-hints` disables every hint.

## Global backend options

Some options are plain `cl::opt`s read globally rather than per loop, so they only take effect as operator-level
`clang_flags=[...]`:

- `-mllvm -aie-postpipeliner-maxii=N -mllvm -aie-postpipeliner-maxtry-ii=M` widen the post-RA II search range (defaults
  60 / 20; mlir-aie's `mm_bfp_mixed.cc` uses 120 / 50). They help only loops whose schedule exists above the default
  search window; they do not fix register-pressure or acyclic-resource failures. No pragma or loop hint widens the window
  (`AIE_TRY_INITIATION_INTERVAL(75)` on a loop with ResMII ~72 left it unpipelined). With maxii 140 / maxtry 80, the
  compiler pipelined a spilling SiLU epilogue loop with stack accesses inside its body and produced wrong results: never
  ship these flags on a kernel file with spilling loops.
- `-mllvm -aie-pipeliner-max-stagecount=N` (default 3): the pre-RA pipeliner rejects schedules with more stages
  (`AIEBasePipelinerLoopInfo` `canAcceptII`). For a hardware loop without a pragma, a schedule over the cap with II < 11
  (`-aie-postpipeliner-cutoff`) is left to the post-RA pipeliner instead. The option never applies to loops with an II
  pragma, and the pre-RA pipeliner skips every loop whose MII > 27 (`-pipeliner-max-mii`). Per loop, use
  `AIE_TRY_INITIATION_INTERVAL(N)`: on gemm_q's header copies it gives exactly the flag's schedules (II 9 -> 6 and
  13 -> 7, with 5 and 4 stages) without touching other kernels.

Flags change the kernel's cache key and apply to every loop in the file; measure all loops after adding one.
