# Loop pragmas and per-loop hints (Peano)

All macros live in `src/artlite/operators/operators.h` (auto-included into
every RTC kernel). Place them immediately before the `for` statement; they
apply to that loop only. Never write raw `#pragma clang loop`.

## Contents
- Pragma macros
- Per-loop backend hints (`AIE_LOOP_HINT`)
- Global backend options (clang_flags)

## Pragma macros

| macro | effect |
|---|---|
| `AIE_LOOP_RANGE(min,)` / `AIE_LOOP_RANGE(min, max)` | trip-count bounds; gate pipelining, ZOL, guard removal. Unneeded when the bound is a Jinja-rendered constant |
| `AIE_LOOP_MIN_ITERATION_COUNT(N)` / `AIE_LOOP_MAX_ITERATION_COUNT(N)` | the two halves of `AIE_LOOP_RANGE`. The minimum also caps the post-RA pipeliner's stages ("trip count too small for the stages" in `variant_search --why`): give the smallest real trip count, not a safe small one (sd15's epilogue loops 4 -> 16: II 4 -> 2, 3 -> 2, 2 -> 1). A loop entered with fewer iterations than the minimum runs wrong |
| `AIE_PREPARE_FOR_POSTPIPELINING` | `pipeline(disable)`: skip the pre-RA pipeliner so the post-RA pipeliner schedules the loop |
| `AIE_TRY_INITIATION_INTERVAL(N)` | pre-RA pipeliner: search II from the MII up to N (Peano turns on `-pipeliner-pragma-as-max-ii`), with no stage-count cap and no hand-off to the post-RA pipeliner; post-RA: the exact solver's target II. The per-loop form of `-aie-pipeliner-max-stagecount` (below). Compile error together with `AIE_PREPARE_FOR_POSTPIPELINING` |
| `AIE_LOOP_UNROLL(N)` / `AIE_LOOP_UNROLL_FULL` / `AIE_LOOP_NO_UNROLL` | unrolling control. `UNROLL_FULL` on a short loop whose index selects vector lanes (`extract<8>(j)`, `insert(j, ...)`) turns a switch into constant lanes; `NO_UNROLL` on outer loops keeps program memory and stack in check |
| `AIE_LOOP_VERSIONING` | IR pass clones an innermost loop into pipelined high-trip / plain low-trip copies behind a runtime guard (for runtime trip counts only) |

## Per-loop backend hints

`AIE_LOOP_HINT(key, value)` attaches `llvm.loop.hint.<key>` metadata that
selected backend passes read as a per-loop override of their `cl::opt`.
Only these keys have consumers; anything else is silently dead metadata.
Enum options take integers (use the macros).

| macro | key | effect |
|---|---|---|
| `AIE_LOOP_GPR_REALLOC` | `aie-gpr-realloc` | WAW rewriter also re-allocates scalar/pointer registers |
| `AIE_LOOP_REALLOC_LOOPAWARE` | `aie-realloc-loopaware` | rewriter's LRU wraps around the loop back-edge |
| `AIE_LOOP_AGGRESSIVE_REALLOC` | `aie-aggressive-realloc` | also de-allocate live-through registers |
| `AIE_LOOP_WAW_MIN_LAT(N)` | `aie-waw-reg-rewrite-min-lat` | minimum latency considered for WAW rewriting (default 3) |
| `AIE_LOOP_REALLOC_II_BIAS(N)` | `aie-realloc-ii-bias` | MinII bias in swpaware mode |
| `AIE_LOOP_REG_REWRITE_{BASIC,LATENCYAWARE,SWPAWARE,SWPAWARE_AUTO}` | `aie-reg-rewrite-mode` | rewriting mode (0/2/3/4; default auto=1) |
| `AIE_LOOP_IGNORE_MEM_DEPS` | `aie-safe-to-ignore-memory-deps` | scheduler ignores ALL memory dependences in this loop, including the compiler's own spill/reload pairs: only for spill-free loops whose dependences you know are false (a spilling k=32 MAC loop miscompiled); prefer distinct `__restrict` helper parameters |
| `AIE_LOOP_OLP_WAR_RENAME` | `aie-olp-war-rename` | WAR renaming in epilogues of outer-loop-pipelined loops |
| `AIE_LOOP_OUTER_PIPELINING` / `_NO_OUTER_PIPELINING` | `aie-enable-outer-loop-pipelining` | IR outer-loop pipeliner (put on the OUTER loop) |
| `AIE_LOOP_OUTER_MIN_TRIP_COUNT(N)`, `_NO_SPLIT_PROLOGUE`, `_LEAN_STAGE0`, `_SPECULATIVE`, `_HW_LOOP` | `aie-outer-loop-pipelining-*`, `aie-outer-loop-hw-loop` | outer-loop pipeliner tuning |

These are cheap to sweep. On the 4-accumulator MMUL loop none of them moved
II on its own (the fix was structural, machine-model.md "Worked example"); on loops bound
by false dependences or rename-able WAW chains they can.

`AIE_LOOP_REG_REWRITE_BASIC` is the one that pays off, per loop only. On
every loop of every Artlite kernel (1358 loops) it made 23 faster, 50 slower
and left the rest unchanged:

- wins: a post-RA pipelined loop that the default rewrite left above its slot
  bound (the Q3_K dequant's code-byte loop: II 34 -> 28, the mv bound), a
  pre-RA pipelined transpose loop (the decode GEMV's `_dtab`: II 13 -> 11);
- losses on a whole file: up to +39% per kernel (the GemmQ SiLU and SD 1.5
  row epilogues, `silu_lut`).

`AIE_LOOP_REG_REWRITE_LATENCYAWARE` matched it on the Q3_K loop only. The
default rewrite's choices shift between llvm-aie builds: when a toolchain
update moves a loop's II, try this hint first.

Debug switch: `-mllvm --aie-ignore-loop-hints` disables every hint.

## Global backend options

Some options are plain `cl::opt`s read globally, not per loop, so they only
take effect as operator-level `clang_flags=[...]`:

- `-mllvm -aie-postpipeliner-maxii=N -mllvm -aie-postpipeliner-maxtry-ii=M`:
  widen the post-RA II search range (defaults 60 / 20; mlir-aie's
  `mm_bfp_mixed.cc` uses 120 / 50).
  Helps only loops whose schedule exists above the default search window;
  it does not fix register-pressure or acyclic-resource failures. No pragma
  or loop hint widens the window (`AIE_TRY_INITIATION_INTERVAL(75)` on a
  loop with ResMII ~72 left it unpipelined). With maxii 140 / maxtry 80 it
  pipelined a spilling loop (a SiLU epilogue) with stack slots in its body
  and the results were wrong: never ship it on a kernel file with spilling
  loops.
- `-mllvm -aie-pipeliner-max-stagecount=N` (default 3): the pre-RA
  pipeliner rejects schedules with more stages (AIEBasePipelinerLoopInfo
  `canAcceptII`); for a hardware loop without a pragma, a schedule over the
  cap with II < 11 (`-aie-postpipeliner-cutoff`) is left to the post-RA
  pipeliner instead. It never applies to loops with an II pragma, and the
  pre-RA pipeliner skips every loop whose MII > 27 (`-pipeliner-max-mii`).
  Per loop, use `AIE_TRY_INITIATION_INTERVAL(N)`: on gemm_q's header copies
  it gives exactly the flag's schedules (II 9 -> 6 and 13 -> 7, 5 and 4
  stages) without touching other kernels.

Flags change the kernel's cache key and apply to every loop in the file;
measure all loops after adding one.
