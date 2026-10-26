# Reading compiler feedback

## Contents
- measure_ii: the fast path
- Remark flags for a clang build
- Warnings that always print
- Assembly checks
- Symptom -> cause -> fix

## measure_ii: the fast path

- **An operator**: `ARTLITE_MEASURE_II=1 python -c "from artlite.operators import X; X(...).compile()"`, or its
  pytest with the variable set; `ARTLITE_IGNORE_CACHE=1` forces a rebuild. Per function: stack and code size; per
  loop: scheduler, II, NS, ZOL, iterations, cost with calls, resource bound, spills; the paths of the `.cc`, `.s`,
  annotated `.s` and remarks YAML in `~/.cache/artlite/rtc_<hash>/`.
- **A standalone `.cc`** starting with `#include "operators.h"`: `artlite.measure_ii.measure(path)` (pass `clang_cmd`
  when the operator adds flags).
- Fields, the printed lines and which II numbers are real: the **measure-ii** skill. In short: only the final
  `!Passed pipeliner/schedule` remark (after register allocation) describes the code that runs; a loop of several
  blocks is never pipelined.
- **If measure_ii disagrees with the assembly, fix measure_ii** (`src/artlite/measure_ii.py`, tests in
  `tests/test_measure_ii.py`). It is the tool everyone reads.

## Remark flags (clang)

```
-Rpass=postpipeliner
-Rpass-missed='postpipeliner|aie-multi-slot-pseudo'
-Rpass-analysis='aie-hardware-loops|aie-asm-printer'
-fsave-optimization-record -foptimization-record-file=<path>
```

| pass | reports |
|---|---|
| `postpipeliner` Passed / Missed | II, NS, prologue / epilogue bundles / why no schedule ("Longest circuit does not fit II") |
| `aie-hardware-loops` | per loop `Zero-Overhead-Loop: true/false` |
| `aie-asm-printer` | per block `BundleCount` (= cycles for straight-line code; = II for a pipelined body) and `ByteCount` |
| `aie-multi-slot-pseudo` | a load without a DM bank annotation |

Debug builds of llvm-aie (`-mllvm -debug-only=...`) show why the post-pipeliner rejects an II (`variant_search --why`
runs one).

## Warnings (no flags needed)

- `-Wpass-failed`: an `AIE_*` pragma was dropped.
- `No memory bank assigned to load in function ...`: add a `__aie_dm_resource_*` qualifier and keep it through casts.
- `checks.py` (every compile, AGENTS.md): stack, kernel signatures, rounding, runtime-library arithmetic.

## Assembly checks (`kernel.cc.s`, `ARTLITE_GENERATE_ASM=1`)

- **Loop body**: the lines between the loop label and `.L_LEnd*`, one bundle per line. Count the per-slot ops there
  (machine-model.md).
- **Spills**: `Folded Spill` / `Folded Reload` inside the body mean the reported II is not what you pay; pre-RA
  `NS > 4` usually means spills too. Restructure (fewer live accumulators, restructurings.md R1) or switch pipeliner.
- **Frame size**: `paddxm [sp], #N` in the prologue. Overflow corrupts the neighbouring buffer silently, often in an
  alternating-buffer (parity) pattern (aie2p-coding section 9).
- **Calls**: `jl` to `__divsi3`, `__mulsf3`, `__gtsf2`, ... in a hot function means division or scalar float math
  survived (`llvm-nm X.o | grep __` lists them).
- **Switches**: a jump table or a chain of compares and branches in a loop body usually comes from a runtime lane
  index (`extract<8>(j)`). Unroll the loop.
- **Size**: `llvm-nm --print-size --size-sort` on `kernel.o` / the core ELF. Sudden growth means spills, an
  unexpected full unroll or a soft-float library pulled in (program memory: SKILL.md section 7).
- **ZOL**: measure_ii's "not a zero-overhead loop" can be wrong: check the `.s` for the `ls` / `le` setup.

## Symptom -> cause -> fix

| symptom | usual cause | fix |
|---|---|---|
| pre-RA pipelined, body > II or spills | register allocation broke the pre-RA plan | `AIE_PREPARE_FOR_POSTPIPELINING` |
| post-RA II far above ResMII, "out of resources" at every II | a shared port (shuffles + panel pushes on L_RM_PORT; variant_search's `port` column) | fewer shuffles: broadcast + transpose expansions, transposes before the arithmetic (restructurings.md R7) |
| post-RA II far above ResMII, no spills | a value carried across iterations, a pointer used at both ends of the iteration, or an unproven memory dependence | load inside the iteration; separate walking load / store pointers; distinct `__restrict` parameters (SKILL.md section 4) |
| alu-bound loop (`lshl`, `and`, `or`) | indices decoded from a flat loop counter | `add_2d_ptr` / `add_3d_ptr` walks (restructurings.md R8) |
| "Minimal Initiation Interval too large" / RecMII = the whole chain | loads and stores through pointers into one buffer (no noalias scopes) | the loop in its own function with distinct `__restrict` parameters (SKILL.md section 2) |
| gathers (`vldb.4x64`) in a loop that does not pipeline: RecMII through the next iteration's loads | the gathers carry no memory operand, so each orders against every load and store: llvm-aie gives them one only when each base of the address vector is a global or a pointer parameter of its own (not one pointer plus an offset, not one pointer used twice) | static tables, or each table copy a parameter (`km.table_pass(table_arg=True)`: II 49 -> 22) |
| not pipelined, "Not a single basic block" | a branch, a switch (runtime lane index) or a call in the body | select instead of if; unroll lane loops; hoist the branch; loop fission |
| not pipelined, "The loop structure is not supported" | a runtime trip count or a call-like construct | constant trip count (Jinja), mask the tail |
| a long elementwise chain, II = its latency | fp32-emulated steps in one dependence chain | the LUT / identity rewrite or loop fission (restructurings.md R3) |
| II jumps after adding a stream | a third bank shares a load slot with the heaviest stream, or a third panel stream | reorder the first loads or change bank annotations (machine-model.md); two panel streams per loop (SKILL.md section 4) |
| ZOL false | several exits or an unknown trip count | constant trip count |
| large prologue + epilogue vs `iters * II` | high NS on a short trip | fewer stages, a longer trip (merge loops), or accept |
| fine schedule, slow on the NPU (cycles per window 20-30% above) | bank conflicts, read-modify-write of L1 accumulators, spills next to streams | accumulators in registers; `pin_l1_buffers` (bank-conflicts.md) |
| a compiler crash, wrong results only in some variants | a Peano / aie_api trap | aie2p-coding section 9 |
| `section '.text' will not fit in region 'program'` | inlined copies of big helpers, unrolls, control code | SKILL.md section 7 |
| stack check fails after a change | an inlined fp32 chain or an unroll of pipelined code | `noinline` the helper; re-measure `stack N B` |
