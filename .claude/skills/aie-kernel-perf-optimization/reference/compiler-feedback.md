# Reading compiler feedback

## Contents
- measure_ii: the fast path
- Remark flags for a clang build
- Warnings that always print
- Assembly checks
- Symptom -> cause -> fix

## measure_ii: the fast path

- **An operator**: run `ARTLITE_MEASURE_II=1 python -c "from artlite.operators import X; X(...).compile()"`, or the
  operator's pytest with the variable set; `ARTLITE_IGNORE_CACHE=1` forces a rebuild. The report gives, per function,
  stack and code size; per loop, the scheduler, II, NS, ZOL, iterations, cost with calls, resource bound, and spills;
  and the paths of the `.cc`, `.s`, annotated `.s`, and remarks YAML in `rtc_<hash>/` of the cache
  (`artlite-cache path`).
- **A standalone `.cc`** starting with `#include "operators.h"`: call `artlite.measure_ii.measure(path)` (pass
  `clang_cmd` when the operator adds flags).
- The fields, the printed lines, and which II numbers are real are described in the **measure-ii** skill. In short:
  only the final `!Passed pipeliner/schedule` remark (after register allocation) describes the code that runs, and
  multi-block loops are never pipelined.
- **If measure_ii disagrees with the assembly, fix measure_ii** (`src/artlite/measure_ii/`, tests in
  `tests/test_measure_ii.py`). It is the shared source of truth.

## Remark flags (clang)

```
-Rpass=postpipeliner
-Rpass-missed='postpipeliner|aie-multi-slot-pseudo'
-Rpass-analysis='aie-hardware-loops|aie-asm-printer'
-fsave-optimization-record -foptimization-record-file=<path>
```

| pass | reports |
|---|---|
| `postpipeliner` Passed / Missed | II, NS, prologue / epilogue bundles, or why no schedule was found ("Longest circuit does not fit II") |
| `aie-hardware-loops` | `Zero-Overhead-Loop: true/false` per loop |
| `aie-asm-printer` | `BundleCount` per block (= cycles for straight-line code; = II for a pipelined body) and `ByteCount` |
| `aie-multi-slot-pseudo` | a load without a DM bank annotation |

Debug builds of llvm-aie (`-mllvm -debug-only=...`) show why the post-pipeliner rejects an II (`variant_search --why`
runs one).

## Warnings (no flags needed)

- `-Wpass-failed`: an `AIE_*` pragma was dropped.
- `No memory bank assigned to load in function ...`: add a `__aie_dm_resource_*` qualifier and keep it through casts.
- `checks.py` (runs on every compile, `docs/development.md` "Operators"): checks stack, kernel signatures, rounding,
  and runtime-library arithmetic.

## Assembly checks (`kernel.cc.s`, `ARTLITE_GENERATE_ASM=1`)

- **Loop body**: the lines between the loop label and `.L_LEnd*`, one bundle per line. Count the per-slot ops there
  (`machine-model.md`).
- **Spills**: `Folded Spill` / `Folded Reload` inside the body mean the reported II is not what you pay; a pre-RA
  `NS > 4` usually means spills too. Restructure (fewer live accumulators, `restructurings.md` R1) or switch
  to the other pipeliner.
- **Frame size**: `paddxm [sp], #N` in the prologue. A stack overflow silently corrupts the neighbouring buffer, often in
  an alternating-buffer (parity) pattern (`aie2p-coding` section 9).
- **Calls**: `jl` to `__divsi3`, `__mulsf3`, `__gtsf2`, ... in a hot function means that division or scalar float math
  survived (`llvm-nm X.o | grep __` lists them).
- **Switches**: a jump table or a chain of compares and branches in a loop body usually comes from a runtime lane index
  (`extract<8>(j)`). Unroll the loop.
- **Size**: run `llvm-nm --print-size --size-sort` on `kernel.o` or the core ELF. Sudden growth means spills, an
  unexpected full unroll, or a soft-float library pulled in (program memory: `SKILL.md` section 7).
- **ZOL**: measure_ii's "not a zero-overhead loop" can be wrong; check the `.s` for the `ls` / `le` setup.

## Symptom -> cause -> fix

| symptom | usual cause | fix |
|---|---|---|
| pre-RA pipelined, but body > II or spills | register allocation broke the pre-RA plan | `AIE_PREPARE_FOR_POSTPIPELINING` |
| post-RA II far above ResMII, "out of resources" at every II | a shared port is saturated (shuffles + panel pushes on `L_RM_PORT`; `variant_search`'s `port` column) | use fewer shuffles: broadcast + transpose expansions, transposes before the arithmetic (`restructurings.md` R7) |
| post-RA II far above ResMII, no spills | a loop-carried dependency, a pointer kept live across the whole iteration, or unprovable memory independence | load inside the iteration; use separate walking load / store pointers; use distinct `__restrict` parameters (`SKILL.md` section 4) |
| alu-bound loop (`lshl`, `and`, `or`) | indices decoded from a flat loop counter | use `add_2d_ptr` / `add_3d_ptr` walks (`restructurings.md` R8) |
| "Minimal Initiation Interval too large" / RecMII = the whole chain | loads and stores through pointers into one buffer (no noalias scopes) | move the loop into a separate function with distinct `__restrict` parameters (`SKILL.md` section 2) |
| gathers (`vldb.4x64`) in a loop that does not pipeline: RecMII runs through the next iteration's loads | the gathers carry no memory operand, so each one is ordered against every load and store; llvm-aie gives them one only when each base in the address vector is a separate global or pointer parameter (not one pointer plus an offset, not one pointer used twice) | use static tables, or pass each table copy as a parameter (`km.table_pass(table_arg=True)`: II 49 -> 22) |
| not pipelined, "Not a single basic block" | a branch, a switch (runtime lane index), or a call in the body | use a select instead of an if; unroll lane loops; hoist the branch; split the loop (fission) |
| not pipelined, "The loop structure is not supported" | a runtime trip count or a call-like construct | use a constant trip count (Jinja) and mask the tail |
| a long elementwise chain with II = its latency | fp32-emulated steps in one dependence chain | use a LUT or algebraic identity rewrite, or split the loop (`restructurings.md` R3) |
| II jumps after adding a stream | a third bank shares a load slot with the heaviest stream, or a third panel stream was added | reorder the first loads or change bank annotations (`machine-model.md`); keep two panel streams per loop (`SKILL.md` section 4) |
| ZOL false | several exits or an unknown trip count | use a constant trip count |
| large prologue + epilogue vs. `iters * II` | high NS on a short trip count | use fewer stages or a longer trip (merge loops), or accept it |
| good schedule, but 20-30% more cycles per window on the NPU than static cycles | bank conflicts, read-modify-write of L1 accumulators, spills next to streams | keep accumulators in registers; `pin_l1_buffers` (`bank-conflicts.md`) |
| a compiler crash, or wrong results only in some variants | a Peano / aie_api trap | `aie2p-coding` section 9 |
| `section '.text' will not fit in region 'program'` | inlined copies of big helpers, unrolls, control code | `SKILL.md` section 7 |
| stack check fails after a change | an inlined fp32 chain, or unrolling pipelined code | mark the helper `noinline`; re-measure `stack N B` |
