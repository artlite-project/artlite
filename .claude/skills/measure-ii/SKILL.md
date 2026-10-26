---
name: measure-ii
description: "How to use artlite.measure_ii to compile an AIE C++ kernel file and extract the final per-loop schedule (II, NS, ZOL, trip counts, cycle cost, spills, resource bound) plus compiler optimization remarks; and artlite.variant_search, which ranks variants of an operator's functions (pragmas, rewrites, parameters) per loop with slot and port counts, post-pipeliner failure reasons and optional NPU timing. Use when reading an ARTLITE_MEASURE_II report, measuring a loop's II, comparing pragma or source variants, or when measure_ii's output disagrees with the assembly."
---

# Using `artlite.measure_ii` with an arbitrary C++ file

`artlite.measure_ii` wraps Peano compilation for AIE2P and parses the LLVM
optimization record YAML plus the emitted assembly, so you can quickly answer:

- What is the **final initiation interval (II)** of each loop?
- Which pipeliner produced the loop (**pre-RA** / **post-RA**), or was it left
  unpipelined (and why)?
- Is it a **hardware zero-overhead loop (ZOL)** and **how many times does it
  iterate** (the `lc` count is recovered from the assembly)?
- What is the loop cost in cycles: `prologue + iters*II + epilogue`?
- Are there **spills/reloads** or missing memory-bank annotations?

## What the numbers mean (important!)

The loops are those of the final code: the assembly's loop comments
(`=>This Loop Header: Depth=N`, `in Loop: Header=`) give every loop, its
nesting and its blocks, also the branch loops (an `if`, a `break`, an outer
loop) that no scheduling remark describes.

The Peano pipeline reports several II numbers per loop; only one describes the
code that actually runs:

- **`!Passed pipeliner/schedule` with a `Pipeliner:` field** (emitted by
  `AIEInterBlockScheduling`, after both pipeliners *and* register allocation)
  is authoritative, and exists for single-block loops only.
  `Pipeliner: postpipeliner` → post-RA pipelined (II/NS from
  the schedule); `Pipeliner: prepipeliner` → pre-RA pipelined (II = actual
  final loop-region length, usually search-II plus the moves regalloc
  inserted); `Pipeliner: loop-aware` → the single-block loop stayed unpipelined
  (II = body length). Prologue/Epilogue bundle counts included in all cases.
  A loop of several blocks is never pipelined: the tool lists it with its
  own bundles (`N blocks + K inner loops, body B bundles`).
- The `!Analysis pipeliner/schedule` remark ("Schedule found with Initiation
  Interval") is the pre-RA MachinePipeliner **search plan**, made before code
  generation and register allocation. It is routinely 1-2 cycles smaller than
  the final loop. The tool shows it only as a `note:`, never as the loop's II.
- Pre-RA rejections ("Unable to find schedule", "The loop structure is not
  supported", "Not a single basic block: N") are shown **only for loops that
  stayed unpipelined**; a pre-RA failure followed by a successful post-RA
  schedule is an intermediate step and is suppressed.
- Source lines (`@kernel.cc:L`, `loop.kernel_line`) are the loop *statement*
  (its `for`/`while`/`do`). The pre-RA remarks name loops only by a source
  location; the tool matches them with the final loops by the source lines of
  each loop's instructions (`.loc` in `<src>.annotated.s`), the statement
  nesting (within one function body: a loop of an inlined helper lies outside
  its caller's loop statement), trip and block counts. A loop no remark names
  (an outer loop the pre-RA pipeliner never reported) takes the innermost
  unmatched loop statement around its own lines. The compiler sometimes reports an outer loop
  at the location of the inner loop that starts its body (or at a branch in
  the body); the tool moves it to its own statement. `@?`: no source loop
  matched this loop (its schedule is still right). `loop @L: unattributed`: a
  pre-RA loop that matches no loop of the final code (e.g. fully peeled).
- A bundle's source lines are its `.loc`'s inlining chain (the comment
  `vector.hpp:1098:29 @[ kernel_lib.h:41:27 @[ kernel.cc:315:9 ] ]`); the
  innermost frame in a loop statement is the loop's own code. A header
  helper's loop inlined into the measured file prints its call site:
  `for.body.i @kernel_lib.h:40 via kernel.cc:315` (`loop.call_site`), which
  tells the copies of one helper apart.

Trip counts:

- A for-header bounded by a template parameter (`c < N`) takes the instance's argument from the demangled
  name (`conv8<256>`).

- A hardware loop executes exactly `lc` times; `lc` is set by
  `add.nc lc, rX, #imm` and read with start-of-bundle semantics (a `mov` to rX
  in the same bundle is invisible; any other write to rX, e.g. `lshl`, `mul`,
  makes the count unknown). The tool binds each `lc` setup to the next ZOL
  header (the label written to `ls`). Register values flow along the control
  flow (jumps, fall-through, a ZOL's back edge): a branch of an if-chain sees
  the value set before its jump; a call clobbers the return-value registers
  (r0-r7, p0-p5), the compiler keeps others across calls it knows.
- For pipelined loops the pipeliner peels `NS-1` iterations into the
  prologue/epilogue, so the source trip count is `lc + (NS-1)`.
- When there is no ZOL, the tool falls back to parsing the
  `for (i = A; i < B; i += S)` header from the kernel source (constants only;
  also `<=`, `!=`, `i++`) and marks it `(from source)` / `[iters from src]`;
  otherwise the user-supplied `trip_count` is used (`@32 iters = ...`). A
  `break`, `return` or `goto` in the loop makes that count a maximum:
  `may exit early`, the cost becomes `<=`.
- An iteration count that contradicts the source-level work reveals kernel
  bugs: 74 iterations of a 7-store loop for 512 stores is an out-of-bounds
  tail (use a per-iteration unroll factor that divides the element count
  exactly).

## Environment

```bash
source .venv/bin/activate  # in the repo root
```

## Minimal example

```python
from pathlib import Path
from artlite.measure_ii import measure

report = measure(Path("/path/to/kernel.cc"), verbose=True)
```

## Signature

```python
def measure(
    source: str | Path,
    *,
    clang_cmd: list[str] | None = None,
    verbose: bool = True,
    trip_count: int = 32,
) -> KernelReport:
```

- `clang_cmd`: optional full `clang++` argument list; defaults to
  `artlite.operators.base.CLANG_CMD_COMMON` + `AIE_TARGET`. Must match the
  operator's flags (e.g. defines like `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`)
  or the numbers describe different code than what runs.
- `trip_count`: fallback for loops whose trip count cannot be recovered.
- Identical (source text, clang_cmd) pairs are measured once per process;
  repeats print a one-line skip notice.

## Reading the `KernelReport`

- `report.loops`: list of `LoopReport`, one per loop of the final code.
  - `loop.function`: the symbol (mangled for C++ functions); `loop.block`: IR
    name of the header block; `loop.header`: its asm label (`.LBBf_n`).
  - `loop.depth`, `loop.parent`, `loop.inner` (header labels),
    `loop.num_blocks` (inner loops' blocks included).
  - `loop.sched`: `"post-RA"` / `"pre-RA"` / `"none"` (final disposition).
  - `loop.ii`, `loop.ns`, `loop.prologue_bundles`, `loop.epilogue_bundles`
    (`ii` is None for a loop of several blocks); `loop.prologue_of` /
    `loop.epilogue_of`: the neighbour loop that shares the block (below),
    `loop.prologue_cycles`: the prologue in this loop's cost (0 when shared).
  - `loop.zol`: hardware-loop status; `loop.hw_iters`: recovered `lc`;
    `loop.trip_count`: implied source trip count.
  - `loop.kernel_line`: loop-statement line in the compiled source (None:
    unknown, or the loop is in a header, see `loop.source_ref` and
    `loop.call_site`);
    `loop.source_trip`: trip count parsed from that statement;
    `loop.early_exit`: a break/return/goto makes it a maximum.
  - `loop.estimated_cycles()` / `loop.cost_formula()`: `P + iters*II + E`,
    bundles only (no calls, no inner loops).
  - `loop.body_bundles`: ground truth from `aie-asm-printer`, the loop's own
    blocks (not its inner loops'); for a pipelined loop it must equal `ii`
    (mismatch is an issue). `loop.branches`: conditional jumps besides the
    back edge.
  - `loop.search_ii`: stale pre-RA search plan (reference only).
  - `loop.prereject_reasons`.
  - `loop.loop_spills`, `loop.loop_reloads` (+ `spill_bytes`,
    `reload_bytes`): spill/reload *instructions* in the loop's own blocks;
    `loop.regalloc_spills`, `loop.regalloc_reloads`: the register allocator's
    count of spilled/reloaded *registers* for the loop, inner loops included.
  - `loop.loop_calls` (names) / `loop.callees` (symbols): calls per iteration.
- `report.unattributed_loops`: pre-RA loops that match no final loop.
- `report.pipelined_loops`, `report.best_ii`, `report.best_post_ra_ii`.
- `report.blocks`: bundle/byte sizes per IR block name; `report.layout`: per
  function, per block in layout order (the printout folds sizes into
  per-function `code N bundles / M B` totals); a scheduled loop's header block
  has its `schedule` (II, NS), which places the final remark on it.
- `report.issues`: deduplicated, actionable warnings only.

## Interpreting the printed report

```
fused_dequant_dot_core_2048_4_g32<32, 2048>, stack 0 B, code 108 bundles / 544 B
  for.body @kernel.cc:21: post-RA pipelined, II=14, NS=3, ZOL, 30 iters (trip 32: 2 peeled)
    cost 31 + 30*14 + 63 = 514 cy [iters from ZOL]; body 14 bundles (112 B)
```

- `II=14, NS=3`: final schedule; one loop iteration every 14 cycles, 3
  overlapping stages.
- `ZOL, 30 iters (trip 32: 2 peeled)`: hardware loop counter runs 30 times;
  the source loop has 32 iterations, of which `NS-1 = 2` were peeled into the
  prologue/epilogue.
- `cost = prologue(31) + iters*II + epilogue(63)`: cycles spent per entry into
  this loop. Epilogue 63 of 514 cycles here is worth reducing: that is what
  the "ramp+drain" note flags when `prologue+epilogue` exceeds the steady
  state.
- `body 14 bundles (112 B)` from the asm-printer cross-checks II and shows
  program-memory cost.

A one-block loop also prints its resource bound: the busiest issue slot or
shared register / memory port of its body (the counts of `variant_search`,
tables in `artlite/aie2p_isa.py`), with the ops of one or two binding ones:

```
gq_dq_5e0a71c2, stack 256 B, code 116 bundles / 704 B
  for.body @kernel.cc:583: post-RA pipelined, II=40, NS=2, ZOL, 7 iters (trip 8: 1 peeled)
    cost 51 + 7*40 + 25 = 356 cy [iters from ZOL]; body 40 bundles (272 B)
    res bound 40 (rm: vshuffle 24, vst.flush.512.conv 8, vst.push.576.conv.bfp16ebs8.fp32 8); slots lda 5, ldb 9, st 16, vec 12, mv 28
```

II near it: resource-bound (fewer ops on that slot or port); II far above it:
a recurrence, register pressure or the scheduler (`variant_search --why`).

The remark's prologue and epilogue are whole blocks (the preheader, the
exit). Between consecutive loops one block is both; it counts in the first
loop's cost only, the second prints it in parentheses:

```
for.body @kernel.cc:122: post-RA pipelined, II=12, NS=2, ZOL, 15 iters (trip 16: 1 peeled)
  cost 13 + 15*12 + 15 = 208 cy [iters from ZOL; epilogue shared: the prologue of for.body13 @kernel.cc:125]
for.body13 @kernel.cc:125: post-RA pipelined, II=12, NS=2, ZOL, 15 iters (trip 16: 1 peeled)
  cost (15) + 15*12 + 6 = 186 cy [iters from ZOL; prologue shared: counted in for.body @kernel.cc:122]
```

`not pipelined, body N bundles, ZOL (reason)` lines: the loop runs
sequentially; `body bundles` is its per-iteration cost. Outer call-loops
around a noinline core are expected to look like this.

Function names are demangled with their template arguments (`srec<32, 4>`,
the anonymous namespace dropped). Inner loops are indented under their outer
loop. An inlined function's exit block is shortened (`pop.exit186.7`) and
followed by its asm label, to find it in `<src>.s`:

```
gmp0f3dcf07_srec<32, 4>, stack 0 B, code 117 bundles / 832 B
  for.cond4.preheader @gp2.cc:252: not pipelined, 2 blocks + 1 inner loop, body 57 bundles, no ZOL (Not a single basic block: 3)
    cost 2*(57 + 114) = 342 cy [iters from src]
    pop.exit186.7 (.LBB17_2) @gp2.cc:254: post-RA pipelined, II=38, NS=2, ZOL, 3 iters (trip 4: 1 peeled)
      cost 40 + 3*38 + 17 = 171 cy [iters from ZOL]; body 38 bundles (336 B)
```

- `N blocks + K inner loops, body B bundles`: a loop of several blocks (never
  pipelined); `B` counts its own blocks, which hold its inner loops' prologue
  and epilogue.
- Its cost is `iters*(B + inner loops' iterations + calls)`: here
  `2*(57 + 3*38)`. The inner loop's own line shows the cost of one entry
  (`40 + 3*38 + 17`).
- `<=` with `[all paths]`: the loop has branches (an `if`), every block is
  counted in every iteration, an upper bound. `[may exit early]`: a `break`
  (or the like), the trip count is a maximum.

Calls are not in the bundle count; each loop that calls lists them with the
callee's cycles per call (its code outside loops + its loops + its calls,
`<=` when it branches), and its cost adds them:

```
if.end28 @kernel.cc:776: not pipelined, body 18 bundles, no ZOL (The loop structure is not supported)
  cost <= 7 + 5*(18 + 641) + 8 = 3310 cy [iters from src; with calls; may exit early]
  ⚠ calls per iteration (not in the bundle count): gmp5df2e9e1_unit (641 cy)
```

`[excluding calls]`: a callee's cost is unknown (a runtime-library function
such as `__divsi3`, or a trip count in the callee): the cost counts the loop's
bundles only. `cost(iters) ...; @32 iters = X cy`: the trip count is unknown,
`X` uses the `trip_count` fallback.

Spill code shows two counts, of different things:

```
⚠ spill code in the loop body: 12 spill store(s) (768 B), 16 reload load(s) (1024 B); the register allocator: 3 spill(s), 4 reload(s) of registers, inner loops included (a register wider than 64 B moves in 64-byte parts)
```

- The first are the asm instructions in the loop's own blocks (their `N-byte
  [Folded] Spill/Reload` comments): what the loop executes each iteration.
- The second is the register allocator's remark for the loop statement:
  registers, before they are split into 64-byte moves, and summed over inner
  loops. A 2048-bit accumulator (`dm`) spill is 1 there and 4 stores in the
  asm (as here: 12 = 3*4, 16 = 4*4). The function-level remark is printed as
  `(function @L): the register allocator spilled S, reloaded R register(s)`.

## Variant search (variants of an operator's functions)

Comparing pragmas or rewrites of a loop: use `artlite.variant_search`, not a throwaway script. It
renders the operator's kernel, patches the target functions per variant, measures all variants in
parallel and ranks them in one table per loop.

```bash
python -m artlite.variant_search 'tests.decode.mmv.MmvOperator(fmt="Q5_K", n_tiles=4)' \
    --fn _q5k --pragmas none [--sub NAME OLD NEW] [--why] [--npu --region codes=_q5k]
```
```
mmvh_q5k_mm loop 1/1: for.body @kernel.cc:261, depth 1 (in base); ranked by cycles per entry
  variant  pragma    sched  II  NS  iters  cycles  body  spill  stack  lda  ldb  st  vec   mv  alu  lng    port  res bound
  base             post-RA  30   2    7+1     266    30      -      0    9   16   2   19  30*    0    0  qwb 16      30 mv
  none       none   pre-RA  38   2    7+1     325    38    0/3    128    9   19   2   19  28*    0    0  qwb 18      28 mv
  base: mv 30: mov 28, vmov 2
        post-pipeliner (debug llvm-aie): ResMII 30; tried II 30..30 -> II 30 (Physical)
          Physical: RecMII 27
  none: mv 28: mov 26, vmov 2
        post-pipeliner (debug llvm-aie): not a post-pipeliner candidate: No tripcount def
```

- Variants (a name collects several flags; `base` = as rendered, always first; `@file` reads text):
  `--pragmas` shorthands `post`, `none`, `iiN`, `post+iiN` replace the pipelining pragma of the
  innermost rolled loops of the targets and of the `always_inline` helpers they call (`_mac` under `_mm`; a
  helper's pragma applies to all its callers; a loop that calls a helper with loops is not innermost);
  `--loop K` picks others (K counts the targets' rolled loops in source order, then the helpers');
  `--sub V OLD NEW` (must match once);
  `--define V FN @def.cc` (a whole function); patches may name functions as `kernel.cc` does (`<stem>_<name>_<hash>`:
  the hash follows the code, so a patched function and its callers get new ones) or as rendered (`__artlite_<stem>_<name>`);
  `--param V key=value` (an operator / Jinja argument);
  `--variants FILE.py` with `VARIANTS = [Variant(...)]`.
- Program memory: the summary's `code B` is the whole object. For one core's bytes, walk the call graph (`jl`,
  tail `j`) from its entry kernels in each variant's `kernel.cc.s` with the report's per-function block bytes
  (scriptable from `search()`; aie-kernel-perf-optimization section 7).
- Columns: `iters` = ZOL count + peeled iterations; `cycles` per entry with prologue and epilogue; the
  loop body's ops per slot (llvm-aie's slot of each mnemonic; a `lng` op takes alu and mv); `port`: the busiest
  register / memory port that ops of several slots share (`rm` L_RM_PORT: shuffles, selects, broadcasts and every
  bfp16 push / flush; `wm`, `qwa`, `qwb`, `qrs`, `dwm`, `ada`: aie-kernel-perf-optimization
  `reference/machine-model.md`); `res bound` = the busiest slot or port, `*` on the binding ones, whose ops are
  listed below the table. A loop with II far above its slots and "out of resources" at every II is often port-bound.
- A one-block loop left unpipelined (every loop with `--why`) is compiled again by the debug build of
  llvm-aie (`ARTLITE_DEBUG_CLANG`, default `../llvm-aie/build/bin/clang++`): ResMII, the IIs tried,
  RecMII / ScarceRegMII per mode, what failed at each II and the chain of the most frequent "latency
  not met". The search stops at II 60 (`-aie-postpipeliner-maxii`) or after 20 IIs (`-maxtry-ii`).
  The report warns when that build's llvm-aie commit is not the wheel's (and lists the commits between them
  that touch the AIE backend or CodeGen): its traces may then differ from the shipped schedule.
- Slot and port tables: `artlite/aie2p_isa.py`, generated from llvm-aie's TableGen at the wheel's commit by
  `python -m artlite.gen_isa_tables` (after a toolchain upgrade; `test_variant_search.py` checks them when the
  `../llvm-aie` checkout and its `llvm-tblgen` are there).
- `--npu`: the builds compile in parallel, then each runs alone in its own process: the time of one
  dispatch (median, p10 over `--reps`) and, per `--region NAME=FN[*N]`, the counters of the operator's
  `win=NAME` build (cycles, windows, memory and lock stall) next to the static cycles of N calls of FN.
- Python: `search(factory, kwargs, functions=..., variants=...)`, `print_report`, `run_npu`.

## Troubleshooting

- **Compilation fails**: `report.returncode`, `report.diagnostics`.
- **Loop not pipelined**: read the printed reason. Typical fixes: flatten
  branches ("Not a single basic block"), remove call-like constructs
  ("loop structure is not supported"), or accept it when pre-RA fails but
  post-RA delivered a schedule.
- **Spills appear**: per-loop `spill code in the loop body` (asm and
  register-allocator counts); prefer `__restrict`, DM-bank annotations,
  smaller unroll factors, or `AIE_PREPARE_FOR_POSTPIPELINING`.
- **`@?` or `unattributed`**: the loop's schedule is right but its source line
  could not be tied to it; check the asm label and the `.loc` lines in
  `<src>.annotated.s`, and improve the matching in `measure_ii.py`.
- **Missing bank annotations**: add `__aie_dm_resource_a/b/c/d` qualifiers.

## Improvements

measure_ii is error-prone. When the assembly or the compiler's remarks disagree with its report, do not just note
it: fix `src/artlite/measure_ii.py` (tests in `tests/test_measure_ii.py`), so that the next report is right.