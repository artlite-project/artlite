---
name: measure-ii
description: "How to use artlite.measure_ii to compile an AIE C++ kernel file and extract the final per-loop schedule (II, NS, ZOL, trip counts, cycle cost, spills, resource bound) plus compiler optimization remarks; and artlite.variant_search, which ranks variants of an operator's functions (pragmas, rewrites, parameters) per loop with slot and port counts, post-pipeliner failure reasons and optional NPU timing. Use when reading an ARTLITE_MEASURE_II report, measuring a loop's II, comparing pragma or source variants, or when measure_ii's output disagrees with the assembly."
---

# Using `artlite.measure_ii` with an arbitrary C++ file

`artlite.measure_ii` wraps Peano compilation for AIE2P and parses the LLVM optimization record YAML plus the emitted
assembly, so you can quickly answer:

- What is the **final initiation interval (II)** of each loop?
- Which pipeliner produced the loop (**pre-RA** / **post-RA**), or was it left unpipelined (and why)?
- Is it a **hardware zero-overhead loop (ZOL)**, and **how many times does it iterate** (the `lc` count is recovered
  from the assembly)?
- What is the loop cost in cycles: `prologue + iters*II + epilogue`?
- Are there **spills/reloads** or missing memory-bank annotations?

Every shipped kernel's measurements are kept per app in `tests/schedule_ledger.jsonl`. In `tests/schedule_ledger.py`,
`record` compares the tree against it and `rank APP` weights a prefill chunk's loops; `test_app_designs.py` checks it.
A change to a kernel shows up there as the loops and functions whose schedule moved.

## What the numbers mean (important!)

The loops listed are those of the final code. The assembly's loop comments (`=>This Loop Header: Depth=N`,
`in Loop: Header=`) give every loop with its nesting and its blocks, including loops with branches (an `if`, a
`break`, an outer loop) that no scheduling remark describes.

The Peano pipeline reports several II numbers per loop; only one of them describes the code that actually runs:

- **`!Passed pipeliner/schedule` with a `Pipeliner:` field** (emitted by `AIEInterBlockScheduling`, after both
  pipeliners *and* register allocation) is authoritative, and it exists only for single-block loops.
  `Pipeliner: postpipeliner` means post-RA pipelined (II / NS come from the schedule); `Pipeliner: prepipeliner` means
  pre-RA pipelined (II is the actual length of the final loop region, usually the search II plus the moves that
  register allocation inserted); `Pipeliner: loop-aware` means the single-block loop stayed unpipelined (II is the body
  length). Prologue and epilogue bundle counts are included in all cases. A loop with several blocks is never
  pipelined; the tool lists it with the bundles of its own blocks (`N blocks + K inner loops, body B bundles`).
- The `!Analysis pipeliner/schedule` remark ("Schedule found with Initiation Interval") is the pre-RA
  MachinePipeliner's **search plan**, made before code generation and register allocation. It is routinely 1-2 cycles
  shorter than the final loop. The tool shows it only as a `note:`, never as the loop's II.
- Pre-RA rejections ("Unable to find schedule", "The loop structure is not supported", "Not a single basic block: N")
  are shown **only for loops that stayed unpipelined**; a pre-RA failure followed by a successful post-RA schedule is
  an intermediate step and is suppressed.
- Source lines (`@kernel.cc:L`, `loop.kernel_line`) point at the loop *statement* (its `for` / `while` / `do`). The
  pre-RA remarks identify loops only by a source location, so the tool matches them to the final loops using the
  source lines of each loop's instructions (`.loc` in `<src>.annotated.s`), the statement nesting (within one function
  body; a loop of an inlined helper lies outside its caller's loop statement), and the trip and block counts. A loop
  that no remark names (such as an outer loop that the pre-RA pipeliner never reported) is assigned the innermost
  unmatched loop statement that encloses its lines. The compiler sometimes reports an outer loop at the location of
  the inner loop that starts its body (or at a branch in the body); the tool moves it to the correct statement. `@?`
  means no source loop matched this loop (its schedule is still correct). `loop @L: unattributed` is a pre-RA loop that
  matches no loop of the final code (for example, because it was fully peeled).
- A bundle's source lines come from the inlining chain of its `.loc` (the comment
  `vector.hpp:1098:29 @[ kernel_lib.h:41:27 @[ kernel.cc:315:9 ] ]`); the innermost frame inside a loop statement
  marks code that belongs to that loop. A loop from a header helper that is inlined into the measured file prints its
  call site, `for.body.i @kernel_lib.h:40 via kernel.cc:315` (`loop.call_site`), which distinguishes the copies of one
  helper.

Trip counts:

- For a `for` header bounded by a template parameter (`c < N`), the tool takes the instance's argument from the
  demangled name (`conv8<256>`).

- A hardware loop executes exactly `lc` times. `lc` is set by `add.nc lc, rX, #imm` and read with start-of-bundle
  semantics: a `mov` to `rX` in the same bundle is invisible, and any other write to `rX` (e.g. `lshl`, `mul`) makes
  the count unknown. The tool binds each `lc` setup to the next ZOL header (the label written to `ls`). Register values
  are tracked along the control flow (jumps, fall-through, a ZOL's back edge): a branch of an if-chain sees the value
  set before its jump, and a call clobbers the return-value registers (`r0-r7`, `p0-p5`), while the compiler preserves
  the others across calls it knows.
- For pipelined loops, the pipeliner peels `NS-1` iterations into the prologue / epilogue, so the source trip count is
  `lc + (NS-1)`.
- When there is no ZOL, the tool falls back to parsing the `for (i = A; i < B; i += S)` header from the kernel source
  (constants only; `<=`, `!=` and `i++` also work) and marks the result `(from source)` / `[iters from src]`;
  otherwise it uses the user-supplied `trip_count` (`@32 iters = ...`). A `break`, `return` or `goto` in the loop makes
  that count a maximum: the report says `may exit early`, and the cost becomes `<=`.
- An iteration count that contradicts the source-level work reveals kernel bugs: 74 iterations of a 7-store loop for
  512 stores means an out-of-bounds tail (use a per-iteration unroll factor that divides the element count exactly).

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

- `clang_cmd`: an optional full `clang++` argument list; defaults to
  `artlite.operators.toolchain.clang_cmd_common()` + `AIE_TARGET`. It must match the operator's flags (e.g. defines
  like `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`); otherwise the numbers describe different code than what runs.
- `trip_count`: the fallback for loops whose trip count cannot be recovered.
- Identical (source text, `clang_cmd`) pairs are measured once per process; repeats print a one-line skip notice.

## Reading the `KernelReport`

- `report.loops`: a list of `LoopReport`, one per loop of the final code.
  - `loop.function`: the symbol (mangled for C++ functions); `loop.block`: the IR name of the header block;
    `loop.header`: its asm label (`.LBBf_n`).
  - `loop.depth`, `loop.parent`, `loop.inner` (header labels), `loop.num_blocks` (including inner loops' blocks).
  - `loop.sched`: `"post-RA"` / `"pre-RA"` / `"none"` (the final disposition).
  - `loop.ii`, `loop.ns`, `loop.prologue_bundles`, `loop.epilogue_bundles` (`ii` is None for a loop with several
    blocks); `loop.prologue_of` / `loop.epilogue_of`: the neighbouring loop that shares the block (see below);
    `loop.prologue_cycles`: the prologue cycles counted in this loop's cost (0 when shared).
  - `loop.zol`: the hardware-loop status; `loop.hw_iters`: the recovered `lc`; `loop.trip_count`: the implied source
    trip count.
  - `loop.kernel_line`: the loop-statement line in the compiled source (None when unknown or when the loop is in a
    header; see `loop.source_ref` and `loop.call_site`); `loop.source_trip`: the trip count parsed from that statement;
    `loop.early_exit`: set when a `break` / `return` / `goto` makes the count a maximum.
  - `loop.estimated_cycles()` / `loop.cost_formula()`: `P + iters*II + E`, counting bundles only (no calls, no inner
    loops).
  - `loop.body_bundles`: the ground truth from `aie-asm-printer`, counting the loop's own blocks (not those of its inner
    loops); for a pipelined loop it must equal `ii` (a mismatch is reported as an issue). `loop.branches`: conditional
    jumps other than the back edge.
  - `loop.search_ii`: the stale pre-RA search plan (for reference only).
  - `loop.prereject_reasons`.
  - `loop.loop_spills`, `loop.loop_reloads` (plus `spill_bytes`, `reload_bytes`): spill / reload *instructions* in the
    loop's own blocks; `loop.regalloc_spills`, `loop.regalloc_reloads`: the register allocator's count of spilled /
    reloaded *registers* for the loop, including inner loops.
  - `loop.loop_calls` (names) / `loop.callees` (symbols): calls per iteration.
- `report.unattributed_loops`: pre-RA loops that match no final loop.
- `report.pipelined_loops`, `report.best_ii`, `report.best_post_ra_ii`.
- `report.blocks`: bundle / byte sizes per IR block name; `report.layout`: the same per function, per block in layout
  order (the printout folds the sizes into per-function `code N bundles / M B` totals). A scheduled loop's header block
  carries its `schedule` (II, NS), which attaches the final remark to it.
- `report.issues`: deduplicated, actionable warnings only.

## Interpreting the printed report

```
fused_dequant_dot_core_2048_4_g32<32, 2048>, stack 0 B, code 108 bundles / 544 B
  for.body @kernel.cc:21: post-RA pipelined, II=14, NS=3, ZOL, 30 iters (trip 32: 2 peeled)
    cost 31 + 30*14 + 63 = 514 cy [iters from ZOL]; body 14 bundles (112 B)
```

- `II=14, NS=3`: the final schedule; one loop iteration starts every 14 cycles, with 3 overlapping stages.
- `ZOL, 30 iters (trip 32: 2 peeled)`: the hardware loop counter runs 30 times; the source loop has 32 iterations, of
  which `NS-1 = 2` were peeled into the prologue / epilogue.
- `cost = prologue(31) + iters*II + epilogue(63)`: the cycles spent per entry into this loop. Here the epilogue's 63 of
  514 cycles is worth reducing; the "ramp+drain" note flags this case when `prologue+epilogue` exceeds the steady
  state.
- `body 14 bundles (112 B)` from the asm printer cross-checks II and shows the program-memory cost.

A one-block loop also prints its resource bound: the busiest issue slot or shared register / memory port of its body
(the same counts as `variant_search`, with tables in `artlite/aie2p_isa.py`), together with the ops of the one or two
binding resources:

```
gq_dq_5e0a71c2, stack 256 B, code 116 bundles / 704 B
  for.body @kernel.cc:583: post-RA pipelined, II=40, NS=2, ZOL, 7 iters (trip 8: 1 peeled)
    cost 51 + 7*40 + 25 = 356 cy [iters from ZOL]; body 40 bundles (272 B)
    res bound 40 (rm: vshuffle 24, vst.flush.512.conv 8, vst.push.576.conv.bfp16ebs8.fp32 8); slots lda 5, ldb 9, st 16, vec 12, mv 28
```

An II near the bound means the loop is resource-bound (reduce the ops on that slot or port). An II far above it
points to a recurrence, register pressure or the scheduler (`variant_search --why`).

The remark's prologue and epilogue are whole blocks (the preheader and the exit). Between consecutive loops, one block
is both; it counts only in the first loop's cost, and the second loop prints it in parentheses:

```
for.body @kernel.cc:122: post-RA pipelined, II=12, NS=2, ZOL, 15 iters (trip 16: 1 peeled)
  cost 13 + 15*12 + 15 = 208 cy [iters from ZOL; epilogue shared: the prologue of for.body13 @kernel.cc:125]
for.body13 @kernel.cc:125: post-RA pipelined, II=12, NS=2, ZOL, 15 iters (trip 16: 1 peeled)
  cost (15) + 15*12 + 6 = 186 cy [iters from ZOL; prologue shared: counted in for.body @kernel.cc:122]
```

A `not pipelined, body N bundles, ZOL (reason)` line means the loop runs sequentially; `body bundles` is its
per-iteration cost. Outer loops that call a noinline core are expected to look like this.

Function names are demangled with their template arguments (`srec<32, 4>`, with the anonymous namespace dropped).
Inner loops are indented under their outer loop. An inlined function's exit block is shortened (`pop.exit186.7`) and
followed by its asm label, so you can find it in `<src>.s`:

```
gmp0f3dcf07_srec<32, 4>, stack 0 B, code 117 bundles / 832 B
  for.cond4.preheader @gp2.cc:252: not pipelined, 2 blocks + 1 inner loop, body 57 bundles, no ZOL (Not a single basic block: 3)
    cost 2*(57 + 114) = 342 cy [iters from src]
    pop.exit186.7 (.LBB17_2) @gp2.cc:254: post-RA pipelined, II=38, NS=2, ZOL, 3 iters (trip 4: 1 peeled)
      cost 40 + 3*38 + 17 = 171 cy [iters from ZOL]; body 38 bundles (336 B)
```

- `N blocks + K inner loops, body B bundles`: a loop with several blocks (never pipelined); `B` counts the loop's own
  blocks, which hold its inner loops' prologues and epilogues.
- Its cost is `iters*(B + inner loops' iterations + calls)`, here `2*(57 + 3*38)`. The inner loop's line shows the
  cost of one entry into it (`40 + 3*38 + 17`).
- `<=` with `[all paths]`: the loop has branches (an `if`), and every block is counted in every iteration, giving an
  upper bound. `[may exit early]`: a `break` (or similar) makes the trip count a maximum.

Calls are not in the bundle count. Each loop that calls functions lists them with the callee's cycles per call (its
code outside loops + its loops + its calls, `<=` when it branches), and the loop's cost adds them:

```
if.end28 @kernel.cc:776: not pipelined, body 18 bundles, no ZOL (The loop structure is not supported)
  cost <= 7 + 5*(18 + 641) + 8 = 3310 cy [iters from src; with calls; may exit early]
  ⚠ calls per iteration (not in the bundle count): gmp5df2e9e1_unit (641 cy)
```

`[excluding calls]` means a callee's cost is unknown (a runtime-library function such as `__divsi3`, or an unknown
trip count in the callee), so the cost counts only the loop's bundles. `cost(iters) ...; @32 iters = X cy` means the
trip count is unknown and `X` uses the `trip_count` fallback.

Spill code shows two counts, which measure different things:

```
⚠ spill code in the loop body: 12 spill store(s) (768 B), 16 reload load(s) (1024 B); the register allocator: 3 spill(s), 4 reload(s) of registers, inner loops included (a register wider than 64 B moves in 64-byte parts)
```

- The first counts the asm instructions in the loop's own blocks (from their `N-byte [Folded] Spill/Reload`
  comments): what the loop executes on each iteration.
- The second comes from the register allocator's remark for the loop statement: it counts registers before they are
  split into 64-byte moves, summed over inner loops. A 2048-bit accumulator (`dm`) spill counts as 1 there and as 4
  stores in the asm (as here: 12 = 3*4, 16 = 4*4). The function-level remark is printed as
  `(function @L): the register allocator spilled S, reloaded R register(s)`.

## Variant search (variants of an operator's functions)

To compare pragmas or rewrites of a loop, use `artlite.variant_search` rather than a throwaway script. It renders
the operator's kernel, patches the target functions per variant, measures all variants in parallel and ranks them in
one table per loop.

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

- Variants: a variant name collects several flags; `base` (the kernel as rendered) always comes first, and `@file`
  reads the text from a file.
  - `--pragmas` with the shorthands `post`, `none`, `iiN` and `post+iiN` replaces the pipelining pragma of the
    innermost rolled loops in the targets and in the `always_inline` helpers they call (`_mac` under `_mm`). A helper's
    pragma applies to all its callers, and a loop that calls a helper containing loops is not innermost. `--loop K`
    selects other loops (K counts the targets' rolled loops in source order, then the helpers' loops).
  - `--sub V OLD NEW` replaces text (it must match exactly once).
  - `--define V FN @def.cc` replaces a whole function. Patches may name functions as `kernel.cc` does
    (`<stem>_<name>_<hash>`; the hash follows the code, so a patched function and its callers get new hashes) or as
    rendered (`__artlite_<stem>_<name>`).
  - `--param V key=value` sets an operator / Jinja argument.
  - `--variants FILE.py` reads `VARIANTS = [Variant(...)]`.
- Program memory: the summary's `code B` covers the whole object. To get one core's bytes, walk the call graph (`jl`,
  tail `j`) from its entry kernels in each variant's `kernel.cc.s`, using the report's per-function block bytes
  (scriptable from `search()`; aie-kernel-perf-optimization section 7).
- Columns: `iters` is the ZOL count plus peeled iterations; `cycles` is per entry, including prologue and epilogue;
  then come the loop body's ops per slot (llvm-aie's slot for each mnemonic; a `lng` op occupies both alu and mv).
  `port` is the busiest register / memory port shared by ops from several slots (`rm` is L_RM_PORT: shuffles,
  selects, broadcasts and every bfp16 push / flush; for `wm`, `qwa`, `qwb`, `qrs`, `dwm`, `ada` and `lck`, see
  aie-kernel-perf-optimization `reference/machine-model.md`). `res bound` is the busiest slot or port; `*` marks the
  binding ones, whose ops are listed below the table. A loop whose II is far above its slot counts and that fails with
  "out of resources" at every II is often port-bound.
- A one-block loop left unpipelined (or every loop, with `--why`) is compiled again by the debug build of llvm-aie
  (`ARTLITE_DEBUG_CLANG`, default `../llvm-aie/build/bin/clang++`). It reports ResMII, the IIs tried, RecMII /
  ScarceRegMII per mode, what failed at each II, and the chain of the most frequent "latency not met". The search
  stops at II 60 (`-aie-postpipeliner-maxii`) or after 20 IIs (`-maxtry-ii`). The report warns when that build's
  llvm-aie commit differs from the wheel's (and lists the commits in between that touch the AIE backend or CodeGen),
  because its traces may then differ from the shipped schedule.
- Slot and port tables live in `artlite/aie2p_isa.py`: ports per mnemonic, plus per first-operand register kind where
  a mnemonic's variants differ (`PORT_KINDS`; e.g. a `vst` of a vector register reads QEY_RS). They are generated from
  llvm-aie's TableGen at the wheel's commit by `python -m artlite.gen_isa_tables` (run it after a toolchain upgrade);
  `test_variant_search.py` checks them when the `../llvm-aie` checkout and its `llvm-tblgen` are present.
- `--npu`: the builds compile in parallel, then each variant runs alone in a separate process. It reports the time of
  one dispatch (median and p10 over `--reps`) and, for each `--region NAME=FN[*N]`, the counters of the operator's
  `win=NAME` build (cycles, windows, memory and lock stalls) next to the static cycles of N calls of FN.
- Python API: `search(factory, kwargs, functions=..., variants=...)`, `print_report`, `run_npu`.

## Troubleshooting

- **Compilation fails**: `report.returncode`, `report.diagnostics`.
- **Loop not pipelined**: read the printed reason. Typical fixes are flattening branches ("Not a single basic
  block") and removing call-like constructs ("loop structure is not supported"); when pre-RA fails but post-RA
  delivered a schedule, accept it.
- **Spills appear**: check the per-loop `spill code in the loop body` line (asm and register-allocator counts); try
  `__restrict`, DM-bank annotations, smaller unroll factors, or `AIE_PREPARE_FOR_POSTPIPELINING`.
- **`@?` or `unattributed`**: the loop's schedule is correct, but its source line could not be tied to it. Check the
  asm label and the `.loc` lines in `<src>.annotated.s`, and improve the matching in `measure_ii/match.py`.
- **Missing bank annotations**: add `__aie_dm_resource_a/b/c/d` qualifiers.

## Improvements

measure_ii is error-prone. When the assembly or the compiler's remarks disagree with its report, do not just note the
discrepancy; fix `src/artlite/measure_ii/` so that the next report is correct. `remarks.py` parses the remarks,
`asm.py` the assembly and its trip counts, `statements.py` and `match.py` the source loops; `costs.py`, `slots.py` and
`render.py` produce the printed report; the tests are in `tests/test_measure_ii.py`.