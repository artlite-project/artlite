# Development

The rules every change follows and the commands are in [AGENTS.md](../AGENTS.md), the map of the code in [Code
map](code-map.md); contributions are accepted under the agreement in [CONTRIBUTING.md](../CONTRIBUTING.md). Building:
[Installation](install.md), the development venv: [Installation: Development](install.md#development).

## Tests

```bash
pytest tests/<package>/test_<module>.py -k <config-id>   # one module's tests (NPU)
pytest tests -m smoke                                    # after a change: about a minute with the designs cached (NPU)
pytest tests --run-slow ...                              # also the whole-model tests
pytest tests --dump-designs DIR                          # render every design, no NPU
ruff check <changed files>                               # the tree passes: fix whatever it reports
```

GitHub runs `ruff check` on each push to `main` and each pull request, and imports every module on Python 3.12
(`requires-python`'s oldest, which evaluates annotations as it defines a function or class, where 3.14 defers them:
`tests/test_no_torch.py`); the tests need an NPU, so they run on your machine. The full suite is slow: run the test file of what you changed, and the smoke set after a shared change. The
tests need no compile plumbing: pytest plans the selected tests' designs in a subprocess and builds them in the
background while the tests run. Whole-model tests (`slow`) need the model files; the long-prompt tests need the essay
corpus (`python tests/datasets/fetch_datasets.py`); the reference tests need a llama.cpp CPU build
(`ARTLITE_LLAMA_CPP_DIR`; Inkling 2's: one with `inkling2_loop`, master of 2026-11-22 on) and skip without one. Their
references (llama.cpp's runs, the CPU models' logits) are cached in `tests/refs/` (gitignored), keyed by their inputs;
a new llama.cpp build or reference source replaces an entry's older version. An entry no test reads any more stays
until deleted; a read marks it used, so
`find tests/refs -mindepth 2 -maxdepth 2 -mtime +30 -exec rm -rf {} +` removes those unused for 30 days.

The C++ runtime's tests (gtest: `tests/csrc/npu_test.cpp` on the NPU, the other files host-only) build in a separate
tree, with the venv active (`build/` belongs to the editable install):

```bash
cmake -S . -B build-tests -DARTLITE_BUILD_TESTS=ON
cmake --build build-tests -j
ctest --test-dir build-tests --output-on-failure
```

## Operators

How they compile and load: [Design](design.md#kernel-compilation-pipeline); where the framework's modules are:
[Code map](code-map.md).

- One module per operator: an `ArtcOperator` subclass implementing
  - `get_kernel_src()`: C++ from a Jinja2 template (`JINJA_ENV`: undefined variables are errors) with shapes, trip
    counts and strides as compile-time constants, its names under `kernel_prefix` (AGENTS.md's Rules). Kernel
    templates are files next to their module, `<module>.cc.jinja` (several: `<module>.<part>.cc.jinja`), named by
    their path from `src/` in Python (`templating.load`, `templating.source`) and in `{% include %}` (on a line of its
    own: whole lines), never C++ in Python strings;
  - `get_mlir_artifact()`: the array design in `aie.iron` (ObjectFifos, Workers, runtime sequence);
  - `get_arg_spec()`: buffer directions, shapes, dtypes.
- Reuse the shared pieces instead of copying (`artlite/operators/`, the headers in `artlite/include/`): `kmath/`
  (exp2, recip, sigmoid, SiLU, GELU, ...: `km.exp2(...)` in a template; arrays through the gather LUT:
  `km.act(km.SILU)`, `km.table_pass`; a head's norm and rope: `km.rms_norm`, `km.rope`; a softmax's P:
  `km.p_pass1`, `km.p_pass2`), `kernel_lib.h` (`inv_rms`, bfp16 panels, lane shuffles, vectors from 128-bit
  pieces: `al::concat`, `al::gather`; an fp32 value's bf16 hi / lo parts: `al::split2`; the attention preps' stage
  rows and panels: `al::stage_row`, `al::encode_hilo`, `al::vpan`), `design.py` (`COLS`, `ROWS`, `range32`: Worker
  loops with an i32 induction variable, `linear_tap`), `bfp16.py` (host bfp16ebs8 conversions and panels),
  `l1_layout.py` (`pin_l1_buffers`), `util.py` (template `Attr`, workspace `regions`, bf16 constants, `arg_bytes`).
  Words a kernel and the host share (metadata, state, command words, flags) are declared in `words.py` layouts and
  reach a template as that layout's `consts()`, never by a name prefix of a module's globals.
- A new operator lives with what it serves (an `artlite` subsystem, or its model's package: AGENTS.md's Layers) and is
  tested in `tests/<package>/test_<module>.py`. Tests need no compile plumbing: pytest plans the selected tests in a
  subprocess and builds their designs in the background.
- `compile()` = render (`build_spec()`, in the calling thread) + build (Peano: `kernel.cc` -> `kernel.o`; `aiecc`:
  `design.mlir` -> `main.pdi` + `insts.bin`). `kernel.cc` has a section per operator (a fused ELF's: each definition
  in the first section that uses it, a later one lists what it shares). `compile_all(ops)` builds many in
  parallel: use it wherever several designs are compiled before anything runs (`parallel_map` for other compile
  work). `ARTLITE_COMPILE_JOBS` bounds the builds; `python -m artlite.parallel_compile CMD ...` fills the cache for
  any command.
- Every compile runs `operators/checks.py`: each core's deepest kernel call against its Worker `stack_size`, kernel
  signatures against the design, rounding kernels that do not set the rounding mode, runtime-library arithmetic
  (`ARTLITE_SKIP_CHECKS=1` skips them).
- Builds are cached per content hash in `rtc_<hash>/` of the toolchain's cache directory (`artlite-cache path`;
  inspect the artifacts there; `ARTLITE_CACHE_DIR` moves the cache, `ARTLITE_IGNORE_CACHE=1` rebuilds; the layout,
  last use and cleaning: `artlite/cache.py`). A render manifest (`render_cache.py`, keyed by
  class, constructor arguments and the sources' hash) skips rendering cached designs; a module global changed at
  runtime is not in that key, so a monkeypatched test needs distinct operator names.
- First use (`op.device` or the first dispatch) opens a HW context and uploads the program. `dispatch(*buffers)`
  submits in sequence order and waits, without host <-> device syncs; `run(*tensors)` is the one-shot wrapper;
  `artlite.from_numpy` / `artlite.empty(..., device=)` give device buffers (`Tensor.numpy()`: a zero-copy view); `compile(device=)` chains operators
  on one device.
- NPU2: 8 columns of compute rows 2-5, a memtile (row 1, 512 KB) and a shim (row 0). A core has 64 KB of data
  memory in four 16 KB regions (budget ~63 KB with the stack) and 16 KB of program memory.

## NPU pitfalls

- A runtime-sequence argument stays below 4 GB (ELF relocation addends are 32-bit).
- An ELF's control code goes into the process's device heap (64 MB heap BOs up to 512 MB; one before Linux 7.2, where
  `Library`'s pieces shrink to fit, the least recently used step transactions are given up for room
  (`overlay.Transactions`) and a heap that is still full raises `HeapFullError`); every ConfigureOp inlines its design's
  configuration. Long programs: `fused.Library` + `Library.run`.
  Thousands of entries in one aiecc ELF make aiecc's time explode.
- `FusedOperator` refuses to run while a buffer read before it is written was never written by the host (declare
  intentionally empty ones `uninitialized`). The host writes only input and scratch buffers: output BOs are never
  flushed, so cached host writes land over the NPU's (garbage in 64-byte runs).
- No kernel statics in fused or overlay designs (rewritten on every reconfiguration; they also collide with pinned
  L1 buffers): scratch and tables are worker `Buffer`s. Every kernel entry that rounds sets
  `aie::set_rounding(conv_even)` (the default is floor).
- 128-byte vector loads need 64-byte alignment: an unaligned `load_v<64>` of bf16 silently rounds the address down.
- Shim BDs: innermost dimension <= 1023 words (a longer one can make aiecc run for very long instead of failing);
  at most 4 queued tasks per shim channel.
- Host-built transactions (overlays): issue a phase's input fills and drains before the bulk of its weight fills, or
  the queues deadlock; clflush host writes before a dispatch and fetch NPU writes before reading them. A dispatch ends
  with every task it queued complete: the next one rewrites the BDs from its start, and a task still queued runs with
  its words.


## Kernel tools

```bash
# rebuild one test's designs, print the Peano / aiecc commands, keep the assembly (kernel.cc.s in the cache)
ARTLITE_IGNORE_CACHE=1 ARTLITE_LOG_COMMANDS=1 ARTLITE_GENERATE_ASM=1 pytest tests/operators/test_fused.py -v -s

# the final schedule of each loop of a kernel: II, stages, trip counts, spills, the resource bound
ARTLITE_MEASURE_II=1 python <<EOF
from tests.decode.mmv import MmvOperator
MmvOperator(fmt="Q4_K", n_tiles=4).compile()
EOF

# variants of an operator's functions (pragmas, rewrites, parameters) ranked per loop
python -m artlite.variant_search --help

# every shipped kernel's schedules and every core's program memory against the committed baseline, no NPU
python -m tests.schedule_ledger record [APP ...]            # the differences, worst first
python -m tests.schedule_ledger record [APP ...] --update   # refresh tests/schedule_ledger.jsonl (commit it)
python -m tests.schedule_ledger rank APP                    # a prefill chunk's loops by weight and slack

# build everything a script compiles, ARTLITE_COMPILE_JOBS at a time, then run it
python -m artlite.parallel_compile python script.py
```

`tests/test_app_designs.py` checks each app's ledger: a change to kernel code or a toolchain update refreshes the
baseline in the same commit, whose diff of `tests/schedule_ledger.jsonl` shows what it did to the kernels' loops
(pipelined or not, II, spills), cycles, code bytes, stack and the cores' program memory.
`record` alone prints the differences worst first (lost pipelining, II / spills up, cores below 256 B free): a
toolchain bump's survey. `rank APP` (`-p 1024`: the chunk's length): a prefill chunk's kinds of items by their busiest
cores' static cycles and its loops by weight and by slack (II above the resource bound): which kernel work a
session can win back.

On-NPU profiling (hardware counters between a kernel's `event0()` / `event1()` markers, or event traces, as
Chromium JSON for Perfetto) is switched on by `ARTLITE_PROFILE` (below).

## Checking a change

- Refactoring check: `pytest tests --dump-designs DIR` before and after, then
  `python tests/design_diff.py BEFORE AFTER --asm` (identical MLIR; changed kernel sources compile to the same
  asm). The apps' full designs are dumped too (`test_app_designs.py`); the slow tests are not. A change of shared
  kernel code (`kmath/`, `include/kernel_lib.h`) re-renders only the tests it touches: `--affected REGEX --compile` on
  BEFORE before the change, `--partial --asm` after (`design_diff.py`'s docstring); on the NPU then only the tests of
  the designs whose code changed.
- Host-side refactoring check (the overlays' step building and patching, metadata and command writes, weight
  packing): `python tests/txn_dump.py DIR` before and after, then `python tests/txn_dump.py --diff BEFORE AFTER` (every
  app's decode overlay and media encoder on a fake device, no NPU: per dispatch the transaction's words and the host
  buffers' digests must be identical).
- Data layouts (shim taps, memtile walks, joins) are tested on the host with `tests/dma_model.py`
  (`tests/gemm/test_dma.py` is the pattern).

## Environment variables

Every `ARTLITE_*` variable is declared in `artlite/settings.py` (its kind, default and how its value is read); nothing
else reads one. `import artlite` warns about a variable that is set but not declared, naming the closest one. This
table is the registry's (`tests/test_settings.py` compares it).

<!-- settings -->
| variable | effect | default |
|---|---|---|
| `ARTLITE_MODELS_DIR` | the models directory. Read at import. | ~/.artlite/models |
| `ARTLITE_LLAMA_CPP_DIR` | the llama.cpp CPU build the reference tests load. Read at import. | /src/llama.cpp |
| `ARTLITE_CACHE_DIR` | the build cache. Read at import. | ~/.cache/artlite |
| `ARTLITE_CACHE_GC` | the build cache's automatic cleaning, after a process that built something, once a day: `auto` (`superseded=1d,unused=30d`), `off`, or rules that replace some of these (`superseded=4h,size=30G`; also `legacy`, `interval`); `artlite-cache clean` applies them too. | auto |
| `ARTLITE_BACKEND` | the NPU backend: `xdna` (the amdxdna driver) or `xrt` (XRT). Read at import. | the platform's own (xdna on Linux, xrt on Windows) |
| `ARTLITE_COMPILE_JOBS` | concurrent design builds (1: serial). | the CPU count |
| `ARTLITE_IGNORE_CACHE` | rebuild every design instead of loading it from the cache. | off; `1` turns it on |
| `ARTLITE_LOG_COMMANDS` | print every Peano and aiecc command line. | off; `1` turns it on |
| `ARTLITE_GENERATE_ASM` | also write each built kernel's assembly (`kernel.cc.s` in its cache directory). | off; `1` turns it on |
| `ARTLITE_MEASURE_II` | print the per-loop schedule report of every kernel compiled (measure-ii skill). | off; `1` turns it on |
| `ARTLITE_SKIP_CHECKS` | skip the build-time design checks (stack depth, kernel signatures, rounding mode). | off; `1` turns it on |
| `ARTLITE_DEBUG_CLANG` | an assertions build of llvm-aie's `clang++`, for `variant_search`'s post-pipeliner traces. | llvm-aie's build directory next to the checkout |
| `ARTLITE_PLANNING` | a planning process: `compile()` hands its spec to the precompiler instead of building (set by `python -m artlite.parallel_compile`). | off; `1` turns it on |
| `ARTLITE_PLAN_SINK` | `<key>:<address>` of the precompiler's listener that planning processes send to (set with `ARTLITE_PLANNING`). | none |
| `ARTLITE_VERIFY_TXN` | check every host-built transaction where it is built or loaded (`txn.verify`: AxQoS 0, valid BDs, token controllers, the shim channels' queues; the tests turn it on). | off; `1` turns it on |
| `ARTLITE_PM_PACKETS` | a library configuration's tile writes go as control packets where that is cheaper (off: all in the transaction, and no configuration without the array reset). | on; `0` turns it off |
| `ARTLITE_PDI_RESET` | every library configuration with the array reset (none without it after another of the same piece). | off; `1` turns it on |
| `ARTLITE_PACE` | the prefill GEMMs' pace levels: `auto` (calibrated per machine and cached), `off`, `recalibrate`, or levels (`gu=3,qkv=2`, or one level for every class). | auto |
| `ARTLITE_LM_SCREEN` | a Q6_K LM head is screened: the NPU reads its high plane (39% of its bytes; F16 / BF16 heads: a Q4_0 plane, 28%; a Q8_0 head where a model asks: its high plane, 53%), the host computes the logits that matter exactly (`decode.screen`; off: the NPU reads the whole head). | on; `0` turns it off |
| `ARTLITE_LARGE_PAGES` | Windows: host buffers of 2 MB and more on 2 MB pages (`MEM_LARGE_PAGES`) where the account holds the "Lock pages in memory" right, else 4 KB pages (`artlite-probe --pages` reports which). Read at first use. | off; `1` turns it on |
| `ARTLITE_NB_LEAK_WARNINGS` | nanobind's leak warnings at exit (set to anything to show them). | off; any value turns it on |
| `ARTLITE_ROUTER_MODELS` | a router's model server: the router's `--models-max`, which divides the default prompt cache's part of the RAM (set by the router). | 1 |
| `ARTLITE_ARG_<OPTION>` | `artlite-server`'s options, one variable per option (`ARTLITE_ARG_CTX_SIZE`). | the option's own |
| `ARTLITE_PROFILE` | on-NPU profiling: `1` / `counters` (event-window counters) or `trace` (stream traces); `0`, `off`, `false`, `no` or empty: off (profiling skill). Read at first use. | 0 |
| `ARTLITE_PROFILE_DIR` | profiling: the output directory. Read at first use. | ./artlite_profile |
| `ARTLITE_PROFILE_COUNTERS` | profiling: counter event pairs, `[module:]start:stop` comma-separated (modules `m`, `mt`, `sh`). Read at first use. | core 33:34, 34:33, 33:33, 34:34 |
| `ARTLITE_PROFILE_EVENTS` | trace mode: core event names or hex ids, comma-separated. Read at first use. | none |
| `ARTLITE_PROFILE_TRACE_SIZE` | trace mode: the trace buffer's size in bytes. Read at first use. | 4194304 |
| `ARTLITE_PROFILE_TILES` | trace mode: the traced tiles: `all`, `first:N` or `;`-separated `col,row`. Read at first use. | all |
| `ARTLITE_PROFILE_LATERAL` | trace mode: pre-lower with lateral routing (a spare shim). Read at first use. | off; `1` turns it on |
| `ARTLITE_PROFILE_DISTRIBUTE` | trace mode: pre-lower with distributed channels (two S2MM). Read at first use. | off; `1` turns it on |
| `ARTLITE_PROFILE_CONVERT` | convert the `.ttbin` dumps to JSON at exit. Read at first use. | on; any value but `1` turns it off |
| `ARTLITE_PROFILE_SAMPLE_EVERY` | counters mode: sample every Nth dispatch (at least 1). Read at first use. | 1 |
| `ARTLITE_DIAGNOSE_ROOT` | `artlite-probe --diagnose`: the file of the root readings that the sudo user's run continues with (set by the tool). | none |
<!-- /settings -->

`OMP_WAIT_POLICY` is not Artlite's: the tests set `PASSIVE` unless it is set (the idle OpenMP threads of torch, which
only the tests' CPU references load, sleep instead of spinning ~1.2 CPUs while the NPU works).

## Skills

[`.claude/skills/`](../.claude/skills/) holds the AIE2P knowledge Artlite's coding agents work from, in plain
Markdown that people can read too:

| skill | |
|---|---|
| [aie2p-coding](../.claude/skills/aie2p-coding/SKILL.md) | AIE2P facts and limits for writing kernels and designs: tiles, memory, vector and MMUL shapes, DMA, toolchain traps |
| [aie-kernel-perf-optimization](../.claude/skills/aie-kernel-perf-optimization/SKILL.md) | making kernels and designs fast: where the time goes, loop schedules, proven restructurings, rejected ideas |
| [measure-ii](../.claude/skills/measure-ii/SKILL.md) | `artlite.measure_ii` and `artlite.variant_search` |
| [profiling](../.claude/skills/profiling/SKILL.md) | hardware counters and event traces on the NPU |
| [npu-prefill](../.claude/skills/npu-prefill/SKILL.md) | a model's prefill on top of its decode overlay |
| [iron-porting](../.claude/skills/iron-porting/SKILL.md) | porting an IRON operator to an Artlite operator |
| [artlite-bench](../.claude/skills/artlite-bench/SKILL.md) | measuring a model for the README's charts |
| [update-deps](../.claude/skills/update-deps/SKILL.md) | upgrading the pinned mlir-aie and Peano wheels |

## References

- AIE API headers: mlir-aie's `third_party/aie_api/`.
- MLIR-AIE usage: mlir-aie's `{programming_guide,programming_examples,mlir_exercises}` (large; grep).
- Peano backend (scheduler, pipeliners): llvm-aie's `llvm/lib/Target/AIE/`; the `__builtin_aie2p_*`
  intrinsics: its `clang/include/clang/Basic/BuiltinsAIE2P.def`.
- AMD manuals, cited by number: AM027 / AM029 (AIE-ML v2 = AIE2P architecture manual / register reference), AM020 /
  AM025 (AIE-ML v1 = AIE2), UG1079 / UG1076 (AI Engine kernel and graph programming, tools and flows; chess /
  Vitis, AIE1-oriented). AM029 is Versal's register layout: its memory-module and memtile offsets are the NPU's, its
  shim (NOC / PL module) and core register-file, debug and counter offsets are not (Core_Status: 0x38004 there,
  0x32004 here). The NPU's map: mlir-aie's `third_party/aie-codegen/aie-regdb/globalparams/xaie2pgbl_params.h`.
