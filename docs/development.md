# Development

The rules every change follows, the commands and the map of the code are in [AGENTS.md](../AGENTS.md); contributions
are accepted under the agreement in [CONTRIBUTING.md](../CONTRIBUTING.md). Building: [Installation](install.md), the
development venv: [Installation: Development](install.md#development).

## Tests

```bash
pytest tests/<package>/test_<module>.py -k <config-id>   # one module's tests (NPU)
pytest tests -m smoke                                    # after a change: about a minute with the designs cached (NPU)
pytest tests --run-slow ...                              # also the whole-model tests
pytest tests --dump-designs DIR                          # render every design, no NPU
ruff check <changed files>                               # the tree passes: fix whatever it reports
```

GitHub runs `ruff check` on each push to `main` and each pull request; the tests need an NPU, so they run on your
machine. The full suite is slow: run the test file of what you changed, and the smoke set after a shared change. The
tests need no compile plumbing: pytest plans the selected tests' designs in a subprocess and builds them in the
background while the tests run. Whole-model tests (`slow`) need the model files; the long-prompt tests need the essay
corpus (`python tests/datasets/fetch_datasets.py`); the Qwen reference tests need a llama.cpp CPU build
(`ARTLITE_LLAMA_CPP_DIR`) and skip without one.

The C++ runtime's tests (gtest: `tests/csrc/npu_test.cpp` on the NPU, the other files host-only) build in their own tree, with the venv active (`build/` belongs
to the editable install):

```bash
cmake -S . -B build-tests -DARTLITE_BUILD_TESTS=ON
cmake --build build-tests -j
ctest --test-dir build-tests --output-on-failure
```

## Operators

An operator is one file: an `ArtcOperator` subclass that renders its kernel's C++ (`get_kernel_src()`), its array
design in IRON (`get_mlir_artifact()`) and its buffer arguments (`get_arg_spec()`). It lives with what it serves: a
subsystem the models share (`src/artlite/`: `gemm/`, `decode/`, `prefill/`, `moe/`, `vision/`) or one model's package
(`src/models/<model>/`), and is tested in `tests/<package>/test_<module>.py`. How they compile and load:
[Design](design.md#kernel-compilation-pipeline); the framework and the shared pieces to reuse: AGENTS.md's map and
"How an operator works".

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

# build everything a script compiles, ARTLITE_COMPILE_JOBS at a time, then run it
python -m artlite.parallel_compile python script.py
```

On-NPU profiling (hardware counters between a kernel's `event0()` / `event1()` markers, or event traces, as
Chromium JSON for Perfetto) is switched on by `ARTLITE_PROFILE` (below).

## Environment variables

Every `ARTLITE_*` variable is declared in `artlite/settings.py` (its kind, default and how its value is read); nothing
else reads one. `import artlite` warns about a variable that is set but not declared, naming the closest one. This
table is the registry's (`tests/test_settings.py` compares it).

<!-- settings -->
| variable | effect | default |
|---|---|---|
| `ARTLITE_MODELS_DIR` | the models directory. Read at import. | ~/.artlite/models |
| `ARTLITE_LLAMA_CPP_DIR` | the llama.cpp build the Qwen reference tests load. Read at import. | /src/llama.cpp |
| `ARTLITE_CACHE_DIR` | the build cache. Read at import. | ~/.cache/artlite |
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
