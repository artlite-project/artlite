# Development

The rules every change follows, the commands and the map of the code are in [AGENTS.md](../AGENTS.md); contributions
are accepted under the agreement in [CONTRIBUTING.md](../CONTRIBUTING.md). Building: [Installation](install.md).

## Tests

```bash
pytest tests/rtc/test_<operator>.py -k <config-id>   # one operator's tests (NPU)
pytest tests/rtc -m smoke                            # after a change: about a minute with the designs cached (NPU)
pytest tests/rtc --run-slow ...                      # also the whole-model tests
pytest tests/rtc --dump-designs DIR                  # render every design, no NPU
ruff check <changed files>                           # the tree passes: fix whatever it reports
```

GitHub runs `ruff check` on each push to `main` and each pull request; the tests need an NPU, so they run on your
machine. The full suite is slow: run the test file of what you changed, and the smoke set after a shared change. The
tests need no compile plumbing: pytest plans the selected tests' designs in a subprocess and builds them in the
background while the tests run. Whole-model tests (`slow`) need the model files; the long-prompt tests need the essay
corpus (`python tests/datasets/fetch_datasets.py`); the Qwen reference tests need a llama.cpp build
(`ARTLITE_LLAMA_CPP_DIR`).

The C++ runtime's tests (gtest, `tests/npu_test.cpp`) build in their own tree, with the venv active (`build/` belongs
to the editable install):

```bash
cmake -S . -B build-tests -DARTLITE_BUILD_TESTS=ON
cmake --build build-tests -j
ctest --test-dir build-tests --output-on-failure
```

## Operators

An operator is one file: an `ArtcOperator` subclass that renders its kernel's C++ (`get_kernel_src()`), its array
design in IRON (`get_mlir_artifact()`) and its buffer arguments (`get_arg_spec()`), exported from
`src/artlite/operators/__init__.py` and tested in `tests/rtc/test_<name>.py`. How they compile and load:
[Design](design.md#kernel-compilation-pipeline); the shared pieces to reuse: AGENTS.md's "How an operator works".

## Kernel tools

```bash
# rebuild one test's designs, print the Peano / aiecc commands, keep the assembly (kernel.cc.s in the cache)
ARTLITE_IGNORE_CACHE=1 ARTLITE_LOG_COMMANDS=1 ARTLITE_GENERATE_ASM=1 pytest tests/rtc/test_silu.py -v -s

# the final schedule of each loop of a kernel: II, stages, trip counts, spills, the resource bound
ARTLITE_MEASURE_II=1 python <<EOF
from tests.rtc.mmv import MmvOperator
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

| variable | |
|---|---|
| `ARTLITE_MODELS_DIR` | the models directory (default `~/.artlite/models`) |
| `ARTLITE_CACHE_DIR` | the build cache (default `~/.cache/artlite`) |
| `ARTLITE_COMPILE_JOBS` | concurrent design builds (default: the CPU count; 1: serial) |
| `ARTLITE_IGNORE_CACHE=1` | rebuild every design instead of loading it from the cache |
| `ARTLITE_LOG_COMMANDS=1` | print every Peano and aiecc command line |
| `ARTLITE_GENERATE_ASM=1` | also write each built kernel's assembly (`kernel.cc.s` in its cache directory) |
| `ARTLITE_MEASURE_II=1` | print the per-loop schedule report of every kernel compiled ([measure-ii skill](../.claude/skills/measure-ii/SKILL.md)) |
| `ARTLITE_SKIP_CHECKS=1` | skip the build-time design checks (stack depth, kernel signatures, rounding mode) |
| `ARTLITE_PM_PACKETS=0` | a library configuration's tile writes all in the transaction, none as control packets |
| `ARTLITE_PROFILE=1`, `counters`, `trace` | on-NPU profiling; `ARTLITE_PROFILE_DIR` (default `./artlite_profile`) and the other `ARTLITE_PROFILE_*` variables tune it ([profiling skill](../.claude/skills/profiling/SKILL.md)) |
| `ARTLITE_DEBUG_CLANG` | an assertions build of llvm-aie's `clang++`, for `variant_search`'s post-pipeliner traces |
| `ARTLITE_LLAMA_CPP_DIR` | the llama.cpp build the Qwen reference tests load |
| `ARTLITE_NB_LEAK_WARNINGS=1` | nanobind's leak warnings at exit (off by default) |
| `ARTLITE_ARG_<OPTION>` | `artlite-server`'s options ([artlite-server](server.md#options)) |

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
