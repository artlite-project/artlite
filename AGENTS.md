# Artlite

Artlite is a lightweight, fully open replacement for XRT on AMD Ryzen AI NPUs. A small C++ shim (`csrc/`) talks to
the `amdxdna` driver directly: no xclbin, no plugin layers, zero-copy buffers. Operators are compiled at runtime
("RTC") from Jinja2 C++ templates specialized per shape, plus an IRON (`aie.iron`) array design lowered through
MLIR-AIE. Python packages: `artlite` (nanobind bindings `artlite._native`, the operator framework
`artlite.operators`, the subsystems the models share) and `models` (`src/models/`: each model on the NPU and its host
side, the commands `artlite-cli`, `artlite-bench`, `artlite-server`), built with scikit-build-core. On Windows 11 the
same runtime goes through XRT (the XRT backend, `csrc/shim_xrt/`; on Linux too where XRT is installed:
`ARTLITE_BACKEND=xrt`; setup and differences: `docs/install.md`, Windows).

Target: AIE2P (NPU2) only, i.e. AMD Strix Point, Strix Halo and Krackan. All these CPUs are znver5, so `csrc/` may
use any modern SIMD (AVX-512 included).

## Rules

- **Never write a nonzero shim BD AxQoS.** 1 and 15 hung the NPU (only a reboot recovered it); 8 changes nothing,
  even next to CPU or iGPU memory load.
- **Never kill a process that holds an NPU context** (let it finish or time out). A killed process has wedged the
  NPU beyond a driver reload. Give compile-only scripts, which open no context, short timeouts instead.
- **Kernel names**: every namespace-scope name of a kernel source (entry points, helpers, tables, types, macros) is
  `{{ p }}_<name>` with `p = kernel_prefix(stem)`, and the design declares its kernels under the same names. The build
  names each definition by its code (`operators/kernel_source.py`), so a fused program's `kernel.cc` holds every
  definition once; a name written out in full collides as soon as two configurations meet in one fused program.
- **Weights stay at their quantization's size.** Never expand them on the host; prefill and decode read one weight
  image in place. Lossless, same-size re-layouts are fine (images are packed at every load: nothing on disk).
- **No file holds a copy of the weights.** Every load converts them from the model's files on the fly, with the native
  AVX-512 packers (`csrc/artlite/tile_pack.cpp`: the GGUF tile formats; `slot_pack.cpp`: the vision overlay's B slots;
  both on `pack_run.h`'s threads and read-ahead, `memory.h`'s non-temporal stores); preprocessing them (padding,
  permutations, scales, folded normalizations) happens at load, never into a file. Caches keep only what the code
  builds (designs, transactions, and a vision network's commands, which carry its biases and norm parameters).
- Operators are self-contained: never import IRON operators.
- Layers (Map): `models` -> `artlite`'s subsystems (`decode`, `prefill`, `moe`, `vision`, `gemm`, `formats`) -> the
  framework (`artlite.operators`, `artlite.overlay`) -> the runtime. Code that two models share lives in `artlite`, in
  the subsystem it belongs to, never in one model's package for another to import (a model package imports another
  only to run that model: qwen35 runs MiniCPM-V's and Qwen3-VL's vision encoders); `artlite` never imports `models`.
- `src/` holds what the apps and tools run: test harnesses (standalone operators of one kernel), CPU references and
  other test-only code live in `tests/`, a directory per package of `src/`
  (`tests/<subsystem or model>/test_<module>.py`; the top-level modules' tests at the top). Layout inverses and
  readbacks stay next to the code they mirror.
- No absolute paths in code, tests or scripts: derive them from `__file__` or package constants. Model files only
  through `artlite.model_config`: the models dir (`ARTLITE_MODELS_DIR`, default `~/.artlite/models`; a model's files
  in `<HF org>/<HF repo>/`) is spelled only in `model_config.MODELS_DIR`: get paths from its helpers (`llama_gguf`,
  `qwen3_gguf`, `gemma_gguf`, `qwen_gguf`, ...), in tests and research scripts too.
- `ARTLITE_*` environment switches are declared once in `artlite/settings.py` (and in `docs/development.md`'s
  table): read one only through its object (`settings.PACE.get()`), never `os.environ`.
- Licenses: Artlite is LGPL-2.1-or-later and its own files carry no license header. Code from another project keeps
  its copyright and license notices plus a "Modified for Artlite" line, and is listed with its license in a LICENSE
  next to it (`csrc/shim_xdna/LICENSE`).
- Python: annotate all parameters and return types; functions with more than 3 parameters take keyword-only
  arguments. Code runs on Python 3.12 (`requires-python`) and every newer version a venv may have, with no
  compatibility code for older ones; `from __future__ import annotations` only where 3.12 needs it (an annotation
  naming a class defined later in the module or imported only under `TYPE_CHECKING`, an attribute of a package
  still loading, such as `artlite.Device` in a module that `import artlite` loads, or a module the class body
  rebinds, such as a dataclass field `np: int = 8` before `-> np.ndarray`). GitHub imports every module on 3.12.
- Production code (`src/`) runs on numpy and ml_dtypes, never torch: torch is a dev dependency, the tests' CPU
  references (ruff refuses the import outside `tests/`; `tests/test_no_torch.py` imports every module without it).
  Host work numpy does badly goes to `csrc/` (AVX-512), the rest to the NPU.
- Use `uv`, never `pip`.
- Tests: run the test file of what you changed, not the whole suite (slow), and no long LLM measuring sessions unless
  asked. Whole-model tests are marked `slow` and run only with `--run-slow`.
- Commit whenever it helps.
- User documentation lives in `docs/`; README.md stays an overview that links there. A change to a tool's options,
  a model's files or a requirement updates its page.
- This file holds the rules and commands most tasks need, a short map, and where the rest is. Design notes go in module
  docstrings, reusable lessons in `.claude/skills/`; research scripts, measurements and history stay out of the repo.
  Fix or remove a stale fact wherever you find it.
- Comments, docstrings and skills describe the present: no dates, commit ids, removed code, fixed bugs or "was / now"
  stories (git keeps those). A rejected alternative stays only as a comparison that justifies the current code.
  Skills are condensed: each fact in one place, pointed to from the others.

## Setup and commands

```bash
source .venv/bin/activate                                           # in every shell
uv venv && uv sync && uv pip install --no-build-isolation --no-deps -ve .  # first setup (uv sync: with the dev group)
uv pip install --no-build-isolation --no-deps -ve .                 # rebuild after csrc/ changes or added / moved modules
artlite-cli -m <org>/<repo>/<file>.gguf -p "..." -n 100             # run a model (NPU)
artlite-bench -m <org>/<repo>/<file>.gguf -p 512,4096 -n 128        # its speeds, as llama-bench (NPU)
artlite-server -m <org>/<repo>/<file>.gguf                          # OpenAI API + web chat on :8080, as llama-server (NPU)
artlite-server                                                      # router mode: the models dir's models, one process each
artlite-probe                                                       # NPU clock, dispatch, DRAM rates: docs/probe.md
pytest tests/<package>/test_<module>.py -k <config-id>              # needs the NPU
pytest tests -m smoke                                               # after a change: ~1 min with cached designs (NPU)
ARTLITE_BACKEND=xrt pytest tests -m smoke                           # the same on the XRT backend (Windows' code path)
pytest tests --run-slow ...                                         # also the whole-model tests
pytest tests --dump-designs DIR                                     # render every design, no NPU
ruff check <changed files>                                          # the tree passes: fix all it reports
cmake -S . -B build-tests -DARTLITE_BUILD_TESTS=ON && cmake --build build-tests -j \
  && ctest --test-dir build-tests                                   # C++ runtime gtests (NPU)
```

- Rebuild with `--no-deps`: without it uv reinstalls `llvm_aie`, and the new `clang++` timestamp invalidates every
  cached design.
- `build/` is scikit-build's editable-install tree; configure standalone CMake builds elsewhere (`build-tests/`).
  libdrm and gtest are installed system-wide.
- The toolchain is pinned in `pyproject.toml` (`mlir_aie`, `llvm_aie` = Peano). `aiecc` is on PATH; Peano's
  `clang++` is `artlite.operators.toolchain.clangpp()`. Upgrade with the `update-deps` skill.
- Style: match the surrounding code (lines up to ~120 columns) and don't reformat code you are not changing
  (`ruff format` is off). `ruff check` (rules in `pyproject.toml`) passes on the whole tree, so whatever it reports
  came from your change.

## Map

Module by module, with how the models run on the NPU: `docs/code-map.md` (read it to find where something lives; a new
or moved module updates its row).

| path | what |
|---|---|
| `csrc/` | the C++ runtime (device, buffers, HW contexts, dispatch, profiler) and the host's hot code (weight packers, dequantization, sampling, the MoE prefill's host step, BPE); the Python module `bindings/`, the XRT backend `shim_xrt/` |
| `src/artlite/` | the runtime's Python side: `settings.py` (every `ARTLITE_*` switch), `backend.py`, control programs and overlays (`txn.py`, `elf_parser.py`, `overlay.py`), model files (`model_config.py`, `gguf_model.py`, `hf_model.py`, `paro.py`), `npu_regs.py`, `parallel_compile.py`, the schedule tools (`measure_ii/`, `variant_search/`) |
| `src/artlite/operators/` | the operator framework (`base.py`, `build.py`, `templating.py`, `kernel_source.py`, `checks.py`, `words.py`), `kmath/` (math functions per contract), `paging.py` (code pages), `fused/` (several designs in one ELF) |
| `src/artlite/include/` | the kernels' headers: `operators.h`, `kernel_lib.h` |
| `src/artlite/{formats,gemm,decode,prefill,moe,vision}/` | the subsystems the models share: the GGUF tile formats, the GEMMs on them, the decode overlay, the chunk prefill, MoE layers, the vision overlay (image networks, speech) |
| `src/models/<model>/` | a language model per GGUF architecture (`llama`, `gemma4`, `qwen35`, `lfm`, `spark`, `flint`, `inkling2`), the media encoders (`minicpmv`, `lfm3vl`, `lfm3a`, `qwen3vl`, `inkling2v`), MOSS-Transcribe-Diarize (`moss_tr`: a checkpoint's Qwen3 decoder and its audio tower), Stable Diffusion 1.5 (`sd15`) |
| `src/models/server/` | `artlite-server`: the HTTP APIs, the NPU workers, the prompt cache, tool calls, router mode |
| `src/models/*.py` | what the models share: `Model`, `NpuModel`, `OverlayModel`, several sequences (`lanes.py`), sampling, chat templates, structured output, drafts; `artlite-cli`, `artlite-bench`, `artlite-probe` |
| `tests/` | a directory per package of `src/`; the shared helpers (`conftest.py`, `common.py`, `numerics.py`, `model_files.py`), llama.cpp's goldens (`goldens/`), CPU references, `dma_model.py`, `design_diff.py`, `txn_dump.py`, `schedule_ledger.py`, the C++ gtests (`csrc/`), `datasets/` |
| `docs/` | the user documentation (index: `docs/README.md`); the README's charts (`data/`, `charts.py`) |
| `scripts/`, `flake.nix`, `.github/` | the FCLK hold of Strix Halo, NixOS's development shell, GitHub's workflows (`ruff check`, the imports on Python 3.12) and templates |

## Where the rest is

- Writing or changing an operator, a kernel template or an array design: read `docs/development.md` "Operators"
  first (the `ArtcOperator` contract, the shared pieces to reuse, compile, cache and dispatch, a core's budget).
- One NPU dispatch must finish within the driver's hang detection (2 s): split long work into pieces well under
  it. A killed job fails with `DispatchError ... TIMEOUT (state 8)` and "TDR timeout detected" in dmesg;
  `health["txn_op_idx"]` is the last control-code operation the firmware ran (0xFFFFFFFF: none reported).
- The other NPU pitfalls (the device heap, `FusedOperator`'s buffers, kernel statics, rounding, alignment, shim BD
  limits, host-built transactions): `docs/development.md` "NPU pitfalls"; read them before writing a kernel, a
  design, a dispatch or a transaction.
- Performance or accuracy work: load the matching skill first (`aie-kernel-perf-optimization`, `measure-ii`,
  `profiling`, `npu-prefill`, `iron-porting`, `aie2p-coding`, `artlite-bench`).
- Measure one NPU job at a time: each variant in its own process, processes alternated, median and p10 over several
  reps, an idle CPU (a memory-heavy CPU job slows DMA-bound NPU work a lot).
- A change that keeps the per-element arithmetic order must produce bit-identical output, not just pass the
  tolerance test.
- A change of kernel code or of the toolchain refreshes `tests/schedule_ledger.jsonl` in the same commit
  (`python -m tests.schedule_ledger record APP ... --update`); refactoring checks (design and transaction dumps
  before and after, data layouts on the host): `docs/development.md` "Checking a change".
- Long prompts use the essay corpus (`tests/datasets/`): the essays concatenated in file-name order.
- References (the AIE API headers, MLIR-AIE's examples, Peano's backend and intrinsics, the AMD manuals by
  number): `docs/development.md` "References".
