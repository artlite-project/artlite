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

- **Never write a nonzero shim BD AxQoS.** It hung the NPU; only a reboot recovered it.
- **Never kill a process that holds an NPU context** (let it finish or time out). A killed process has wedged the
  NPU beyond a driver reload. Give compile-only scripts, which open no context, short timeouts instead.
- **Kernel names**: every namespace-scope name of a kernel source (entry points, helpers, tables, types, macros) is
  `{{ p }}_<name>` with `p = kernel_prefix(stem)`, and the design declares its kernels under the same names. The build
  names each definition by its code (`operators/kernel_source.py`), so a fused program's `kernel.cc` holds every
  definition once; a name written out in full collides as soon as two configurations meet in one fused program.
- **Weights stay at their quantization's size.** Never expand them on the host; prefill and decode read one weight
  image in place. Lossless, same-size re-layouts are fine (images are packed at every load: nothing on disk).
- Operators are self-contained: never import IRON operators.
- Layers (Map): `models` -> `artlite`'s subsystems (`decode`, `prefill`, `moe`, `vision`, `gemm`, `formats`) -> the
  framework (`artlite.operators`, `artlite.overlay`) -> the runtime. Code that two models share lives in `artlite`, in
  the subsystem it belongs to, never in one model's package for another to import (a model package imports another
  only to run that model: qwen35 runs MiniCPM-V's vision encoder); `artlite` never imports `models`.
- `src/` holds what the apps and tools run: test harnesses (standalone operators of one kernel), CPU references and
  other test-only code live in `tests/`, a directory per package of `src/`
  (`tests/<subsystem or model>/test_<module>.py`; the top-level modules' tests at the top). Layout inverses and
  readbacks stay next to the code they mirror.
- No absolute paths in code, tests or scripts: derive them from `__file__` or package constants. Model files only
  through `artlite.model_config` (below).
- Licenses: Artlite is LGPL-2.1-or-later and its own files carry no license header. Code from another project keeps
  its copyright and license notices plus a "Modified for Artlite" line, and is listed with its license in a LICENSE
  next to it (`csrc/shim_xdna/LICENSE`).
- Python: annotate all parameters and return types; functions with more than 3 parameters take keyword-only
  arguments. Code runs on Python 3.12 (`requires-python`) and every newer version a venv may have, with no
  compatibility code for older ones; `from __future__ import annotations` only where 3.12 needs it (an annotation
  naming a class defined later in the module or imported only under `TYPE_CHECKING`, or an attribute of a package
  still loading, such as `artlite.Device` in a module that `import artlite` loads).
- Use `uv`, never `pip`.
- Tests: run the test file of what you changed, not the whole suite (slow), and no long LLM measuring sessions unless
  asked. Whole-model tests are marked `slow` and run only with `--run-slow`.
- Commit whenever it helps.
- User documentation lives in `docs/`; README.md stays an overview that links there. A change to a tool's options,
  a model's files or a requirement updates its page.
- This file holds rules, commands and the map of the code. Design notes go in module docstrings, reusable lessons
  in `.claude/skills/`; research scripts, measurements and history stay out of the repo. Fix or remove a stale
  fact wherever you find it.
- Comments, docstrings and skills describe the present: no dates, commit ids, removed code, fixed bugs or "was / now"
  stories (git keeps those). A rejected alternative stays only as a comparison that justifies the current code.
  Skills are condensed: each fact in one place, pointed to from the others.

## Setup and commands

```bash
source .venv/bin/activate                                           # in every shell
uv venv && uv sync --extra examples && uv pip install --no-build-isolation -ve .  # first setup
uv pip install --no-build-isolation --no-deps -ve .                 # rebuild after csrc/ changes or added / moved modules
artlite-cli -m <org>/<repo>/<file>.gguf -p "..." -n 100             # run a model (NPU)
artlite-bench -m <org>/<repo>/<file>.gguf -p 512,4096 -n 128        # its speeds, as llama-bench (NPU)
artlite-server -m <org>/<repo>/<file>.gguf                          # OpenAI API + web chat on :8080, as llama-server (NPU)
artlite-server                                                      # router mode: the models dir's models, one process each
artlite-probe                                                       # the NPU's clock, dispatch and DRAM rates vs Strix Halo's
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
  `clang++` is `artlite.operators.base.CLANGPP`. Upgrade with the `update-deps` skill.
- Style: match the surrounding code (lines up to ~120 columns) and don't reformat code you are not changing
  (`ruff format` is off). `ruff check` (rules in `pyproject.toml`) passes on the whole tree, so whatever it reports
  came from your change.

## Map

| path | what |
|---|---|
| `csrc/` | the C++ runtime (device, buffers, HW contexts, ELF dispatch, profiler); API and errors: `docs/design.md`; the XRT backend: `shim_xrt/` (the shim over XRT, contexts from in-memory xclbins) and `artlite/kernel_xrt.cpp` (dispatch as XRT runs) |
| `src/artlite/backend.py`, `portable.py` | which backend is `artlite._native` (`ARTLITE_BACKEND`) and what the Python side does differently on XRT; the operating system's part of the Python side (file locks, process groups) |
| `src/artlite/{txn,elf_parser}.py`, `overlay.py` | control programs: transactions built on the host (the `insts.bin` format), full ELFs and the patching of their control code; the host runtime of the overlays, designs run by host-built transactions (`Sequence`, `OverlayDevice`, `Transaction`, `WeightBuffer`) |
| `src/artlite/operators/` | the operator framework: `base.py` (`ArtcOperator`: render, compile, cache, dispatch), `kernel_source.py` (kernel names by content, `kernel_prefix`; the translation unit `kernel.cc` of one or several operators), `checks.py`, `render_cache.py`, and the pieces operators share (How an operator works) |
| `src/artlite/operators/kmath.py` | math functions generated per contract (`km` in every template: exp2, recip, expm1, softplus, sigmoid, silu; arrays through the gather LUT: `act`, `table_pass` and the tables; `silu_tanh`), each variant's bit-exact numpy model in `tests/operators/kmath_sim.py` (`tests/operators/test_kmath.py`) |
| `src/artlite/operators/paging.py` | code pages: core programs beyond 16 KB of program memory, loaded by the cores themselves (the page linker and its checks, page streams, the pager's kernel source) |
| `src/artlite/operators/fused/` | several designs in one ELF: `FusedOperator` (a runlist in one dispatch), `Library` (programs assembled on the host; their configurations' tile writes as control packets: `ctrlpkt`) |
| `src/artlite/include/` | the kernels' headers: `operators.h` (loop-pragma macros; the precompiled header of every kernel), `kernel_lib.h` (shared kernel helpers, namespace `al`) |
| `src/artlite/formats/` | the GGUF tile formats: one slot per format family (`formats.common.Slot`) |
| `src/artlite/gemm/` | the GEMMs on the weight tiles: GemmQ, GemmQd (a dequantizing core per column), their MAC loop |
| `src/artlite/decode/` | the decode overlay of the language models: its design (`design.py`) and the main / helper cores' GEMV phases, the weight image (`image.py`), the KV record format (`kv.py`), the attention core of Gemma and Llama (`attn.py`), the models' shared host side (`model.py`) |
| `src/artlite/prefill/` | the chunk prefill on the decode overlay's buffers: chunk programs, the GEMM planner, the residual norms, the attention of a column per KV head (Llama's) |
| `src/artlite/moe/` | MoE layers: the routing on the decode overlay's cores, the prefill's router, expert work lists and gathers |
| `src/artlite/vision/` | the vision overlay: a whole image network per dispatch on one static design (host-built transactions, core roles as code pages the cores load); a network's tensors and layers (`layers.py`) |
| `src/artlite/parallel_compile.py` | all parallel compilation |
| `src/artlite/{model_config,gguf_model,measure_ii,variant_search,profiling,probe}.py` | model files, schedule tools, profiling, `artlite-probe`'s designs |
| `src/models/{llama,gemma4,qwen35}/` | the language models, a package per GGUF architecture: on the NPU (the weight image `image.py`, the decode design `decode.py` and its host side `overlay.py`, the prefill `prefill.py`, the model's own operators) and on the host (tokenizer, chat template, inputs); `models.open_model` picks one by a GGUF file's architecture |
| `src/models/minicpmv/` | MiniCPM-V 4.7's vision side (its mmproj): the encoder on the vision overlay (`encoder.py`), an image's embeddings and tokens (`npu.py`), its slices and positions (`host.py`); qwen35's language model takes them (`models/sequence.py`: token ids and embedding rows) |
| `src/models/sd15/` | Stable Diffusion 1.5 text to image from a diffusers directory (`models.open_pipeline`): CLIP tokenizer, samplers, the pipeline over a `Backend` of three parts (text encoder, UNet, VAE decoder), the fp32 torch CPU backend, the NPU UNet and VAE decoder (`npu.py`) on the vision overlay: the 64x64 level (`level64.py`), the 32x32 .. 8x8 levels (`unet.py`; 32x32 in bands), the step (`model.py`), the VAE decoder (`vae.py`) |
| `src/models/server/` | `artlite-server`: the HTTP API (FastAPI: llama-server's routes and shapes), the NPU worker and its prompt cache, llama.cpp's prebuilt web UI (pinned); router mode (`router.py`: a model server process per model, `child.py`; `catalog.py`: the models directory's models, presets) |
| `src/models/common.py`, `cli.py`, `bench.py`, `probe.py` | the shared model interface (`Model`, `NpuModel`: generation loop, sampling; `OverlayModel`: steps and prompts on an overlay), `artlite-cli`, `artlite-bench` and `artlite-probe` (its designs: `artlite.probe`) |
| `tests/` | the tests, a directory per package of `src/` (`operators/`, `decode/`, ..., `llama/`, `qwen35/`, ...; the top-level modules' at the top): `conftest.py` (the smoke set, `--dump-designs`), `common.py` (the essay corpus, cached CPU references in `refs/`, gitignored), `<model>/ref.py` (model references), `llama/cpu.py`, `gemma4/cpu.py` (fp32 CPU models), `decode/attn_ref.py` (the decode attention cores' host model), `llama_cpp.py` (llama.cpp's CPU build), `operators/elementwise.py` (1-D stream operators, the activations' harnesses), `decode/mmv.py` (the decode GEMV's host model and its harness `MmvOperator`), `dma_model.py` (the host DMA model), `design_diff.py` (design dumps compared), `csrc/` (the C++ runtime's gtests) |
| `tests/datasets/` | the long-prompt corpus (gitignored; `fetch_datasets.py` fetches it, `common.essay()` reads it) |
| `.github/` | GitHub: the `ruff check` workflow (on pushes to `main` and pull requests), the issue and pull request templates, the security policy |
| `docs/` | the user documentation (index: `docs/README.md`): installation, the tools' options, models (`models/`: tested files, plans, one page per model), benchmark methodology (the charts' test system and measured files), design, development; the README's performance charts: `data/*.json` (Artlite's and other runtimes' measurements), `charts.py` draws them as `images/*.svg`; `diagram.py` draws the README's how-it-works diagram; `images/`: also the logo and the docs' pictures |

## How an operator works

- One file per operator: an `ArtcOperator` subclass implementing
  - `get_kernel_src()`: C++ from a Jinja2 template (`JINJA_ENV`: undefined variables are errors) with shapes, trip
    counts and strides as compile-time constants, its names under `kernel_prefix` (Rules);
  - `get_mlir_artifact()`: the array design in `aie.iron` (ObjectFifos, Workers, runtime sequence);
  - `get_arg_spec()`: buffer directions, shapes, dtypes.
- Reuse the shared pieces instead of copying (`artlite/operators/`, the headers in `artlite/include/`): `kmath.py`
  (exp2, recip, sigmoid, SiLU, GELU, ...: `km.exp2(...)` in a template; arrays through the gather LUT:
  `km.act(km.SILU)`, `km.table_pass`), `kernel_lib.h` (`inv_rms`, bfp16 panels, lane shuffles), `design.py` (`COLS`,
  `ROWS`, `range32`: Worker loops with an i32 induction variable, `linear_tap`), `bfp16.py` (host bfp16ebs8 conversions
  and panels), `l1_layout.py` (`pin_l1_buffers`), `util.py` (template `Attr`, workspace `regions`, bf16 constants,
  `arg_bytes`).
- A new operator lives with what it serves (an `artlite` subsystem, or its model's package: Layers) and is tested in
  `tests/<package>/test_<module>.py`. Tests need no compile plumbing: pytest plans the selected tests in a subprocess
  and builds their designs in the background.
- `compile()` = render (`build_spec()`, in the calling thread) + build (Peano: `kernel.cc` -> `kernel.o`; `aiecc`:
  `design.mlir` -> `main.pdi` + `insts.bin`). `kernel.cc` has a section per operator (a fused ELF's: each definition
  in the first section that uses it, a later one lists what it shares). `compile_all(ops)` builds many in
  parallel: use it wherever several designs are compiled before anything runs (`parallel_map` for other compile
  work). `ARTLITE_COMPILE_JOBS` bounds the builds; `python -m artlite.parallel_compile CMD ...` fills the cache for
  any command.
- Every compile runs `operators/checks.py`: each core's deepest kernel call against its Worker `stack_size`, kernel
  signatures against the design, rounding kernels that do not set the rounding mode, runtime-library arithmetic
  (`ARTLITE_SKIP_CHECKS=1` skips them).
- Builds are cached per content hash in `~/.cache/artlite/rtc_<hash>/` (inspect the artifacts there;
  `ARTLITE_CACHE_DIR` moves it, `ARTLITE_IGNORE_CACHE=1` rebuilds). A render manifest (`render_cache.py`, keyed by
  class, constructor arguments and the sources' hash) skips rendering cached designs; a module global changed at
  runtime is not in that key, so a monkeypatched test needs its own operator names.
- First use (`op.device` or the first dispatch) opens a HW context and uploads the program. `dispatch(*buffers)`
  submits in sequence order and waits, without host <-> device syncs; `run(*tensors)` is the one-shot wrapper;
  `artlite.from_torch` / `artlite.empty(..., device=)` give zero-copy buffers; `compile(device=)` chains operators
  on one device.
- NPU2: 8 columns of compute rows 2-5, a memtile (row 1, 512 KB) and a shim (row 0). A core has 64 KB of data
  memory in four 16 KB regions (budget ~63 KB with the stack) and 16 KB of program memory.
- Compile-time switches: `ARTLITE_MEASURE_II=1` (per-loop II, bounds, stack frames), `ARTLITE_GENERATE_ASM=1` (keep
  `kernel.cc.s`), `ARTLITE_LOG_COMMANDS=1`, `ARTLITE_PROFILE=...` (the `profiling` skill).

## Models

- `artlite.model_config` reads the GGUF metadata (`LlamaConfig`, `GemmaConfig`, `QwenConfig`). Model files live in
  `<models dir>/<HF org>/<HF repo>/`; the models dir (`ARTLITE_MODELS_DIR`, default `~/.artlite/models`) is spelled
  only in `model_config.MODELS_DIR`: get paths from its helpers (`llama_gguf`, `gemma_gguf`, `qwen_gguf`, ...), in
  tests and research scripts too.
- Decode: each language model runs one overlay design whose cores interpret metadata in their inputs, driven by
  host-built transactions (`artlite/decode/`: the models' shared host side `model.py`, the KV record format `kv.py`,
  the attention core of Gemma and Llama `attn.py`).
- Prefill: layer-major chunk programs (`artlite/prefill/chunks.py`) of GEMMs on the decode's weight image
  (`artlite/prefill/gemm.py`), the residual norms (`norm.py`: bf16 for Qwen and Llama, fp32 for Gemma) and Llama's
  attention (`attn.py`), writing the caches the decode continues from (the `npu-prefill` skill).
- Image networks (Stable Diffusion's UNet and VAE decoder, MiniCPM-V's vision encoder) run whole on the vision overlay
  (`artlite/vision/`), a dispatch per network.
- Each model's design is in its package's module docstrings.

## Pitfalls

- One NPU dispatch must finish within the driver's hang detection (2 s): split long work into pieces well under
  it. A killed job fails with `DispatchError ... TIMEOUT (state 8)` and "TDR timeout detected" in dmesg;
  `health["txn_op_idx"]` is the last control-code operation the firmware ran.
- A runtime-sequence argument stays below 4 GB (ELF relocation addends are 32-bit).
- An ELF's control code goes into the process's device heap (64 MB heap BOs up to 512 MB; one before Linux 7.2, where
  `Library`'s pieces shrink to fit and a full heap raises `HeapFullError`); every ConfigureOp inlines its design's
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

## Performance and accuracy work

- Load the matching skill first: `aie-kernel-perf-optimization` (where the time goes, bounds, pipelining, L1
  placement, numerics, data movement, rejected ideas), `measure-ii` (loop schedules, `variant_search`), `profiling`
  (counters and traces), `npu-prefill` (prefills on the decode image), `iron-porting` (porting an IRON operator),
  `aie2p-coding` (hardware facts, DMA limits and Peano gotchas for a new kernel or design), `artlite-bench` (a
  model's TTFT, prefill, decode and best MTP S for the README charts).
- Measure one NPU job at a time: each variant in its own process, processes alternated, median and p10 over several
  reps, an idle CPU (a memory-heavy CPU job slows DMA-bound NPU work a lot). Cycles convert at 1.81 GHz.
- A change that keeps the per-element arithmetic order must produce bit-identical output, not just pass the
  tolerance test.
- Refactoring check: `pytest tests --dump-designs DIR` before and after, then
  `python tests/design_diff.py BEFORE AFTER --asm` (identical MLIR; changed kernel sources compile to the same
  asm). The apps' full designs are dumped too (`test_app_designs.py`); the slow tests are not. A change of shared
  kernel code (`kmath.py`, `include/kernel_lib.h`) re-renders only the tests it touches: `--affected REGEX --compile` on
  BEFORE before the change, `--partial --asm` after (`design_diff.py`'s docstring); on the NPU then only the tests of
  the designs whose code changed.
- Data layouts (shim taps, memtile walks, joins) are tested on the host with `tests/dma_model.py`
  (`tests/gemm/test_dma.py` is the pattern).
- Long prompts use the essay corpus (`tests/datasets/`): the essays concatenated in file-name order.
- Research scripts and reports stay outside the repo: move conclusions that must survive into docstrings or a
  skill.

## References

- AIE API headers: mlir-aie's `third_party/aie_api/`.
- MLIR-AIE usage: mlir-aie's `{programming_guide,programming_examples,mlir_exercises}` (large; grep).
- Peano backend (scheduler, pipeliners): llvm-aie's `llvm/lib/Target/AIE/`; the `__builtin_aie2p_*`
  intrinsics: its `clang/include/clang/Basic/BuiltinsAIE2P.def`.
- AMD manuals, cited by number: AM027 / AM029 (AIE-ML v2 = AIE2P architecture manual / register reference), AM020 /
  AM025 (AIE-ML v1 = AIE2), UG1079 / UG1076 (AI Engine kernel and graph programming, tools and flows; chess /
  Vitis, AIE1-oriented). AM029 is Versal's register layout: its memory-module and memtile offsets are the NPU's, its
  shim (NOC / PL module) and core register-file, debug and counter offsets are not (Core_Status: 0x38004 there,
  0x32004 here). The NPU's map: mlir-aie's `third_party/aie-rt/driver/src/global/xaie2pgbl_params.h`.
