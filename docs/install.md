# Installation

## Hardware

An AMD processor with an XDNA 2 NPU, whose AI Engine array is AIE2P: Strix Point and Krackan Point (Ryzen AI 300),
Strix Halo (Ryzen AI Max). Artlite is developed and measured on a Ryzen AI Max+ 395. The XDNA 1 NPUs of Phoenix and
Hawk Point (AIE2) are not supported: Artlite's kernels and designs are written for AIE2P only.

The NPU reads the weights from system memory, so a model needs about its file's size of RAM plus its KV cache
(Qwen3.6 27B Q4_K_M: 16 GB of weights).

## Software (Linux)

| | |
|---|---|
| Linux with the `amdxdna` driver | AMD's [xdna-driver](https://github.com/amd/xdna-driver) builds the driver module and installs the NPU firmware; Artlite is developed with it. Linux has its own `amdxdna` since 6.14, with the same ioctls (tested with Linux 7.2; before 7.2 a process's control code and PDIs must fit in 64 MB) |
| access to `/dev/accel/accel0` | the device node belongs to the `render` group: `sudo usermod -aG render $USER`, then log in again |
| libdrm | the runtime's build links it (`pkg-config libdrm`) |
| CMake 3.20 or newer, a C++23 compiler (GCC 13+, Clang 17+) | the runtime and its Python bindings |
| Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) | uv installs everything else into a venv |

`uv sync` installs the AIE toolchains (mlir-aie, Peano) into the venv; XRT and Vitis are not needed.

## Build (Linux)

```bash
git clone https://github.com/artlite-project/artlite && cd artlite
uv venv && source .venv/bin/activate        # activate it in every shell
uv sync --extra examples                    # dependencies and toolchain wheels; examples: what the tools need
uv pip install --no-build-isolation -ve .   # Artlite, editable: scikit-build-core builds csrc/ in build/
```

Rebuild after changing `csrc/`, or after adding or moving Python modules:

```bash
uv pip install --no-build-isolation --no-deps -ve .
```

- `--no-deps`: without it uv reinstalls `llvm_aie`, and the new `clang++` timestamp invalidates every cached design.
- `--no-build-isolation`: builds against the venv's nanobind, so its headers stay for IDEs (an isolated build
  removes them afterwards).

## Windows

Artlite runs on Windows 11 through XRT instead of the driver's ioctls (`csrc/shim_xrt/`). What differs from Linux:

| | |
|---|---|
| NPU driver | the Ryzen AI driver 32.0.203.304 or newer (Task Manager > Performance > NPU shows it) |
| XRT | its headers, `xrt_coreutil.lib`, `xclbinutil.exe` and `aiebu-asm.exe` in one directory (usually `C:\Xilinx\XRT`): an XRT 2.21 build or SDK, version 2.21 or newer |
| C++ compiler | MSVC 2022 (Build Tools, "Desktop development with C++"), C++23 (`/std:c++latest`) |
| CMake, Python, uv | as above |

The build finds XRT through `XILINX_XRT`, and the module it builds remembers that directory for `xclbinutil` and
`aiebu-asm`. Set `XILINX_XRT` for the build only: while it is set, XRT looks for its driver library there, and every
run fails with `No such library '...\xrt_core.dll'`.

```powershell
git clone https://github.com/artlite-project/artlite; cd artlite
uv venv; .venv\Scripts\activate
$env:XILINX_XRT = "C:\Xilinx\XRT"   # XRT's directory, usually this one
uv sync --extra examples
uv pip install --no-build-isolation -ve .
Remove-Item Env:XILINX_XRT
```

- Each program is an xclbin that wraps its PDI (`main.xclbin`, from aiecc; a fused ELF's beside the ELF, of its first
  PDI, packed with `xclbinutil` in aiecc's form): XRT loads nothing else.
- A fused ELF's control program (fused operators, the prefill's library programs) runs as a transaction of its
  xclbin's kernel: its addresses are resolved on the host as on Linux, PDI loads included, and its DDR patches become
  no-ops (the firmware would take their addresses from the command's buffers).
- The NPU reaches only the data buffers a command lists, at most 5. A program of more buffers lists the regions they
  lie in: buffers under 256 MB share 1 GB regions of the process, a larger one has its own. Instruction buffers
  (control code, PDIs, the scratchpad) need not be listed.
- Each hardware context maps a buffer at its own device address (`Buffer.paddr_in`), and only once a context exists
  in the process (XRT's rule): a program opens its context before it resolves addresses.
- Not available: the profiler's register reads and `Device.clocks()`, `busy_ns()`, `last_async_error()` (they return 0 or
  nothing).
- The planning step of the tests and `python -m artlite.parallel_compile` work as on Linux.

## Models directory

`-m` takes a model file or directory, or a path relative to the models directory: `ARTLITE_MODELS_DIR`, by default
`~/.artlite/models`. Inside it, models follow Hugging Face's names, `<account>/<repository>/<file>`:

```bash
uvx --from huggingface_hub hf download lmstudio-community/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir ~/.artlite/models/lmstudio-community/Llama-3.2-1B-Instruct-GGUF
artlite-cli -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Hello"
```

Which files run: [Models](models/README.md).

## Build cache

The first run of a model compiles its NPU designs into `~/.cache/artlite` (`ARTLITE_CACHE_DIR`), keyed by their
content: later runs load them. Stable Diffusion's npu backend and MiniCPM-V's vision encoder also keep their built
weight images and dispatches there (~1 GB and ~0.7 GB per model, the latest build only). The cache only grows; deleting it is safe (the next runs rebuild
what they need).

## Troubleshooting

| symptom | cause |
|---|---|
| no `/dev/accel/accel0` | the driver is not loaded: `sudo modprobe amdxdna`; `dmesg \| grep amdxdna` shows why (firmware not found, unsupported device) |
| `Permission denied` on `/dev/accel/accel0` | your user is not in the `render` group (above) |
| `artlite.DispatchError` with state `TIMEOUT` | the driver's hang detection stopped a command; its `health` is the firmware's report ([Design: error handling](design.md#error-handling)) |
