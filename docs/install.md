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
| Linux with the `amdxdna` driver | AMD's [xdna-driver](https://github.com/amd/xdna-driver) builds the driver module and installs the NPU firmware; Artlite is developed with it. Linux has its own `amdxdna` since 6.14, with the same ioctls (tested with Linux 7.2, and with Linux 7.0's driver: before 7.2 it gives a process one 64 MB device heap, where Artlite runs a prompt's control code in smaller pieces) |
| access to `/dev/accel/accel0` | the device node belongs to the `render` group: `sudo usermod -aG render $USER`, then log in again |
| the locked-memory limit unlimited | the driver pins the NPU's buffers (a model's weights: GBs) within `ulimit -l`, which is 8 MB on many distributions: below |
| libdrm | the runtime's build links it (`pkg-config libdrm`) |
| CMake 3.20 or newer, a C++23 compiler (GCC 13+, Clang 17+) | the runtime and its Python bindings |
| Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) | uv installs everything else into a venv |

`uv sync` installs the AIE toolchains (mlir-aie, Peano) into the venv; XRT and Vitis are not needed.

If `ulimit -l` prints a number, not `unlimited`, raise the limit (Artlite raises its own up to the hard limit, which
only root can raise), then log in again:

```bash
printf '* soft memlock unlimited\n* hard memlock unlimited\n' | sudo tee /etc/security/limits.d/99-memlock.conf
```

A shell or tmux server started before keeps the old limit. A systemd service (`artlite-server` as one) takes its own:
`LimitMEMLOCK=infinity` in its unit.

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

After a `git pull`, run `uv sync --extra examples` and then the rebuild: a commit may pin other toolchain wheels or
change `csrc/`. `artlite-cli`, `artlite-server`, `artlite-bench`, `artlite-probe` and the tests warn when the venv's
`mlir_aie` or `llvm_aie` is not the version `pyproject.toml` pins; the tools warn, and the tests do not start, when the
native module is not built from the checkout's `csrc/` (a build records the hash of its sources).

## Windows

Artlite runs on Windows 11 through XRT instead of the driver's ioctls (the XRT backend, `csrc/shim_xrt/`). What
differs from Linux:

| | |
|---|---|
| NPU driver | the Ryzen AI driver 32.0.203.304 or newer (Task Manager > Performance > NPU shows it) |
| XRT | its headers, `xrt_coreutil.lib` and `aiebu-asm.exe` in one directory (usually `C:\Xilinx\XRT`): an XRT 2.21 build or SDK, version 2.21 or newer |
| C++ compiler | MSVC 2022 (Build Tools, "Desktop development with C++"), C++23 (`/std:c++latest`) |
| CMake, Python, uv | as above |

The build finds XRT through `XILINX_XRT`, and the module it builds remembers that directory for `aiebu-asm` (aiecc
packs fused ELFs with it; mlir-aie's Windows wheels lack it). Set `XILINX_XRT` for the build only: while it is set,
XRT looks for its driver library there, and every run fails with `No such library '...\xrt_core.dll'`.

```powershell
git clone https://github.com/artlite-project/artlite; cd artlite
uv venv; .venv\Scripts\activate
$env:XILINX_XRT = "C:\Xilinx\XRT"   # XRT's directory, usually this one
uv sync --extra examples
uv pip install --no-build-isolation -ve .
Remove-Item Env:XILINX_XRT
```

- Power mode: `artlite-cli` and `artlite-server` raise the NPU's power mode from `default` to `performance` at their
  start (`xrt-smi configure --pmode performance`, as FastFlowLM does), and keep any other mode set by hand;
  `artlite-bench` measures the mode that is set. On a Strix Point laptop (driver 32.0.20102.3930) `default` and
  `performance` ran the array at its top clock alike; `turbo` cut the dispatch round trip from 94 to 58 us;
  `balanced` runs the array at 1267 MHz, `powersaver` at 800 (`artlite-probe`).
- A hardware context opens from an xclbin that the runtime builds around the program's PDI in memory (aiecc's
  sections): programs are built as on Linux.
- A fused ELF's control program (fused operators, the prefill's library programs) runs as a transaction of the
  context's kernel: its addresses are resolved on the host as on Linux, PDI loads included, and its DDR patches become
  no-ops (the firmware would take their addresses from the command's buffers).
- The NPU reaches only the data buffers a command lists, at most 5. A program of more buffers lists the regions they
  lie in: buffers under 256 MB share 1 GB regions of the process, a larger one has its own. Instruction buffers
  (control code, PDIs, the scratchpad) need not be listed.
- Each hardware context maps a buffer at its own device address (`Buffer.paddr_in`), and only once a context exists
  in the process (XRT's rule): a program opens its context before it resolves addresses.
- Not available: the profiler's register reads (counters; traces work) and `Device.clocks()`, `busy_ns()`,
  `last_async_error()` (they return 0 or nothing).
- The planning step of the tests and `python -m artlite.parallel_compile` work as on Linux.

### The XRT backend on Linux

Where XRT (2.21 or newer) and its NPU plugin (xdna-driver's `libxrt_driver_xdna`) are installed, the Linux build also
makes the XRT backend, `artlite._native_xrt` (`-DARTLITE_XRT=OFF` skips it). `ARTLITE_BACKEND=xrt` selects it for a
process: the tests and the tools then run Windows' code path on Linux, but for the register reads above. Prefill is a
few percent slower there than on the default backend: the firmware runs a control program slower as a transaction
than as an ELF.

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
| `Cannot allocate memory` / `Resource temporarily unavailable` from a BO, "as ulimit -l is 8192 KB" | the locked-memory limit (above) |
| `artlite.DispatchError` with state `TIMEOUT` | the driver's hang detection stopped a command; its `health` is the firmware's report ([Design: error handling](design.md#error-handling)) |
| Strix Halo: the machine resets during NPU work while the iGPU is busy too (a browser, a video call, llama.cpp); the next boot's kernel log says `Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood event` | an AMD bug in Strix Halo's data fabric under GPU and NPU load at once, not Artlite's ([ROCm/TheRock#8709](https://github.com/ROCm/TheRock/issues/8709)); holding FCLK avoided it in Artlite's tests, though not on every board ([below](#strix-halo-resets-under-npu-and-igpu-load)) |
| opening `/dev/accel/accel0` fails; `dmesg` says `SVA bind device failed` | the IOMMU is off (`amd_iommu=off` on the kernel command line, a common iGPU tuning tip): the NPU needs it. Remove the parameter; `iommu=pt` keeps the NPU working |
| `artlite-probe` says `host buffers: 4 KB pages` on Linux | transparent huge pages are off (`/sys/kernel/mm/transparent_hugepage/enabled` reads `[never]`): the NPU then translates a page per short DMA row, and its write rows run at a fifth of their rate. `madvise` (most distributions' default) or `always` gives Artlite's buffers 2 MB pages |

### Strix Halo: resets under NPU and iGPU load

As far as known only Strix Halo is affected: other machines can skip this. Before a reset the kernel log
(`journalctl -k`) usually has deferred errors: `[Hardware Error]: Deferred error, no action required.`, with
`MC28_STATUS` and `Syndrome: 0x000000005b000000`. In a stress test (an NPU prefill beside llama.cpp on the iGPU) the
errors and the resets stopped with the data fabric's clock (FCLK) held at its top level, 2000 MHz, and came back when
it was released. The hold costs power and heat: use it on mains power, while the NPU works. On some other boards the
hold was not enough ([ROCm/TheRock#8709](https://github.com/ROCm/TheRock/issues/8709)): there the BIOS's "Eco Mode"
disabled, or no NPU work while the iGPU computes, stopped the resets.

```bash
D=/sys/class/drm/card0/device                                  # the iGPU (amdgpu)
echo manual | sudo tee $D/power_dpm_force_performance_level
echo 7 | sudo tee $D/pp_dpm_fclk                               # the top level
grep '\*' $D/pp_dpm_fclk                                       # ~8 s later: 7: 2000Mhz * (another level: write 7 again)
echo auto | sudo tee $D/power_dpm_force_performance_level      # the release (a reboot also releases it)
```
