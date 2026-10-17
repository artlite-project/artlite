---
name: iron-porting
description: Port an IRON operator (XRT / xclbin, several files - op.py, design.py, aie_kernels/*.cc) to an Artlite RTC operator (one ArtcOperator file - Jinja kernel source, aie.iron design, arg spec - plus its test). Use when porting or re-implementing an operator from IRON or an mlir-aie example, or when writing a new standalone Artlite operator modelled on one.
---

# Porting an IRON operator to Artlite

| | IRON | Artlite |
|---|---|---|
| files | `iron/operators/<name>/` in IRON: `op.py`, `design.py`, `reference.py`, `test.py` | one file `src/artlite/operators/<purpose>/<name>.py` + `tests/rtc/test_<name>.py` |
| base classes | `MLIROperator`, `ChanneledUnaryOperator`, `BinaryElementwiseOperator` | `ArtcOperator` |
| kernel | a C++ file in `aie_kernels/{aie2,aie2p,generic}/` | `get_kernel_src()`: a Jinja template rendered per shape (one `str`) |
| design | `design.py` via `PythonGeneratedMLIRArtifact` | `get_mlir_artifact()`, the same `aie.iron` API |
| build | `XclbinArtifact`, `KernelObjectArtifact`, `SourceArtifact` | `compile()`: Peano `kernel.cc` -> `kernel.o`, aiecc -> PDI + insts (a single operator) or one ELF (fused operators, libraries) |
| runtime | XRT via `aie.utils.NPUKernel` | artlite (`Device`, `Program`, buffers) |

AGENTS.md "How an operator works" is the contract (render and build, the shared helpers, tests without compile
plumbing, exports). Never import IRON components: only the `aie.iron` Python API, the `aie_api` C++ headers and the
artlite runtime.

## 1. The operator class

```python
import torch
from aie.ir import Module

from ..base import ArgSpec, ArtcOperator
from ..design import COLS


class MyOperator(ArtcOperator):
    def __init__(self, *, size: int, tile_size: int, num_columns: int = 1):
        # IRON's __post_init__ validation goes here, before super().__init__()
        assert 1 <= num_columns <= COLS, f"num_columns must be 1-{COLS}"
        assert size % (tile_size * num_columns) == 0, "size must be divisible by tile_size * num_columns"
        self.size, self.tile_size, self.num_columns = size, tile_size, num_columns
        super().__init__()

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.size,), torch.bfloat16), ArgSpec("out", (self.size,), torch.bfloat16)]

    def get_kernel_src(self) -> str:
        return _KERNEL.render(name=self.kernel_name, ts=self.tile_size)

    def get_mlir_artifact(self) -> Module:
        from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
        from aie.iron.device import NPU2
        ...
        return Program(NPU2(), rt).resolve_program()
```

A standalone 1-D elementwise operator is a test harness, not an Artlite operator: subclass `StreamOperator` in
`tests/rtc/elementwise.py` (validation, arg spec and the whole design; the subclass supplies `kernel_name` and
`get_kernel_src()`: `SiluOperator`, `GeluOperator`; two inputs: `AddOperator` in `tests/rtc/test_fused.py`).

## 2. The kernel

Inline the kernel in a template, build it from the shared helpers of `kernel_lib.h` (namespace `al`: exp2, SiLU,
GELU, `inv_rms`, bfp16 panels, lane shuffles, `al::div_small`), or copy IRON's kernel and turn its runtime sizes into
template parameters. Shapes, trip counts and strides rendered as constants let Peano fold, unroll and pipeline (zero
overhead loops, no runtime trip counts); use `constexpr` for derived values.

```python
from ..base import JINJA_ENV  # StrictUndefined: a misspelled template variable is an error, not an empty string
from ..kernel_source import kernel_prefix

_KERNEL = JINJA_ENV.from_string(r"""
extern "C"
void {{ p }}_gelu(bfloat16 __aie_dm_resource_a *__restrict input, bfloat16 __aie_dm_resource_b *__restrict output)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int j = 0; j < {{ ts }}; j += 32) {
        aie::vector<bfloat16, 32> x = aie::load_v<32>(input), xr = aie::load_v<32>(input);
        input += 32;
        aie::store_v(output, al::gelu_lut(x, xr));
        output += 32;
    }
}
""")
```

- `operators.h` (and with it `kernel_lib.h`) is prepended to every kernel: no includes.
- Every namespace-scope name is `{{ p }}_<name>`, helpers too, with `p = kernel_prefix("<stem>")` (a class attribute;
  the stem names the operator in `kernel.cc`), and the design declares the kernels the same way
  (`Kernel(f"{self.p}_gelu", "kernel.o", ...)`): the build names each definition by its code, so a fused ELF or library
  (one translation unit) holds equal code once and never mixes up two configurations (AGENTS.md "Rules").
- The speed rules (512-bit vectors, bank qualifiers, `__restrict` parameters, no division / branches in loop bodies):
  aie-kernel-perf-optimization section 2.

## 3. The design

- Copy the ObjectFifo, Worker, Kernel and Runtime logic from IRON's `design.py`, including the
  `TensorAccessPattern`s of a multi-column distribution (`from aie.helpers.taplib import TensorAccessPattern`).
- Device `NPU2()` (not `npu1_1col` or `npu2_1col`: Artlite targets AIE2P only); `Program(...).resolve_program()`
  without a placer.
- Worker loops: IRON's `range_()` (`aie.iron.controlflow`) for a constant bound, `design.range32()` for a bound read at
  run time, `forever_()` with `Worker(while_true=False)` for a core's infinite loop (aie-kernel-perf-optimization
  section 7). Shared design pieces (`COLS`, `ROWS`, `range32`, `linear_tap`, `stream_walk`): `operators/design.py`.
- All kernel functions compile into one `kernel.o`: every `Kernel` names it.

```python
matvec = Kernel(f"my_op_matvec_bf16_{params}", "kernel.o", [np.int32, np.int32, L1_A_ty, L1_B_ty, np.int32])
silu_mul = Kernel(f"my_op_silu_mul_bf16_{params}", "kernel.o", [L1_C_ty, np.int32])
```

- AIE2P has hardware `vtanh` / `vexp2` and the gather LUTs of `act_lut.py`: drop IRON's `lut_based_ops.o`.

## 4. The test

```python
import pytest
import torch

from artlite.operators import MyOperator
from .common import verify_close


@pytest.mark.parametrize("size,tile_size", [(1024, 256), (2048, 512)])
def test_my_op(size: int, tile_size: int):
    op = MyOperator(size=size, tile_size=tile_size).compile()
    x = torch.randn(size, dtype=torch.bfloat16)
    verify_close(op.run(x), reference(x), rel_tol=0.04, abs_tol=1e-6)
```

- Parametrize over the sizes and tile sizes IRON tests (edge cases included); `torch.equal` for bit-identical
  results, `verify_close` for floating point. Keep the body before `compile()` cheap: the planning pass runs it up to
  the first use of the NPU.
- Export the operator from `src/artlite/operators/__init__.py` (import and `__all__`).

## Patterns

- **Elementwise** (GELU, SiLU, add, mul): fuse it into the kernel that produces or consumes the data (GemmQ / GemmQd's
  gate/up epilogue, the decode overlay's `F_PLI` epilogue), with the activations of `kernel_lib.h` / `act_lut.py`;
  accuracy over the bf16 domain through a `StreamOperator` harness (`tests/rtc/elementwise.py`).
- **GEMV / GEMM** on the GGUF tile formats (`formats/`, one slot per format): the decode overlay's GEMV kernels
  (`overlay/gemv.py`, harness `MmvOperator` in `tests/rtc/mmv.py`), the prefill GEMMs `GemmQOperator` /
  `GemmQdOperator` (MAC loop `gemm/mac.py`).
- **Several operators in one dispatch**: `FusedOperator` (a runlist in one ELF) or `fused.Library` (programs assembled
  on the host), `operators/fused/`.

## Constraints

- `num_columns` 1-8; a shim tile has 2 MM2S + 2 S2MM channels.
- 32 bf16 lanes (512 bits) per vector op; other widths only after checking the loop's II (the **measure-ii** skill).
- Elementwise tile sizes in multiples of 32; operators may add constraints.
- Toolchain: Peano `clang++` is `artlite.operators.base.CLANGPP` (target `aie2p-none-unknown-elf`); aiecc runs inside
  `ArtcOperator.compile` / `FusedOperator` (single operators `--get-pdi --get-npu-insts`, fused ones `--get-full-elf`
  plus `--expand-load-pdis` unless `expand_load_pdis=False`).
