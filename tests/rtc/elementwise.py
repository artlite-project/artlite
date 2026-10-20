"""1-D elementwise operators for the tests: harnesses of the activations the model kernels call (``SiluOperator``:
``al::silu`` and act_lut's table SiLU; ``GeluOperator``: ``al::gelu_lut``) and the simplest operators for the
infrastructure tests (fused ELFs, profiling, the render manifest). ``StreamOperator``: every core streams its chunk
of all arguments through one kernel."""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from aie.ir import Module
from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from ml_dtypes import bfloat16

from artlite.operators import act_lut
from artlite.operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from artlite.operators.design import COLS, ROWS, linear_tap

_NP_DTYPE: dict[torch.dtype, type] = {
    torch.bfloat16: bfloat16, torch.float32: np.float32, torch.int32: np.int32, torch.int16: np.int16,
    torch.int8: np.int8, torch.uint8: np.uint8,
}


def vec(n: int, dtype: type | torch.dtype) -> type[np.ndarray]:
    """The IRON type of a buffer of n elements (numpy or torch element type)."""
    return np.ndarray[(n,), np.dtype[_NP_DTYPE[dtype] if isinstance(dtype, torch.dtype) else dtype]]


def stream_design(*, kernel: str, size: int, tile: int, ins: Sequence[type | torch.dtype],
                  outs: Sequence[type | torch.dtype], cores: int, depth: int = 2, stack_size: int | None = None) -> Module:
    """The design of a 1-D streaming operator: the arguments (inputs of the element types ``ins``,
    then outputs ``outs``; ``size`` elements each) are split into ``cores`` contiguous chunks, one per
    core, and every core calls ``kernel`` on consecutive tiles of ``tile`` elements of all of them.

    One ObjectFifo of ``depth`` objects per argument and core, named ``in<k>_<core>`` /
    ``out<k>_<core>`` (IRON resolves FIFOs in name order: by argument, then core)."""
    chunk = size // cores
    assert size % cores == 0 and chunk % tile == 0, f"{size} elements do not split into {cores} x {tile}-element tiles"
    types = [*ins, *outs]
    names = [f"in{k}" for k in range(len(ins))] + [f"out{k}" for k in range(len(outs))]
    fifos = [[ObjectFifo(vec(tile, dt), name=f"{nm}_{c:02d}", depth=depth) for c in range(cores)]
             for nm, dt in zip(names, types)]
    fn = Kernel(kernel, "kernel.o", [vec(tile, dt) for dt in types])

    def core(*args: object) -> None:
        *handles, kern = args
        for _ in range_(chunk // tile):
            kern(*[h.acquire(1) for h in handles])
            for h in handles:
                h.release(1)

    stack = {} if stack_size is None else {"stack_size": stack_size}
    workers = [Worker(core, [*(f[c].cons() for f in fifos[:len(ins)]), *(f[c].prod() for f in fifos[len(ins):]), fn],
                      **stack) for c in range(cores)]
    fills = [[f[c].prod() for c in range(cores)] for f in fifos[:len(ins)]]
    drains = [[f[c].cons() for c in range(cores)] for f in fifos[len(ins):]]
    taps = [linear_tap(size, c * chunk, chunk) for c in range(cores)]

    def seq(*args: object) -> None:
        bufs = args[:len(types)]
        tg = TaskGroup()
        for c in range(cores):
            for k, hs in enumerate(fills):
                hs[c].fill(bufs[k], taps[c], group=tg)
        for c in range(cores):
            for k, hs in enumerate(drains):
                hs[c].drain(bufs[len(ins) + k], taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [*(vec(size, dt) for dt in types), [*fills, *drains]])
    return Program(NPU2(), rt, workers=workers).resolve_program()


class StreamOperator(ArtcOperator):
    """Base of the 1-D elementwise operators (see ``stream_design``): arguments of ``size`` elements,
    the inputs ``dtypes_in`` then the outputs ``dtypes_out``, split over ``num_columns *
    num_channels`` cores (``num_channels`` rows per column), each calling ``kernel_name`` on tiles
    of ``tile`` elements.

    A core's ``tile_size`` elements (default: its whole chunk) go through FIFO objects of ``tile``
    elements, at most ``MAX_TILE_BYTES`` per argument; objects above 8 KB are single-buffered.
    Subclasses provide ``kernel_name`` and ``get_kernel_src`` (and ``stack_size`` when the kernel
    needs more than the Worker default)."""

    MAX_TILE_BYTES: ClassVar[int] = 8192
    MAX_CHANNELS: ClassVar[int] = 1

    def __init__(self, *, size: int, num_columns: int, tile_size: int | None, dtypes_in: Sequence[torch.dtype],
                 dtypes_out: Sequence[torch.dtype], num_channels: int = 1):
        self.size = size
        self.num_columns = num_columns
        self.num_channels = num_channels
        self.dtypes_in = tuple(dtypes_in)
        self.dtypes_out = tuple(dtypes_out)
        cores = num_columns * num_channels
        self.tile_size = ts = tile_size or size // cores
        max_ch = min(self.MAX_CHANNELS, ROWS)
        assert size > 0, "size must be positive"
        assert 1 <= num_columns <= COLS, f"num_columns must be 1-{COLS} for NPU2"
        assert 1 <= num_channels <= max_ch, f"num_channels must be 1-{max_ch} (1 row per channel, {ROWS} rows per column)"
        assert ts > 0 and ts % 32 == 0, "tile_size must be a positive multiple of 32"
        assert size % (cores * ts) == 0, (
            f"size ({size}) must be divisible by num_columns*num_channels*tile_size ({cores * ts})")
        self.itemsize = max(dt.itemsize for dt in (*self.dtypes_in, *self.dtypes_out))
        self.tile = next(t for t in range(min(ts, self.MAX_TILE_BYTES // self.itemsize), 0, -32) if ts % t == 0)
        super().__init__()

    @property
    def kernel_name(self) -> str:
        raise NotImplementedError

    @property
    def stack_size(self) -> int | None:
        """The Worker stack size (None: the IRON default of 1 KB)."""
        return None

    def get_arg_spec(self) -> list[ArgSpec]:
        return ([ArgSpec("in", (self.size,), dt) for dt in self.dtypes_in]
                + [ArgSpec("out", (self.size,), dt) for dt in self.dtypes_out])

    def get_mlir_artifact(self) -> Module:
        return stream_design(kernel=self.kernel_name, size=self.size, tile=self.tile, ins=self.dtypes_in,
                             outs=self.dtypes_out, cores=self.num_columns * self.num_channels,
                             depth=1 if self.tile * self.itemsize > 8192 else 2, stack_size=self.stack_size)


_SILU = JINJA_ENV.from_string(r"""
extern "C"
void {{ name }}(bfloat16 __aie_dm_resource_a *__restrict input, bfloat16 __aie_dm_resource_b *__restrict output)
{
    event0();
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if lut %}
    {{ lut }}_{{ table }}_lut<{{ ts }}>(input, output);
{% else %}
    // al::silu, 16 lanes per step (see kernel_lib.h)
    for (int j = 0; j < {{ ts }}; j += 16) {
        aie::vector<bfloat16, 16> x = aie::load_v<16>(input); input += 16;
        aie::store_v(output, aie::accum<accfloat, 16>(al::silu<16>(x)).template to_vector<bfloat16>());
        output += 16;
    }
{% endif %}
    event1();
}
""")


class SiluOperator(StreamOperator):
    """SiLU on a 1-D bf16 vector: ``lut`` act_lut's table SiLU (~0.85 cycles per element), else ``al::silu``
    (software exp2, two Newton steps for the reciprocal; not pipelined: one long fp32 chain per 16 lanes, spilling,
    ~16 cycles per element)."""

    def __init__(self, *, size: int, num_columns: int = 1, tile_size: int | None = None, lut: bool = False,
                 table: str = "silu"):
        """``table``: the act_lut table of the lut form (silu, sigmoid)."""
        self.lut, self.table = lut, table
        super().__init__(size=size, num_columns=num_columns, tile_size=tile_size, dtypes_in=(torch.bfloat16,),
                         dtypes_out=(torch.bfloat16,))

    @property
    def kernel_name(self) -> str:
        return f"{self.table}_{'lut' if self.lut else 'exact'}_bf16_{self.tile}"

    @property
    def stack_size(self) -> int | None:
        return 0xA00  # a 2.2 KB frame

    def get_kernel_src(self) -> str:
        tab = {"silu": act_lut.SILU, "sigmoid": act_lut.SIGMOID}[self.table]
        return ((tab.render() if self.lut else "")
                + _SILU.render(name=self.kernel_name, ts=self.tile, lut=self.lut and act_lut.LUT, table=self.table))


_GELU = JINJA_ENV.from_string(r"""
extern "C"
void {{ name }}(bfloat16 __aie_dm_resource_a *__restrict input, bfloat16 __aie_dm_resource_b *__restrict output)
{
    event0();
    aie::set_rounding(aie::rounding_mode::conv_even);
    // al::gelu_lut, 32 lanes per step (see kernel_lib.h)
    for (int j = 0; j < {{ ts }}; j += 32) {
        aie::vector<bfloat16, 32> x = aie::load_v<32>(input), xr = aie::load_v<32>(input);
        input += 32;
        aie::store_v(output, al::gelu_lut(x, xr));
        output += 32;
    }
    event1();
}
""")


class GeluOperator(StreamOperator):
    """GELU in its tanh form (``al::gelu_lut``) on a 1-D bf16 vector, up to 4 cores per column."""

    MAX_TILE_BYTES = 16384
    MAX_CHANNELS = 4

    def __init__(self, *, size: int, num_columns: int = 1, tile_size: int | None = None, num_channels: int = 1):
        super().__init__(size=size, num_columns=num_columns, tile_size=tile_size, num_channels=num_channels,
                         dtypes_in=(torch.bfloat16,), dtypes_out=(torch.bfloat16,))

    @property
    def kernel_name(self) -> str:
        return f"gelu_lut_bf16_{self.tile}"

    @property
    def stack_size(self) -> int | None:
        return 0xD00

    def get_kernel_src(self) -> str:
        return _GELU.render(name=self.kernel_name, ts=self.tile)


def emulate(table: act_lut.ActTable, x: np.ndarray) -> np.ndarray:
    """A table activation's kernel result (``ActTable.render``; bit-exact on the NPU)."""
    x = act_lut._bf16(x)
    tm = np.float32(table.t_max)
    v, r, r2 = table._reduce(np.clip(x, -tm, tm) if table.full else np.minimum(np.abs(x), tm))
    s = ((v - table.base) / table.seg).astype(np.int64)
    c0, c1, c2 = (act_lut._f32(table.coefficients[s, i]) for i in range(3))
    acc = act_lut._f32(act_lut._f32(c0 + act_lut._f32(c1 * r)) + act_lut._f32(c2 * r2))
    return act_lut._bf16(acc if table.full else act_lut._f32(acc + np.maximum(x, np.float32(0))))
