"""1-D elementwise operators for the tests: harnesses of the activations the model kernels call (``SiluOperator``,
``GeluOperator``: kmath's table activations, ``kmath.act``; ``KmOperator``: any kmath function of
fp32 vectors) and the simplest operators for the infrastructure tests (fused ELFs, profiling, the render manifest).
``StreamOperator``: every core streams its chunk of all arguments through one kernel."""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from aie.ir import Module
from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from ml_dtypes import bfloat16
from numpy.typing import DTypeLike

from artlite.operators import kmath
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.design import COLS, ROWS, linear_tap
from artlite.operators.templating import JINJA_ENV


def vec(n: int, dtype: DTypeLike) -> type[np.ndarray]:
    """The IRON type of a buffer of n elements."""
    return np.ndarray[(n,), np.dtype[np.dtype(dtype).type]]


def stream_design(*, kernel: str, size: int, tile: int, ins: Sequence[DTypeLike], outs: Sequence[DTypeLike], cores: int,
                  depth: int = 2, stack_size: int | None = None) -> Module:
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

    def __init__(self, *, size: int, num_columns: int, tile_size: int | None, dtypes_in: Sequence[DTypeLike],
                 dtypes_out: Sequence[DTypeLike], num_channels: int = 1):
        self.size = size
        self.num_columns = num_columns
        self.num_channels = num_channels
        self.dtypes_in = tuple(np.dtype(d) for d in dtypes_in)
        self.dtypes_out = tuple(np.dtype(d) for d in dtypes_out)
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


_ACT = JINJA_ENV.from_string(r"""
{{ act.src }}
extern "C"
void {{ name }}(bfloat16 __aie_dm_resource_a *__restrict input, bfloat16 __aie_dm_resource_b *__restrict output)
{
    event0();
{% if repeat > 1 %}
    AIE_LOOP_NO_UNROLL
    for (int r = 0; r < {{ repeat }}; r++)
{% endif %}
    {{ act }}<{{ ts }}>(input, output);
    event1();
}
""")


class SiluOperator(StreamOperator):
    """SiLU (or another kmath table's activation: ``table``) on a 1-D bf16 vector (``kmath.act``: ~0.5 cycles per
    element); ``repeat``: computed that many times per tile (the same output: a long compute-bound dispatch)."""

    def __init__(self, *, size: int, num_columns: int = 1, tile_size: int | None = None, table: str = "silu",
                 repeat: int = 1):
        self.table, self.repeat = table, repeat
        super().__init__(size=size, num_columns=num_columns, tile_size=tile_size, dtypes_in=(bfloat16,),
                         dtypes_out=(bfloat16,))

    @property
    def kernel_name(self) -> str:
        return f"{self.table}_lut_bf16_{self.tile}" + (f"_r{self.repeat}" if self.repeat > 1 else "")

    def get_kernel_src(self) -> str:
        return _ACT.render(name=self.kernel_name, ts=self.tile, act=kmath.act(kmath.TABLES[self.table]), repeat=self.repeat)


_VTANH = JINJA_ENV.from_string(r"""
extern "C"
void {{ name }}(bfloat16 *__restrict input, bfloat16 *__restrict output)
{
    for (int j = 0; j < {{ ts }}; j += 32)
        aie::store_v(output + j, al::tanh_acc<32>(aie::accum<accfloat, 32>(aie::load_v<32>(input + j))));
}
""")


class VtanhOperator(StreamOperator):
    """vtanh (``al::tanh_acc``) of a bf16 vector: the hardware's table, read for ``kmath_sim.load_vtanh``."""

    def __init__(self, *, size: int):
        super().__init__(size=size, num_columns=1, tile_size=None, dtypes_in=(bfloat16,),
                         dtypes_out=(bfloat16,))

    @property
    def kernel_name(self) -> str:
        return f"vtanh_bf16_{self.tile}"

    def get_kernel_src(self) -> str:
        return _VTANH.render(name=self.kernel_name, ts=self.tile)


_INV = JINJA_ENV.from_string(r"""
extern "C"
void {{ name }}(float *__restrict input, float *__restrict output)
{
    for (int j = 0; j < {{ ts }}; j += 16)
        aie::store_v(output + j, aie::{{ op }}(aie::load_v<16>(input + j)));
}
""")


_ROW = JINJA_ENV.from_string(r"""
{{ fn.src }}
{% for name, t in tables.items() %}
alignas(64) static const {{ t.ctype }} {{ kname }}_{{ name }}[{{ t.vals | length }}] = { {{ t.vals | join(", ") }} };
{% endfor %}

extern "C"
void {{ kname }}({{ "bfloat16" if fn.func == "rms_norm" else "float" }} *__restrict input, float *__restrict output)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int r = 0; r < {{ ts // n }}; r++) {
{% if fn.func == "rms_norm" %}
        {{ fn }}(input + {{ n }} * r, {{ (kname ~ "_w, ") if fn.params.weight }}output + {{ n }} * r{{ (", " ~ n) if fn.params.n is not integer }});
{% else %}
        for (int i = 0; i < {{ n }}; i += 32)
            aie::store_v(output + {{ n }} * r + i, aie::load_v<32>(input + {{ n }} * r + i));
        {{ fn }}(output + {{ n }} * r, {{ kname }}_cs{{ (", " ~ rot) if fn.params.rot is not integer }}{{ (", " ~ dist) if fn.params.dist is not integer }});
{% endif %}
    }
}
""")


class RowOperator(StreamOperator):
    """A kmath row function (``kmath.rms_norm``: bf16 rows -> fp32, ``kmath.rope``: fp32 rows rotated) on rows of
    ``n`` values, its constant arguments (``w``, ``cs``: numpy arrays) in the kernel; ``rot`` and ``dist``: the
    arguments of a rope of run-time extents (``n`` is a norm's)."""

    def __init__(self, *, size: int, fn: kmath.Fn, n: int, rot: int = 0, dist: int = 0, **tables: np.ndarray):
        self.fn, self.n, self.rot, self.dist, self.tables = fn, n, rot, dist, tables
        dt = bfloat16 if fn.func == "rms_norm" else np.float32
        super().__init__(size=size, num_columns=1, tile_size=None, dtypes_in=(dt,), dtypes_out=(np.float32,))
        assert self.tile % n == 0, (self.tile, n)

    @property
    def kernel_name(self) -> str:
        return "kmrow_" + self.fn.name.removeprefix(kmath.KM + "_")

    @property
    def stack_size(self) -> int | None:
        return 0x800  # (a 64-lane norm with a weight and a run-time n: ~1.1 KB)

    def get_kernel_src(self) -> str:
        def table(a: np.ndarray) -> dict[str, object]:
            if a.dtype == bfloat16:
                return {"ctype": "bfloat16", "vals": [f"bfloat16({float(v)!r}f)" for v in a]}
            return {"ctype": "float", "vals": [f"{float(v)!r}f" for v in a.astype(np.float32)]}
        return _ROW.render(fn=self.fn, kname=self.kernel_name, ts=self.tile, n=self.n, rot=self.rot, dist=self.dist,
                           tables={k: table(v) for k, v in self.tables.items()})


_SOFTMAX = JINJA_ENV.from_string(r"""
{{ p1.src }}
{{ p2.src }}

extern "C"
void {{ name }}(float *__restrict input, float *__restrict output)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    uint8_t *__restrict pp = (uint8_t *)output;
{% if lsum %}
    float *__restrict l = (float *)(pp + {{ l_off }});
    aie::store_v(l, aie::zeros<float, 32>());
    aie::store_v(l + {{ lsum }}, aie::zeros<float, 32>());
{% endif %}
{% if split %}
    alignas(64) bfloat16 pP[{{ n }}], pF[{{ n }}];
    {{ "((int32_t *)(pp + %d))[0] = " % flag_off if track }}{{ p1 }}<{{ n // 32 }}>(input, pP, pF);
    {{ p2 }}<{{ n // 64 }}>(pP, pF, {{ "l, " if lsum }}pp);
{% else %}
    alignas(64) float s[{{ n }}];
    for (int i = 0; i < {{ n }}; i += 32)
        aie::store_v(s + i, aie::load_v<32>(input + i));
    {{ p1 }}<{{ n // 32 }}>(s);
    {{ p2 }}<{{ n // 64 }}>((const bfloat16 *)s, (const bfloat16 *)(s + {{ n // 2 }}), {{ "l, " if lsum }}pp);
{% endif %}
}
""")


class SoftmaxOperator(StreamOperator):
    """``kmath.p_pass1`` then ``p_pass2`` of ``e192`` (a bias-192 exp2 of 32 lanes) on one tile of ``size`` scores A (fp32,
    in the order of the layout: "split" step t's records at 64 t + 32 u; "inplace" the records of u = 0, 1 in two halves,
    step t at 32 t of its half) -> the output's bytes: the P panels (``hi``: 72 B per step of 64 scores), at ``lo_off``
    the panels of x - hi (``lo``), at ``l_off`` the sums l (``lsum``: 2 x 32 fp32) and at ``flag_off`` pass 1's
    ``track`` result (int32). ``rounding``: pass 2 sets the rounding mode itself."""

    def __init__(self, *, size: int, e192: kmath.Fn, layout: str = "split", track: bool = False, lsum: bool = False,
                 lo: bool = False, rounding: bool = False):
        assert size % 64 == 0, size
        self.e192, self.layout, self.track, self.lsum, self.lo = e192, layout, track, lsum, lo
        self.lo_off = -(-72 * size // 64 // 64) * 64
        self.l_off = self.lo_off * (2 if lo else 1)
        self.flag_off = self.l_off + 256
        assert self.flag_off + 4 <= 4 * size, (size, self.flag_off)  # (the output's bytes: size fp32)
        self.p1 = kmath.p_pass1(e192, layout=layout, track=track)
        self.p2 = kmath.p_pass2(e192, layout=layout, lsum=32 if lsum else None, lo=self.lo_off if lo else None,
                                rounding=rounding)
        super().__init__(size=size, num_columns=1, tile_size=None, dtypes_in=(np.float32,), dtypes_out=(np.float32,))

    @property
    def kernel_name(self) -> str:
        return f"kmsoft_{self.p1.name.removeprefix(kmath.KM + '_')}_{self.p2.name.removeprefix(kmath.KM + '_')}_{self.size}"

    @property
    def stack_size(self) -> int | None:
        return 0x2000  # (the pass-1 arrays: 4 B per score)

    def get_kernel_src(self) -> str:
        return _SOFTMAX.render(p1=self.p1, p2=self.p2, name=self.kernel_name, n=self.size, split=self.layout == "split",
                               track=self.track, lsum=32 if self.lsum else None, l_off=self.l_off,
                               flag_off=self.flag_off)


class InvOperator(StreamOperator):
    """``aie::inv`` or ``aie::invsqrt`` (``op``) of an fp32 vector: the scalar unit's functions, read for
    ``kmath_sim.load_inv``."""

    def __init__(self, *, size: int, op: str):
        assert op in ("inv", "invsqrt"), op
        self.op = op
        super().__init__(size=size, num_columns=1, tile_size=None, dtypes_in=(np.float32,), dtypes_out=(np.float32,))

    @property
    def kernel_name(self) -> str:
        return f"{self.op}_f32_{self.tile}"

    def get_kernel_src(self) -> str:
        return _INV.render(name=self.kernel_name, ts=self.tile, op=self.op)


class GeluOperator(StreamOperator):
    """GELU in its tanh form (``kmath.act(kmath.GELU)``) on a 1-D bf16 vector, up to 4 cores per column."""

    MAX_TILE_BYTES = 16384
    MAX_CHANNELS = 4

    def __init__(self, *, size: int, num_columns: int = 1, tile_size: int | None = None, num_channels: int = 1):
        super().__init__(size=size, num_columns=num_columns, tile_size=tile_size, num_channels=num_channels,
                         dtypes_in=(bfloat16,), dtypes_out=(bfloat16,))

    @property
    def kernel_name(self) -> str:
        return f"gelu_lut_bf16_{self.tile}"

    def get_kernel_src(self) -> str:
        return _ACT.render(name=self.kernel_name, ts=self.tile, act=kmath.act(kmath.GELU), repeat=1)


_KM = JINJA_ENV.from_string(r"""
{{ fn.src }}
{% if fn.params.get("f32") %}
extern "C"
void {{ name }}(float *__restrict input, float *__restrict output)
{  // (x bf16 values as fp32)
    aie::set_rounding(aie::rounding_mode::conv_even);
    alignas(64) bfloat16 x[{{ ts }}], t[{{ ts }}];
    for (int j = 0; j < {{ ts }}; j += 32)
        aie::store_v(x + j, aie::accum<accfloat, 32>(aie::load_v<32>(input + j)).template to_vector<bfloat16>());
    {{ fn }}<{{ ts }}>(x, t, output);
}
{% elif fn.arg == "ptr" %}
extern "C"
void {{ name }}(bfloat16 *__restrict input, bfloat16 *__restrict output)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ fn }}<{{ ts }}>(input, output{{ ", 0" if fn.params.get("zero") else "" }});
}
{% else %}
extern "C"
void {{ name }}(float __aie_dm_resource_a *__restrict input, float __aie_dm_resource_b *__restrict output)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int j = 0; j < {{ ts }}; j += {{ fn.lanes }}) {
        const aie::vector<float, {{ fn.lanes }}> x = aie::load_v<{{ fn.lanes }}>(input + j);
{% set arg = {"accum": "aie::accum<accfloat, " ~ fn.lanes ~ ">(x)",
               "bfloat16": "aie::accum<accfloat, " ~ fn.lanes ~ ">(x).template to_vector<bfloat16>()"}.get(fn.arg, "x") %}
{% set r = fn ~ "(" ~ arg ~ ")" %}
        aie::store_v(output + j, {{ {"accum": r ~ ".template to_vector<float>()",
                                     "bfloat16": "aie::accum<accfloat, " ~ fn.lanes ~ ">(" ~ r ~ ").template to_vector<float>()"}.get(fn.ret, r) }});
    }
}
{% endif %}
""")


class KmOperator(StreamOperator):
    """A kmath function (``km.<func>(**kwargs)``: an fp32 vector, fp32 accumulator or bf16 vector argument and result)
    on a 1-D fp32 vector, one core; an array function (``kmath.act``: a table's name as ``table``) on a bf16 one."""

    def __init__(self, *, size: int, func: str, **kwargs: object):
        self.func, self.kwargs = func, kwargs
        args = {k: kmath.TABLES[v] if k == "table" else v for k, v in kwargs.items()}
        self.fn = getattr(kmath, func)(**args)
        dt = bfloat16 if self.fn.arg == "ptr" and not self.fn.params.get("f32") else np.float32
        super().__init__(size=size, num_columns=1, tile_size=None, dtypes_in=(dt,), dtypes_out=(dt,))

    @property
    def kernel_name(self) -> str:
        return "kmtest_" + self.fn.name.removeprefix(kmath.KM + "_")

    @property
    def stack_size(self) -> int | None:
        return 0x2000 if self.fn.params.get("f32") else 0x1000  # (softplus inlined: 3.4 KB; f32: x and t in the frame)

    def get_kernel_src(self) -> str:
        return _KM.render(fn=self.fn, name=self.kernel_name, ts=self.tile)


_ROT = JINJA_ENV.from_string(r"""
{{ fn.src }}
extern "C"
void {{ name }}(int32_t *__restrict input, int32_t *__restrict output)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *__restrict x = (const bfloat16 *)input;
    const bfloat16 *__restrict w = x + {{ 512 * G }};
    const uint8_t *__restrict coef = (const uint8_t *)(w + {{ 128 * G }});
    float *__restrict u = (float *)output;
    bfloat16 *__restrict b0 = (bfloat16 *)(u + {{ 512 * G }}), *__restrict b1 = b0 + {{ 1024 * G }};
    for (int c = 0; c < {{ 128 * G }}; c += 8) {  // u = x w' ([group][channel][4]), its bf16 copy (each 4 channels twice)
        const aie::vector<bfloat16, 32> xv = aie::load_v<32>(x + 4 * c);
        const aie::vector<bfloat16, 8> w8 = aie::load_v<8>(w + c);
        const auto [a, b] = aie::interleave_zip(w8.template grow<16>(), w8.template grow<16>(), 1);
        const auto [c0, c1] = aie::interleave_zip(a, a, 1);
        const aie::accum<accfloat, 32> u0 = aie::mul(xv, aie::concat(c0, c1));
        aie::store_v(u + 4 * c, u0.template to_vector<float>());
        const aie::vector<bfloat16, 32> h = u0.template to_vector<bfloat16>();
        aie::store_v(b0 + 8 * c, aie::concat(h.template extract<16>(0), h.template extract<16>(0)));
        aie::store_v(b0 + 8 * c + 32, aie::concat(h.template extract<16>(1), h.template extract<16>(1)));
    }
    event0();
    {{ fn }}({{ "u, " if f32 }}b0, b0, b1, b1, b0, b1, coef);
    event1();
{% if not f32 %}
    for (int c = 0; c < {{ 128 * G }}; c += 4)  // (the state's last copy -> u as fp32)
        aie::store_v(u + 4 * c, aie::accum<accfloat, 16>(aie::load_v<16>({{ "b0" if rounds % 2 == 0 else "b1" }} + 8 * c)).template to_vector<float>());
{% endif %}
}
""")


class RotOperator(StreamOperator):
    """``kmath.rotate`` (one core, ``groups`` groups per call, ``calls`` calls): the input per call [x: G x 128 x 4 bf16
    ([group][channel][lane]) | w': G x 128 bf16 | the coefficients: G x rounds x 384 B], the output per call u (G x 512
    fp32), then the two bf16 copies (and padding): u = x w', then the rounds (``f32``: the state in fp32, else bf16)."""

    MAX_TILE_BYTES: ClassVar[int] = 4 * 6144  # (a call's argument: four groups)

    def __init__(self, *, groups: int = 4, calls: int = 2, rounds: int = 8, f32: bool = True):
        self.fn, self.rounds, self.groups, self.f32 = kmath.rotate(rounds, groups=groups, f32=f32), rounds, groups, f32
        n = groups * (256 + 64 + 96 * rounds)
        self.group_ints = max(-(-n // 32) * 32, groups * 1536)  # (the output: u and the two copies)
        super().__init__(size=calls * self.group_ints, num_columns=1, tile_size=None, dtypes_in=(np.int32,),
                         dtypes_out=(np.int32,))
        self.tile = self.group_ints  # (a call per group set)

    @property
    def kernel_name(self) -> str:
        return "kmrot_" + self.fn.name.removeprefix(kmath.KM + "_")

    def get_kernel_src(self) -> str:
        return _ROT.render(fn=self.fn, name=self.kernel_name, G=self.groups, f32=self.f32, rounds=self.rounds)
