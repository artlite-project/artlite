"""kernel_lib.h's lane helpers on the NPU: vectors from 128-bit pieces (``al::gather`` / ``al::concat``: vshuffles) and
from 256-bit ones (aie::concat), bit for bit against numpy (data movement only); the row helpers (``al::max8``,
``al::sum8``, ``al::rows4``, ``al::rows4_inv``, ``al::bcast8``, ``al::group8``, ``al::max16``) and ``al::fmax``: maxima and
broadcasts bit for bit, sums as ``kmath_sim``'s adder along the helper's additions, inverses within the scalar inv's
ULP."""

from collections.abc import Callable

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.operators.templating import JINJA_ENV

from . import kmath_sim
from .elementwise import StreamOperator

N = 1024  # int32 words per argument

_SRC = JINJA_ENV.from_string(r"""
// blocks of K rows (pitch P elements) of pieces of E elements -> per block and piece column q, the K rows' piece q
// (lanes [row][E])
extern "C"
void {{ name }}(const int32 *__restrict in_, int32 *__restrict out_)
{
    const {{ T }} *__restrict in = (const {{ T }} *)in_;
    {{ T }} *__restrict out = ({{ T }} *)out_;
    for (int i = 0; i < {{ blocks }}; i++)
        for (int q = 0; q < {{ P // E }}; q++)
            aie::store_v(out + {{ K * P }} * i + {{ K * E }} * q, al::gather<{{ K }}, {{ E }}>(in + {{ K * P }} * i + {{ E }} * q, {{ P }}));
}
""")

CASES = {  # id: element type, its numpy view, rows K, piece E, pitch P
    "bf16-8": ("bfloat16", np.uint16, 8, 8, 64), "bf16-4": ("bfloat16", np.uint16, 4, 8, 64),
    "bf16-2": ("bfloat16", np.uint16, 2, 8, 64), "int8-4": ("int8", np.uint8, 4, 16, 64),
    "fp32-8": ("float", np.int32, 8, 4, 32), "fp32x8-8": ("float", np.int32, 8, 8, 64),  # (the last: 256-bit pieces)
}


class GatherOperator(StreamOperator):
    """al::gather of the rows of ``CASES[case]``."""

    def __init__(self, case: str):
        self.case = case
        super().__init__(size=N, num_columns=1, tile_size=None, dtypes_in=(np.int32,), dtypes_out=(np.int32,))

    @property
    def kernel_name(self) -> str:
        return f"kl_gather_{self.case.replace('-', '_')}"

    def get_kernel_src(self) -> str:
        t, view, k, e, p = CASES[self.case]
        return _SRC.render(name=self.kernel_name, T=t, K=k, E=e, P=p, blocks=4 * N // np.dtype(view).itemsize // (k * p))


@pytest.mark.parametrize("case", CASES)
def test_gather(case: str):
    _, view, k, e, p = CASES[case]
    x = np.random.default_rng(1).integers(-2**31, 2**31, N, dtype=np.int64).astype(np.int32)
    got = GatherOperator(case).run(x).view(view)
    np.testing.assert_array_equal(got, x.view(view).reshape(-1, k, p // e, e).transpose(0, 2, 1, 3).reshape(-1))


_ROWS = JINJA_ENV.from_string(r"""
extern "C"
void {{ name }}(const int32 *__restrict in_, int32 *__restrict out_)
{
    const {{ T }} *__restrict in = (const {{ T }} *)in_;
    {{ TO }} *__restrict out = ({{ TO }} *)out_;
    for (int i = 0; i < {{ blocks }}; i++) {
        const {{ T }} *__restrict x = in + {{ NI }} * i;
        aie::store_v(out + {{ NO }} * i, {{ expr }});
    }
}
""")


def _rows(x: np.ndarray, n: int) -> np.ndarray:
    return x.reshape(len(x), -1, n)


def _sum8(x: np.ndarray) -> np.ndarray:
    """Row sums ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)) on the accumulators' adder."""
    r = _rows(x, 8)
    p = [kmath_sim.add(r[..., 2 * m], r[..., 2 * m + 1]) for m in range(4)]
    return kmath_sim.add(kmath_sim.add(p[0], p[1]), kmath_sim.add(p[2], p[3]))


ROW_CASES: dict[str, tuple[str, str, int, int, Callable[[np.ndarray], np.ndarray]]] = {
    # id: element type, expression of x, lanes in, lanes out, numpy result per block (blocks, lanes in) -> (blocks, out)
    "max8-bf16": ("bfloat16", "al::max8(aie::load_v<32>(x))", 32, 32, lambda x: np.repeat(_rows(x, 8).max(2), 8, 1)),
    "max8-bf16-16": ("bfloat16", "al::max8(aie::load_v<16>(x))", 16, 16, lambda x: np.repeat(_rows(x, 8).max(2), 8, 1)),
    "max8-fp32": ("float", "al::max8(aie::load_v<32>(x))", 32, 32, lambda x: np.repeat(_rows(x, 8).max(2), 8, 1)),
    "max8-fp32-64": ("float", "al::max8(aie::load_v<64>(x))", 64, 64, lambda x: np.repeat(_rows(x, 8).max(2), 8, 1)),
    "max16": ("float", "al::max16(aie::load_v<16>(x))", 16, 16, lambda x: np.repeat(x.max(1, keepdims=True), 16, 1)),
    "max16-4": ("float", "al::max16<4>(aie::load_v<16>(x))", 16, 16, lambda x: np.repeat(x[:, :4].max(1, keepdims=True), 16, 1)),
    "fmax": ("float", "al::fmax(aie::load_v<64>(x), aie::load_v<64>(x + 64))", 128, 64,
             lambda x: np.maximum(x[:, :64], x[:, 64:])),
    "group8-max": ("float", "al::group8<true>(aie::load_v<64>(x))", 64, 8, lambda x: _rows(x, 8).max(2)),
    "rows4": ("float", "al::rows4(aie::load_v<16>(x))", 16, 32, lambda x: np.repeat(x[:, :4], 8, 1)),
    "rows4-1": ("float", "al::rows4<1>(aie::load_v<16>(x))", 16, 32, lambda x: np.repeat(x[:, 4:8], 8, 1)),
    "bcast8": ("float", "al::bcast8(aie::load_v<8>(x))", 8, 64, lambda x: np.repeat(x, 8, 1)),
    "sum8": ("float", "al::sum8(aie::load_v<32>(x))", 32, 16, _sum8),
    "group8-sum": ("float", "al::group8<false>(aie::load_v<64>(x))", 64, 8, _sum8),
    "rows4_inv": ("float", "al::rows4_inv(al::sum8(aie::load_v<32>(x)))", 32, 32, lambda x: np.repeat(1 / _sum8(x), 8, 1)),
}


class RowsOperator(StreamOperator):
    """The row helper ``ROW_CASES[case]`` on consecutive blocks of a tile."""

    def __init__(self, case: str):
        self.case = case
        super().__init__(size=N, num_columns=1, tile_size=None, dtypes_in=(np.int32,), dtypes_out=(np.int32,))

    @property
    def kernel_name(self) -> str:
        return f"kl_rows_{self.case.replace('-', '_')}"

    def get_kernel_src(self) -> str:
        t, expr, ni, no, _ = ROW_CASES[self.case]
        words = 2 if t == "bfloat16" else 1  # (elements per int32 word)
        return _ROWS.render(name=self.kernel_name, T=t, TO=t, expr=expr, NI=ni, NO=no, blocks=N * words // max(ni, no))


def _row_data(rng: np.random.Generator, n: int, sums: bool) -> np.ndarray:
    """Rows of 8: random signs over 2^-20 .. 2^20, and rows all negative, all equal, of +-0, tied at their maximum, with
    +-inf (sums: normal values only, the adder flushes denormals and inf - inf has no sum)."""
    x = (rng.standard_normal(n) * np.exp2(rng.integers(-20, 20, n))).astype(np.float32)
    if sums:
        return x
    r = x.reshape(-1, 8)
    kind = rng.integers(0, 6, len(r))
    r[kind == 0] = -np.abs(r[kind == 0])
    r[kind == 1] = r[kind == 1][:, :1]
    r[kind == 2] = np.where(np.arange(8) % 2, np.float32(0.0), np.float32(-0.0))
    tie = np.flatnonzero(kind == 3)
    r[tie, rng.integers(0, 8, len(tie))] = r[tie].max(1)
    inf = np.flatnonzero(kind == 4)
    r[inf, 3] = -np.inf
    r[inf[::2], 5] = np.inf
    return x


@pytest.mark.parametrize("case", ROW_CASES)
def test_rows(case: str):
    t, _, ni, no, ref = ROW_CASES[case]
    sums = "sum" in case or "inv" in case
    x = _row_data(np.random.default_rng(2), 4 * N // 4 * (2 if t == "bfloat16" else 1), sums)
    if t == "bfloat16":
        x = x.astype(bfloat16)
    got = RowsOperator(case).run(x.view(np.int32))
    blocks = len(x) // max(ni, no)
    xs = x[:blocks * ni].astype(np.float32).reshape(blocks, ni)
    out = got.view(bfloat16 if t == "bfloat16" else np.float32)[:blocks * no].astype(np.float32).reshape(blocks, no)
    want = ref(xs)
    if case == "sum8":  # (row r's sum in lane 4 r)
        np.testing.assert_array_equal(out[:, ::4].view(np.int32), want.view(np.int32))
    elif case == "group8-sum":
        np.testing.assert_array_equal(out.view(np.int32), want.view(np.int32))
    elif case == "rows4_inv":
        np.testing.assert_allclose(out, want, rtol=2.0**-22)
    else:  # (maxima: by value, so that +0 and -0 compare equal)
        np.testing.assert_array_equal(out, want)
