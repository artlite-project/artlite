"""kernel_lib.h's lane helpers on the NPU: vectors from 128-bit pieces (``al::gather`` / ``al::concat``: vshuffles) and
from 256-bit ones (aie::concat), bit for bit against numpy (data movement only)."""

import numpy as np
import pytest

from artlite.operators.templating import JINJA_ENV

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
