"""The activation of a split gate / up block's prefill (``prefill.gate_up``): h = bf16(act(gate) up) as the down's bfp16
panels, bit for bit the gate / up epilogue's arithmetic on the host (kmath.silu_tanh's model; a GELU form: kmath.act's
through its table; the padding columns of GU junk)."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

import artlite
from artlite.operators import bfp16, kmath
from artlite.operators.design import COLS
from artlite.prefill.gate_up import PART, GateUpActOperator

from ..operators import kmath_sim
from ..operators.elementwise import VtanhOperator

# (tokens, groups per column, a part's slot W, the down's K, act): Qwen3.8-27B's (34 groups), MiniCPM-V's (7 groups, K
# padded to 4096), bartowski's Spark-X 4B (19 groups, GELU's erf form, K padded to 10240)
CASES = [(256, 34, 96, 17408, "silu"), (64, 7, 72, 4096, "silu"), (1024, 16, 64, 8192, "silu"),
         (256, 19, 96, 10240, "gelu_erf")]


@pytest.fixture(scope="module")
def vtanh() -> None:
    """kmath_sim's vtanh: the NPU's table (``VtanhOperator``)."""
    bits = np.arange(65536, dtype=np.int32).astype(np.uint16).view(bfloat16)
    kmath_sim.load_vtanh(VtanhOperator(size=65536).compile().run(bits).astype(np.float32))


@pytest.mark.parametrize(("T", "groups", "W", "K", "act"), CASES, ids=[f"T{T}-g{g}-W{w}-{a}" for T, g, w, _, a in CASES])
@pytest.mark.usefixtures("vtanh")
def test_gate_up_act(T: int, groups: int, W: int, K: int, act: str) -> None:
    rng = np.random.default_rng(groups)
    op = GateUpActOperator(T=T, groups=groups, W=W, K=K, act=act).compile()
    gu = (rng.standard_normal((T, COLS, groups, 2, W)) * 2).astype(bfloat16)
    dev = op.device
    gb, hb = artlite.from_numpy(gu.reshape(-1), dev), artlite.empty((T * K * 9 // 16,), dtype=bfloat16, device=dev)
    op.dispatch(gb, hb)
    got = bfp16.from_panels(hb.numpy(), T, K)
    g = gu[..., 0, :PART].astype(np.float32)
    u = torch.from_numpy(gu[..., 1, :PART].astype(np.float32))
    fn = kmath.silu_tanh(walk=(2, 128)) if act == "silu" else kmath.act(kmath.TABLES[act], table_arg=True, walk=(2, 128))
    y = torch.from_numpy(np.asarray(kmath_sim.emulate(fn, g), np.float32))
    want = (y.bfloat16().float() * u).bfloat16().float().reshape(T, COLS * groups * PART).numpy()
    n = COLS * groups * PART
    assert np.array_equal(got[:, :n], bfp16.values(want))
