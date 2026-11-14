"""The activation of a split gate / up block's prefill (``prefill.gate_up``): h = bf16(silu(gate) up) as the down's bfp16
panels, bit for bit the gate / up epilogue's arithmetic on the host (kmath.silu_tanh's model; the padding columns of GU
junk)."""

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

# (tokens, groups per column, a part's slot W, the down's K): Qwen3.8-27B's (34 groups), MiniCPM-V's (7 groups, K padded
# to 4096)
CASES = [(256, 34, 96, 17408), (64, 7, 72, 4096), (1024, 16, 64, 8192)]


@pytest.fixture(scope="module")
def vtanh() -> None:
    """kmath_sim's vtanh: the NPU's table (``VtanhOperator``)."""
    bits = np.arange(65536, dtype=np.int32).astype(np.uint16).view(bfloat16)
    kmath_sim.load_vtanh(VtanhOperator(size=65536).compile().run(bits).astype(np.float32))


@pytest.mark.parametrize(("T", "groups", "W", "K"), CASES, ids=[f"T{T}-g{g}-W{w}" for T, g, w, _ in CASES])
@pytest.mark.usefixtures("vtanh")
def test_gate_up_act(T: int, groups: int, W: int, K: int) -> None:
    rng = np.random.default_rng(groups)
    op = GateUpActOperator(T=T, groups=groups, W=W, K=K).compile()
    gu = (rng.standard_normal((T, COLS, groups, 2, W)) * 2).astype(bfloat16)
    dev = op.device
    gb, hb = artlite.from_numpy(gu.reshape(-1), dev), artlite.empty((T * K * 9 // 16,), dtype=bfloat16, device=dev)
    op.dispatch(gb, hb)
    got = bfp16.from_panels(hb.numpy(), T, K)
    g = gu[..., 0, :PART].astype(np.float32)
    u = torch.from_numpy(gu[..., 1, :PART].astype(np.float32))
    y = torch.from_numpy(np.asarray(kmath_sim.emulate(kmath.silu_tanh(walk=(2, 128)), g), np.float32))
    want = (y.bfloat16().float() * u).bfloat16().float().reshape(T, COLS * groups * PART).numpy()
    n = COLS * groups * PART
    assert np.array_equal(got[:, :n], bfp16.values(want))
