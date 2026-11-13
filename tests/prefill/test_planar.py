"""The prefill's planar-projection planning on the host (``prefill.planar``): passes cover the rows, windows, GEMM choice."""

import pytest

from artlite.operators.design import COLS
from artlite.prefill import planar as P


@pytest.mark.parametrize("fmt,R,rows", [("Q6_K", 11, [4096]), ("Q5_K", 13, [2048, 512, 512]), ("Q3_K", 21, [3072]),
                                        ("Q4_0", 32, [2560]), ("Q8_0", 8, [1024, 256])])
@pytest.mark.parametrize("few", [False, True])
def test_plan_planar_covers_the_rows(fmt: str, R: int, rows: list[int], few: bool) -> None:
    plan = P.plan_planar(rows, R, few_passes=few, fmt=fmt)
    assert plan.passes == sum(plan.block_passes)
    for r, p, nt in zip(rows, plan.block_passes, [-(-r // R) for r in rows], strict=True):
        assert p * plan.tiles >= nt  # (every row tile in some pass)
    assert plan.width % 2 == 0 and all(pitch % plan.width == 0 for pitch in plan.pitches)
    assert all(lp >= p for lp, p in zip(plan.line_pitches, plan.pitches, strict=True))
    cols = plan.out_cols([32 * b for b in range(len(rows))], list(plan.line_pitches))
    assert len(cols) == COLS and all(len(c) == plan.passes for c in cols)
    lp = plan.line_pitches
    for c, cs in enumerate(cols):  # (block b's first pass at its column offset in column c's slice)
        assert all(cs[sum(plan.block_passes[:b])] == 32 * b + c * lp[b] for b in range(len(rows)))


def test_gemm_mac_rows() -> None:
    assert [P.gemm_mac_rows(T, "gu", "Q4_K") for T in (64, 256, 512)] == [0, 0, 0]
    assert [P.gemm_mac_rows(T, "qkv", "Q4_K") for T in (64, 256, 512)] == [0, 0, 3]
    assert P.gemm_mac_rows(512, "down", "Q4_K") == 0 and P.gemm_mac_rows(512, "qkv", "BFP16") == 0
    assert [P.gemm_mac_rows(T, "o", "Q6_K", True) for T in (64, 128, 129)] == [1, 2, 3]
