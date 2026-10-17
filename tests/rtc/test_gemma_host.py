"""Host side of the Gemma decode overlay's attention (``operators/gemma4/attn.py``): the sub-steps (the KV caches:
test_overlay_kv.py). No NPU."""

import pytest

from artlite.operators.gemma4 import attn as AT


@pytest.mark.parametrize(("pg", "r"), [(16, 4), (16, 2)])
@pytest.mark.parametrize("S", [1, 2, 3, 4])
def test_substeps(pg: int, r: int, S: int) -> None:
    """The sub-steps cover the slots in order, each within one unit of pg positions (a pair of records) and of r slots
    at most, at most U of them."""
    for p in range(2 * pg):
        subs = AT.substeps(p, S, pg, r)
        assert [a for a0, n in subs for a in range(a0, a0 + n)] == list(range(S)) and len(subs) <= AT.U
        assert all(1 <= n <= r and (p + a) // pg == (p + a + n - 1) // pg for a, n in subs)
