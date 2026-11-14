"""LFM3's conv history in the overlay's ACT regions (``lfm/conv.py``): the spans that the snapshot, the reset, the prefill's
base move and the readback share. No NPU."""

import numpy as np
import pytest

from models.lfm import conv as CV


def test_history_parts_cover_the_rows() -> None:
    """The two spans hold the HIST rows of a region once each, in order, inside the region."""
    parts = CV.history_parts(1000)
    assert sum(n for _, _, n in parts) == CV.HIST * CV.ROW
    assert [b for _, b, _ in parts] == [0, parts[0][2]]
    rows = np.zeros(CV.region_bytes() + 1000, np.int8)
    for a, _, n in parts:
        rows[a:a + n] += 1
    assert rows.max() == 1 and rows.sum() == CV.HIST * CV.ROW
    assert all(1000 <= a and a + n <= 1000 + CV.region_bytes() for a, _, n in parts)


@pytest.mark.parametrize("S", [1, 2, 3, 4])
def test_history_drains_follow_the_parts(S: int) -> None:
    """A step's new history (rows 0 .. 5 + S) drains into the spans' first bytes."""
    parts = CV.history_parts(0)
    for (off, nbytes), (a, b, n) in zip(CV.hist_drains(S), parts, strict=False):
        assert off == a and nbytes <= n
    assert sum(n for _, n in CV.hist_drains(S)) == (CV.WIDTH - 1 + S) * CV.ROW
