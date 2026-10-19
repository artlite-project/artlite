"""MiniCPM-V 4.7's image host side (models/qwen35/minicpmv.py), no NPU: the token layout of an image and its positions
against transformers' ``MiniCPMV4_7Model.get_rope_index`` (the values of llama.cpp PR #30510's
``test_decoder_pos_minicpmv4_7``, from transformers 5.19, for the tokens after ``<image>``; the official ``<image>``
corner and the template's "\\n" after the image, which that PR leaves out)."""

import numpy as np
import pytest

from models.qwen35 import minicpmv as M

# (overview w x h, slices cols x rows of w x h tokens, the text position after the image, the (T, H, W) of the tokens
# after <image> through the last marker), <image> at position 5 = the canvas origin
CASES = [
    ((5, 2), (2, 2), (7, 2), 20, [
        (5, 5, 5), (5, 5, 8), (5, 5, 11), (5, 5, 15), (5, 5, 18), (5, 8, 5), (5, 8, 8), (5, 8, 11),
        (5, 8, 15), (5, 8, 18), (5, 9, 19), (5, 5, 5), (5, 5, 5), (5, 5, 6), (5, 5, 7), (5, 5, 8),
        (5, 5, 9), (5, 5, 10), (5, 5, 11), (5, 6, 5), (5, 6, 6), (5, 6, 7), (5, 6, 8), (5, 6, 9),
        (5, 6, 10), (5, 6, 11), (5, 6, 11), (5, 5, 12), (5, 5, 12), (5, 5, 13), (5, 5, 14), (5, 5, 15),
        (5, 5, 16), (5, 5, 17), (5, 5, 18), (5, 6, 12), (5, 6, 13), (5, 6, 14), (5, 6, 15), (5, 6, 16),
        (5, 6, 17), (5, 6, 18), (5, 6, 18), (5, 6, 19), (5, 7, 5), (5, 7, 5), (5, 7, 6), (5, 7, 7),
        (5, 7, 8), (5, 7, 9), (5, 7, 10), (5, 7, 11), (5, 8, 5), (5, 8, 6), (5, 8, 7), (5, 8, 8),
        (5, 8, 9), (5, 8, 10), (5, 8, 11), (5, 8, 11), (5, 7, 12), (5, 7, 12), (5, 7, 13), (5, 7, 14),
        (5, 7, 15), (5, 7, 16), (5, 7, 17), (5, 7, 18), (5, 8, 12), (5, 8, 13), (5, 8, 14), (5, 8, 15),
        (5, 8, 16), (5, 8, 17), (5, 8, 18), (5, 8, 18)]),
    ((4, 3), (0, 0), (0, 0), 10, [
        (5, 5, 5), (5, 5, 6), (5, 5, 7), (5, 5, 8), (5, 6, 5), (5, 6, 6), (5, 6, 7), (5, 6, 8),
        (5, 7, 5), (5, 7, 6), (5, 7, 7), (5, 7, 8), (5, 8, 9)]),
]


@pytest.mark.parametrize(("ov", "grid", "cell", "text", "want"), CASES, ids=["2x2-slices", "overview"])
def test_positions(ov: tuple, grid: tuple, cell: tuple, text: int, want: list) -> None:
    g = M.Grid(ov=(ov[1], ov[0]), grid=(grid[1], grid[0]), cell=(cell[1], cell[0]))
    lay = g.layout()
    pos, nxt = g.positions(5)
    assert pos.shape == (len(lay), 3)
    assert lay[0] == M.IMAGE and tuple(pos[0]) == (5, 4, 4)  # (<image>: just outside the canvas)
    assert lay[-1] == M.NEWLINE and tuple(pos[-1]) == (5, 5, 5)  # (the template's "\n" after the image)
    assert [tuple(p) for p in pos[1:-1]] == want
    assert nxt == text
    # the layout: the markers where the official processor puts them
    n_ov = ov[0] * ov[1]
    assert lay[1:1 + n_ov] == [M.MEDIA] * n_ov and lay[1 + n_ov] == M.IMAGE_END
    assert lay.count(M.SLICE) == lay.count(M.SLICE_END) == grid[0] * grid[1]
    assert lay.count(M.NEWLINE) == max(grid[1] - 1, 0) + 1
    assert lay.count(M.MEDIA) == n_ov + grid[0] * grid[1] * cell[0] * cell[1]
    # without the template's newline: the same positions, one token fewer
    p2, n2 = g.positions(5, newline_after=False)
    assert np.array_equal(p2, pos[:-1]) and n2 == nxt and len(g.layout(newline_after=False)) == len(lay) - 1
