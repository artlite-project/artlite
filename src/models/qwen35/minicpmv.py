"""MiniCPM-V 4.7's images in the qwen35 language model (its mmproj, projector ``minicpmv4_7``): the token layout of an
image and its M-RoPE positions.

An image is an overview (``ov``: tokens h x w) and, unless it is small, a grid of slices (``rows`` x ``cols`` of ``cell``
tokens each). Its tokens, as the official processor lays them out (transformers' ``MiniCPMV4_7Processor``)::

    <image> [ov] </image> <slice> [S00] </slice> <slice> [S01] </slice> \\n <slice> [S10] </slice> ...

the slice rows joined by "\\n"; the chat template's "\\n" after the image follows.

Positions are (T, H, W) per token, as transformers' ``MiniCPMV4_7Model.get_rope_index`` gives them (``positions``),
llama.cpp's RoPE sections by its ``rope.section_order`` zyxt (section 0: T, 1: H, 2: W). With O the image's first
position (the text before it ends at O - 1): every token of the image has T = O; ``<image>`` (O - 1, O - 1) in (H, W),
just outside the canvas; the overview's tokens stretched over the canvas (the slices' grid, or the overview's own:
``torch.linspace(0, canvas - 1, n).round()``, float32, ties to even), the slices' tokens in their cells; ``</image>``
one past the canvas's far corner, ``<slice>`` / ``</slice>`` on their slice's top-left / bottom-right cell, a row's
"\\n" just past the right edge of the row's last cell row, the template's "\\n" after the image (O, O); the text after
the image from O + max(canvas_h, canvas_w) + 1. llama.cpp's mtmd (7182ef71e) numbers the markers like text and starts
the text after the image at O + its tiles and separators; its PR #30510 moves both to these values, leaving the
``<image>`` corner and the template's "\\n" (it emits none). The attention stays causal in token order (llama.cpp
breaks ties of its KV position by (H, W)).
"""

from dataclasses import dataclass

import numpy as np
import torch

# token ids of the markers (the GGUF's vocabulary; the official config's image_start_id .. newline_id)
IMAGE, IMAGE_END, SLICE, SLICE_END, IMAGE_ID, IMAGE_ID_END, NEWLINE = 248078, 248079, 248088, 248089, 248090, 248091, 198
MEDIA = -1  # an image token in a layout (its embedding row in place of a token's)


@dataclass(frozen=True)
class Grid:
    """The tokens of one image: the overview's (h, w), the slices' grid (rows, cols; (0, 0): none) and one slice's
    (h, w) tokens."""

    ov: tuple[int, int]
    grid: tuple[int, int] = (0, 0)
    cell: tuple[int, int] = (0, 0)

    @property
    def sliced(self) -> bool:
        return self.grid[0] * self.grid[1] > 0

    @property
    def canvas(self) -> tuple[int, int]:
        """(h, w) of the canvas in tokens: the slices' grid, or the overview."""
        return (self.grid[0] * self.cell[0], self.grid[1] * self.cell[1]) if self.sliced else self.ov

    def layout(self, *, newline_after: bool = True) -> list[int]:
        """The image's token ids, ``MEDIA`` for an embedding row (the template's "\\n" after it: ``newline_after``)."""
        out = [IMAGE] + [MEDIA] * (self.ov[0] * self.ov[1]) + [IMAGE_END]
        rows, cols = self.grid
        for r in range(rows):
            if r:
                out.append(NEWLINE)
            for _ in range(cols):
                out += [SLICE] + [MEDIA] * (self.cell[0] * self.cell[1]) + [SLICE_END]
        return out + [NEWLINE] * newline_after

    def positions(self, origin: int, *, newline_after: bool = True) -> tuple[np.ndarray, int]:
        """(T, H, W) (tokens of ``layout``, 3) of the image from position ``origin``, and the position of the text after
        it (module docstring)."""
        O, (ch, cw), (oh, ow) = origin, self.canvas, self.ov
        pos = []
        pos.append((O, O - 1, O - 1))  # <image>
        hs = torch.linspace(0, ch - 1, oh).round().long().numpy()
        ws = torch.linspace(0, cw - 1, ow).round().long().numpy()
        pos += [(O, O + h, O + w) for h in hs for w in ws]
        pos.append((O, O + ch, O + cw))  # </image>
        rows, cols = self.grid
        sh, sw = self.cell
        for r in range(rows):
            if r:  # the "\n" ending row r - 1
                pos.append((O, O + r * sh - 1, O + cols * sw))
            for c in range(cols):
                h0, w0 = r * sh, c * sw
                pos.append((O, O + h0, O + w0))  # <slice>
                pos += [(O, O + h0 + i, O + w0 + j) for i in range(sh) for j in range(sw)]
                pos.append((O, O + h0 + sh - 1, O + w0 + sw - 1))  # </slice>
        if newline_after:
            pos.append((O, O, O))
        return np.asarray(pos, np.int64), O + max(ch, cw) + 1
