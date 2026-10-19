"""MiniCPM-V 4.7's images in the qwen35 language model (its mmproj, projector ``minicpmv4_7``): an image's slices and
pixels, its token layout and its M-RoPE positions.

Slicing (``slice_image``), as transformers' ``MiniCPMV4_6ImageProcessor`` and the training code (MiniCPM-V's
``finetune/dataset.py``): an image of more than ``image_size`` squared pixels gets a grid of slices (``slice_grid``:
of ``max_slices`` +- 1 slices, the grid whose cols / rows is nearest the image's aspect in log, in double: llama.cpp's
mtmd computes it in float, where ties between transposed grids pick the one of fewer columns: 3 x 2 slices of a 1024 x
1024 image instead of 2 x 3) and an overview of about that area; a small image is only scaled up to it. Sides align to
4 patches (56 pixels: the encoder's two 2 x 2 merges, in both files), each size taken from the original by Pillow's
bicubic filter (the training's; llama.cpp ports it), the pixels normalized by the mmproj's mean and std.

An image is an overview (``ov``: tokens h x w) and, unless it is small, a grid of slices (``rows`` x ``cols`` of ``cell``
tokens each). Its tokens, as the official processor lays them out (transformers' ``MiniCPMV4_7Processor``)::

    <image> [ov] </image> <slice> [S00] </slice> <slice> [S01] </slice> \\n <slice> [S10] </slice> ...

the slice rows joined by "\\n", after the text ``<image_id>k</image_id>`` (k counting a prompt's images from 0); the chat
template's "\\n" after the image follows.

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

import io
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

from artlite.model_config import ClipConfig

from .sequence import MEDIA, Sequence

# token ids of the markers (the GGUF's vocabulary; the official config's image_start_id .. newline_id)
IMAGE, IMAGE_END, SLICE, SLICE_END, IMAGE_ID, IMAGE_ID_END, NEWLINE = 248078, 248079, 248088, 248089, 248090, 248091, 198
# the markers' rows in the mmproj's v.tok_embd_sep (the language model's embeddings of them, unquantized)
SEP_ROWS = {IMAGE_END: 0, SLICE: 1, SLICE_END: 2, NEWLINE: 3}


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

    def sequence(self, embd: np.ndarray, sep: np.ndarray, *, newline_after: bool = True) -> Sequence:
        """The image's tokens (``layout``, the template's "\\n" after it: ``newline_after``) from position 0: the
        encoder's rows ``embd`` (its entries' tokens in order: the overview's, then the slices' row by row) for its
        ``MEDIA`` tokens, the mmproj's ``v.tok_embd_sep`` rows ``sep`` for the markers inside it (``SEP_ROWS``);
        ``<image>`` and the template's "\\n" as text tokens."""
        ids = np.asarray(self.layout(newline_after=newline_after), np.int64)
        given = np.ones(len(ids), bool)
        given[0] = False
        given[-1] &= not newline_after
        media = ids[given] == MEDIA
        rows = np.empty((int(given.sum()), embd.shape[1]), np.float32)
        assert embd.shape[0] == media.sum(), (embd.shape, int(media.sum()))
        rows[media] = embd
        rows[~media] = sep[[SEP_ROWS[int(t)] for t in ids[given][~media]]]
        pos, nxt = self.positions(0, newline_after=newline_after)
        return Sequence(ids, given, rows, pos, nxt)


IMAGE_PAD = "<|image_pad|>"  # (the chat template's place of an image)


def image_prompt(text: str, images: list[object], encode: Callable[[str], list[int]],
                 sequence: Callable[..., Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``IMAGE_PAD`` per image -> its tokens: each placeholder becomes the text
    ``<image_id>k</image_id>`` (k counting the prompt's images from 0) and the image's tokens (``sequence(image,
    newline_after=)``: ``Grid.sequence``'s); the template's "\\n" after a placeholder is the image's last token, at its
    official position."""
    parts = text.split(IMAGE_PAD)
    if len(parts) != len(images) + 1:
        raise ValueError(f"{len(images)} image(s) for {len(parts) - 1} place(s) in the prompt")
    seq = Sequence.text(encode(parts[0]))
    for k, (im, tail) in enumerate(zip(images, parts[1:], strict=True)):
        nl = tail.startswith("\n")
        seq = seq + encode(f"<image_id>{k}</image_id>") + sequence(im, newline_after=nl) + encode(tail[nl:])
    return seq


# ----------------------------------------------------------------------
# Slicing
# ----------------------------------------------------------------------


def _divide(length: float, d: int) -> int:
    return max(round(length / d) * d, d)  # (Python's round: ties to even)


def best_resize(h: float, w: float, scale: int, align: int, *, upscale: bool = False) -> tuple[int, int]:
    """(h, w) of an image (or a slice) of more than ``scale`` squared pixels (or any, ``upscale``) brought to about
    that area at its aspect, the sides rounded to multiples of ``align``."""
    if h * w > scale * scale or upscale:
        r = w / h
        h = int(scale / math.sqrt(r))
        w = int(h * r)
    return _divide(h, align), _divide(w, align)


def slice_grid(h: int, w: int, max_slices: int, scale: int) -> tuple[int, int] | None:
    """(rows, cols) of an image's slices, None for an image of at most one slice's area."""
    log_ratio = math.log(w / h)
    multiple = min(math.ceil(w * h / (scale * scale)), max_slices)
    if multiple <= 1:
        return None
    best, err = (1, 1), math.inf
    for n in (multiple - 1, multiple, multiple + 1):
        if n == 1 or n > max_slices:
            continue
        for rows in range(1, n + 1):
            if n % rows == 0:
                e = abs(log_ratio - math.log((n // rows) / rows))
                if e < err or (e == err and rows > best[0]):
                    best, err = (rows, n // rows), e
    return best


def refine_size(h: int, w: int, grid: tuple[int, int], scale: int, align: int) -> tuple[int, int]:
    """(h, w) of the image that the grid's slices cut."""
    rows, cols = grid
    sh, sw = best_resize(_divide(h, rows) / rows, _divide(w, cols) / cols, scale, align, upscale=True)
    return sh * rows, sw * cols


def load_image(src: str | bytes) -> Image.Image:
    """An image file (a path or its bytes) as RGB, turned upright by its EXIF orientation (transformers'
    ``load_image``; an alpha channel dropped; a JPEG's pixels within 3 levels of llama.cpp's stb_image)."""
    with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _pixels(im: Image.Image, clip: ClipConfig) -> np.ndarray:
    x = np.asarray(im, np.float32).transpose(2, 0, 1) / np.float32(255)
    return (x - np.float32(clip.mean)[:, None, None]) / np.float32(clip.std)[:, None, None]


def slice_image(im: Image.Image, clip: ClipConfig, max_slices: int | None = None) -> tuple[list[np.ndarray], Grid]:
    """An RGB image's entries for the encoder (the overview, then the slices row by row: pixels (3, H, W), float32)
    and its token grid; ``max_slices``: the mmproj's (9) by default (MiniCPM-V's card uses 36 for images)."""
    w, h = im.size
    scale, align, tok = clip.image_size, clip.patch_size * 4, clip.patch_size * clip.merge
    grid = slice_grid(h, w, max_slices or clip.max_slices, scale)
    oh, ow = best_resize(h, w, scale, align, upscale=grid is None)
    entries = [_pixels(im.resize((ow, oh), Image.Resampling.BICUBIC), clip)]
    if grid is None:
        return entries, Grid(ov=(oh // tok, ow // tok))
    rh, rw = refine_size(h, w, grid, scale, align)
    ref = im.resize((rw, rh), Image.Resampling.BICUBIC)
    sh, sw = rh // grid[0], rw // grid[1]
    entries += [_pixels(ref.crop((c * sw, r * sh, (c + 1) * sw, (r + 1) * sh)), clip)
                for r in range(grid[0]) for c in range(grid[1])]
    return entries, Grid(ov=(oh // tok, ow // tok), grid=grid, cell=(sh // tok, sw // tok))
