"""What the image encoders' host sides share (Qwen3-VL's ``models.qwen3vl.host``, LFM3-VL's ``models.lfm3vl.host``): an
image at its encoder size as uint8 pixels, their normalization, the grid of merged tokens and its sequence, a chat
prompt's placeholders replaced by images. Each model keeps its ``target_size`` (the two compute the size in float32
and in float64: llama.cpp's and transformers' calculations) and its tokens' positions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .sequence import Sequence, splice


def resized(im: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """An RGB image at ``size`` (width, height), taken from the original by Pillow's bicubic filter (llama.cpp's
    resampler): uint8 (H, W, 3)."""
    if size != im.size:
        im = im.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(im.convert("RGB"), np.uint8)


def normalized(pixels: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """uint8 pixels (H, W, 3) -> the encoder's input (3, H, W) float32: (p / 255 - mean) / std (llama.cpp's)."""
    x = pixels.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    return (x - np.float32(mean)[:, None, None]) / np.float32(std)[:, None, None]


@dataclass(frozen=True)
class Grid:
    """An image's merged tokens: gh rows of gw, one position each (a model whose positions are other overrides
    ``positions``)."""

    gh: int
    gw: int

    @property
    def tokens(self) -> int:
        return self.gh * self.gw

    def positions(self) -> tuple[np.ndarray, int]:
        """(T, H, W) of the image's tokens (tokens, 3) from position 0, and the position of the text after them."""
        return np.repeat(np.arange(self.tokens, dtype=np.int64)[:, None], 3, 1), self.tokens

    def sequence(self, embd: np.ndarray) -> Sequence:
        """The image's tokens from position 0: the encoder's rows ``embd`` (tokens, hidden; raster order)."""
        assert embd.shape[0] == self.tokens, (embd.shape, self)
        return Sequence.media(embd, *self.positions())


def marker_prompt(text: str, marker: str, images: list[object], encode: Callable[[str], list[int]],
                  sequence: Callable[[object], Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``marker`` per image -> its tokens: each placeholder becomes the image's
    tokens (``sequence(image)``: ``Grid.sequence``'s); the template's ``<|vision_start|>`` and ``<|vision_end|>`` around
    it stay text."""
    if text.count(marker) != len(images):
        raise ValueError(f"{len(images)} image(s) for {text.count(marker)} place(s) in the prompt")
    return splice(text, marker, [sequence(im) for im in images], encode)
