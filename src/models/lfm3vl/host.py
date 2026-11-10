"""LFM3-VL's images in the lfm3 language models (their mmproj, projector ``lfm3vl``: LFM3-VL-2B's, d1-2B's): an image's
size and pixels, its tokens.

Size (``target_size``): the whole image at its aspect, no tiles: each side rounded to a multiple of ``align`` (the
patch 16 times the 2 x 2 merge: 32 pixels), then scaled into [min_tokens, max_tokens] merged tokens (transformers'
``smart_resize`` of Qwen2-VL in float64, round half to even, as llama.cpp's ``mtmd_image_preprocessor_lfm3vl``; the
bounds: the mmproj's ``image_min_pixels`` / 1024 = 64 tokens and llama.cpp's cap of 4096: ``MIN_TOKENS``,
``MAX_TOKENS``), taken from the original by Pillow's bicubic filter (llama.cpp's is Pillow's with torchvision's int16
weights: a pixel at most a level apart).

Tokens: the chat template's ``<|vision_start|><|image|><|vision_end|>`` per image (``image_prompt``), the placeholder
replaced by the image's merged tokens (gh x gw, raster order: ``Grid.sequence``), one position each (the language
model's positions are 1-D).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..sequence import MEDIA, Sequence

MIN_TOKENS, MAX_TOKENS = 64, 4096  # (an image's merged tokens: the mmproj's min pixels / 1024, llama.cpp's cap)
IMAGE = "<|image|>"  # (the chat template's place of an image, between <|vision_start|> and <|vision_end|>)


def target_size(width: int, height: int, *, align: int = 32, min_tokens: int = MIN_TOKENS,
                max_tokens: int = MAX_TOKENS) -> tuple[int, int]:
    """(width, height) of an image's encoder input: multiples of ``align`` (a merged token's pixels per side) of
    between min_tokens and max_tokens tokens at about the image's aspect (module docstring)."""
    f, h, w = align, float(height), float(width)
    min_px, max_px = min_tokens * f * f, max_tokens * f * f
    hb, wb = round(h / f) * f, round(w / f) * f  # (Python's round: half to even, as C's nearbyint)
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb, wb = math.floor(h / beta / f) * f, math.floor(w / beta / f) * f
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return max(f, wb), max(f, hb)


def image_pixels(im: Image.Image, *, align: int = 32, min_tokens: int = MIN_TOKENS,
                 max_tokens: int = MAX_TOKENS) -> np.ndarray:
    """An RGB image at its encoder size (``target_size``): uint8 (H, W, 3)."""
    w, h = target_size(*im.size, align=align, min_tokens=min_tokens, max_tokens=max_tokens)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(im.convert("RGB"), np.uint8)


def normalized(pixels: np.ndarray) -> np.ndarray:
    """uint8 pixels (H, W, 3) -> the encoder's input (3, H, W) float32: (p / 255 - 0.5) / 0.5 (llama.cpp's)."""
    x = pixels.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    return (x - np.float32(0.5)) / np.float32(0.5)


@dataclass(frozen=True)
class Grid:
    """An image's merged tokens: gh rows of gw."""

    gh: int
    gw: int

    @property
    def tokens(self) -> int:
        return self.gh * self.gw

    def sequence(self, embd: np.ndarray) -> Sequence:
        """The image's tokens: the encoder's rows ``embd`` (tokens, hidden; raster order), a position each."""
        assert embd.shape[0] == self.tokens, (embd.shape, self)
        n = self.tokens
        pos = np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1)
        return Sequence(np.full(n, MEDIA, np.int64), np.ones(n, bool), np.ascontiguousarray(embd, np.float32), pos, n)


def image_prompt(text: str, images: list[object], encode: Callable[[str], list[int]],
                 sequence: Callable[[object], Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``IMAGE`` per image -> its tokens: each placeholder becomes the image's
    tokens (``sequence(image)``: ``Grid.sequence``'s); the template's ``<|vision_start|>`` and ``<|vision_end|>``
    around it stay text."""
    parts = text.split(IMAGE)
    if len(parts) != len(images) + 1:
        raise ValueError(f"{len(images)} image(s) for {len(parts) - 1} place(s) in the prompt")
    seq = Sequence.text(encode(parts[0]))
    for im, tail in zip(images, parts[1:], strict=True):
        seq = seq + sequence(im) + encode(tail)
    return seq
