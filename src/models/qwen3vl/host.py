"""Qwen3-VL's images in the qwen35 / qwen35moe language models (their mmproj, projector ``qwen3vl_merger``): an image's
size and pixels, its tokens and their M-RoPE positions.

Size (``target_size``): the whole image at its aspect, no slices: each side rounded to a multiple of ``align`` (the
patch 16 times the 2 x 2 merge: 32 pixels), then scaled into [min_tokens, max_tokens] merged tokens (llama.cpp's
``calc_size_preserved_ratio`` in float32 with Python's round, as transformers' ``smart_resize``; llama.cpp's
default bounds 8 and 4096 tokens: ``MIN_TOKENS``, ``MAX_TOKENS``), taken from the original by Pillow's bicubic filter
(llama.cpp's resampler is Pillow's).

Tokens: the chat template's ``<|vision_start|><|image_pad|><|vision_end|>`` per image (``image_prompt``), the
placeholder replaced by the image's merged tokens (gh x gw, raster order: ``Grid.sequence``). Positions (T, H, W), as
transformers' ``Qwen3VLModel.get_rope_index`` and llama.cpp's mtmd (``MTMD_POS_TYPE_MROPE``) give them: with O the
position after ``<|vision_start|>``, token (r, c) at (O, O + r, O + c) and the text after the image from O + max(gh,
gw); llama.cpp's ``rope.section_order`` tyxz (sections 0, 1, 2: T, H, W, interleaved: ``QwenConfig.rope_theta``).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..sequence import MEDIA, Sequence

MIN_TOKENS, MAX_TOKENS = 8, 4096  # (an image's merged tokens: llama.cpp's bounds for Qwen-VL)
IMAGE_PAD = "<|image_pad|>"  # (the chat template's place of an image, between <|vision_start|> and <|vision_end|>)


def target_size(width: int, height: int, *, align: int = 32, min_tokens: int = MIN_TOKENS,
                max_tokens: int = MAX_TOKENS) -> tuple[int, int]:
    """(width, height) of an image's encoder input: multiples of ``align`` (a merged token's pixels per side) of
    between min_tokens and max_tokens tokens at about the image's aspect (module docstring)."""
    f32 = np.float32
    f, area = align, align * align
    min_px, max_px = min_tokens * area, max_tokens * area
    rnd = lambda x: int(np.rint(f32(x) / f32(f))) * f  # (np.rint: ties to even, as Python's round)
    ceil = lambda x: int(np.ceil(f32(x) / f32(f))) * f
    floor = lambda x: int(np.floor(f32(x) / f32(f))) * f
    w, h = max(f, rnd(width)), max(f, rnd(height))
    if w * h > max_px:
        beta = np.sqrt(f32(height) * f32(width) / f32(max_px))
        w, h = max(f, floor(f32(width) / beta)), max(f, floor(f32(height) / beta))
    elif w * h < min_px:
        beta = np.sqrt(f32(min_px) / (f32(height) * f32(width)))
        w, h = ceil(f32(width) * beta), ceil(f32(height) * beta)
    return w, h


def image_pixels(im: Image.Image, *, align: int = 32, min_tokens: int = MIN_TOKENS,
                 max_tokens: int = MAX_TOKENS) -> np.ndarray:
    """An RGB image at its encoder size (``target_size``): uint8 (H, W, 3)."""
    w, h = target_size(*im.size, align=align, min_tokens=min_tokens, max_tokens=max_tokens)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(im.convert("RGB"), np.uint8)


def normalized(pixels: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """uint8 pixels (H, W, 3) -> the encoder's input (3, H, W) float32: (p / 255 - mean) / std (llama.cpp's)."""
    x = pixels.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    return (x - np.float32(mean)[:, None, None]) / np.float32(std)[:, None, None]


@dataclass(frozen=True)
class Grid:
    """An image's merged tokens: gh rows of gw."""

    gh: int
    gw: int

    @property
    def tokens(self) -> int:
        return self.gh * self.gw

    def positions(self) -> tuple[np.ndarray, int]:
        """(T, H, W) of the image's tokens (tokens, 3) from position 0, and the position of the text after them."""
        r, c = np.divmod(np.arange(self.tokens, dtype=np.int64), self.gw)
        return np.stack([np.zeros_like(r), r, c], 1), max(self.gh, self.gw)

    def sequence(self, embd: np.ndarray) -> Sequence:
        """The image's tokens from position 0: the encoder's rows ``embd`` (tokens, hidden; raster order)."""
        assert embd.shape[0] == self.tokens, (embd.shape, self)
        pos, nxt = self.positions()
        n = self.tokens
        return Sequence(np.full(n, MEDIA, np.int64), np.ones(n, bool), np.ascontiguousarray(embd, np.float32), pos, nxt)


def image_prompt(text: str, images: list[object], encode: Callable[[str], list[int]],
                 sequence: Callable[[object], Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``IMAGE_PAD`` per image -> its tokens: each placeholder becomes the
    image's tokens (``sequence(image)``: ``Grid.sequence``'s); the template's ``<|vision_start|>`` and
    ``<|vision_end|>`` around it stay text."""
    parts = text.split(IMAGE_PAD)
    if len(parts) != len(images) + 1:
        raise ValueError(f"{len(images)} image(s) for {len(parts) - 1} place(s) in the prompt")
    seq = Sequence.text(encode(parts[0]))
    for im, tail in zip(images, parts[1:], strict=True):
        seq = seq + sequence(im) + encode(tail)
    return seq
