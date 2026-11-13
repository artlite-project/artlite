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

from ..images import TOKEN_PX, marker_prompt, resized
from ..images import Grid as MergedGrid
from ..sequence import Sequence

MIN_TOKENS, MAX_TOKENS = 8, 4096  # (an image's merged tokens: llama.cpp's bounds for Qwen-VL)
IMAGE_PAD = "<|image_pad|>"  # (the chat template's place of an image, between <|vision_start|> and <|vision_end|>)


def target_size(width: int, height: int, *, align: int = TOKEN_PX, min_tokens: int = MIN_TOKENS,
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


def image_pixels(im: Image.Image, *, align: int = TOKEN_PX, min_tokens: int = MIN_TOKENS,
                 max_tokens: int = MAX_TOKENS) -> np.ndarray:
    """An RGB image at its encoder size (``target_size``): uint8 (H, W, 3)."""
    return resized(im, target_size(*im.size, align=align, min_tokens=min_tokens, max_tokens=max_tokens))


@dataclass(frozen=True)
class Grid(MergedGrid):
    """An image's merged tokens: gh rows of gw, at M-RoPE positions."""

    def positions(self) -> tuple[np.ndarray, int]:
        """(T, H, W) of the image's tokens (tokens, 3) from position 0, and the position of the text after them."""
        r, c = np.divmod(np.arange(self.tokens, dtype=np.int64), self.gw)
        return np.stack([np.zeros_like(r), r, c], 1), max(self.gh, self.gw)


def image_prompt(text: str, images: list[object], encode: Callable[[str], list[int]],
                 sequence: Callable[[object], Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``IMAGE_PAD`` per image -> its tokens (``images.marker_prompt``; the
    image's tokens: ``Grid.sequence``'s)."""
    return marker_prompt(text, IMAGE_PAD, images, encode, sequence)
