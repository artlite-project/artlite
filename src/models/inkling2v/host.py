"""Inkling 2's images in its language model (the mmproj of projector ``inkling2v``; the model: ``models.inkling2``): an
image's size, pixels and padding, its tokens.

Size (``target_size``): the image as it is, no resize (transformers' ``Inkling2ImageProcessor``), padded at the bottom
and the right to multiples of ``TOKEN_PX`` (a merged token: patches of 20 pixels merged 2 x 2) with the pixel value -1
on the 0 .. 255 scale (normalized like the others: (p / 255 - 0.5) / 0.5, so a padding pixel is 1 / 127.5 below
black). Only an image of more than ``max_tokens`` merged tokens so padded is resized (llama.cpp's
``mtmd_image_preprocessor_inkling2v``: its cap of 4096, ``MAX_TOKENS``): to ``calc_size_preserved_ratio``'s multiples of
40 (each side rounded half away from zero, then scaled under the cap by the floor, in float32), by Pillow's bicubic
filter (llama.cpp's resampler).

Tokens: the chat template's ``<|content_image|><|unused_200054|><|end_message|>`` per image, the placeholder ``IMAGE``
replaced by the image's merged tokens (gh x gw, raster order: ``Grid.sequence``), one position each (the language
model's positions are 1-D); their rows enter the language model as they are (a text token's row takes
``token_embd_norm``, an image's does not: llama.cpp's ``inkling2``).
"""

import math
from collections.abc import Callable

import numpy as np
from PIL import Image

from ..images import Grid, marker_prompt, resized
from ..sequence import Sequence

PATCH, MERGE = 20, 2
TOKEN_PX = PATCH * MERGE  # (a merged token's pixels per side)
MAX_TOKENS = 4096  # (an image's merged tokens at most: llama.cpp's cap)
IMAGE = "<|unused_200054|>"  # (the chat template's place of an image, after <|content_image|>)
PAD = -1  # (the padding pixels' value on the 0 .. 255 scale)


def target_size(width: int, height: int, *, max_tokens: int = MAX_TOKENS) -> tuple[int, int]:
    """(width, height) the image is taken at: its own unless its padded grid has more than ``max_tokens`` merged
    tokens, else llama.cpp's ``calc_size_preserved_ratio`` (module docstring; multiples of ``TOKEN_PX``)."""
    f = TOKEN_PX
    if -(-width // f) * -(-height // f) <= max_tokens:
        return width, height
    f32 = np.float32
    rnd = lambda x: math.floor(f32(x) / f32(f) + 0.5) * f  # (C's std::round of a positive value)
    floor = lambda x: int(np.floor(f32(x) / f32(f))) * f
    w, h = max(f, rnd(width)), max(f, rnd(height))
    if w * h > max_tokens * f * f:
        beta = np.sqrt(f32(height) * f32(width) / f32(max_tokens * f * f))
        w, h = max(f, floor(f32(width) / beta)), max(f, floor(f32(height) / beta))
    return w, h


def image_pixels(im: Image.Image, *, max_tokens: int = MAX_TOKENS) -> np.ndarray:
    """An RGB image at its encoder size (``target_size``; not padded): uint8 (H, W, 3)."""
    return resized(im, target_size(*im.size, max_tokens=max_tokens))


def grid(px: np.ndarray) -> Grid:
    """The merged tokens of pixels (H, W, 3) padded to multiples of ``TOKEN_PX``."""
    return Grid(-(-px.shape[0] // TOKEN_PX), -(-px.shape[1] // TOKEN_PX))


def image_prompt(text: str, images: list[object], encode: Callable[[str], list[int]],
                 sequence: Callable[[object], Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``IMAGE`` per image -> its tokens (``images.marker_prompt``; the
    image's tokens: ``Grid.sequence``'s)."""
    return marker_prompt(text, IMAGE, images, encode, sequence)
