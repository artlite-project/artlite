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

Videos (``VideoPlan``: transformers' ``Lfm3VlVideoProcessor`` and LFM3-Omni's processor as llama.cpp's mtmd runs them,
``MTMD_VIDEO_TEMPLATE_LFM3`` / ``_LFM3_OMNI``): frames sampled at 2 a second, n = trunc(n_src / fps x 2) clamped to 4 ..
768 (with the soundtrack, LFM3-Omni: 384) and to the video's frames, the source frames round(linspace(0, n_src - 1, n));
one size for every frame (Qwen3-VL's video rule: ``video_size``; a frame of more than ``MAX_TOKENS`` merged tokens,
which llama.cpp does not cap, takes an image's size instead: a video under 3 s of more than 4 Mpx a frame);
consecutive frames in pairs, one temporal patch each
(the patch embedding's two kernels on its two frames; an odd last frame alone: both kernels on it, as on an image).
The chat template's ``<|vision_start|><|video|><|vision_end|>`` becomes per pair its timestamp ``<t seconds>`` (the
mean of its source frames' times, "%.1f"), ``<|vision_start|>``, its merged tokens, ``<|vision_end|>`` and, with a
soundtrack, the pair's rows of it (``video_sequence``): the soundtrack is one clip (``models.video.soundtrack``), split
at each pair's first frame (``VideoPlan.slices``) and its encoder rows by round(samples / 1280) at those points, the
last pair taking the rest (``VideoPlan.rows``).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..sequence import MEDIA, Sequence, splice

MIN_TOKENS, MAX_TOKENS = 64, 4096  # (an image's merged tokens: the mmproj's min pixels / 1024, llama.cpp's cap)
IMAGE = "<|image|>"  # (the chat template's place of an image, between <|vision_start|> and <|vision_end|>)
VIDEO = "<|video|>"  # (the chat template's place of a video, between <|vision_start|> and <|vision_end|>)
VISION_START, VISION_END = "<|vision_start|>", "<|vision_end|>"
FPS = 2.0  # (a video's frames sampled a second)
FRAMES, FRAMES_HEARD = (4, 768), 384  # (a video's frames at least, at most; at most with its soundtrack)
VIDEO_PIXELS = (4096, 25165824)  # (all of a video's frames' pixels: at least, at most)
ROW_SAMPLES = 1280  # (the soundtrack's samples a row of the speech encoder: 8 mel frames of 160)


def token_bounds(min_pixels: int, max_pixels: int, min_tokens: int | None = None,
                 max_tokens: int | None = None) -> tuple[int, int]:
    """An image's merged tokens at least and at most: ``min_tokens`` / ``max_tokens`` where given, else the mmproj's
    ``image_min_pixels`` / ``image_max_pixels`` over 1024, the maximum at most ``MAX_TOKENS`` (llama.cpp's)."""
    return min_tokens or max(1, min_pixels // 1024), max_tokens or min(MAX_TOKENS, max_pixels // 1024)


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
    if text.count(IMAGE) != len(images):
        raise ValueError(f"{len(images)} image(s) for {text.count(IMAGE)} place(s) in the prompt")
    return splice(text, IMAGE, [sequence(im) for im in images], encode)


def video_size(width: int, height: int, n: int, *, align: int = 32) -> tuple[int, int]:
    """(width, height) of a video's frames for n of them (Qwen3-VL's video ``smart_resize`` in float64, as llama.cpp's:
    sides below ``align`` scaled up first; the pixels of t_bar = 2 round(n / 2) frames bounded by ``VIDEO_PIXELS``, the
    scale computed over n frames)."""
    f, h, w = align, float(height), float(width)
    if h < f or w < f:
        s = max(f / h, f / w)
        h, w = float(int(h * s)), float(int(w * s))
    lo, hi = VIDEO_PIXELS
    t = round(n / 2) * 2
    hb, wb = round(h / f) * f, round(w / f) * f
    if t * hb * wb > hi:
        beta = math.sqrt(n * h * w / hi)
        hb, wb = math.floor(h / beta / f) * f, math.floor(w / beta / f) * f
    elif t * hb * wb < lo:
        beta = math.sqrt(lo / (n * h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return max(f, wb), max(f, hb)


@dataclass(frozen=True)
class VideoPlan:
    """A video's sampled frames (module docstring): the source frame of each (``indices``), the frames' size
    (``width`` x ``height``), the source's rate (``fps``: a float32's value; times in float64, as llama.cpp's)."""

    indices: tuple[int, ...]
    width: int
    height: int
    fps: float

    @classmethod
    def of(cls, n_frames: int, fps: float, width: int, height: int, *, heard: bool) -> "VideoPlan":
        """The plan of a video stream of n_frames at fps (``models.video.Info``), with its soundtrack (``heard``) or
        not."""
        n = int(n_frames / fps * FPS)
        n = min(max(n, FRAMES[0]), FRAMES_HEARD if heard else FRAMES[1], n_frames)
        step = (n_frames - 1) / (n - 1) if n > 1 else 0.0
        idx = tuple(int(np.rint(i * step)) for i in range(n))
        w, h = video_size(width, height, n)
        if w * h > MAX_TOKENS * 1024:  # (the tower's largest image)
            w, h = target_size(width, height)
        return cls(idx, w, h, fps)

    @property
    def grid(self) -> Grid:
        return Grid(self.height // 32, self.width // 32)

    def pairs(self, n: int | None = None) -> list[tuple[int, ...]]:
        """The temporal patches of the first n frames (default all): frames (2 i, 2 i + 1), an odd last one alone."""
        n = len(self.indices) if n is None else n
        return [tuple(range(i, min(i + 2, n))) for i in range(0, n, 2)]

    def stamps(self, n: int | None = None) -> list[str]:
        """Each pair's timestamp text: the mean of its source frames' times."""
        return [f"<{(self.indices[p[0]] / self.fps + self.indices[p[-1]] / self.fps) / 2.0:.1f} seconds>"
                for p in self.pairs(n)]

    def slices(self, n_samples: int, rate: int = 16000, n: int | None = None) -> list[tuple[int, int]]:
        """Each pair's samples [begin, end) of a soundtrack of n_samples: from the pair's first frame's time to the next
        pair's (the first from 0, the last to the end)."""
        pairs = self.pairs(n)
        first = lambda g: min(n_samples, int(np.rint(self.indices[pairs[g][0]] / self.fps * rate)))
        return [(0 if g == 0 else first(g), n_samples if g == len(pairs) - 1 else first(g + 1))
                for g in range(len(pairs))]

    def rows(self, slices: list[tuple[int, int]], n_rows: int) -> list[tuple[int, int]]:
        """Each pair's rows [begin, end) of the soundtrack's n_rows: a pair with samples ends at round(its samples' end
        / ``ROW_SAMPLES``) (clamped), the last such pair at n_rows (llama.cpp's split of the joined pieces)."""
        last = max((g for g, (b, e) in enumerate(slices) if e > b), default=-1)
        out, at, total = [], 0, 0
        for g, (b, e) in enumerate(slices):
            if e <= b:
                out.append((at, at))
                continue
            total += e - b
            end = n_rows if g == last else min(max(int(np.rint(total / ROW_SAMPLES)), at), n_rows)
            out.append((at, end))
            at = end
        return out


def video_sequence(pairs: list[Sequence], stamps: list[str], encode: Callable[[str], list[int]],
                   rows: list[Sequence] | None = None) -> Sequence:
    """A video's tokens between the chat template's markers (module docstring): per pair its timestamp, its tokens
    (``Grid.sequence``) between ``<|vision_start|>`` and ``<|vision_end|>``, then its soundtrack rows (``rows``, a
    sequence per pair: ``models.lfm3a.host.clip_sequence``); each text piece tokenized alone (llama.cpp's mtmd)."""
    seq = None
    for i, (pair, stamp) in enumerate(zip(pairs, stamps, strict=True)):
        s = Sequence.text(encode(stamp)) + encode(VISION_START) + pair + encode(VISION_END)
        if rows is not None and len(rows[i]):
            s = s + rows[i]
        seq = s if seq is None else seq + s
    return seq
