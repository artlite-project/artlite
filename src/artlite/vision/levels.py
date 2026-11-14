"""The levels of the vision overlay: the geometry of a band of tokens per column (the memtile holds one band's tensors).

A level gives the tile (``m`` tokens, ``ntile`` of them), the token panels per kb (``np``) and the bytes of its tensors
(``ct``: C tiles, ``pn``: panels). ``ImageLevel``: the convolutional networks' (Stable Diffusion's UNet and VAE decoder,
and the tests'); ``TokenLevel``: the ViTs' (one tile per column, ``vit``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..operators.design import COLS
from . import layout as L


class Level(ABC):
    """A level's geometry per column; ``nb``: its tokens in that many bands."""

    nb: int

    @property
    @abstractmethod
    def m(self) -> int:
        """Tokens of a tile."""

    @property
    @abstractmethod
    def ntile(self) -> int:
        """Tiles of a band."""

    @property
    @abstractmethod
    def np(self) -> int:
        """Token panels per kb of a band."""

    def ct(self, C: int) -> int:
        """Bytes of a C-tile tensor of C channels (one band)."""
        return C // 80 * self.ntile * L.ct_bytes(self.m)

    def pn(self, C: int) -> int:
        """Bytes of a panel tensor of C channels (one band)."""
        return C // 8 * self.np * L.PANEL


@dataclass(frozen=True)
class ImageLevel(Level):
    """An image level per column: R image rows of Lg groups of 8 lanes, both images; its tokens in ``nb`` bands (the
    memtile holds one band's tensors: 32x32 two of 2 rows, 64x64 four), a band's in tiles of m tokens (64: 4 groups of a
    row; 8x8: 16). half: one image of H x H, its left and right halves as the two images (``vae``): Lg groups of a half
    row."""

    H: int
    nb: int = 1
    half: bool = False

    @property
    def R(self) -> int:
        return self.H // COLS

    @property
    def Lg(self) -> int:
        return self.H // (16 if self.half else 8)

    @property
    def T(self) -> int:  # (a band's tokens per column)
        return 16 * self.R * self.Lg // self.nb

    @property
    def m(self) -> int:
        return min(self.T, 64)

    @property
    def ntile(self) -> int:
        return self.T // self.m

    @property
    def np(self) -> int:  # (token panels per kb)
        return self.T // 8


@dataclass(frozen=True)
class TokenLevel(Level):
    """A ViT level: ``nb`` bands of one tile per column of ``tm`` tokens (64; 16: the merges' tiles of merged tokens;
    both images: token panel tp of image tp % 2)."""

    nb: int
    tm: int = 64

    @property
    def m(self) -> int:
        return self.tm

    @property
    def ntile(self) -> int:
        return 1

    @property
    def np(self) -> int:
        return self.tm // 8
