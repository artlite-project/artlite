"""The vision overlay's layouts on the host, for the tests of its ops and of the networks built on them: bfp16 rounding of
operands, an image level's tokens per column (and the 64x64 level's row tiles) and their inverses."""

import numpy as np
import torch

from artlite.operators.design import COLS
from artlite.vision import layout as L
from artlite.vision.weights import Slots, job_arrays, pack_arrays

IL = (0, 2, 4, 6, 1, 3, 5, 7)  # (64x64: a row's groups in their storage order, even then odd)


def q(x: torch.Tensor) -> torch.Tensor:
    """x rounded to bfp16ebs8 blocks of 8 along the last dimension."""
    return torch.from_numpy(L.unblocks(L.blocks(np.ascontiguousarray(x.float().reshape(*x.shape[:-1], -1, 8).numpy()))
                                       )).reshape(x.shape)


def level_tokens(x: torch.Tensor, rows: int, L_: int) -> torch.Tensor:
    """(2 images, C, H, W) -> per column its tokens (COLS, rows 2 L_ 8, C): column c's image rows c rows .., token panel
    (row, g, image), lane j = pixel x = g + L_ j."""
    _, C, H, W = x.shape
    assert H == COLS * rows and W == 8 * L_
    v = x.reshape(2, C, COLS, rows, 8, L_)  # i C c row j g
    return v.permute(2, 3, 5, 0, 4, 1).reshape(COLS, rows * L_ * 16, C)


def from_level_tokens(y: torch.Tensor, rows: int, L_: int) -> torch.Tensor:
    C = y.shape[-1]
    return y.reshape(COLS, rows, L_, 2, 8, C).permute(3, 5, 0, 1, 4, 2).reshape(2, C, COLS * rows, 8 * L_)


def qconv_w(w: torch.Tensor) -> torch.Tensor:
    """Conv weights (N, C, 3, 3) rounded as the B stream holds them (blocks of 8 input channels per tap)."""
    return q(w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def il_tokens(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``level_tokens`` of a 64x64 level's rows (8 groups): token panel (row, group IL[k], image)."""
    t = level_tokens(x, rows, 8)
    return t.reshape(COLS, rows, 8, 16, -1)[:, :, list(IL)].reshape(t.shape)


def from_il_tokens(y: torch.Tensor, rows: int) -> torch.Tensor:
    inv = [IL.index(g) for g in range(8)]
    return from_level_tokens(y.reshape(COLS, rows, 8, 16, -1)[:, :, inv].reshape(y.shape), rows, 8)


def pack_to(slots: Slots, out: dict[str, np.ndarray]) -> None:
    """The pending weight jobs of ``slots`` packed into host arrays (uint8) by image name."""
    jobs, slots.pending = slots.pending, []
    pack_arrays(job_arrays(jobs, lambda im: (im, 0, out[im].size)), out.__getitem__)
