"""sd15_v2's memtile tensor layouts and their host packers (tests, the level boundaries' DDR images).

Per column (its tokens: ``NP`` token panels of 8; a tile = the m tokens a core's C tile holds: panels 16 t .. for
m = 128):
- panels: bfp16 [kb = channel / 8][token panel][8 tokens][9 B] (a row: one token's 8 channels, exponent first): the
  GEMM's A operand; V^T panels: the same with [8 channels][9 B] (a row: one channel's 8 tokens);
- C tiles: bf16 objects [channel block of 80][tile][``ct_bytes``]: the C tile [rb][10 cb][8 tokens][8 channels], padded
  to whole output chunks (the cores' C-tile outputs, residual inputs).
Weights of row r (``gemm_weights``): per pass of 80 output channels its k-tiles, each [16-column block 5][k-step 8][half
2][8 outputs][9 B] (the MAC's B pop order).
"""

import numpy as np
import torch

from .. import bfp16 as _bfp16
from . import design as D

PANEL = 72


def blocks(x: np.ndarray) -> np.ndarray:
    """float (..., 8) -> bfp16ebs8 bytes (..., 9) [E, m0..m7] (round to nearest even)."""
    e, m = _bfp16.blocks(x)
    return np.concatenate([e[..., None], m.view(np.uint8)], axis=-1)


def unblocks(b: np.ndarray) -> np.ndarray:
    """bfp16ebs8 bytes (..., 9) -> float (..., 8) (padding panels may overflow to inf)."""
    with np.errstate(over="ignore"):
        return b[..., 1:].view(np.int8).astype(np.float32) * np.ldexp(np.float32(1), b[..., 0].astype(np.int32) - 133)[..., None]


def ct_bytes(m: int) -> int:
    """Bytes of a bf16 C-tile object of m tokens (whole output chunks)."""
    return -(-m * 160 // D.OCH) * D.OCH


def panels(x: torch.Tensor) -> np.ndarray:
    """(tokens, C) -> the panel layout [C / 8][tokens / 8][8][9] (bytes)."""
    t, c = x.shape
    return blocks(np.ascontiguousarray(x.float().reshape(t // 8, 8, c // 8, 8).permute(2, 0, 1, 3).numpy())).reshape(-1)


def from_panels(b: np.ndarray, tokens: int, c: int) -> torch.Tensor:
    v = unblocks(np.asarray(b, np.uint8)[:c // 8 * tokens * 9].reshape(c // 8, tokens // 8, 8, 9))
    return torch.from_numpy(np.ascontiguousarray(v)).permute(1, 2, 0, 3).reshape(tokens, c)


def vt_panels(x: torch.Tensor) -> np.ndarray:
    """(tokens, C) -> panels of the transposes [C / 8][tokens / 8][8 channels][8 tokens] (bytes; the exponent per
    channel)."""
    t, c = x.shape
    return blocks(np.ascontiguousarray(x.float().reshape(t // 8, 8, c // 8, 8).permute(2, 0, 3, 1).numpy())).reshape(-1)


def from_vt_panels(b: np.ndarray, tokens: int, c: int) -> torch.Tensor:
    v = unblocks(np.asarray(b, np.uint8)[:c // 8 * tokens * 9].reshape(c // 8, tokens // 8, 8, 9))
    return torch.from_numpy(np.ascontiguousarray(v)).permute(1, 3, 0, 2).reshape(tokens, c)


def ctiles(x: torch.Tensor, m: int) -> np.ndarray:
    """(tokens, C) -> bf16 C-tile objects [C / 80][tokens / m][ct_bytes(m)] (bytes)."""
    t, c = x.shape
    nt = t // m
    v = x.float().reshape(nt, m // 8, 8, c // 80, 10, 8).permute(3, 0, 1, 4, 2, 5).bfloat16()  # cb80 t rb cb tok ch
    out = np.zeros((c // 80, nt, ct_bytes(m)), np.uint8)
    out[..., :m * 160] = v.reshape(c // 80, nt, -1).view(torch.uint8).numpy().reshape(c // 80, nt, m * 160)
    return out.reshape(-1)


def from_ctiles(b: np.ndarray, tokens: int, c: int, m: int) -> torch.Tensor:
    nt = tokens // m
    raw = np.asarray(b, np.uint8)[:c // 80 * nt * ct_bytes(m)].reshape(c // 80, nt, ct_bytes(m))[..., :m * 160]
    v = torch.from_numpy(np.ascontiguousarray(raw)).view(torch.bfloat16).float()
    return v.reshape(c // 80, nt, m // 8, 10, 8, 8).permute(1, 2, 4, 0, 3, 5).reshape(tokens, c)


def gemm_weights(w: torch.Tensor, row: int, npass: int) -> np.ndarray:
    """(N, K) -> row ``row``'s weight image: its channel slice [row npass 80, ..) as per pass, per k-tile [5 j][8 s][2
    h][8 n][9 B] (K padded to whole k-tiles)."""
    k = w.shape[1]
    nkt = -(-k // D.KT)
    ws = torch.zeros(npass * 80, nkt * D.KT)
    ws[:, :k] = w[row * npass * 80:(row + 1) * npass * 80].float()
    t = ws.reshape(npass, 5, 2, 8, nkt, 8, 8).permute(0, 4, 1, 5, 2, 3, 6)  # p kt j s h n k
    return blocks(np.ascontiguousarray(t.numpy())).reshape(-1)


def kv_slots(k: torch.Tensor, v: torch.Tensor, d: int = 160) -> np.ndarray:
    """An attention's K / V^T slots of 8 heads of d 160 (groups of 32 keys), 80 or 40 (groups of 64): k, v (2 images,
    keys, 8 d) -> [image][head][group][K [key block][d-block][2] | V^T [d-block pair][key panel][2]] (d 40: V^T's
    d-block 4 as [key panel] after the pairs', K and V^T one slot; keys padded to whole groups with zeros:
    ``ops.Attention`` masks them)."""
    n, nd, gk = k.shape[1], d // 8, 4 if d == 160 else 8
    ng = -(-n // (8 * gk))
    kp = torch.zeros(2, 8 * gk * ng, 8 * d)
    vp = torch.zeros(2, 8 * gk * ng, 8 * d)
    kp[:, :n], vp[:, :n] = k.float(), v.float()
    kk = kp.reshape(2, ng, gk // 2, 2, 8, 8, nd, 8).permute(0, 5, 1, 2, 6, 3, 4, 7)  # i g kb kh j h db d -> i h g kb db kh j d
    if d == 40:  # ([d-block pair 2][key panel][2] of d-blocks 0..3, then [key panel] of d-block 4)
        v4 = vp.reshape(2, ng, gk, 8, 8, nd, 8)  # i g kp j h db d
        va = v4[..., :4, :].reshape(2, ng, gk, 8, 8, 2, 2, 8).permute(0, 4, 1, 5, 2, 6, 7, 3)  # -> i h g dp kp dh d j
        vb = v4[..., 4, :].permute(0, 4, 1, 2, 5, 3)  # i g kp j h d -> i h g kp d j
        vv = torch.cat([va.reshape(2, 8, ng, -1), vb.reshape(2, 8, ng, -1)], -1)
    else:
        vv = vp.reshape(2, ng, gk, 8, 8, nd // 2, 2, 8).permute(0, 4, 1, 5, 2, 6, 7, 3)  # i g kp j h dp dh d -> i h g dp kp dh d j
    kb_ = blocks(np.ascontiguousarray(kk.numpy())).reshape(2, 8, ng, -1)
    vb_ = blocks(np.ascontiguousarray(vv.numpy())).reshape(2, 8, ng, -1)
    return np.concatenate([kb_, vb_], axis=-1).reshape(-1)
