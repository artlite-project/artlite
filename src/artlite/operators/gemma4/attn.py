"""Gemma 4's attention on the decode overlay's attention cores (``overlay.attn``): head_dim 256 (sliding-window layers:
a window of W positions, every dim pair (i, i + 128) rotated) or 512 (full layers: pairs (i, i + 256) of i < 64
rotated), q / k RMS-normed per head with one weight per layer each (folded into the score constant: q_norm k_norm
log2(e), attention scale 1), v RMS-normed without a weight, K as bfp16 hi + lo blocks (or one: ``kp``). The 4 columns
of KV head h (4 h .. 4 h + 3) attend together, one chain of 4 with all 4 query heads of the KV head (a drafter's: 2);
sub-steps of at most 4 slots at head_dim 256, 2 at 512. The token objects: t = SMAX kind + a per layer kind (0 SWA,
1 full).
"""

import numpy as np

from ..design import COLS
from ..overlay.attn import (
    F_ATT,
    M_AHD,
    M_AKIND,
    M_ANH,
    M_AOWN,
    M_ATOK,
    M_ATREC,
    M_AWIN,
    M_POS,
    M_QC_HI,
    M_QC_LO,
    M_TOK_A,
    M_TOK_G0,
    M_TOK_NA,
    SMAX,
    Attn,
    U,
    ring_rows,
    score_bits,
    substeps,
    tails,
)
from ..overlay.attn import rope_tables as _rope_tables

__all__ = ["F_ATT", "M_AHD", "M_AKIND", "M_ANH", "M_AOWN", "M_ATOK", "M_ATREC", "M_AWIN", "M_POS", "M_QC_HI", "M_QC_LO",
           "M_TOK_A", "M_TOK_G0", "M_TOK_NA", "SMAX", "U", "ring_rows", "substeps", "tails"]

QH = 4  # query heads per KV head (one chain's columns)
KV_HEADS = COLS // QH
TOKENS = 2 * SMAX  # token objects: t = SMAX kind + a, of the sub-step from slot a on


class GemmaAttn(Attn):
    """The attention cores' sizes for a Gemma model (head dims 256 and 512, 4 query heads per KV head on 8 columns), the
    cache objects of ``a_bytes`` (one head_dim-512 K record each), K panels per record ``kp`` (2: hi + lo)."""

    def __init__(self, cfg: object, *, a_bytes: int = 9216, c_len: int = 64, d_len: int = 128, kp: int = 2):
        hds = (cfg.head_dim_swa, cfg.head_dim_full)
        assert hds == (256, 512) and cfg.num_attention_heads == COLS and all(k == KV_HEADS for k in cfg.kv_heads), hds
        assert a_bytes % (144 * 64) == 0
        # rotated pairs: every pair of a SWA head; the full layers' with a frequency factor of 1 (the others: 1e30)
        f = np.asarray(cfg.rope_freq_factors if cfg.rope_freq_factors is not None else np.ones(256))
        rot = (128, int((f[:256] < 1e10).sum()))
        assert rot[1] % 32 == 0 and np.all(f[:rot[1]] < 1e10) and np.all(f[rot[1]:256] > 1e10)
        super().__init__(hds=hds, rot=rot, chain=QH, heads=QH, kp=kp, qk_norm=True, v_norm=True, eps=cfg.rms_norm_eps,
                         a_bytes=a_bytes, c_len=c_len)
        assert self.d_len == d_len, (self.d_len, d_len)
        self.cfg = cfg
        self.window = cfg.sliding_window


def rope_tables(cfg: object, positions: list[int], rot: tuple[int, int], *, swa: bool) -> np.ndarray:
    """(SMAX, 2 rot) fp32: per slot [cos | sin] of the rotated pairs' angles (unused slots zero)."""
    return _rope_tables(cfg.rope_inv_freq(swa=swa), positions, rot[0] if swa else rot[1])


def score_constant(q_norm: float, k_norm: float) -> tuple[int, int]:
    """The bf16 hi, lo bits of q_norm k_norm log2(e)."""
    return score_bits(q_norm * k_norm)
