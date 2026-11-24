"""Host model of the decode overlay's attention core in Inkling's mode (``artlite.decode.attn``, ``Attn(ink=True)``), one
KV head and its query heads, from the core's own inputs (the q and raw K / V objects the q, k, v phases wrote):

- K, V: the raw rows' convs over the head's history (positions before 0 zero): (x + sum_k t_k x_(P - 3 + k) (+ a full
  layer's per-position deltas: sum_lag d_lag x_(P - lag))) times 1 / sqrt(5) (full: 1 / 3), the taps bf16, the rows
  fp32; K L2-clipped to sqrt(512), cached as bfp16 hi + lo blocks (or one); V RMS-normed to bf16 rows, a record's V
  panels encoded from the rows of its positions up to the attending one (``attn_ref.AttentionRef``'s);
- the query rows: q times its clip scales (bf16) L2-clipped to sqrt(512), times tau on a full layer; r likewise (clip 8);
- scores q k / sqrt(head_dim) + r . E[:, distance] (E rounded to bf16, distances below the extent; a sliding-window
  layer's keys past its window masked), the head's sink an extra logit with no value, exact softmax; o times
  sigmoid(gate) (the gate bf16, the sigmoid rounded to bf16), rounded to bf16.

Not modelled (as ``AttentionRef``): the Q panels' 14 bits, the P panels, exp2's bf16-level polynomial, the products'
hi + lo splits (~2^-16).
"""

import numpy as np
import torch

from artlite.operators import bfp16

from ..numerics import bf16_round as bf
from .attn_ref import hilo_values

QK_CLIP = 512 ** 0.5  # (the core's thresholds: ``InklingConfig.qk_clip``, ``rel_clip``)
REL_CLIP = 8.0


def _clip(x: np.ndarray, thr: float) -> np.ndarray:
    """Each row (last axis) scaled to an L2 norm of at most thr (the core's: no epsilon)."""
    n = np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True))
    return (x * np.minimum(1.0, thr / np.maximum(n, 1e-30))).astype(np.float32)


class InkAttentionRef:
    """One KV head of an Inkling layer as its attention core computes it: ``store`` a position's raw K / V rows (and a
    full layer's taps' deltas), ``attend`` its query heads at the newest position. ``qc`` (heads, hd), ``rc`` (heads, rel):
    the clip scales; ``taps`` (8, hd): K's 4 taps then V's (tap 3 the current row); ``E`` (rel, extent) the relative-bias
    table; ``window``: a sliding-window layer's (0: full); ``kp``: K as hi + lo bfp16 blocks (2) or one (1)."""

    def __init__(self, *, qc: np.ndarray, rc: np.ndarray, taps: np.ndarray, E: np.ndarray, window: int, eps: float,
                 kp: int = 2):
        self.qc, self.rc = bf(qc), bf(rc)
        self.taps = bf(taps).reshape(2, 4, -1)
        self.E = bf(E)
        self.window, self.eps, self.kp = window, eps, kp
        self.hd = self.qc.shape[1]
        self.raw = np.zeros((0, 2, self.hd), np.float32)  # (the raw K | V rows)
        self.K = np.zeros((0, self.hd), np.float32)  # (as cached: bfp16 blocks' values)
        self.rows = np.zeros((0, self.hd), np.float32)  # (V's bf16 rows)

    def _conv(self, part: int, d: np.ndarray | None) -> np.ndarray:
        P = self.raw.shape[0] - 1
        def x(i: int) -> np.ndarray:
            return self.raw[i, part] if i >= 0 else np.zeros(self.hd, np.float32)

        y = x(P).astype(np.float64) + sum(self.taps[part, k].astype(np.float64) * x(P - 3 + k) for k in range(4))
        if d is not None:
            y = y + sum(float(d[lag]) * x(P - lag).astype(np.float64) for lag in range(4))
        return (y * (1 / 3 if d is not None else 1 / 5 ** 0.5)).astype(np.float32)

    def store(self, k: np.ndarray, v: np.ndarray, kd: np.ndarray | None = None, vd: np.ndarray | None = None) -> None:
        """The next position's raw K, V rows (hd,) fp32; kd, vd: a full layer's deltas (4,) of the position."""
        self.raw = np.concatenate([self.raw, np.stack([k, v])[None].astype(np.float32)])
        kc = _clip(self._conv(0, kd)[None], QK_CLIP)
        kn = hilo_values(torch.from_numpy(kc)).numpy() if self.kp == 2 else bfp16.values(kc)
        self.K = np.concatenate([self.K, kn])
        vc = self._conv(1, vd).astype(np.float64)
        vn = bf((vc / np.sqrt((vc ** 2).mean() + self.eps)).astype(np.float32))
        self.rows = np.concatenate([self.rows, vn[None]])

    def _values(self) -> np.ndarray:
        """V as cached for the newest position: the closed records' panels, the open one's encoded from its rows so far."""
        P = self.rows.shape[0] - 1
        out = np.zeros_like(self.rows)
        for r0 in range(0, P + 1, 8):
            rec = np.zeros((8, self.hd), np.float32)
            n = min(8, P + 1 - r0)
            rec[:n] = self.rows[r0:r0 + n]
            out[r0:r0 + n] = bfp16.values(np.ascontiguousarray(rec.T)).T[:n]
        return out

    def attend(self, q: np.ndarray, g: np.ndarray, r: np.ndarray, sink: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """The query heads' o (heads, hd), bf16 values, at the newest position: q, g (heads, hd), r (heads, rel) as the q
        object holds them (bf16), the sinks (heads,) fp32; tau: a full layer's factor of q and r at the position."""
        P = self.K.shape[0] - 1
        qn = _clip(bf(q) * self.qc, QK_CLIP) * tau
        rn = _clip(bf(r) * self.rc, REL_CLIP) * tau
        lo = max(0, P - self.window + 1) if self.window else 0
        dist = P - np.arange(lo, P + 1)
        ext = self.E.shape[1]
        bias = rn.astype(np.float64) @ self.E[:, np.minimum(dist, ext - 1)].astype(np.float64)
        bias = np.where(dist < ext, bias, 0.0)
        s = qn.astype(np.float64) @ self.K[lo:].T.astype(np.float64) / np.sqrt(self.hd) + bias
        m = np.maximum(s.max(-1), sink.astype(np.float64))
        p = np.exp(s - m[:, None])
        o = p @ self._values()[lo:].astype(np.float64) / (p.sum(-1) + np.exp(sink - m))[:, None]
        gate = bf(1 / (1 + np.exp(-bf(g).astype(np.float64))))
        return bf((o * gate).astype(np.float32))
