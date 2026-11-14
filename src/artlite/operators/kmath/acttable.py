"""The activation tables' host side: ``ActTable``, its least-squares (Lawson minimax) fit against the activations' f64
references, its words for the gather, and the fit's disk cache. numpy design-time code; the kernels over it:
``gather``."""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..toolchain import CACHE_DIR


def _bf16(x: object) -> np.ndarray:
    """fp32 -> bf16 rounded to nearest even, as fp32 (finite values)."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def _f32(x: object) -> np.ndarray:
    return np.asarray(x, np.float32)


def bf16_values() -> np.ndarray:
    """Every finite bf16 value, as float32 (bit patterns 0x8000 .. 0xFFFF, then 0x0000 .. 0x7FFF)."""
    x = (np.arange(-32768, 32768).astype(np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return x[np.isfinite(x)]


def silu64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / (1.0 + np.exp(np.minimum(-x, 700.0)))


def sigmoid64(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.minimum(-np.asarray(x, np.float64), 700.0)))


def gelu64(x: np.ndarray) -> np.ndarray:
    """gelu in its tanh form (gelu_pytorch_tanh, ggml_gelu) as x sigmoid(z), z = 2 sqrt(2 / pi) (x + 0.044715 x^3)."""
    x = np.asarray(x, np.float64)
    z = 2.0 * np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return x / (1.0 + np.exp(np.minimum(-z, 700.0)))


def gelu_erf64(x: np.ndarray) -> np.ndarray:
    """gelu in its erf form (x Phi(x): MiniCPM-V's final merger)."""
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def ulp_bf16(t: np.ndarray) -> np.ndarray:
    """The bf16 ULP at t (normal range)."""
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(t), 2.0 ** -126))) - 7)


def _minimax(cols: list[np.ndarray], f: np.ndarray, w: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Coefficients c of sum c_k col_k minimizing max |w (sum c_k col_k + fixed - f)| (Lawson's iteration)."""
    A = np.stack(cols, 1) * w[:, None]
    b = (f - fixed) * w
    u = np.full(len(f), 1.0 / len(f))
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(300):
        s = np.sqrt(u)
        c = np.linalg.lstsq(A * s[:, None], b * s, rcond=None)[0]
        e = np.abs(A @ c - b)
        if e.max() <= 1e-9 * max(np.abs(b).max(), 1e-30):
            break
        u = u * e
        u /= u.sum()
    return c


def _round_bf16(c: float) -> float:
    return float(_bf16(np.float32([c]))[0])


@dataclass(frozen=True)
class ActTable:
    """An activation f with f(x) - f(-x) = x (x sigmoid of an odd function: SiLU, GELU) as relu(x) + g(t), g(t) =
    f(-t), t = min(|x|, t_max), g a table of segments of ``seg`` over [0, t_max] (t + m in one binade of ULP seg).
    ``full``: f itself over [-t_max, t_max] (x clamped; x + m in the binade [m - t_max, m + t_max]: sigmoid).
    ``floor``: the error that counts in units of the result's ULP is at least this absolute one (GELU's tail, below
    1e-7 at t = 5, decays faster than a quadratic per segment follows; relative accuracy there is irrelevant)."""

    name: str
    f64: Callable[[np.ndarray], np.ndarray]
    seg: float
    m: float
    t_max: float
    floor: float = 0.0
    full: bool = False

    @property
    def base(self) -> float:
        """The reduced argument's binade start (its first segment's)."""
        return self.m - self.t_max if self.full else self.m

    @property
    def segs(self) -> int:
        return int((2 if self.full else 1) * self.t_max / self.seg) + 1

    @property
    def groups(self) -> int:
        return -(-self.segs // 4)

    def reduce(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kernel's v, r, bf16(r^2) of t (bf16 values in [0, t_max])."""
        v = _bf16(_f32(self.m) + _f32(t))
        r = _bf16(_f32(_f32(self.m) - v) + _f32(t))
        return v, r, _bf16(_f32(r) * _f32(r))

    @functools.cached_property
    def coefficients(self) -> np.ndarray:
        """(segs, 3): c0 (fp32 values), c1, c2 (bf16 values) of g's segments (cached on disk: ~1 s to fit, keyed by the
        table's parameters and the fitting code)."""
        code = [inspect.getsource(f) for f in (self.f64, ActTable._fit, ActTable.reduce, _minimax, _round_bf16, _bf16, _f32,
                                               bf16_values, ulp_bf16)]
        key = hashlib.sha256(repr((self.name, self.seg, self.m, self.t_max, self.floor, self.full, code)).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, "kmath", f"{self.name}_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        coef = self._fit()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.npy"
        np.save(tmp, coef)
        os.replace(tmp, path)
        return coef

    def _fit(self) -> np.ndarray:
        x = bf16_values()
        if self.full:
            t = x[(np.abs(x) <= self.t_max) & ((x == 0) | (np.abs(x) >= 2.0 ** -40))]
            v, r, r2 = self.reduce(t)
            f = self.f64(t)
            w = 1.0 / np.maximum(ulp_bf16(f), self.floor)
        else:
            t = -x[(x <= 0) & (x >= -self.t_max) & ((x == 0) | (np.abs(x) >= 2.0 ** -40))]
            v, r, r2 = self.reduce(t)
            f = self.f64(-t)
            # (the error of y = relu(x) + g on both sides)
            w = 1.0 / np.maximum(np.minimum(ulp_bf16(f), ulp_bf16(self.f64(t))), self.floor)
        s = ((v - self.base) / self.seg).astype(np.int64)
        coef = np.zeros((self.segs, 3))
        for k in range(self.segs):
            sel = (s == k) & ((t != 0) | self.full)
            rk, r2k, fk, wk = r[sel].astype(np.float64), r2[sel].astype(np.float64), f[sel], w[sel]
            ones, zero = np.ones_like(rk), np.zeros_like(fk)
            if k == 0 and not self.full:  # c0 = 0: tiny t stay relative-accurate
                c2 = _round_bf16(_minimax([rk, r2k], fk, wk, zero)[1])
                c1 = _round_bf16(_minimax([rk], fk, wk, c2 * r2k)[0])
                coef[k] = 0.0, c1, c2
                continue
            c2 = _round_bf16(_minimax([ones, rk, r2k], fk, wk, zero)[2])
            c1 = _round_bf16(_minimax([ones, rk], fk, wk, c2 * r2k)[1])
            c0 = _minimax([ones], fk, wk, c1 * rk + c2 * r2k)[0]
            coef[k] = float(np.float32(c0)), c1, c2
        return coef

    def table_words(self, coefficients: np.ndarray | None = None) -> np.ndarray:
        """One table copy as uint32 words: entries [c2 | c1 (high half), c0], groups of 4 entries stored twice (the
        gather's odd lanes read 32 bytes further); of ``coefficients`` (default: the table's)."""
        coef = self.coefficients if coefficients is None else coefficients

        def bits(c: np.ndarray) -> np.ndarray:
            return _bf16(_f32(c)).view(np.uint32) >> 16

        words = np.zeros((self.groups * 4, 2), np.uint32)
        words[:len(coef), 0] = bits(coef[:, 1]) << 16 | bits(coef[:, 2])
        words[:len(coef), 1] = _f32(coef[:, 0]).view(np.uint32)
        g = words.reshape(self.groups, 8)
        return np.concatenate([g, g], 1).reshape(-1)


SILU = ActTable("silu", silu64, 0.25, 32.0, 24.0)
# GELU: 65 segments of 1/8 over [0, 8] (g(-8) ~ -4e-21): within 0.59 bf16 ULP from x = -4 on, 1.42 in [-6, -4) (absolute
# 1.4e-7), below: within 6e-13; the tanh form through vtanh: 1.7 ULP at ~7 cycles per element
GELU = ActTable("gelu", gelu64, 0.125, 16.0, 8.0)
GELU_ERF = ActTable("gelu_erf", gelu_erf64, 0.125, 16.0, 8.0)
SIGMOID = ActTable("sigmoid", sigmoid64, 0.25, 48.0, 16.0, full=True)
TABLES = {t.name: t for t in (SILU, GELU, GELU_ERF, SIGMOID)}
