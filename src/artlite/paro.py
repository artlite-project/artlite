"""ParoQuant (PARO) linears of a Hugging Face checkpoint (z-lab's ParoQuant, arXiv 2511.10645): weight-only asymmetric
int4 (or int5) with integer zeros and fp16 scales per group of ``group_size`` (128; 64) along K, after a learned
transform of the input per linear: ``x' = x * channel_scales`` then ``krot`` (8) rounds of 64 disjoint Givens rotations
in every group (a perfect matching of its channels, an angle per pair), so that ``y = x' W^T``, ``W = (q - z) s``.

Checkpoint tensors per linear (AWQ packing: logical column j of a word in nibble [0 4 1 5 2 6 3 7][j]): ``qweight`` [K,
N / 8] int32, ``qzeros`` [K / g, N / 8] int32 (not offset by one), ``scales`` [K / g, N] f16, ``theta`` [krot, K / 2]
f16, ``pairs`` [krot, K] i16 (group-local indices: pair t of group G at 2 t, 2 t + 1), ``channel_scales`` [1, K] f16;
int5 (``format: int5-bitplane``): ``qweight`` / ``qzeros`` the codes' low nibbles, ``qweight_hi`` [K, N / 32] /
``qzeros_hi`` [K / g, N / 32] int32 their fifth bits (bit n % 32 of word n // 32). The rounds take their angles as stored
(not negated), in order 0 .. krot - 1; round r maps x_i' = c x_i + s x_j, x_j' = -s x_i + c x_j.

``ParoTensor`` is such a linear in GGUF orientation (rows = N outputs, cols = K), read in place from the checkpoint,
optionally with its rows in another order and its groups permuted (Qwen3.5's V heads in tiled order: whole groups of
head_dim 128 move); ``Rotation`` is its transform: ``matrix`` the exact one (fp64), ``tan_form`` the rounds compiled for the
NPU (module ``Rotation.tan_form``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gguf_model import QTensor

_AWQ_ORDER = np.array([0, 4, 1, 5, 2, 6, 3, 7], np.uint32)  # nibble of logical column j


def awq_unpack(q: np.ndarray) -> np.ndarray:
    """AWQ int32 words (rows, cols / 8) -> codes (rows, cols) uint8."""
    w = np.ascontiguousarray(q).view(np.uint32)
    return ((w[..., None] >> (4 * _AWQ_ORDER)) & 0xF).astype(np.uint8).reshape(q.shape[0], -1)


def bitplane_unpack(q: np.ndarray) -> np.ndarray:
    """A fifth-bit plane (rows, cols / 32) int32 -> bits (rows, cols) uint8 (bit c % 32 of word c // 32)."""
    w = np.ascontiguousarray(q).view(np.uint32)
    return ((w[..., None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8).reshape(q.shape[0], -1)


@dataclass
class Rotation:
    """The input transform of a PARO linear (or of the linears that share it: an MoE layer's experts): ``theta``
    (krot, K / 2), ``pairs`` (krot, K) group-local, ``channel_scales`` (K,), groups of ``group`` channels."""

    theta: np.ndarray
    pairs: np.ndarray
    channel_scales: np.ndarray
    group: int

    @property
    def K(self) -> int:
        return self.channel_scales.size

    @property
    def rounds(self) -> int:
        return self.theta.shape[0]

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x' of rows x (T, K), in fp64 (the exact transform)."""
        T, G, g = x.shape[0], self.K // self.group, self.group
        y = np.asarray(x, np.float64).reshape(T, G, g) * np.asarray(self.channel_scales, np.float64).reshape(1, G, g)
        p = np.asarray(self.pairs, np.int64).reshape(self.rounds, G, g // 2, 2)
        th = np.asarray(self.theta, np.float64).reshape(self.rounds, G, g // 2)
        gi = np.arange(G)[:, None]
        for r in range(self.rounds):
            i, j = p[r, ..., 0], p[r, ..., 1]
            c, s = np.cos(th[r]), np.sin(th[r])
            xi, xj = y[:, gi, i], y[:, gi, j]
            y[:, gi, i], y[:, gi, j] = c * xi + s * xj, -s * xi + c * xj
        return y.reshape(T, self.K)

    def matrix(self) -> np.ndarray:
        """M (G, g, g) fp64: group G's x'_G = x_G M[G] (block-diagonal over the groups)."""
        G, g = self.K // self.group, self.group
        eye = np.zeros((g, G, g))
        eye[np.arange(g), :, np.arange(g)] = 1.0
        return self.apply(eye.reshape(g, self.K)).reshape(g, G, g).transpose(1, 0, 2)

    def tan_form(self, gamma: np.ndarray | None = None) -> TanForm:
        """The rounds as ``u <- u + T * u[partner]`` on storage positions (module docstring of ``TanForm``), from the
        stored parameters in fp64; ``gamma``: a per-channel scale before the transform (a norm's weight) folded into
        D_0."""
        G, g, R = self.K // self.group, self.group, self.rounds
        d = np.asarray(self.channel_scales, np.float64).reshape(G, g).copy()
        if gamma is not None:
            d *= np.asarray(gamma, np.float64).reshape(G, g)
        pos = np.tile(np.arange(g), (G, 1))  # pos[G, c]: the storage position of channel c
        p = np.asarray(self.pairs, np.int64).reshape(R, G, g // 2, 2)
        th = np.asarray(self.theta, np.float64).reshape(R, G, g // 2)
        gi = np.arange(G)[:, None]
        partner = np.empty((R, G, g), np.int64)
        coef = np.empty((R, G, g), np.float64)
        for r in range(R):
            i, j = p[r, ..., 0], p[r, ..., 1]
            c, s = np.cos(th[r]), np.sin(th[r])
            P, Q = pos[gi, i], pos[gi, j]
            di, dj = d[gi, i], d[gi, j]
            cf = np.abs(c) >= np.abs(s)  # (else the outputs swap storage: the tangent stays at most 1)
            t = np.where(cf, s / np.where(cf, c, 1), c / np.where(cf, 1, s))
            partner[r, gi, P], partner[r, gi, Q] = Q, P
            coef[r, gi, P] = np.where(cf, t * dj / di, -t * dj / di)
            coef[r, gi, Q] = np.where(cf, -t * di / dj, t * di / dj)
            d[gi, i], d[gi, j] = np.where(cf, c * di, s * dj), np.where(cf, c * dj, -s * di)
            pos[gi, i], pos[gi, j] = np.where(cf, P, Q), np.where(cf, Q, P)
        chan = np.empty_like(pos)  # chan[G, s]: the channel at storage position s
        chan[gi, pos] = np.arange(g)[None, :]
        return TanForm(partner.reshape(R, -1).astype(np.uint8), coef.reshape(R, -1),
                       np.take_along_axis(d, chan, 1).reshape(-1), chan.reshape(-1), g)


@dataclass
class TanForm:
    """A PARO transform as the NPU runs it: u = x (storage position s of group G holds channel s at first), then per
    round r ``u[s] += T[r, s] * u[partner[r, s]]`` with the old values (the pair (s, partner) updates together); at the
    end storage position s holds channel ``chan[s]`` of x' up to the factor ``D[s]``: x'[chan[s]] = D[s] u[s]. A
    linear whose weight takes its K in storage order (``chan``: a permutation within each group, lossless) needs no
    gather: y = (D u) W[:, chan]^T. partner, chan group-local (uint8 / int64), T and D in fp64 (each caller rounds)."""

    partner: np.ndarray  # (rounds, K) uint8
    T: np.ndarray  # (rounds, K) fp64
    D: np.ndarray  # (K,) fp64
    chan: np.ndarray  # (K,) group-local
    group: int

    def apply(self, x: np.ndarray) -> np.ndarray:
        """D u of rows x (T, K) in fp64: x' in storage order (``chan``)."""
        n, K, g = x.shape[0], x.shape[1], self.group
        u = np.asarray(x, np.float64).reshape(n, K // g, g).copy()
        gi = np.arange(K // g)[:, None]
        for r in range(self.T.shape[0]):
            part = self.partner[r].reshape(K // g, g).astype(np.int64)
            u = u + self.T[r].reshape(1, K // g, g) * u[:, gi, part]
        return (u.reshape(n, K) * self.D)

    def chan_index(self) -> np.ndarray:
        """The global K index of each storage position (K,)."""
        g = self.group
        return (np.arange(self.chan.size) // g * g + self.chan).astype(np.int64)


@dataclass
class ParoTensor(QTensor):
    """A PARO linear in GGUF orientation (``QTensor``: rows = N, cols = K; ``qtype`` "PARO4" / "PARO5"; ``raw``: its
    ``qweight`` as stored) with the checkpoint's arrays (``qweight``, ``qzeros``, ``scales``, int5's ``qweight_hi`` /
    ``qzeros_hi``), ``group`` its group size, its ``rotation`` (shared with the linears that share it), and the order of
    its rows (``rows_of``: output n is checkpoint column rows_of[n]) and groups (``groups_of``: group G is checkpoint
    group groups_of[G], its channels, rotation and codes moving with it); None: as stored."""

    qweight: np.ndarray = field(default=None)  # type: ignore[assignment]
    qzeros: np.ndarray = field(default=None)  # type: ignore[assignment]
    scales: np.ndarray = field(default=None)  # type: ignore[assignment]
    group: int = 128
    rotation: Rotation | None = None
    qweight_hi: np.ndarray | None = None
    qzeros_hi: np.ndarray | None = None
    rows_of: np.ndarray | None = None
    groups_of: np.ndarray | None = None

    @property
    def bits(self) -> int:
        return 5 if self.qweight_hi is not None else 4

    def _k_index(self) -> np.ndarray | None:
        """The checkpoint row of K index k (None: as stored)."""
        if self.groups_of is None:
            return None
        g = self.group
        return (np.asarray(self.groups_of, np.int64)[:, None] * g + np.arange(g)).reshape(-1)

    def fields(self) -> dict[str, np.ndarray]:
        """q (rows, cols) uint8 codes (0 .. 15, int5: 0 .. 31), z (rows, cols / g) uint8 integer zeros, s (rows, cols / g)
        fp32 (fp16 values); w = s (q - z)."""
        q = awq_unpack(self.qweight)  # (K, N)
        z = awq_unpack(self.qzeros)  # (G, N)
        if self.qweight_hi is not None:
            q |= bitplane_unpack(self.qweight_hi) << 4
            z |= bitplane_unpack(self.qzeros_hi) << 4
        s = np.asarray(self.scales, np.float32)
        if (k := self._k_index()) is not None:
            q, z, s = q[k], z[self.groups_of], s[self.groups_of]
        if self.rows_of is not None:
            q, z, s = q[:, self.rows_of], z[:, self.rows_of], s[:, self.rows_of]
        return {"q": np.ascontiguousarray(q.T), "z": np.ascontiguousarray(z.T), "s": np.ascontiguousarray(s.T)}

    def dequantize(self) -> np.ndarray:
        """W (rows, cols) fp32 = s (q - z): the quantized weight (its input transform not applied: ``effective``)."""
        f = self.fields()
        g = self.group
        w = (f["q"].reshape(self.rows, -1, g).astype(np.float32) - f["z"][..., None]) * f["s"][..., None]
        return w.reshape(self.rows, self.cols)

    def effective(self) -> np.ndarray:
        """W_eff (rows, cols) fp64 = W M^T: the plain linear the PARO one computes (y = x W_eff^T), M its rotation's
        exact matrix (the golden references' weights)."""
        w = self.dequantize().astype(np.float64)
        M = self.rot().matrix()  # (G, g, g)
        g = self.group
        return np.einsum("ngk,gjk->ngj", w.reshape(self.rows, -1, g), M).reshape(self.rows, self.cols)

    def rot(self) -> Rotation:
        """Its rotation with its groups in this tensor's order."""
        r = self.rotation
        if self.groups_of is None:
            return r
        g, G = self.group, self.groups_of
        R = r.rounds
        return Rotation(r.theta.reshape(R, -1, g // 2)[:, G].reshape(R, -1), r.pairs.reshape(R, -1, g)[:, G].reshape(R, -1),
                        r.channel_scales.reshape(-1, g)[G].reshape(-1), g)


def check_config(path: str, qc: dict) -> tuple[int, int]:
    """(bits, group size) of a checkpoint's ``quantization_config`` with ``quant_method`` "paroquant"; what Artlite does
    not run is refused with the reason."""
    bits, group, krot = int(qc.get("bits", 4)), int(qc.get("group_size", 128)), int(qc.get("krot", 8))
    fmt = qc.get("format")
    if bits == 4 and fmt not in (None, "int4", "awq"):
        raise NotImplementedError(f"{path}: PARO format {fmt!r} (int4 AWQ packing expected)")
    if bits == 5 and fmt != "int5-bitplane":
        raise NotImplementedError(f"{path}: PARO int5 format {fmt!r} (int5-bitplane expected)")
    if bits not in (4, 5):
        raise NotImplementedError(f"{path}: PARO of {bits} bits (4 and 5 run)")
    if group not in (64, 128):
        raise NotImplementedError(f"{path}: PARO groups of {group} (64 and 128 run)")
    if krot < 1:
        raise ValueError(f"{path}: krot {krot}")
    return bits, group
