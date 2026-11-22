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

from dataclasses import dataclass, field, replace

import numpy as np

from . import _native
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

    @classmethod
    def identity(cls, K: int, group: int = 128, rounds: int = 8) -> Rotation:
        """The transform x' = x of K channels (zero angles; a copy's rounds that only scale: ``tan_form``'s gamma)."""
        return cls(np.zeros((rounds, K // 2)), np.tile(np.arange(K) % group, (rounds, 1)), np.ones(K), group)

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

    def tan_form(self, gamma: np.ndarray | None = None, *, unit: bool = False) -> TanForm:
        """The rounds as ``u <- u + T * u[partner]`` on storage positions (module docstring of ``TanForm``), from the
        stored parameters in fp64; ``gamma``: a per-channel scale before the transform (a norm's weight) folded into
        D_0. ``unit``: D_0 chosen so that the final scales D are 1 and the input scaled instead (``TanForm.w``: x' =
        rounds(x w), no multiply after the rounds; the scales' paths through the rounds do not depend on D_0); gamma's
        zero entries (a dead channel of a fine-tuned norm, Qwen3.8-27B-PARO-int5's layer 7) then a zero in w, the rounds
        those of a weight of 1."""
        G, g = self.K // self.group, self.group
        d0 = np.asarray(self.channel_scales, np.float64).reshape(G, g).copy()
        dead = None
        if gamma is not None:
            gm = np.asarray(gamma, np.float64).reshape(G, g)
            dead = gm == 0  # (a norm's zero weights: those channels' inputs are zero)
            assert unit or not dead.any(), "a zero weight in a scaled tan form"
            d0 *= np.where(dead, 1.0, gm)
        if not unit:
            return self._rounds(d0)[0]
        tf, origin = self._rounds(d0)
        d1 = np.empty_like(d0)
        np.put_along_axis(d1, origin, np.take_along_axis(d0, origin, 1) / tf.D.reshape(G, g), 1)
        out = self._rounds(d1)[0]
        assert np.allclose(out.D, 1, rtol=1e-9, atol=0), np.abs(out.D - 1).max()
        out.D = np.ones_like(out.D)
        # (the input's scales; a dead channel's zero: the rounds as for a weight of 1, its input zeroed first)
        out.w = np.where(dead, 0.0, d0 / d1).reshape(-1) if dead is not None else (d0 / d1).reshape(-1)
        return out

    def _rounds(self, d0: np.ndarray, *, native: bool = True) -> tuple[TanForm, np.ndarray]:
        """The tan form from the initial scales d0 (G, g): and per final storage position the channel whose initial scale
        flows into its final one (G, g). ``native``: the rounds in C++ (``_native.paro_rounds``: the numpy expressions
        below term by term, the angles' cosines and sines numpy's: the same bits; 9 ms -> 0.1 ms a linear of K 5120)."""
        G, g, R = self.K // self.group, self.group, self.rounds
        d = np.ascontiguousarray(d0, np.float64).copy()
        th = np.asarray(self.theta, np.float64).reshape(R, G, g // 2)
        if native:
            partner8 = np.empty((R, G, g), np.uint8)
            coef = np.empty((R, G, g), np.float64)
            pos, origin = np.empty((G, g), np.int64), np.empty((G, g), np.int64)
            _native.paro_rounds(np.ascontiguousarray(np.asarray(self.pairs, np.int64).reshape(R, G, g)),
                                np.ascontiguousarray(np.cos(th)), np.ascontiguousarray(np.sin(th)), d, partner8, coef,
                                pos, origin)
            return self._form(partner8, coef, d, pos, origin)
        origin = np.tile(np.arange(g), (G, 1))  # (per channel: whose initial scale its scale carries)
        pos = np.tile(np.arange(g), (G, 1))  # pos[G, c]: the storage position of channel c
        p = np.asarray(self.pairs, np.int64).reshape(R, G, g // 2, 2)
        gi = np.arange(G)[:, None]
        partner = np.empty((R, G, g), np.int64)
        coef = np.empty((R, G, g), np.float64)
        for r in range(R):
            i, j = p[r, ..., 0], p[r, ..., 1]
            c, s = np.cos(th[r]), np.sin(th[r])
            P, Q = pos[gi, i], pos[gi, j]
            di, dj = d[gi, i], d[gi, j]
            oi, oj = origin[gi, i], origin[gi, j]
            cf = np.abs(c) >= np.abs(s)  # (else the outputs swap storage: the tangent stays at most 1)
            t = np.where(cf, s / np.where(cf, c, 1), c / np.where(cf, 1, s))
            partner[r, gi, P], partner[r, gi, Q] = Q, P
            coef[r, gi, P] = np.where(cf, t * dj / di, -t * dj / di)
            coef[r, gi, Q] = np.where(cf, -t * di / dj, t * di / dj)
            d[gi, i], d[gi, j] = np.where(cf, c * di, s * dj), np.where(cf, c * dj, -s * di)
            origin[gi, i], origin[gi, j] = np.where(cf, oi, oj), np.where(cf, oj, oi)
            pos[gi, i], pos[gi, j] = np.where(cf, P, Q), np.where(cf, Q, P)
        return self._form(partner.astype(np.uint8), coef, d, pos, origin)

    def _form(self, partner: np.ndarray, coef: np.ndarray, d: np.ndarray, pos: np.ndarray, origin: np.ndarray
              ) -> tuple[TanForm, np.ndarray]:
        """``_rounds``' result from the rounds' (partner, coef per storage position; d, pos, origin per channel)."""
        G, g, R = self.K // self.group, self.group, self.rounds
        gi = np.arange(G)[:, None]
        chan = np.empty_like(pos)  # chan[G, s]: the channel at storage position s
        chan[gi, pos] = np.arange(g)[None, :]
        tf = TanForm(partner.reshape(R, -1), coef.reshape(R, -1), np.take_along_axis(d, chan, 1).reshape(-1),
                     chan.reshape(-1), g)
        return tf, np.take_along_axis(origin, chan, 1)


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
    w: np.ndarray | None = None  # (K,) fp64: (``Rotation.tan_form(unit=True)``) the input's scales: u = x w at first

    def objects(self) -> np.ndarray:
        """The NPU's rotation object of each group (G, 3328) uint8: the rounds' coefficients (``kmath.rotate.coefs``) and
        w (bf16; a unit form's: D is 1)."""
        from ml_dtypes import bfloat16

        from .operators.kmath.rotate import coefs

        assert self.w is not None, "a unit tan form (Rotation.tan_form(unit=True))"
        R, g = self.T.shape[0], self.group
        G = self.T.shape[1] // g
        T, part = self.T.reshape(R, G, g), self.partner.reshape(R, G, g)
        out = np.zeros((G, 3328), np.uint8)
        for k in range(G):
            out[k, :3072] = coefs(T[:, k], part[:, k])
        out[:, 3072:] = np.asarray(self.w, np.float32).astype(bfloat16).view(np.uint8).reshape(G, 2 * g)
        return out

    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Each round's Givens pairs per group (G, rounds, 64, 2) (storage positions s < partner[s]) and their
        coefficients (G, rounds, 64, 2) fp64: (T[s], T[partner[s]])."""
        R, g = self.T.shape[0], self.group
        G = self.T.shape[1] // g
        part = self.partner.reshape(R, G, g).astype(np.int64).transpose(1, 0, 2)
        T = self.T.reshape(R, G, g).transpose(1, 0, 2)
        s = np.argsort(part > np.arange(g), axis=-1, kind="stable")[..., g // 2:]  # (positions with partner[s] > s)
        q = np.take_along_axis(part, s, -1)
        return np.stack([s, q], -1), np.stack([np.take_along_axis(T, s, -1), np.take_along_axis(T, q, -1)], -1)

    def pair_objects(self, row: int = 32) -> np.ndarray:
        """The prefill's rotation object of each group (G, 512 rounds + 512) uint8 (``prefill.rotate``, its u in rows of
        ``row`` floats): u lies in each round's pair order (pair i: rows 2 i, 2 i + 1, s < partner[s]); per round 64 x
        [the destinations of the pair's two outputs: their rows in the next round's order (the last round: the positions
        themselves), times ``row``, as u16 | u16] then 64 x [T_s | T_p] bf16; the first round's row of each channel (u16,
        times ``row``) and w (bf16; a unit form's)."""
        from ml_dtypes import bfloat16

        assert self.w is not None, "a unit tan form (Rotation.tan_form(unit=True))"
        pairs, coef = self.pairs()
        G, R = pairs.shape[:2]
        g = self.group
        order = pairs.reshape(G, R, g)  # (round r: the position at each row)
        at = np.empty_like(order)
        np.put_along_axis(at, order, np.broadcast_to(np.arange(g), order.shape).copy(), -1)  # (position -> row)
        dest = np.concatenate([np.take_along_axis(at[:, 1:], order[:, :-1], -1), order[:, -1:]], 1)  # (G, R, g)
        out = np.zeros((G, 512 * R + 4 * g), np.uint8)
        rounds = out[:, :512 * R].reshape(G, R, 512)
        rounds[..., :256] = (row * dest).astype(np.uint16).view(np.uint8).reshape(G, R, 256)
        rounds[..., 256:] = np.asarray(coef, np.float32).astype(bfloat16).reshape(G, R, g).view(np.uint8).reshape(G, R, 256)
        out[:, 512 * R:512 * R + 2 * g] = (row * at[:, 0]).astype(np.uint16).view(np.uint8).reshape(G, 2 * g)
        out[:, 512 * R + 2 * g:] = np.asarray(self.w, np.float32).astype(bfloat16).view(np.uint8).reshape(G, 2 * g)
        return out

    def apply(self, x: np.ndarray) -> np.ndarray:
        """D u of rows x (T, K) in fp64 (times w first: a unit form's): x' in storage order (``chan``)."""
        n, K, g = x.shape[0], x.shape[1], self.group
        u = np.asarray(x, np.float64).reshape(n, K // g, g) * (1 if self.w is None else self.w.reshape(1, K // g, g))
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
    """A PARO linear in GGUF orientation (``QTensor``: rows = N, cols = K; ``qtype`` its tile format, "P4_128",
    "P4_64", "P5_128": bits and group size; ``raw``: its
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
    chan: np.ndarray | None = None  # (K,) group-local: position p of group G holds channel chan[G g + p] (``TanForm``)

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
        if self.chan is not None:
            q = q[np.arange(self.cols) // self.group * self.group + np.asarray(self.chan, np.int64)]
        return {"q": np.ascontiguousarray(q.T), "z": np.ascontiguousarray(z.T), "s": np.ascontiguousarray(s.T)}

    def blocks(self, *, threads: int = 0) -> QTensor:
        """The linear as rows of P4_128 blocks (int5: P5_128; ``_native.awq_rows``: a QTensor the tile packer takes,
        (rows, K / 256 * 144, or 176) uint8 in memory: its rows, groups and channel order as this tensor's)."""
        out = np.empty((self.rows, self.cols // 256 * (176 if self.bits == 5 else 144)), np.uint8)
        self._rows_into(out, threads)
        return QTensor(name=self.name, qtype=self.qtype, rows=self.rows, cols=self.cols, raw=out)

    def _rows_into(self, out: np.ndarray, threads: int) -> None:
        """``blocks``' rows into out (rows, K / 256 * 144 or 176)."""
        from . import _native

        assert self.group == 128, self.group
        five = self.bits == 5
        _native.awq_rows(out, np.ascontiguousarray(self.qweight), np.ascontiguousarray(self.qzeros),
                         np.ascontiguousarray(self.scales).view(np.uint16), self.group,
                         None if self.chan is None else np.ascontiguousarray(self.chan, np.int32),
                         None if self.rows_of is None else np.ascontiguousarray(self.rows_of, np.int64),
                         None if self.groups_of is None else np.ascontiguousarray(self.groups_of, np.int64),
                         np.ascontiguousarray(self.qweight_hi) if five else None,
                         np.ascontiguousarray(self.qzeros_hi) if five else None, threads)

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


@dataclass
class ParoExperts(ParoTensor):
    """E PARO linears of one shape that share one rotation, stacked (an MoE layer's experts, ``ffn_*_exps``: rows E N,
    expert e's outputs at rows e N ..): ``experts`` their ``ParoTensor``s (the stack's ``qweight`` .. are the first
    one's); the stack's ``chan`` is every expert's."""

    experts: list[ParoTensor] = field(default_factory=list)

    def _each(self) -> list[ParoTensor]:
        return [replace(t, chan=self.chan) for t in self.experts]

    def fields(self) -> dict[str, np.ndarray]:
        fs = [t.fields() for t in self._each()]
        return {k: np.concatenate([f[k] for f in fs]) for k in fs[0]}

    def _rows_into(self, out: np.ndarray, threads: int) -> None:
        n = self.rows // len(self.experts)
        for e, t in enumerate(self._each()):
            t._rows_into(out[e * n:(e + 1) * n], threads)

    def effective(self) -> np.ndarray:
        """(float32, expert by expert: a stack's fp64 matrix would be 2 GB)"""
        return np.concatenate([t.effective().astype(np.float32) for t in self._each()])


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
