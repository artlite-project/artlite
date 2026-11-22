"""A rotated overlay's host part: the input regions, metadata and transfers of the phases of a model whose linears take
rotated inputs (a PARO checkpoint's; ``decode.design``'s "Rotated inputs"), shared by the models that run them on the
decode overlay (``llama``, ``qwen35``). A model names each rotated phase's kind (``rot_kind``):

- "in": the copies of the first layer's residual (the embeddings'): a metadata object alone, its F_ROTE copies from the
  residual slices the columns loaded;
- "psq": an F_PSQ phase, its input [metadata object | partial sums object | x], one K group: a producer's F_ROTE copy
  (each producer column's partial sums object, then its slice of x);
- "rotx": an F_ROTX phase, no tiles: per column its slice of the phase's input (``rot_slice``: channels, and slots per
  k block: [gate | up] for F_RACT), rotated, out; in sub-phases of one X object each (``rot_subs``).

The model gives each phase's rotation objects (``rot_objects``: its data region's A objects per column), the copies an
F_ROTE phase emits (``rot_copies``) and routes the outputs (``x_to``: values into a phase's x, ``emit_copies``: an F_ROTE
emission)."""

import numpy as np
from ml_dtypes import bfloat16

from ..operators.design import COLS
from ..overlay import Sequence
from ..paro import Rotation, TanForm
from . import phases as mp
from .design import ROT_GROUP
from .image import Region
from .model import InputRegion
from .protocol import (
    C_LEN,
    F_MAIN,
    F_PSQ,
    F_ROTE,
    M_FLAGS,
    M_NRA,
    M_NRC,
    M_NS,
    M_NT,
    M_NX,
    M_RGN,
    META,
    SLOTS,
    XB,
    XPAY,
    region_bytes,
    rot_input_fill,
    rot_region_bytes,
)
from .protocol import rot_objects as pack_rot_objects
from .step import StepPlan


class RotatedOverlay:
    """The mixin of a ``model.ModelOverlay`` whose phases rotate their inputs (module docstring)."""

    def rot_kind(self, name: str) -> str | None:
        """A phase's kind: "in", "psq", "rotx" or None (another phase)."""
        raise NotImplementedError

    def rot_slice(self, name: str) -> tuple[int, int]:
        """(rotx) A column's slice of the phase's input: (channels, x's slots per k block for S = 4)."""
        raise NotImplementedError

    def rot_objects(self, L: int, name: str) -> int:
        """The rotation objects of a phase per column (0: none)."""
        raise NotImplementedError

    def rot_copies(self, L: int, name: str) -> int:
        """The copies an F_ROTE phase emits."""
        raise NotImplementedError

    def _add_rot_inputs(self, phase: tuple[int, str], extra: int = 0) -> None:
        """The input region of a rotated phase (module docstring; ``extra``: a "rotx" input's bytes per element past its
        x, ``model.ModelOverlay._add_inputs``')."""
        L, name = phase
        kind = self.rot_kind(name)
        if kind == "in":
            regs = [InputRegion(self._take("ACT", XB), 0, 0)]
        elif kind == "rotx":  # (a metadata object per sub-phase: ``rot_subs``, then x)
            n, mul = self.rot_slice(name)
            K, subs = n * COLS, len(self.rot_subs(name, SLOTS))
            off = self._take("ACT", region_bytes(K, mul * SLOTS) + (subs - 1) * XB + extra * K)
            regs = [InputRegion(off + i * XB, 0, K, x0=(subs - i) * XB) for i in range(subs)]
        elif name == "tap":  # (a block drafter's tap: a plain copy of the layer's input, an F_PSQ input's layout)
            regs = [InputRegion(self._take("ACT", rot_region_bytes(self.H)), 0, self.H, x0=XB + XPAY)]
        else:
            r = self.image.region(L, name)
            assert len(mp.group_chunks(r)) == 1, f"{name}: one K group"
            regs = [InputRegion(self._take("ACT", rot_region_bytes(r.K)), 0, r.K, x0=XB + XPAY)]
        self.inp[phase] = regs

    def rot_subs(self, name: str, S: int) -> list[tuple[int, int]]:
        """(rotx) The sub-phases of a rotation phase of S slots: (first group, groups) of the column's slice, an X object
        each. A rotation phase of several X objects deadlocks: the main core takes an X object's rotation objects before
        the next X object, the attention cores (on the same broadcast streams) skip a phase's X objects before its A
        objects, and each stream stalls on the consumer whose L1 objects are full."""
        n, mul = self.rot_slice(name)
        per = XPAY // (2 * mul * self._sp(S) * ROT_GROUP)  # (groups of an X object)
        G = n // ROT_GROUP
        return [(g, min(per, G - g)) for g in range(0, G, per)]

    def _rot_groups(self, L: int, name: str, S: int, flags: int, groups: list[mp.Group]) -> list[mp.Group]:
        """A rotated phase's groups: "in" and "rotx" phases' own; an F_PSQ / F_ROTE phase's from the plain ones
        (``groups``: the phase's as any other's)."""
        per = SLOTS // self._sp(S)  # (strips of an input object)
        nrot = self.rot_objects(L, name)
        kind = self.rot_kind(name)
        if kind == "in":  # (no tiles: the rotation objects only)
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS], m[M_NT], m[M_NRA], m[M_NRC] = F_MAIN | flags, S, nrot, nrot, self.rot_copies(L, name)
            return [mp.Group(0, 0, m, 0, 0)]
        if kind == "rotx":  # (per sub-phase its X object, counted as the other cores' strips: ``decode.attn``, and its
            # rotation objects)
            out = []
            for _, ng in self.rot_subs(name, S):
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_NS], m[M_NX], m[M_RGN] = F_MAIN | flags, S, per, ng
                m[M_NT] = m[M_NRA] = -(-ng // 2)
                out.append(mp.Group(0, 0, m, 0, 0))
            assert sum(int(g.meta[M_NT]) for g in out) == nrot, (name, nrot)
            # (the regions' metadata objects past S's sub-phases: unread, written as zeros: every S writes the same bytes)
            return out + [mp.Group(0, 0, np.zeros(META, np.int32), 0, 0) for _ in self.inp[(L, name)][len(out):]]
        for i, g in enumerate(groups):
            if flags & F_PSQ:  # (the partial sums object counted in: the other cores skip it)
                g.meta[M_NX] += per
            if i + 1 < len(groups):
                g.meta[M_FLAGS] &= ~F_ROTE
            elif flags & F_ROTE:
                g.meta[M_NRA], g.meta[M_NRC] = nrot, self.rot_copies(L, name)
                g.meta[M_NT] += nrot
        return groups

    def _rot_inputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> bool:
        """A rotated phase's input fills (its weights out before); True: its outputs drained too (a "rotx" phase's go
        first: past 4 queued fills a column's next fill waits for its oldest, whose room the cores make only while
        their outputs drain)."""
        name = phase[1]
        kind, sp = self.rot_kind(name), self._sp(st.S)
        for rg in self.inp[phase]:
            if kind == "in":  # (the metadata object: the copies come from the residual)
                for c in range(COLS):
                    seq.fill("X", c, self.dev.address("ACT", rg.off), XB)
            elif kind == "rotx":  # (per sub-phase its metadata object, then its X object: the XB bytes ending with the
                # slice's part, ``rot_subs``)
                self._outputs(seq, st, phase)
                n, mul = self.rot_slice(name)
                x = rg.off + rg.x0
                for i, (g0, _) in enumerate(self.rot_subs(name, st.S)):
                    for c in range(COLS):
                        seq.fill("X", c, self.dev.address("ACT", self.inp[phase][i].off), XB)
                        at = x + 2 * mul * sp * (n * c + ROT_GROUP * g0) - (XB - XPAY)
                        seq.fill("X", c, self.dev.address("ACT", at), XB)
                return True
            elif rg.x0 != XB:  # (F_PSQ: the partial sums object after the metadata object)
                for c in range(COLS):
                    seq.fill("X", c, self.dev.address("ACT", rg.off), *rot_input_fill(rg.K, sp))
            else:
                self._inputs(seq, rg, st.S, phase)
        return False

    def x_to(self, seq: Sequence, c: int, phase: tuple[int, str], *, e0: int, n: int, S: int, mul: int = 1,
             so: int = 0) -> None:
        """n output values per slot of column c (objects (j, slot) of C_LEN values) -> the elements e0 .. e0 + n of a
        phase's x ([k block][slot mul sp][8]: an F_RACT phase's [gate | up]; ``so``: from slot so on), per K group."""
        sp = self._sp(S)
        regions = self.inp[phase]
        for rg in regions[:1] if self.rot_kind(phase[1]) == "rotx" else regions:  # (sub-phases: one x)
            a, b = max(e0, rg.k0), min(e0 + n, rg.k0 + rg.K)
            j = a
            while j < b:  # (repeated BDs: at most 64 iterations)
                m = min(b - j, 64 * C_LEN)
                assert (j - rg.k0) % C_LEN == 0 and m % C_LEN == 0, (j, rg.k0, m)
                nobj, spx = m // C_LEN, mul * sp
                seq.drain("C", c, self.dev.address("ACT", rg.off + rg.x0 + 2 * spx * (j - rg.k0) + 16 * so),
                          S * 2 * C_LEN, (4, 8, 4 * spx, 4), nobj, (4 * spx * 8, nobj))
                j += m

    def sink_out(self, seq: Sequence, c: int, n: int, S: int) -> None:
        """n output values per slot of column c (whole C objects) -> the sink (``rot_sink``: ACT bytes the model takes
        for outputs no phase reads, a rotation's padding groups past its consumer's K)."""
        if n:
            assert n % C_LEN == 0, n
            seq.drain("C", c, self.dev.address("ACT", self.rot_sink), S * 2 * n)

    def emit_copies(self, seq: Sequence, c: int, S: int, to: list[tuple[int, str]]) -> None:
        """An F_ROTE emission of column c: per copy its partial sums object, then its x (the consumer's)."""
        h8 = self.H // COLS
        for ph in to:
            seq.drain("C", c, self.dev.address("ACT", self.inp[ph][0].off + XB + 2 * C_LEN * c), 2 * C_LEN)
            self.x_to(seq, c, ph, e0=h8 * c, n=h8, S=S)


def column_groups(k0: int, n: int, g: int = ROT_GROUP) -> list[np.ndarray]:
    """Per column the rotation groups of a slice of n channels a column from k0 on."""
    return [np.arange((k0 + n * c) // g, (k0 + n * (c + 1)) // g) for c in range(COLS)]


def copy_inverse(rot: Rotation, gamma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of a rotated copy x' = R (w' x) (``rot``'s unit tan form, gamma folded in; the rounds R and w' as the
    NPU runs them, T and w' in bf16) against gamma x, for ``folded``: (gamma / w' (K,), 0 where w' is; R^-1 per group
    transposed (G, g, g)) in fp64."""
    tf = rot.tan_form(gamma, unit=True)
    g, K = rot.group, rot.K
    T = np.asarray(tf.T, np.float32).astype(bfloat16).astype(np.float64)
    ws = np.asarray(tf.w, np.float32).astype(bfloat16).astype(np.float64)
    gm = np.ones(K) if gamma is None else np.asarray(gamma, np.float64).reshape(-1)
    scale = np.where(ws != 0, gm / np.where(ws != 0, ws, 1), 0)
    # (rows x' = x a per group, a's row i the image of e_i: the unit vectors of every group at once, the rounds being
    # block-diagonal)
    a = TanForm(tf.partner, T, np.ones(K), tf.chan, g).apply(np.tile(np.eye(g), (1, K // g)))
    blocks = a.reshape(g, K // g, g).transpose(1, 0, 2)
    return scale, np.linalg.inv(blocks).transpose(0, 2, 1)


def folded(w: np.ndarray, inverse: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """A matrix w (N, K) that reads a rotated copy x' of its input instead of gamma x (``copy_inverse``: the copy's): w
    diag(gamma / w') R^-1 (fp64), its K in the copy's storage order. Exact against the NPU's copy (to its fp32
    arithmetic), not only against the exact transform: logits that pick experts (a router's) agree with the file's fp32
    ones."""
    scale, inv_t = inverse
    G, g, _ = inv_t.shape
    m = (np.asarray(w, np.float64) * scale).reshape(-1, G, g)
    return np.einsum("ngk,gkj->ngj", m, inv_t).reshape(m.shape[0], G * g)


class RotatedImage:
    """The mixin of a ``image.TileImage`` whose linears take rotated inputs: data regions of rotation objects on the A
    runs (``rot_data``: per column the objects of each part, two groups per A object, ``protocol.rot_objects``)."""

    def _rot_init(self) -> None:
        self._rot_objs: dict[tuple[int, str], np.ndarray] = {}

    def rot_data(self, L: int, name: str, parts: list[tuple[str, np.ndarray | None, list[np.ndarray]]]) -> None:
        """A data region of rotation objects: per column the objects of each part (a tensor, a norm weight folded in or
        None, per column its groups: -1, a padding group of zeros) in order."""
        per = []
        for c in range(COLS):
            objs = []
            for t, gamma, groups in parts:
                o = self._rot_tan(t, gamma, 1 + max(int(g.max()) for g in groups if g.size))
                objs += [o[i] if i >= 0 else np.zeros_like(o[0]) for i in groups[c]]
            per.append(pack_rot_objects(np.stack(objs), self.tile_bytes))
        self._rot_objs[(L, name)] = np.stack(per)
        self._add_data(L, name, per[0].size)

    def _rot_tan(self, tensor: str, gamma: np.ndarray | None, groups: int = 0) -> np.ndarray:
        """(G, bytes) the groups' rotation objects of a tensor's tan form (``paro.TanForm.objects``; tensor "": the
        identity of gamma's channels, or of ``groups`` groups: a plain linear's copy of a normed input, or of its input),
        cached."""
        cache = self.__dict__.setdefault("_rot_tans", {})
        key = (tensor, None if gamma is None else gamma.tobytes(), 0 if tensor or gamma is not None else groups)
        if key not in cache:
            rot = self.gguf.tensor(tensor).rot() if tensor else Rotation.identity(
                gamma.size if gamma is not None else groups * ROT_GROUP)
            cache[key] = rot.tan_form(gamma, unit=True).objects()
        return cache[key]

    def rot_slab(self, L: int, name: str, tensor: str, halves: int) -> None:
        """A data slab region (``image.TileImage.slabs``) of rotation objects: per expert half (``halves`` of them, half
        h of an expert: its K's groups 2 h, 2 h + 1, one A object) the groups of the tensor's rotation (no norm weight)."""
        o = self._rot_tan(tensor, None, 4)
        assert o.shape[0] == 4, f"{tensor}: an expert's K of 4 groups"
        self._rot_objs[(L, name)] = np.stack([pack_rot_objects(o[2 * h:2 * h + 2], self.tile_bytes) for h in (0, 1)])
        self.slabs.append(Region(L, name, [], 1, 1, self.tile_bytes, self.kc, data=self.tile_bytes, cols=halves))

    def rot_region(self, layer: int, name: str) -> bool:
        """Whether (layer, name) is a data region of rotation objects."""
        return (layer, name) in self._rot_objs

    def rot_groups(self, layer: int, name: str) -> int:
        """Rotation objects (A objects) of a rotation data region per column."""
        return self._rot_objs[(layer, name)].shape[1]
