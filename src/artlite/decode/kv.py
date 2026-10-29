"""The KV caches of the decode overlays' attention cores, host side: the record format (``Records``), a KV head's cache
as groups of cache objects (``kv_groups``, ``kv_from_groups``), a chain's ranges of the groups and their A fills
(``ranges``, ``fill``), the records a chain's end emits in a step (``emit_order``).

A KV head's cache is linear in the positions: groups of kp + 1 objects of ``a_bytes`` (the A stream's objects: the
image's tile size), kp K objects of ``kper`` K records each and the V object of the group's kp kper records. A record
holds 8 positions: a K record is kp sets of ND panels (a panel: the 8 positions' bfp16 blocks of 8 dims; kp = 2: [hi
panels ND | lo panels ND], the lo blocks the rest of each hi block: K to ~14 bits; kp = 1: one block per 8 dims), a V
record ND panels (a panel: per dim of the 8, a block over the 8 positions); ND = head_dim / 8. The open records' exact
V values sit in a ring next to the caches (``decode.vring``: a slot holds a record's rows).
"""

import itertools
from dataclasses import dataclass

import numpy as np

from ..operators.bfp16 import block_bytes, block_values
from .vring import tails


@dataclass(frozen=True)
class Records:
    """The record format of a head_dim and K panels per record (kp): the bytes of a K record, a V record and its rows
    (a ring slot: [d-block][position 8][8] bf16); per group of cache objects of a_bytes (``objects``: kp K objects, the
    V object) its K records per K object (``kper``), records (``rg``) and positions (``pg``)."""

    hd: int
    kp: int = 2

    @property
    def nd(self) -> int:
        return self.hd // 8

    @property
    def k_bytes(self) -> int:
        return 72 * self.kp * self.nd

    @property
    def objects(self) -> int:
        return self.kp + 1

    @property
    def v_bytes(self) -> int:
        return 72 * self.nd

    @property
    def rows_bytes(self) -> int:
        return 16 * self.hd

    def kper(self, a_bytes: int) -> int:
        k = a_bytes // self.k_bytes
        return k - k % 2 if self.kp == 1 else k  # (a group holds whole pairs of records: P V adds two at a time)

    def rg(self, a_bytes: int) -> int:
        return self.kp * self.kper(a_bytes)

    def pg(self, a_bytes: int) -> int:
        return 8 * self.rg(a_bytes)


def kv_groups(k: np.ndarray, v: np.ndarray, a_bytes: int, kp: int = 2, *, dmajor: bool = False) -> np.ndarray:
    """K, V (P, hd) of a KV head (P whole groups) -> its cache (groups, kp + 1, a_bytes) uint8 (module docstring)."""
    P, hd = k.shape
    rec = Records(hd, kp)
    nd, kper, rg = rec.nd, rec.kper(a_bytes), rec.rg(a_bytes)
    n = P // (8 * rg)
    kr = np.asarray(k, np.float32).reshape(n, rg, 8, nd, 8)  # [g][record][position][db][i]
    parts = [block_bytes(kr)]
    if kp == 2:
        parts.append(block_bytes(kr - block_values(parts[0])))
    kb = np.stack(parts, 2).transpose(0, 1, 2, 4, 3, 5)  # [g][r][hi / lo][db][position][9]
    vr = np.asarray(v, np.float32).reshape(n, rg, 8, nd, 8)
    vp = block_bytes(vr.transpose(0, 1, 3, 4, 2))  # [g][r][db][i][9]: a block over the positions
    if dmajor:
        assert kp == 2 and rg % 2 == 0
        kb = kb.transpose(0, 1, 3, 2, 4, 5)  # [g][r][db][hi / lo][position][9]
        vp = vp.reshape(n, rg // 2, 2, nd, 8, 9).transpose(0, 1, 3, 2, 4, 5)  # [g][pair][db][record of 2][i][9]
    out = np.zeros((n, kp + 1, a_bytes), np.uint8)
    out[:, :kp, :kper * rec.k_bytes] = kb.reshape(n, kp, kper * rec.k_bytes)
    out[:, kp, :rg * rec.v_bytes] = vp.reshape(n, rg * rec.v_bytes)
    return out


def kv_from_groups(groups: np.ndarray, hd: int, *, dmajor: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``kv_groups`` (kp: the groups' objects - 1): K, V (P, hd) float32 as stored."""
    n, no, a_bytes = groups.shape
    rec = Records(hd, no - 1)
    nd, kper, rg, kp = rec.nd, rec.kper(a_bytes), rec.rg(a_bytes), rec.kp
    kb = groups[:, :kp, :kper * rec.k_bytes]
    kb = kb.reshape(n, rg, nd, kp, 8, 9).swapaxes(2, 3) if dmajor else kb.reshape(n, rg, kp, nd, 8, 9)
    k = sum(block_values(kb[:, :, i]) for i in range(kp)).transpose(0, 1, 3, 2, 4)  # [g][r][position][db][i]
    vp = groups[:, kp, :rg * rec.v_bytes]
    vp = vp.reshape(n, rg // 2, nd, 2, 8, 9).swapaxes(2, 3) if dmajor else vp.reshape(n, rg, nd, 8, 9)
    vp = vp.reshape(n, rg, nd, 8, 9)
    v = block_values(vp).transpose(0, 1, 4, 2, 3)
    return k.reshape(n * 8 * rg, hd), np.ascontiguousarray(v).reshape(n * 8 * rg, hd)


def ranges(*, first: int, g: int, chain: int) -> list[tuple[int, int]]:
    """The ranges of a KV head's cache groups of the ``chain`` columns that attend to it together, for a (sub-)step
    whose slots are in group g and attend to groups from ``first`` on: (first group, groups) per column, even shares (the
    last column's holds g), or with fewer groups than columns one each to the last ones (an empty column then merges
    into nothing: a partial merged into an empty one would lose its low bits). A column with nothing to attend gets
    (first, 0)."""
    n = g + 1 - first
    bounds = [first + (i * n) // chain if n >= chain else first + max(0, n - chain + i) for i in range(chain + 1)]
    return [(a, b - a) for a, b in itertools.pairwise(bounds)]


def fill(g0: int, n: int, kp: int = 2) -> tuple[int, int]:
    """The A fill of a column's range (groups g0 .., n of them): (first object, objects) in cache objects of kp + 1 per
    group; an empty range reads one object (of the cache's first group) that the core skips."""
    return ((kp + 1) * g0, (kp + 1) * n) if n else (0, 1)


def emit_order(*, p: int, S: int, hd: int, a_bytes: int, kp: int = 2) -> list[tuple[str, int]]:
    """The records a chain's end emits in a step (kind, record index) in its order: the K records ("K") of positions p
    and p + S - 1 as the K objects come, then per open record (the tail objects) its V panels ("V") and its ring slot's
    rows ("R"); one record of a step of several slots: each twice (the same transfers every step of S)."""
    rec = Records(hd, kp)
    kper, rg = rec.kper(a_bytes), rec.rg(a_bytes)
    rp, rl = p // 8, (p + S - 1) // 8
    recs = [rp, rl] if rl != rp else [rp] * tails(S)
    out = []
    for g in sorted({r // rg for r in recs}):  # groups in order; per group its K objects
        for ph in range(kp):
            out += [("K", r) for r in recs if r // rg == g and r % rg // kper == ph]
    return out + [x for r in recs for x in (("V", r), ("R", r))]


def records(*, p: int, S: int, hd: int, a_bytes: int, kp: int = 2) -> list[tuple[int, int, int]]:
    """The records a chain's end emits in a step (``emit_order``) as (0: from group 0 of the KV head's cache, 1: from
    its ring; byte offset; bytes)."""
    rec = Records(hd, kp)
    kper, rg, T = rec.kper(a_bytes), rec.rg(a_bytes), a_bytes
    out = []
    for kind, r in emit_order(p=p, S=S, hd=hd, a_bytes=a_bytes, kp=kp):
        if kind == "K":
            out.append((0, r // rg * rec.objects * T + r % rg // kper * T + r % kper * rec.k_bytes, rec.k_bytes))
        elif kind == "V":
            out.append((0, v_offset(8 * r, hd=hd, a_bytes=a_bytes, kp=kp), rec.v_bytes))
        else:
            out.append((1, (r & 1) * T, rec.rows_bytes))
    return out


def v_offset(p: int, *, hd: int, a_bytes: int, kp: int = 2) -> int:
    """Byte offset from group 0 of a KV head's cache of the V panels of record p >> 3 (not ``dmajor``)."""
    rec = Records(hd, kp)
    rg, r = rec.rg(a_bytes), p // 8
    return r // rg * rec.objects * a_bytes + kp * a_bytes + r % rg * rec.v_bytes
