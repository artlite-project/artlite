"""The decode overlays' KV caches, host side (``decode/kv.py``): the record layout of a KV head's cache
groups, a chain's ranges of the groups and their A fills, the records a chain's end emits. No NPU."""

import numpy as np
import pytest

from artlite.decode import kv
from artlite.decode.vring import tails

CHAIN = 4  # (Gemma's: the columns of a KV head)


@pytest.mark.parametrize("hd", [256, 512])
def test_kv_groups_roundtrip(hd: int) -> None:
    """kv_groups / kv_from_groups: K (bfp16 hi + lo) to 14 bits below its block's maximum, V to 7."""
    gen = np.random.default_rng(hd)
    pg = kv.Records(hd).pg(9216)
    assert pg == 16 * (9216 // (144 * hd // 8))  # positions per group
    k = gen.standard_normal((3 * pg, hd)).astype(np.float32)
    v = gen.standard_normal((3 * pg, hd)).astype(np.float32)
    g = kv.kv_groups(k, v, 9216)
    assert g.shape == (3, 3, 9216)
    k2, v2 = kv.kv_from_groups(g, hd)
    kmax = np.abs(k.reshape(-1, 8)).max(-1).repeat(8).reshape(k.shape)  # per 8 dims of a position
    vmax = np.abs(v.reshape(-1, 8, hd)).max(-2, keepdims=True)  # per dim over the 8 positions of a record
    assert np.all(np.abs(k2 - k) <= kmax * 2.0 ** -12)
    assert np.all(np.abs(v2.reshape(-1, 8, hd) - v.reshape(-1, 8, hd)) <= vmax * 2.0 ** -6)


def test_kv_groups_layout() -> None:
    """Head_dim 256 in objects of 9280 B (qwen35's): groups of 32 positions; a K block where the cores read it."""
    rng = np.random.default_rng(0)
    k = rng.standard_normal((64, 256)).astype(np.float32)
    v = rng.standard_normal((64, 256)).astype(np.float32)
    g = kv.kv_groups(k, v, 9280)
    assert g.shape == (2, 3, 9280)
    k2, v2 = kv.kv_from_groups(g, 256)
    assert np.abs(k2 - k).max() < 2 ** -13 * np.abs(k).max() and np.abs(v2 - v).max() < 2 ** -6 * np.abs(v).max()
    # record 5 (group 1, its first K object's second record), position 3, d-block 7: the hi block at
    # [record][hi][db][position]
    rk = kv.Records(256).k_bytes
    blk = g[1, 0, rk + 72 * 7 + 9 * 3:rk + 72 * 7 + 9 * 4]
    vals = blk[1:].view(np.int8).astype(np.float32) * 2.0 ** (int(blk[0]) - 133)
    assert np.abs(vals - k[8 * 5 + 3, 56:64]).max() < 2 ** -6 * np.abs(k[8 * 5 + 3, 56:64]).max() + 1e-6


@pytest.mark.parametrize(("first", "g"), [(0, 0), (0, 1), (0, 2), (0, 5), (3, 19), (2, 4)])
def test_ranges(first: int, g: int) -> None:
    """The columns cover first .. g once, in order; the last column holds g; with fewer groups than columns the empty
    ones come first."""
    parts = kv.ranges(first=first, g=g, chain=CHAIN)
    assert len(parts) == CHAIN
    assert [x for g0, n in parts for x in range(g0, g0 + n)] == list(range(first, g + 1)), parts
    assert parts[-1][0] <= g < parts[-1][0] + parts[-1][1]
    ns = [n for _, n in parts]
    assert ns == sorted(ns) if g + 1 - first < CHAIN else min(ns) > 0


@pytest.mark.parametrize(("g0", "n"), [(0, 0), (0, 3), (3, 4), (2, 6)])
def test_fill(g0: int, n: int) -> None:
    """The A fill of a range reads its groups' objects (3 per group) in group order; an empty range one object."""
    a0, n0 = kv.fill(g0, n)
    assert list(range(a0, a0 + n0)) == ([3 * g + i for g in range(g0, g0 + n) for i in range(3)] if n else [0])


@pytest.mark.parametrize("hd", [256, 512])
@pytest.mark.parametrize(("p", "S"), [(0, 1), (7, 1), (7, 2), (30, 4), (15, 3), (9, 4)])
def test_emit_order(hd: int, p: int, S: int) -> None:
    """The K records of positions p and p + S - 1 in group order, then per open record its V panels and ring slot;
    a step of several slots in one record: each twice, a step of one slot: once."""
    out = kv.emit_order(p=p, S=S, hd=hd, a_bytes=9216)
    recs = sorted({p // 8, (p + S - 1) // 8})
    if len(recs) == 1 and S > 1:
        recs *= 2
    assert len(out) == 3 * len(recs) == 3 * tails(S)
    k = [r for kind, r in out if kind == "K"]
    assert sorted(k) == recs and out[:len(recs)] == [("K", r) for r in k]
    assert out[len(recs):] == [x for r in recs for x in (("V", r), ("R", r))]
    rg = kv.Records(hd).rg(9216)
    assert [r // rg for r in k] == sorted(r // rg for r in k)
