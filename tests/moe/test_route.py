"""The MoE routing's host side (``artlite.moe.route``, no NPU): the top k against the oracle, the deal of a
union over the columns, the control packets of a column's items."""

import numpy as np
import pytest

from artlite import txn
from artlite.moe import route as MR
from artlite.operators.design import COLS

from ..qwen35 import ref


def test_top_k() -> None:
    """The cores' top 8 (ties to the lower id) and weights against ``qwen35.ref.moe_route``, ties included."""
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((64, 256)).astype(np.float32) * 2
    lg[0, [3, 9, 200]] = lg[0].max() + 1  # (a three-way tie at the top)
    lg[1, :] = 0.5  # (all tied: the lowest ids)
    ids, w = MR.top_k(lg, 8)
    rids, rw = ref.moe_route(lg, 8)
    assert np.array_equal(ids, rids)
    assert np.abs(w - rw).max() < 2e-6
    assert ids[0, :3].tolist() == [3, 9, 200] and ids[1].tolist() == list(range(8))


@pytest.mark.parametrize("u", [8, 13, 17, 21, 31, 32])
def test_deal(u: int) -> None:
    """A union of u experts over the columns: every half once, whole experts first, at most MAX_ITEMS items and
    ceil(2 u / 8) halves per column (the busiest column of an ideal deal of halves)."""
    union = sorted(np.random.default_rng(u).choice(256, u, replace=False).tolist())
    items = MR.deal(union, COLS)
    halves = [(e, h) for col in items for e, h0, h1 in col for h in range(h0, h1 + 1)]
    assert sorted(halves) == sorted((e, h) for e in union for h in (0, 1))
    assert max(len(col) for col in items) <= MR.MAX_ITEMS
    assert max(sum(h1 - h0 + 1 for _, h0, h1 in col) for col in items) == -(-2 * u // COLS)
    for col in items:  # (whole experts, then halves)
        whole = [h1 - h0 == 1 for _, h0, h1 in col]
        assert whole == sorted(whole, reverse=True)


def test_packet_words() -> None:
    """A column's packets: whole writes of its items' BDs (chained, the halves' run as the address and length), the
    unused BDs zero, the push of the first, the lock; only BDs < 16, AxQoS 0."""
    base, stride = 0x8000_0001_2340_0000, 0x10_0000
    items = [(7, 0, 1), (200, 1, 1)]
    w = MR.packet_words(items, base=base, stride=stride).tolist()
    assert len(w) == MR.PACKET_WORDS
    bds = {}
    i = 0
    while i < len(w):  # (stream header, control header: (n - 1) << 20 | address, n data words)
        assert w[i] & 0x1F == MR.CID
        n, addr = ((w[i + 1] >> 20) & 3) + 1, w[i + 1] & 0xFFFFF
        data = w[i + 2:i + 2 + n]
        if txn.SHIM_BD0 <= addr < txn.SHIM_BD0 + txn.BD_BYTES * txn.SHIM_BDS:
            bd, k = divmod(addr - txn.SHIM_BD0, txn.BD_BYTES)
            bds.setdefault(bd, [0] * 8)[k // 4:k // 4 + n] = data
        else:
            assert (addr, data) in ((txn.QUEUE[txn.MM2S][0], [MR.EXPERT_BDS[0]]),
                                    (txn.SHIM_LOCK0 + 16 * MR.LOCK_DONE, [1]))
        i += 2 + n
    assert sorted(bds) == sorted(MR.EXPERT_BDS)
    for i, bd in enumerate(MR.EXPERT_BDS):
        b = bds[bd]
        if i >= len(items):
            assert not any(b)
            continue
        e, h0, h1 = items[i]
        a = base + (2 * e + h0) * stride
        assert b[:3] == [(h1 - h0 + 1) * stride // 4, a & 0xFFFFFFFF, (a >> 32) & 0xFFFF]
        assert b[5] == MR.BD_W5 and (b[5] >> 28) & 0xF == 0  # (AxQoS 0)
        nxt = MR.EXPERT_BDS[i + 1] if i + 1 < len(items) else None
        assert b[7] == MR.BD_VALID | (MR.BD_USE_NEXT | nxt << 27 if nxt is not None else 0)
