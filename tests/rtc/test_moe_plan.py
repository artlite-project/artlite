"""The MoE prefill's host step (``artlite.operators.moe``: ``plan``, ``ctl``; C++ in ``_native``, no NPU): the routing
against the oracle, the plan's chunks, a control stream followed block by block as the shim runs it."""

import itertools

import numpy as np
import pytest

from artlite import txn
from artlite.operators.design import COLS
from artlite.operators.moe import ctl
from artlite.operators.moe import plan as MP
from artlite.operators.moe.expert_gemm import CHUNK

from . import qwen35_ref as ref


def test_route() -> None:
    """The top 8 (ties to the lower id) and weights against ``qwen35_ref.moe_route``, ties included, from rows in the
    router GEMM's layout (per column 32 experts at its pitch of 64) and from plain (T, E) rows; 1024 tokens (threads)."""
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((1024, 256)).astype(np.float32) * 2
    lg[0, [3, 9, 200]] = lg[0].max() + 1  # (a three-way tie at the top)
    lg[1, :] = 0.5  # (all tied: the lowest ids)
    lg[2, :8], lg[2, 8:] = -0.0, -1  # (-0 and +0 tie)
    lg[2, 5] = 0.0
    rids, rw = ref.moe_route(lg, 8)
    rows = np.full((1024, COLS, 64), np.nan, np.float32)
    rows[:, :, :32] = lg.reshape(1024, COLS, 32)
    for ids, w in (MP.route(lg, 8), MP.route(rows.reshape(1024, -1), 8, E=256, per_col=32, pitch=64)):
        assert np.array_equal(ids, rids)
        assert np.abs(w - rw).max() < 2e-7
    assert rids[0, :3].tolist() == [3, 9, 200] and rids[1].tolist() == rids[2].tolist() == list(range(8))


@pytest.mark.parametrize("T", [24, 2048])
def test_plan(T: int) -> None:
    """A routing's chunks: every (token, slot) pair once, at its ``pos`` row of its expert's chunk; an expert's pairs in
    token order, ceil(r / CHUNK) chunks; chunks dealt in turn (columns within one chunk of each other); padding rows
    the chunk's first token."""
    rng = np.random.default_rng(T)
    ids, _ = MP.route(rng.standard_normal((T, 256)).astype(np.float32) + np.linspace(0, 3, 256, dtype=np.float32), 8)
    cap = -(-(256 + -(-T * 8 // CHUNK)) // COLS)
    plan = MP.Plan.of(ids, cap=cap, E=256)
    counts = plan.counts
    assert max(counts) - min(counts) <= 1 and sum(counts) == sum(-(-r // CHUNK) for r in np.bincount(ids.ravel()))
    g, j = np.divmod(plan.pos, CHUNK)
    c, k = np.divmod(g, cap)
    assert len(set(plan.pos.ravel().tolist())) == ids.size
    for t in range(T):
        for i in range(8):
            assert plan.experts[c[t, i]][k[t, i]] == ids[t, i] and plan.rows[c[t, i]][k[t, i], j[t, i]] == t
    for col in range(COLS):  # (padding rows: the chunk's first token; valid rows in token order)
        for kk in range(counts[col]):
            r = plan.rows[col][kk]
            n = int(((c == col) & (k == kk)).sum())
            assert (r[n:] == r[0]).all() and (np.diff(r[:n]) > 0).all()


def _walk(words: np.ndarray, base: int, bds: tuple[int, int]) -> list[tuple[int, int, dict]]:
    """Follow a stream as the shim's TileControl runs it: block 0 (from word 0, its first stream header the fill's),
    then the block each block's push of a block BD queues. Per block (its start, words, register writes)."""
    blocks, regs = [], {}
    start, length, first = 0, None, True
    while start is not None:
        i, writes, nxt = start, {}, None
        end = start + length if length is not None else None
        while (end is None and i < words.size) or (end is not None and i < end):
            if first:
                first = False
            else:
                assert words[i] & 0x1F == ctl.CID
                i += 1
            n, addr = ((int(words[i]) >> 20) & 3) + 1, int(words[i]) & 0xFFFFF
            data = [int(x) for x in words[i + 1:i + 1 + n]]
            for q, x in enumerate(data):
                writes[addr + 4 * q] = x
                regs[addr + 4 * q] = x
            if addr == txn.QUEUE[txn.MM2S][1] and data[0] & 0xF in bds:  # (a block's push: the next block)
                reg = txn.SHIM_BD0 + txn.BD_BYTES * (data[0] & 0xF)
                a = regs[reg + 4] | (regs[reg + 8] & 0xFFFF) << 32
                nxt = ((a - base) // 4, regs[reg])
            i += 1 + n
            if end is None and nxt is not None:  # (block 0 ends with its push)
                end = i
        blocks.append((start, i - start, writes))
        start, length = nxt if nxt is not None else (None, None)
    return blocks


def test_assemble() -> None:
    """A stream of two templates (block 1 one, then the other) with address slots and lock bits: block 0 fills a whole
    object (b0_words) and queues block 1; every block's tail writes the next block's BD (alternating ``bds``, its
    length and address the next block's, its lock bits) and pushes it; the slots' words hold bases + offsets; the
    blocks cover the stream."""
    base, bds = 0x8012_3450_0000, (5, 14)

    def bd(i: int) -> int:
        return txn.SHIM_BD0 + txn.BD_BYTES * i

    t0 = ctl.template(ctl.ctl(bd(9), [64, 0, 0x4000_0000, 0]), [(3, 4, 0, 16)])  # (packets writing data BDs' words)
    t1 = ctl.template(ctl.ctl(bd(10), [32, 0, 0, 0]) + ctl.ctl(bd(11), [8, 0, 0x4018_0000, 0]),
                      [(3, 4, 1, -8), (9, 10, 0, 0)])
    blocks = np.array([0, 1, 1, 0, 1], np.int32)
    bases = np.array([[0xAB_0000_1000 * (i + 1), 0x12_3456_7000 + i] for i in range(blocks.size)], np.uint64)
    acq = (np.array([2, 3, 2, 3, 2]), np.array([0, 4, 4, 0, 4]))
    out = np.zeros(4096, np.uint32)
    n = ctl.assemble(out, [t0, t1], blocks, bases, base=base, b0_words=40, bds=bds, acquire=acq)
    walked = _walk(out[:n], base, bds)
    assert walked[0][:2] == (0, 40) and len(walked) == blocks.size + 1
    assert sum(w for _, w, _ in walked) == n and all(a + w == b for (a, w, _), (b, _, _) in itertools.pairwise(walked))
    for i, (start, _, writes) in enumerate(walked[1:]):
        body, slots = (t0, t1)[blocks[i]]
        got = out[start:start + body.size]
        for lo, hi, b, off in slots:
            a = int(bases[i, b]) + int(off)
            assert got[lo] == a & 0xFFFFFFFF and got[hi] == body[hi] | a >> 32
        if i + 1 < blocks.size:  # (the tail: the next block's BD, its lock)
            bd = bds[i % 2]
            w7 = writes[txn.SHIM_BD0 + txn.BD_BYTES * bd + 28]
            lk, v = int(acq[0][i + 1]), int(acq[1][i + 1])
            assert w7 & 0x1FFF == (1 << 12 | ((-v) & 0x7F) << 5 | lk if v else 0)
