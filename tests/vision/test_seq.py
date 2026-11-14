"""The sequencer's invariants on the host (``seq.Seq`` over synthetic ops, no device): the queues stay within their depth
however many tasks an op puts on a channel, a memtile BD is rewritten only after its last user's completion, a role's
page enters a row's B stream once, and a row's B parity returns to zero at the end."""

import pytest

from artlite import txn
from artlite.operators.design import COLS, ROWS
from artlite.txn import HardwareLimitError
from artlite.vision import design as D
from artlite.vision.roles import Role
from artlite.vision.seq import ColOps, MemTask, RowOps, Seq

NOP, PAGE, DDR = 0x1_0000_0000, 0x1_0001_0000, 0x1_0010_0000


def task(addr: int, repeat: int = 1) -> MemTask:
    return MemTask([{"addr": addr, "nwords": 256}], repeat=repeat)


def op_cols(ntask: int, *, seed: int = 0, nfill: int = 1) -> dict[int, ColOps]:
    """Per column ``ntask`` sends and receives on every core row, each of other words (a BD each), and ``nfill``
    fills."""
    return {c: ColOps(sends={r: [task(64 * (seed + i + r)) for i in range(ntask)] for r in range(ROWS)},
                      recvs={r: [task(0x8000 + 64 * (seed + i + r)) for i in range(ntask)] for r in range(ROWS)},
                      fills=[MemTask([{"addr": 0x20000 + 64 * (seed + i), "nwords": 64}], ddr=(DDR + c * 4096, 256, None))
                             for i in range(nfill)])
            for c in range(COLS)}


def b_rows(role: Role | None = None, objects: int = 2) -> dict[int, RowOps]:
    return {r: RowOps(runs=[(DDR, objects * D.Q)], b_objects=objects, role=role) for r in range(ROWS)}


def test_queues_stay_within_their_depth() -> None:
    """Ops of more tasks per channel than a queue holds, with and without a barrier: no channel ever has more tasks
    pending than the hardware queue (``Seq.check`` after every op), and the program verifies."""
    s = Seq(nop=NOP, pages={r: PAGE for r in Role})
    for i, (ntask, nfill) in enumerate(((1, 1), (3, 6), (2, 7), (3, 2), (1, 5))):
        s.op(op_cols(ntask, seed=i, nfill=nfill), b_rows(Role.LN if i % 2 else None, 3), barrier=i == 3)
        assert max(len(ch.pending) for ch in s._ch.values()) <= txn.QUEUE_DEPTH
    words = s.finish()
    txn.verify(words, queues=False)


def test_a_bd_is_rewritten_after_its_users_completion() -> None:
    """Every memtile BD the transaction rewrites has its last user complete before (the sequencer's frontier passed
    the op): replayed from the BDs' bookkeeping, per rewrite."""
    s = Seq(nop=NOP)
    writes: list[tuple[int, int, int | None, int]] = []
    real = s.t.mem_bd

    def mem_bd(col: int, bid: int, words: list[int]) -> int:
        writes.append((col, bid, s._mem[col].user[bid], s.frontier))
        return real(col, bid, words)
    s.t.mem_bd = mem_bd
    for i in range(12):
        s.op(op_cols(4, seed=16 * i), b_rows(None, 2))
    s.finish()
    assert writes
    assert all(user is None or user < frontier for _, _, user, frontier in writes)


def test_a_role_page_enters_once() -> None:
    """A row that holds a role's code takes its page again only for another role (an op of resident code leaves the cores' page in place)."""
    s = Seq(nop=NOP, pages={r: PAGE for r in Role})
    runs = []
    real = s._b_chain

    def b_chain(col: int, op: int, rs: list[tuple]) -> None:
        runs.append([r for r in rs if r[0] == PAGE])
        real(col, op, rs)
    s._b_chain = b_chain
    for role in (Role.GN, Role.GN, Role.LN, Role.LN, None, Role.LN):
        s.op({}, b_rows(role, 2))
    s.finish()
    assert [bool(r) for r in runs[:ROWS * 6:ROWS]] == [True, False, True, False, False, False]


def test_check_refuses_a_broken_lock_count() -> None:
    """A lock counting past 63 fails the check."""
    s = Seq(nop=NOP)
    s.op(op_cols(1), b_rows(None, 2))
    s._prod[2] = 64
    with pytest.raises(HardwareLimitError, match="counts to 63"):
        s.check()


def test_rewrite_completes_the_last_user() -> None:
    """A memtile BD is rewritten only after its last user is complete."""
    s = Seq(nop=NOP)
    for _ in range(3):
        s.op(op_cols(1), b_rows(None, 2))
    mem = s._mem[0]
    bid = min((b for b, u in enumerate(mem.user) if u is not None), key=lambda b: mem.user[b])
    u = mem.user[bid]
    assert u is not None and u + 1 < s.n
    s._rewrite(0, bid, [0] * 8)
    assert s.frontier > u and mem.held[bid] == [0] * 8
