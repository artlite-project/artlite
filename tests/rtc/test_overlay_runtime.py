"""Host side of the transaction-run overlays (``artlite.txn``, ``operators/overlay/runtime.py``): transaction encoding,
shim BD iteration, the rolling task sequence. No NPU."""

import types

from artlite import txn
from artlite.operators.design import COLS
from artlite.operators.overlay.runtime import Sequence


def test_txn_encoding() -> None:
    """A host-built transaction decodes to the operations that were added; BD words are found where
    shim_task says; a BD's dimensions."""
    t = txn.Txn()
    t.token_controller(3, txn.S2MM, 1)
    at = t.shim_task(3, txn.MM2S, 0, 5, address=0x1_8000_1000, nbytes=4096)
    t.shim_task(3, txn.S2MM, 1, 12, address=0x8000_2000, nbytes=128, token=True)
    t.sync(col=3, direction=txn.S2MM, channel=1)
    t.shim_bd(2, 7, address=0x8000_0000, nbytes=4 * 576 * 4 * 3, dims=(576, 4, 2304, 576))
    words = t.words()
    assert list(words[at:at + 3]) == [1024, 0x8000_1000, 1] == txn.bd_words(0x1_8000_1000, 4096)
    ops = txn.decode(words)
    assert [name for name, _ in ops] == ["maskwrite32", "blockwrite", "write32", "blockwrite", "write32", "sync",
                                         "blockwrite"]
    assert ops[1][1]["addr"] == txn.tile_address(3, 0, txn.SHIM_BD0 + 5 * txn.BD_BYTES)
    assert (ops[1][1]["data"][5] >> 20) & 0xF == 0 and (ops[1][1]["data"][5] >> 24) & 0xF == 2  # AxQoS 0, AxCache 2
    assert ops[1][1]["data"][3] == 0 and ops[1][1]["data"][4] & 0x3FFFFFFF == 0  # a contiguous BD
    assert ops[2][1] == {"addr": txn.tile_address(3, 0, txn.QUEUE[txn.MM2S][0]), "value": 5}
    assert ops[4][1]["value"] == (1 << 31) | 12
    assert ops[5][1] == {"direction": txn.S2MM, "row": 0, "col": 3, "nrow": 1, "ncol": 1, "channel": 1}
    d = ops[6][1]["data"]
    assert d[3] == 576 << 20 and d[4] & 0x3FFFFFFF == (4 << 20) | 2303 and d[5] == (2 << 24) | 575
    assert (d[5] >> 20) & 0xF == 0  # AxQoS stays 0
    # the stream of that BD: strips of 576 words from 4 regions 2304 words apart, then the next strips
    off = txn.bd_offsets(4 * 576 * 4 * 3, (576, 4, 2304, 576)) // 4
    assert off[0] == 0 and off[575] == 575 and off[576] == 2304 and off[4 * 576] == 576 and off[-1] == 3 * 2304 + 3 * 576 - 1


def test_txn_bd_iteration() -> None:
    """A shim BD with iteration = (step, count): word 6 holds Iteration_Current 0, Iteration_Wrap
    count - 1 and Iteration_Stepsize step - 1 (a repeated task advances its address by step words per
    execution: the prefill's attention-output drain)."""
    t = txn.Txn()
    t.shim_bd(1, 12, address=0x8000_0000, nbytes=4 * 288, iteration=(1728, 12))
    t.shim_bd(1, 13, address=0x8000_0000, nbytes=4 * 288)
    (_, a), (_, b) = txn.decode(t.words())
    assert a["data"][6] == (11 << 20) | 1727 and a["data"][6] >> 26 == 0
    assert b["data"][6] == 0


def test_rolling_sequence() -> None:
    """Sequence(rolling=True): a channel never holds more than 4 tasks no wait covered (the 5th waits for its
    oldest), a BD is rewritten only after its task completed, a task carries a token only where a wait takes it (a
    wait covers its channel's earlier tasks: in-order completion), every token is taken, and sync(fills=True)
    leaves no task uncovered."""
    op = types.SimpleNamespace(ENDPOINTS={"A": (txn.MM2S, 0), "X0": (txn.MM2S, 1), "C": (txn.S2MM, 0)},
                               BD_BASE={"A": 0, "X0": 4, "C": 8})
    seq = Sequence(op, rolling=True)
    for i in range(11):
        seq.fill("A", i % 2, 0x8000_0000 + 4096 * i, 4096)
        if i % 3 == 0:
            seq.fill("X0", 1, 0x9000_0000 + 1024 * i, 1024)
            seq.drain("C", i % 4, 0xA000_0000 + 512 * i, 512)
            seq.drain("C", i % 4, 0xA100_0000 + 512 * i, 512)
    seq.sync()  # (the drains)
    seq.fill("A", 0, 0x8000_0000, 4096, repeat=2)
    seq.sync(fills=True)
    queues = {txn.tile_address(c, 0, txn.QUEUE[d][ch]): (c, d, ch)
              for c in range(COLS) for d, ch in op.ENDPOINTS.values()}
    queued: dict[tuple[int, int, int], list[tuple[int, bool]]] = {}  # (BD, token) of the tasks no wait covered
    pushes = tokens = waits = 0
    for name, a in txn.decode(seq.words()):
        if name == "write32" and a["addr"] in queues:
            q = queued.setdefault(queues[a["addr"]], [])
            bd, token = a["value"] & 0xF, bool(a["value"] >> 31)
            assert bd not in [b for b, _ in q], "a BD rewritten while its task is queued"
            q.append((bd, token))
            assert len(q) <= 4, "more than 4 queued tasks on a channel"
            pushes += 1
            tokens += token
        elif name == "sync":  # per column of the run: its tasks up to the first with a token complete
            for c in range(a["col"], a["col"] + a["ncol"]):
                q = queued[(c, a["direction"], a["channel"])]
                k = next((i for i, (_, t) in enumerate(q) if t), None)
                assert k is not None, "a wait without a token to take"
                del q[:k + 1]
                waits += 1
    assert pushes == 11 + 4 + 8 + 1 and not any(queued.values())
    assert tokens == waits < pushes, (tokens, waits, pushes)


def test_rolling_inputs() -> None:
    """Sequence(rolling=True, inputs=("X",)): a sync first waits for the X fills (one wait per run of columns), so the
    next phase's X fills follow the drain wait without waits of their own; the queue and token rules still hold."""
    op = types.SimpleNamespace(ENDPOINTS={"A": (txn.MM2S, 0), "X": (txn.MM2S, 1), "C": (txn.S2MM, 0)},
                               BD_BASE={"A": 0, "X": 4, "C": 8})
    seq = Sequence(op, rolling=True, inputs=("X",))
    for c in range(COLS):
        seq.fill("A", c, 0x8000_0000 + 0x10000 * c, 1 << 20)
    for phase in range(9):
        for c in range(COLS):
            seq.fill("X", c, 0x9000_0000 + 4352 * phase, 4352)
        for c in range(COLS):
            seq.drain("C", c, 0xA000_0000 + 128 * (8 * phase + c), 128)
        seq.sync()
    seq.sync(fills=True)
    ops = txn.decode(seq.words())
    x_queue = {txn.tile_address(c, 0, txn.QUEUE[txn.MM2S][1]) for c in range(COLS)}
    x_waits = [a for name, a in ops if name == "sync" and a["direction"] == txn.MM2S and a["channel"] == 1]
    assert len(x_waits) == 9 and all(a["col"] == 0 and a["ncol"] == COLS for a in x_waits)
    after_drain_wait = False  # (the ops between a drain wait and the next X push: no X wait)
    for name, a in ops:
        if name == "sync":
            after_drain_wait = a["direction"] == txn.S2MM or (after_drain_wait and a["channel"] != 1)
            assert not (after_drain_wait and a["direction"] == txn.MM2S and a["channel"] == 1)
        elif name == "write32" and a["addr"] in x_queue:
            after_drain_wait = False
