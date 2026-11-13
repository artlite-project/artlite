"""Host-built transactions (``artlite.txn``): the hardware limits of the encoders, ``txn.verify``, the parser, the notes
of a hang, and the placement checks of pinned L1 buffers. No NPU."""

import types
from contextlib import nullcontext

import numpy as np
import pytest

from artlite import _native, overlay, txn
from artlite.operators import l1_layout
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence
from artlite.txn import HardwareLimitError

OP = types.SimpleNamespace(ENDPOINTS={"A": (txn.MM2S, 0), "X": (txn.MM2S, 1), "C": (txn.S2MM, 0)},
                           BD_BASE={"A": 0, "X": 4, "C": 8})


def test_field_limits() -> None:
    """A field or id the hardware cannot encode raises HardwareLimitError (a ValueError, also under ``python -O``) with
    the offending values."""
    t = txn.Txn()
    bad = [
        (lambda: t.shim_bd(0, 16, address=0x8000_0000, nbytes=64), "16"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=4 * 1024, dims=(1024, 1, 1, 1)), "(1024, 1, 1, 1)"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=4 * 8, dims=(4, 2, 1 << 21, 1)), "dims"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=36, dims=(4, 2, 4, 8)), "36 bytes"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=64, iteration=(8, 65)), "(8, 65)"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=64, packet=32), "packet id 32"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=64, next_bd=16), "next BD id 16"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=64, acquire=(16, 1)), "(16, 1)"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=64, release=(0, 64)), "(0, 64)"),
        (lambda: t.shim_bd(0, 0, address=1 << 48, nbytes=64), "0x1000000000000"),
        (lambda: t.shim_bd(0, 0, address=0x8000_0000, nbytes=0), "0 bytes"),
        (lambda: t.shim_push(0, txn.MM2S, 0, 0, repeat=257), "repeat 257"),
        (lambda: t.mem_bd(0, 48, [0] * 8), "BD id 48"),
        (lambda: t.mem_push(0, txn.MM2S, 0, 24, repeat=1), "BD 24"),
        (lambda: txn.mem_lock(64), "lock id 64"),
        (lambda: txn.mem_bd_words(addr=2, nwords=4), "address 2"),
        (lambda: txn.mem_bd_words(addr=0, nwords=4, dims=[(1024, 1)]), "size 1024"),
        (lambda: txn.mem_bd_words(addr=0, nwords=4, iteration=(4, 65)), "(4, 65)"),
        (lambda: txn.mem_bd_words(addr=0, nwords=4, acquire=(1, 99, True)), "99"),
        (lambda: txn.bd_packet_bits(-1), "packet id -1"),
    ]
    for build, text in bad:
        with pytest.raises(HardwareLimitError, match=text):
            build()
    assert issubclass(HardwareLimitError, ValueError)
    assert txn.bd_packet_bits(31) == txn.BD_PACKET | 31 << 19
    t.shim_bd(0, 15, address=0x8000_0000, nbytes=4 * 1023 * 1023, dims=(1023, 1023, 1 << 20, 1 << 20))  # (the limits)


def _bd_words(qos: int = 0, valid: bool = True) -> list[int]:
    """A shim BD's 8 words, contiguous, with the given AxQoS."""
    return [16, 0x8000_0000, 0, 0, txn.BD_BURST_512B, txn.BD_AXCACHE | qos << 20, 0, txn.BD_VALID if valid else 0]


def _controlled(col: int = 0) -> txn.Txn:
    t = txn.Txn()
    t.token_controller(col, txn.MM2S, 0)
    return t


def test_verify_sequence() -> None:
    """The transactions that ``Sequence`` builds (queues full, waits, tokens taken, ``sync(fills=True)``) pass."""
    seq = Sequence(OP, inputs=("X",))
    for i in range(11):
        seq.fill("A", i % 2, 0x8000_0000 + 4096 * i, 4096)
        if i % 3 == 0:
            seq.fill("X", 1, 0x9000_0000 + 1024 * i, 1024)
            seq.drain("C", i % 4, 0xA000_0000 + 512 * i, 512)
    seq.sync()
    for c in range(COLS):
        seq.fill("X", c, 0x9000_0000, 1024, repeat=2)
        seq.drain("C", c, 0xA000_0000, 512)
    seq.sync(fills=True)
    words = seq.words()
    txn.verify(words)
    txn.verify(words, queues=False)


def test_verify_array_tasks() -> None:
    """Tasks the array pushes itself retire without a trace in the words: the queue model is skipped, the static
    checks still run."""
    seq = Sequence(OP, inputs=("X",))
    for i in range(4):
        seq.fill("A", 0, 0x8000_0000 + 4096 * i, 4096)
    seq.device_task("A", 0)
    seq.drain("C", 0, 0xA000_0000, 512)
    seq.device_done("A", 0)
    seq.fill("A", 0, 0x8000_0000, 4096)
    seq.sync(fills=True)
    txn.verify(seq.words(), queues=False)


def test_verify_qos() -> None:
    """A BD blockwrite with a nonzero AxQoS, a clear valid bit or a zero length is refused, naming the operation."""
    t = _controlled()
    t.blockwrite(txn.tile_address(2, 0, txn.SHIM_BD0 + 3 * txn.BD_BYTES), _bd_words())
    txn.verify(t.words())
    for words, text in ((_bd_words(qos=2), "AxQoS 2"), (_bd_words(valid=False), "valid bit"),
                        ([0, *_bd_words()[1:]], "length 0")):
        t = _controlled()
        t.blockwrite(txn.tile_address(2, 0, txn.SHIM_BD0 + 3 * txn.BD_BYTES), words)
        with pytest.raises(HardwareLimitError, match=rf"operation 1: shim BD 3 of column 2: {text}"):
            txn.verify(t.words(), queues=False)
    t = _controlled()  # (a partial write that reaches word 5 is checked too)
    t.blockwrite(txn.tile_address(2, 0, txn.SHIM_BD0 + 3 * txn.BD_BYTES + 20), [txn.BD_AXCACHE | 1 << 21])
    with pytest.raises(HardwareLimitError, match="AxQoS 2"):
        txn.verify(t.words(), queues=False)


def test_verify_rewritten_bd() -> None:
    """A BD rewritten while a task that uses it (alone or in a chain) is queued is refused; after a sync that retires the
    task it is not."""
    t = txn.Txn()
    t.token_controller(0, txn.MM2S, 0)
    t.shim_task(0, txn.MM2S, 0, 3, address=0x8000_0000, nbytes=64, token=True)
    t.shim_bd(1, 3, address=0x8000_1000, nbytes=64)  # (another column's BD 3)
    t.shim_bd(0, 4, address=0x8000_1000, nbytes=64)
    t.sync(col=0, direction=txn.MM2S, channel=0)
    t.shim_task(0, txn.MM2S, 0, 3, address=0x8000_2000, nbytes=64, token=True)
    t.sync(col=0, direction=txn.MM2S, channel=0)
    txn.verify(t.words())
    t = txn.Txn()
    t.shim_bd(0, 6, address=0x8000_0000, nbytes=64)
    t.shim_bd(0, 5, address=0x8000_0000, nbytes=64, next_bd=6)
    t.shim_push(0, txn.MM2S, 0, 5)  # (a chain: BDs 5 and 6)
    t.shim_bd(0, 6, address=0x8000_1000, nbytes=64)
    with pytest.raises(HardwareLimitError, match=r"operation 3: shim BD 6 of column 0 rewritten while operation 2's task"):
        txn.verify(t.words())


def test_verify_token_controller() -> None:
    """A channel's first token push comes after its token controller (shim and memory tile)."""
    t = txn.Txn()
    t.shim_task(3, txn.MM2S, 0, 0, address=0x8000_0000, nbytes=64, token=True)
    with pytest.raises(HardwareLimitError, match=r"operation 1: a token push on MM2S channel 0 of column 3"):
        txn.verify(t.words(), queues=False)
    t = txn.Txn()
    t.token_controller(3, txn.MM2S, 1)  # (another channel's)
    t.shim_task(3, txn.MM2S, 0, 0, address=0x8000_0000, nbytes=64, token=True)
    with pytest.raises(HardwareLimitError, match="before the channel's token controller"):
        txn.verify(t.words(), queues=False)
    t = txn.Txn()
    t.mem_push(2, txn.MM2S, 5, 24, token=True)
    with pytest.raises(HardwareLimitError, match=r"MM2S channel 5 of column 2 \(row 1\)"):
        txn.verify(t.words(), queues=False)
    t = txn.Txn()
    t.mem_token_controller(2, txn.MM2S, 5)
    t.mem_push(2, txn.MM2S, 5, 24, token=True)
    t.mem_push(2, txn.MM2S, 5, 25)
    txn.verify(t.words())


def test_verify_queue_model() -> None:
    """A fifth queued task, a sync with no token task to take, a task still queued at the end, and (with a poll of the
    queue status) the retirement of the oldest tasks."""
    def pushes(n: int, *, token_at: int | None = None) -> txn.Txn:
        t = _controlled()
        for i in range(n):
            t.shim_task(0, txn.MM2S, 0, i, address=0x8000_0000, nbytes=64, token=i == token_at)
        return t

    with pytest.raises(HardwareLimitError, match=r"5 tasks queued on MM2S channel 0 of column 0 \(at most 4\)"):
        txn.verify(pushes(5).words())
    t = pushes(4, token_at=1)
    t.sync(col=0, direction=txn.MM2S, channel=0)  # (retires tasks 0 and 1)
    t.shim_task(0, txn.MM2S, 0, 4, address=0x8000_0000, nbytes=64)
    t.shim_task(0, txn.MM2S, 0, 5, address=0x8000_0000, nbytes=64, token=True)
    t.sync(col=0, direction=txn.MM2S, channel=0)
    txn.verify(t.words())  # (the second sync's first token is the last task's: it retires the queue)
    t = pushes(2)
    t.sync(col=0, direction=txn.MM2S, channel=0)
    with pytest.raises(HardwareLimitError, match=r"operation \d+: a sync on MM2S channel 0 of column 0 with no token"):
        txn.verify(t.words())
    with pytest.raises(HardwareLimitError, match=r"end: 2 tasks still queued on MM2S channel 0 of column 0"):
        txn.verify(pushes(2).words())
    txn.verify(pushes(2).words(), queues=False)
    # a poll of the queue status (as ``Run.guarded`` puts before a push) retires the oldest tasks until it is below 4
    status = txn.tile_address(0, 0, txn.QUEUE_STATUS[txn.MM2S][0])
    t = pushes(4)
    t.maskpoll32(status, 0, txn.QUEUE_DEPTH << 20)
    t.shim_task(0, txn.MM2S, 0, 4, address=0x8000_0000, nbytes=64, token=True)
    t.sync(col=0, direction=txn.MM2S, channel=0)
    txn.verify(t.words())
    t = pushes(4)
    t.maskpoll32(status, 0, txn.QUEUE_DEPTH << 20)
    t.shim_task(0, txn.MM2S, 0, 3, address=0x8000_0000, nbytes=64)  # (BD 3 is of a task still queued)
    with pytest.raises(HardwareLimitError, match="BD 3 of column 0 rewritten"):
        txn.verify(t.words())


def test_decode() -> None:
    """``decode`` reads every operation ``op_offsets`` knows (no-ops, PDI loads, the scratchpad), and refuses an unknown
    opcode or a body that ends inside an operation with a ValueError."""
    t = txn.Txn()
    t.write32(0x100, 7)
    body = [*t.words()[4:], txn.OP_NOOP, txn.OP_LOAD_PDI, 0, 1, 2, txn.OP_CREATE_SCRATCHPAD, 0, 0, 0]
    words = np.array([*t.words()[:2], 4, 4 * (4 + len(body)), *body], np.uint32)
    assert [n for n, _ in txn.decode(words)] == ["write32", "noop", "load_pdi", "create_scratchpad"]
    assert [p for p, _ in txn.op_offsets(words[4:])] == [0, 6, 7, 11]
    for bad in ([0x55, 0, 0, 0], [txn.OP_WRITE, 0]):
        with pytest.raises(ValueError):
            txn.op_offsets(np.array(bad, np.uint32))


def test_hang_notes() -> None:
    """A DispatchError note names the operations around the reported one and the last sync; the firmware's sentinel,
    an index past the end, and a transaction that does not decode give plain notes; the note never raises."""
    t = _controlled()
    t.shim_task(0, txn.MM2S, 0, 0, address=0x8000_0000, nbytes=64, token=True)
    t.sync(col=0, direction=txn.MM2S, channel=0)
    t.write32(0x100, 1)
    t.write32(0x104, 2)
    words = t.words()
    n = len(txn.decode(words))
    note = txn.hang_note(words, {"txn_op_idx": 5}, what="a transaction")
    assert f"operations 4 .. 6 of {n} (txn_op_idx 5)" in note and "the last sync before it: operation 3" in note
    assert "column 0" in note and "MM2S channel 0" in note
    for idx in (txn.NO_OP_INDEX, n, n + 7):
        note = txn.hang_note(words, {"txn_op_idx": idx}, what="a transaction")
        assert "the firmware reports no operation index" in note and f"txn_op_idx {idx}" in note
        assert ".." not in note  # (no window of operations)
    assert "no operation index" in txn.hang_note(words, None, what="t")
    assert "not decodable" in txn.hang_note(np.array([1, 2, 3, 4, 5, 6, 7], np.uint32), {"txn_op_idx": 1}, what="t")
    assert txn.reported_op({"txn_op_idx": 3}, 4) == 3 and txn.reported_op({"txn_op_idx": 4}, 4) is None
    assert txn.reported_op({}, 4) is None and txn.reported_op(None, 4) is None


def _fake_device(monkeypatch: pytest.MonkeyPatch, kernel: object, *, prologue: bool) -> OverlayDevice:
    """An OverlayDevice without a device: its loaded transaction, optionally a stale prologue, and ``kernel``."""
    monkeypatch.setattr(overlay, "npu_turn", nullcontext)
    monkeypatch.setattr(_native, "turn_foreign", lambda: 0, raising=False)
    dev = object.__new__(OverlayDevice)
    dev._prog = types.SimpleNamespace(_kernel=lambda: kernel)
    dev._insts, dev.buffers, dev._turns, dev._submitted = None, {}, 0, None
    dev._loaded = _transaction_words(9)
    dev._prologue = types.SimpleNamespace(buffer=None, words=_transaction_words(3)) if prologue else None
    dev.stale = prologue
    return dev


def _transaction_words(n: int) -> np.ndarray:
    t = txn.Txn()
    for i in range(n):
        t.write32(0x100 + 4 * i, i)
    return t.words()


def _hang(idx: int) -> _native.DispatchError:
    e = _native.DispatchError("hang")
    e.health = {"txn_op_idx": idx}
    return e


def test_dispatch_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure of the main transaction is noted as its own; of the prologue's dispatch (the code pages) as the
    prologue's only; the firmware's sentinel as no operation index; a note that cannot be made adds a plain one."""
    class Kernel:
        def __init__(self, fail: dict[int, int]):
            self.fail, self.runs = fail, []

        def dispatch(self, buffer: object, bufs: list, **kw: object) -> None:
            self.runs.append(buffer)
            if len(self.runs) in self.fail:
                raise _hang(self.fail[len(self.runs)])

    dev = _fake_device(monkeypatch, Kernel({1: 5}), prologue=False)
    with pytest.raises(_native.DispatchError) as info:
        dev.dispatch()
    assert len(info.value.__notes__) == 1 and "the loaded transaction: operations 4 .. 6 of 9" in info.value.__notes__[0]

    dev = _fake_device(monkeypatch, Kernel({1: txn.NO_OP_INDEX}), prologue=True)
    with pytest.raises(_native.DispatchError) as info:  # (the prologue fails first)
        dev.dispatch()
    assert len(info.value.__notes__) == 1
    assert "the prologue's transaction (code pages): 3 operations; the firmware reports no operation index" in \
        info.value.__notes__[0]

    dev = _fake_device(monkeypatch, Kernel({2: 1}), prologue=True)  # (the prologue passes, the main one fails)
    with pytest.raises(_native.DispatchError) as info:
        dev.dispatch()
    assert len(info.value.__notes__) == 1 and "the loaded transaction: operations 0 .. 2 of 9" in info.value.__notes__[0]

    dev = _fake_device(monkeypatch, Kernel({1: 2}), prologue=False)
    dev._loaded = np.array([1, 2, 3], np.uint32)  # (not a transaction)
    with pytest.raises(_native.DispatchError) as info:
        dev.dispatch()
    assert "not decodable" in info.value.__notes__[0]
    e = _hang(1)
    dev._note(e, types.SimpleNamespace())  # (no words: the note fails, the error stays)
    assert "no transaction position" in e.__notes__[0]


def test_placement() -> None:
    """Pinned L1 buffers of a core tile: 64-byte aligned, inside L1, pairwise disjoint (HardwareLimitError naming them)."""
    ok = {(0, 2): [("a", 0, 1024), ("b", 1024, 64), ("c", 0xFFC0, 64)], (0, 3): [("a", 0, 0x10000)]}
    l1_layout.check_placement(ok)
    with pytest.raises(HardwareLimitError, match=r"tile \(1, 3\): pinned buffer x at 0x20 is not 64-byte aligned"):
        l1_layout.check_placement({(1, 3): [("x", 0x20, 64)]})
    with pytest.raises(HardwareLimitError, match=r"pinned buffer y \[0xffc0, 0x10040\) is outside L1"):
        l1_layout.check_placement({(0, 2): [("y", 0xFFC0, 128)]})
    with pytest.raises(HardwareLimitError, match=r"pinned buffers a \[0x0, 0x400\) and b \[0x3c0, 0x4c0\) overlap"):
        l1_layout.check_placement({(0, 2): [("b", 0x3C0, 256), ("a", 0, 1024)]})
