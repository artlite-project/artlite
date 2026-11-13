"""``Library``: operators compiled into one ELF, each once, as a library of blocks: never dispatched whole, it runs
programs assembled on the host (``program.Assembler``), each item an operator with its arguments at any device
addresses (``items``), in pieces of one hardware context (``run``).

Every argument of an operator is its own slot in the ELF's scratch (a BD's address tells its argument); no data BOs
are allocated: the items address the caller's buffers.
"""

import bisect
import contextlib
import itertools
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from ... import _native, portable, txn
from ...elf_parser import ParsedElf
from ...overlay import SPIN_MARGIN_US, irq_cpu, npu_turn
from ..base import ArtcOperator
from . import locks, program
from .elf import FusedElf
from .layout import Runlist, distinct, parse_name
from .program import PIECE_BYTES, Assembler, Item


class _Flight(NamedTuple):
    """A piece dispatched: its start on the NPU, and what maps the operation index of a hang to its item: the program's
    items, the indices of the piece's first and end, ``Assembler.item_ops`` (None: not known)."""

    start: float
    items: Sequence[Item]
    first: int
    end: int
    item_ops: list[int] | None


class _Piece(NamedTuple):
    """An assembled piece of a program: its items (the list it was assembled from), its control code, the PDI parity
    after it and the operation count through the end of each of its items (``Assembler.item_ops``)."""

    items: Sequence[Item]
    code: np.ndarray
    parity: int
    item_ops: list[int]


class PieceCache:
    """The assembled pieces of the programs a Library runs again, and the shortest seen duration of each. A program's
    items are a list the caller keeps (e.g. ChunkPrefill's per chunk size): an entry is of that list object, found by
    its ``id`` and confirmed by identity (an entry keeps its list alive, so its id is not reused while it is held)."""

    ENDS_KEPT = 4096  # (durations kept)

    def __init__(self, limit: int):
        """``limit``: bytes of pieces kept (all dropped when they would exceed it)."""
        self.limit = limit
        self._pieces: dict[tuple[int, int, int, int], _Piece] = {}
        self._bytes = 0
        self._ends: dict[tuple[int, int, int], tuple[Sequence[Item], float]] = {}

    def get(self, items: Sequence[Item], a: int, b: int, parity: int) -> _Piece | None:
        """The piece of ``items[a:b]`` assembled from PDI parity ``parity``."""
        hit = self._pieces.get((id(items), a, b, parity))
        return hit if hit is not None and hit.items is items else None

    def put(self, a: int, b: int, parity: int, piece: _Piece) -> None:
        """Keep the piece of ``piece.items[a:b]`` assembled from ``parity``."""
        if self._bytes + piece.code.nbytes > self.limit:
            self.clear()
        self._pieces[(id(piece.items), a, b, parity)] = piece
        self._bytes += piece.code.nbytes

    def clear(self) -> None:
        self._pieces.clear()
        self._bytes = 0

    def end(self, items: Sequence[Item], a: int, b: int) -> float | None:
        """The shortest seen duration (s) of the piece ``items[a:b]``."""
        hit = self._ends.get((id(items), a, b))
        return hit[1] if hit is not None and hit[0] is items else None

    def note_end(self, items: Sequence[Item], a: int, b: int, seconds: float) -> None:
        if len(self._ends) >= self.ENDS_KEPT:
            self._ends.clear()
        old = self.end(items, a, b)
        self._ends[(id(items), a, b)] = (items, seconds if old is None else min(seconds, old))


class Pieces:
    """The instruction BOs of the process's Library programs: two (piece k + 1 assembled while k runs), one pair for
    every Library (one runs at a time: ``lock``), allocated by a run. A piece takes PIECE_BYTES where the device heap
    holds both and RESERVE besides (allocations after them: a vision encoder's PDIs), the heap grown if it must; a heap
    that cannot grow (Linux before 7.2: one 64 MB heap per process) gets pieces of what it has left but RESERVE, in
    whole MB, at least MIN_BYTES: a smaller piece only adds dispatches (a gap of tens of us each). An allocation that
    does not fit such a heap frees the pair between runs (``release``), and the next run takes what is left then."""

    RESERVE = 8 << 20
    MIN_BYTES = 4 << 20

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.bos: list = []
        self.nbytes = 0
        # (per BO, the program a Library left there for a run again: (weakref to the Library, its key, bytes, parity
        # after it))
        self.held: list[tuple | None] = [None, None]

    def ensure(self, device: Any) -> int:
        """The pair on ``device`` (allocated at the first call); returns a piece's bytes."""
        if self.bos:
            return self.nbytes
        nbytes = PIECE_BYTES
        if device.heap_free() < 2 * PIECE_BYTES + self.RESERVE and not device.grow_heap() and device.heap_fixed:
            nbytes = min(PIECE_BYTES, max(self.MIN_BYTES, (device.heap_free() - self.RESERVE) // 2 >> 20 << 20))
        while True:
            try:
                self.bos = [device.alloc_insts_buffer(nbytes) for _ in range(2)]
                break
            except _native.HeapFullError:  # (a fragmented heap)
                if not device.heap_fixed or nbytes <= self.MIN_BYTES:
                    raise
                nbytes = max(self.MIN_BYTES, nbytes - (2 << 20))
        self.nbytes = nbytes
        return nbytes

    def release(self) -> bool:
        """Free the pair unless a run holds it (the shim's heap reclaim: ``_native.set_heap_reclaim``); returns
        whether it did."""
        if not self.bos or not self.lock.acquire(blocking=False):
            return False
        try:
            self.bos, self.held, self.nbytes = [], [None, None], 0
        finally:
            self.lock.release()
        return True

    def holds(self, slot: int, lib: "Library", key: tuple) -> tuple | None:
        """What BO ``slot`` holds for ``lib``'s program of ``key``: (bytes, parity after it), or None."""
        held = self.held[slot]
        return held[2:] if held is not None and held[0]() is lib and held[1] == key else None

    def write(self, slot: int, piece: np.ndarray, held: tuple | None) -> Any:
        """Copy ``piece`` into BO ``slot`` (``held``: (Library, key, parity after it) of a program run again, or
        None); returns the BO."""
        bo = self.bos[slot]
        np.copyto(bo.map()[:piece.nbytes], piece.view(np.uint8))
        bo.sync_to_device(piece.nbytes, 0)
        self.held[slot] = None if held is None else (weakref.ref(held[0]), held[1], piece.nbytes, held[2])
        return bo


_PIECES = Pieces()
_native.set_heap_reclaim(_PIECES.release)


class Library(FusedElf):
    LIBRARY = True
    # bytes of assembled pieces kept for programs run again: every piece of a prefill's chunk sizes at both parities
    # (Qwen3.6-35B-A3B's 2048-token chunk: 83 pieces, 44 MB; a large piece assembled again ~1.3 ms of host time, on the
    # critical path where the previous piece is short)
    PIECES_KEPT = 512 << 20
    SPIN_US = SPIN_MARGIN_US  # (a piece's wait polls from this long before its shortest seen end: ``wait``)

    def __init__(self, name: str, operators: Iterable[ArtcOperator]):
        """``operators``: the library's entries (each operator object once: repeats are dropped), in order."""
        ops = distinct(operators)
        super().__init__(name=name, runlist=[(op, *[f"e{e}_{k}" for k in range(len(op.get_arg_spec()))])
                                             for e, op in enumerate(ops)])
        self._entry = {id(op): e for e, op in enumerate(ops)}
        self._asm: Assembler | None = None
        self._cache = PieceCache(self.PIECES_KEPT)

    def _piece(self, items: Sequence[Item], a: int, b: int, parity: int) -> _Piece:
        """The piece of ``items[a:b]`` assembled from PDI parity ``parity`` (kept for a program run again); the
        assembler's parity is not changed."""
        hit = self._cache.get(items, a, b, parity)
        if hit is None:
            asm = self._asm
            saved, asm.parity = asm.parity, parity
            try:
                hit = _Piece(items, asm.assemble(items[a:b]), asm.parity, asm.item_ops)
            finally:
                asm.parity = saved
            self._cache.put(a, b, parity, hit)
        return hit

    def __contains__(self, op: ArtcOperator) -> bool:
        return id(op) in self._entry

    def items(self, runlist: Runlist, buffers: Mapping[str, Any]) -> list[Item]:
        """A program's items: per run of ``runlist`` ((operator, *buffer names)) its entry and the device addresses of
        its arguments in ``buffers`` (host buffers by name; ``"NAME[a:b]"`` its bytes a .. b), each at least its
        argument's bytes (``get_arg_spec``: the NPU reads and writes that many from the address, past a smaller buffer
        into whatever the process maps after it)."""
        ctx = self.load()._ctx  # (on XRT: open, its addresses)

        def address(op: ArtcOperator, i: int, name: str) -> int:
            base, start, end = parse_name(name)
            size = getattr(buffers[base], "_raw", buffers[base]).size
            end = size if end is None else end
            if end > size or end - start < op.get_arg_spec()[i].nbytes:
                raise ValueError(f"{self.name}: {type(op).__name__}'s argument {i} needs "
                                 f"{op.get_arg_spec()[i].nbytes} bytes, {name} has {min(end, size) - start}")
            return txn.device_address(buffers[base], start, ctx)

        return [(self._entry[id(op)], [address(op, i, n) for i, n in enumerate(names)]) for op, *names in runlist]

    def _load(self, elf: ParsedElf) -> None:
        super()._load(elf)
        code, bds = self._control_program(elf, lambda bd: 0)  # (the items' addresses: written per item)
        slots = [(s.offset, s.offset + s.nbytes, e, k)
                 for e, (_, *names) in enumerate(self.runlist) for k, s in enumerate(map(self.layout.span, names))]
        census = locks.census(os.path.dirname(self.elf_path)) if program.quiet_enabled() else None
        self._asm = Assembler(code, bds, slots, census)
        assert len(self._asm.blocks) == len(self.runlist), (len(self._asm.blocks), len(self.runlist))
        self._dispatch_bufs = list(self._pdis.values())
        if self._asm.packet_bytes:  # (the configurations' control packets: ``ctrlpkt``)
            bo = self._device.alloc_host_buffer(-(-self._asm.packet_bytes // 4096) * 4096)
            self._asm.place(bo.map().view(np.uint32), txn.device_address(bo, 0, self._ctx))
            bo.sync_to_device()
            self._dispatch_bufs.append(bo)

    def prepare(self) -> None:
        """The hardware context and the process's piece BOs now (else at the first run): before a timed run, as the
        first context of a process wakes the NPU (~40 ms) and the BOs take ~8 ms."""
        self._kernel()
        with _PIECES.lock:
            _PIECES.ensure(self._device)

    def run(self, items: Sequence[Item], *, splits: Iterable[int] = (), bufs: Sequence[Any],
            hooks: Mapping[int, Callable[[], None]] | None = None) -> int:
        """Dispatch the program of ``items`` (``items()``; consecutive items of one design share its configuration): a
        new piece of control code (a dispatch) at each item index of ``splits``, and where a piece would exceed the
        process's piece BOs (``Pieces``). ``bufs``: the data BOs the items address. ``hooks``: per item index of
        ``splits``, a host function called before the piece it starts is dispatched (while the previous one runs, the
        one before it ended: e.g. inputs that only later items read, outputs of two pieces back). Returns the
        dispatches.

        Each piece is dispatched as soon as it is assembled and its hook ran, then the previous one is waited for: the
        firmware starts it at the previous one's end (a boundary waited for first: ~35 us more of idle NPU, the wait's
        wake-up and the submit), unless another process uses the NPU (``_native.turn_peers``): then the previous one
        ends first, and if that process waits for the NPU's turn (``overlay.npu_turn``) its dispatches run before this
        one (each piece configures the array again), so that it waits for one piece at most, not also for one queued
        behind it. The calling thread runs on the CPU that takes the NPU's interrupts meanwhile
        (``irq_cpu``): it waits there (napping, then polling from shortly before the piece's shortest seen end), and the
        host work between pieces runs on that warm core (a thread let go after each wait ran it ~2.5x slower, on a CPU
        the scheduler woke for it)."""
        cpu = irq_cpu()
        with portable.on_cpu(cpu), _PIECES.lock:
            try:
                return self._run(items, splits=splits, bufs=bufs, hooks=hooks, cpu=cpu)
            except BaseException:  # (a hook's exception: the pieces dispatched end first, and with them the NPU's turn)
                with contextlib.suppress(_native.DispatchError):
                    self._kernel().dispatch_wait()
                raise

    def _run(self, items: Sequence[Item], *, splits: Iterable[int], bufs: Sequence[Any],
             hooks: Mapping[int, Callable[[], None]] | None, cpu: int) -> int:
        asm = self.load()._asm
        kern = self._kernel()
        limit = _PIECES.ensure(self._device)
        bounds = asm.bounds(items, set(splits), limit)
        refs = self._dispatch_bufs + [b for b in bufs if all(b is not d for d in self._dispatch_bufs)]
        key = None
        submit = 0.0

        def wait(piece: _Flight, keep: int) -> float:
            """Wait for ``piece`` until ``keep`` pieces after it are pending: napping on the interrupts' CPU, then
            polling from shortly before its shortest seen end; returns its end. A DispatchError names the item that
            holds the operation the firmware reports."""
            start = piece.start
            est = self._cache.end(piece.items, piece.first, piece.end)
            after = int((start + est - submit) * 1e6) - self.SPIN_US if est is not None else 1 << 30  # (from submit)
            try:
                kern.dispatch_wait(0, max(0, after) if cpu >= 0 else -1, max(cpu, 0), keep)
            except _native.DispatchError as e:
                self._note(e, piece)
                raise
            end = time.perf_counter()
            self._cache.note_end(piece.items, piece.first, piece.end, end - start)
            return end

        if len(bounds) == 2 and not hooks:  # (a program of one piece run again: still in the BO of its parity)
            key = (hash(tuple((e, tuple(addrs)) for e, addrs in items)), asm.parity)
            held = _PIECES.holds(asm.parity, self, key)
            if held is not None:
                with npu_turn():
                    submit = time.perf_counter()
                    kern.dispatch_elf_async(_PIECES.bos[asm.parity], refs, held[0])
                cached = self._cache.get(items, 0, len(items), asm.parity)
                asm.parity = held[1]
                wait(_Flight(submit, items, 0, len(items), cached.item_ops if cached is not None else None), 0)
                return 1
        k, starts = 0, []
        last, end = None, 0.0  # (the piece dispatched last; the end of the one before it)
        for k, (a, b) in enumerate(itertools.pairwise(bounds)):
            starts.append(asm.parity)
            slot = asm.parity if key is not None else k % 2
            piece, asm.parity, item_ops = self._piece(items, a, b, asm.parity)[1:]
            if piece.nbytes > limit:
                raise ValueError(f"{self.name}: items {a}..{b} take {piece.nbytes} bytes of control code, more than "
                                 f"a piece BO's {limit}")
            # (slot k % 2: piece k - 2's, which has ended)
            bo = _PIECES.write(slot, piece, (self, key, asm.parity) if key is not None else None)
            if hooks and a in hooks:
                hooks[a]()
            if last is not None and _native.turn_peers():  # (none queued ahead of another process's turn)
                end = wait(last, 0)
                last = None
                if _native.turn_contended():  # (its dispatches between the pieces)
                    _native.turn_pass()
            with npu_turn():  # (the piece's submit time: once the turn is this process's)
                submit = time.perf_counter()
                kern.dispatch_elf_async(bo, refs, piece.nbytes)
            if last is not None:
                end = wait(last, 1)  # (piece k - 1; piece k runs on from its end)
            last = _Flight(max(submit, end), items, a, b, item_ops)
        if key is not None and asm.parity != slot:
            self._ahead(items, key, asm)
        elif asm.parity != starts[0]:  # (the next run's pieces start at the other parities: assembled while this one ends)
            for (a, b), p in zip(itertools.pairwise(bounds), starts):
                self._piece(items, a, b, p ^ 1)
        wait(last, 0)
        return k + 1

    def _note(self, e: _native.DispatchError, piece: _Flight) -> None:
        """Add to a DispatchError where in the program the firmware was: the item and entry (operator) that hold the
        operation its health reports. Never raises."""
        try:
            n = f"{self.name}: piece of items {piece.first}..{piece.first + len(piece.item_ops or ())}"
            if piece.item_ops is None:
                e.add_note(f"{n} (no operation map)")
                return
            i = txn.reported_op(e.health, piece.item_ops[-1])
            if i is None:
                e.add_note(f"{n}: {txn.no_op_note(e.health, piece.item_ops[-1])}")
                return
            k = bisect.bisect_right(piece.item_ops, i)
            entry, addrs = piece.items[piece.first + k]
            start = piece.item_ops[k - 1] if k else 0
            e.add_note(f"{n}: operation {i} is in item {piece.first + k} (entry {entry}: "
                       f"{type(self.runlist[entry][0]).__name__}, operations {start}..{piece.item_ops[k]}; its arguments "
                       f"at {[hex(a) for a in addrs]})")
        except Exception as note_error:  # noqa: BLE001 (the dispatch's error is the one to see)
            e.add_note(f"(no program position: {note_error!r})")

    def _ahead(self, items: Sequence[Item], key: tuple, asm: Assembler) -> None:
        """While a program of one piece runs from the BO of one parity: the program again from the other (its
        configurations' PDI slots swapped), assembled into that BO, so that a run again needs neither (a prompt's
        second chunk of a size: 6 ms)."""
        p = asm.parity
        key = (key[0], p)  # (the key of a run that starts at this parity)
        if _PIECES.holds(p, self, key) is not None:
            return
        piece = self._piece(items, 0, len(items), p)
        _PIECES.write(p, piece.code, (self, key, piece.parity))
