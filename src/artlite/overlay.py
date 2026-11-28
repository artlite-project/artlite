"""Host-side runtime of the overlays, the designs that run on host-built transactions (``artlite.txn``): the decode
overlay (``decode.design`` and its models'), the vision overlay (``vision.design``).

``Sequence`` turns a list of DMA transfers (fills of a column's input streams, drains of its output streams) into
the transaction of a dispatch; ``sync`` closes a group of transfers that must complete before the next group is
issued (an output that the next phase reads back from DDR). ``OverlayDevice`` owns the compiled design's hardware
context and dispatches transactions on host buffers; ``Transaction`` keeps one in an instruction buffer of its
own, to be patched (``txn.bd_words``) and dispatched again (its last durations time a spin on the CPU of the NPU's
interrupts near its end: ``irq_cpu``); ``Transactions`` keeps those an owner runs again (a model's steps), the least
recently used of them given up where the device heap is full and cannot grow (``heap_spare``). ``WeightBuffer`` packs
a weight image in a thread. ``npu_turn`` holds the NPU's turn between processes (``csrc/artlite/turn.h``) across a
check and the dispatches it decides.

The design's operator (``OverlayOperator``) names its streams: ``ENDPOINTS`` (stream -> (direction, shim
channel)), ``BD_BASE`` (stream -> the first of the shim BDs its transfers use in turn: 16 BDs per shim tile;
``BD_COUNT``: how many, default txn.QUEUE_DEPTH), ``PACKET`` (stream -> the packet id of its transfers on a
packet-switched channel), and checks with ``check_endpoints`` that aiecc gave each stream the channel it names.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import itertools
import os
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, ClassVar, Protocol

import numpy as np

from . import _native, backend, parallel_compile, portable, txn
from ._native import Buffer, Device, DispatchError

SPIN_MARGIN_US = 1000  # a transaction's spin starts this long before its shortest recent duration
# A device's dispatches this far apart: the NPU may have been powered down in between (the driver suspends it 5 s after
# its last job, Linux's ``power/autosuspend_delay_ms``; Windows gates the array after ~4.5 s; a sleep of the whole
# system), and with it the array's memories (``OverlayDevice.set_prologue``). Well under those: a prologue on an array
# that kept its state takes 0.3 ms.
IDLE_S = 1.0


@functools.cache
def irq_cpu() -> int:
    """The CPU that takes the NPU's completion interrupts (the amdxdna mailbox's, the most of them in
    /proc/interrupts), or -1 when unknown. A dispatch polls for its end there (``Transaction``): woken from its
    deepest C-state by the interrupt, that CPU would delay each completion by 0.3 - 0.8 ms."""
    try:
        with open("/proc/interrupts") as f:
            ncpu = len(f.readline().split())
            counts = [0] * ncpu
            for line in f:
                if "xdna_mailbox" in line:
                    for i, v in enumerate(line.split()[1:1 + ncpu]):
                        counts[i] += int(v)
    except (OSError, ValueError):
        return -1
    return counts.index(max(counts)) if any(counts) else -1


@contextlib.contextmanager
def npu_turn() -> Iterator[None]:
    """This process's turn on the NPU meanwhile: no other process's dispatch runs between the dispatches inside (each
    pending dispatch holds it too, until a wait retires it)."""
    _native.turn_enter()
    try:
        yield
    finally:
        _native.turn_leave()


class OverlayOperator(Protocol):
    """What ``Sequence`` and ``OverlayDevice`` need of a design's operator (an ``ArtcOperator``)."""

    ENDPOINTS: ClassVar[dict[str, tuple[int, int]]]
    BD_BASE: ClassVar[dict[str, int]]

    def compile(self, device: object = None) -> object: ...

    def check_endpoints(self, insts: np.ndarray) -> None: ...


class Sequence:
    """The transaction of one dispatch, built from stream transfers.

    ``fill`` / ``drain`` return the word index of the transfer's BD in ``words()`` (``drain``: in a one-element list):
    its length and address may be rewritten there before another dispatch (``txn.bd_words``)."""

    # tile register writes (column, row, register, value) every transaction starts with: performance counters and PC
    # events of the designs driven by host-built transactions (the profiling skill); set before the model builds its
    # transactions (they are kept)
    TILE_WRITES: ClassVar[tuple[tuple[int, int, int, int], ...]] = ()

    def __init__(self, op: OverlayOperator, *, inputs: tuple[str, ...] = ()):
        """Every transfer is issued at once; a channel with 4 queued tasks first waits for its oldest: the transaction
        runs ahead of the array, ``sync`` waits for the drains only. A task carries a task-complete token only when a
        wait needs it (its push is patched when the wait is issued): a channel completes its tasks in order, so one wait
        per channel covers its earlier tasks too, and every waited token costs the firmware ~0.3 us (a sync that waits
        for each of a phase's 2-3 drains per column costs a phase boundary ~5 us). ``inputs``: fill streams the drains
        of every sync depend on (a phase's inputs, not weight runs that span phases): a sync waits for them first, one
        wait per run of columns, while the phase still runs, so the next phase's fills after the sync find empty queues
        instead of each waiting for its column's oldest (Gemma's step: 1366 waits instead of 3751, 0.2 ms less).

        A task the array pushes itself (``device_task``: control packets that write and push BDs the host does not
        use) takes a queue slot without a token: the channel's later host task's token covers it."""
        self.op = op
        self.txn = txn.Txn()
        self.inputs = inputs
        self._bd_count: dict[str, int] = getattr(op, "BD_COUNT", {})
        self._packet: dict[str, int] = getattr(op, "PACKET", {})
        self._next_bd: dict[tuple[int, str], int] = {}
        self._controlled: set[tuple[int, int, int]] = set()
        # per channel the push ops not waited for (None: a task the array pushed)
        self._queued: dict[tuple[int, str], list[int | None]] = {}
        self._array_tasks = False  # (``device_task`` was used: the words do not show those tasks' retirement)
        for col, row, reg, value in self.TILE_WRITES:
            self.txn.write32(txn.tile_address(col, row, reg), value)

    def _controller(self, col: int, direction: int, channel: int) -> None:
        if (col, direction, channel) not in self._controlled:
            self.txn.token_controller(col, direction, channel)
            self._controlled.add((col, direction, channel))

    def fill(self, stream: str, col: int, address: int, nbytes: int, dims: tuple | None = None, repeat: int = 1,
             iteration: tuple[int, int] | None = None, *, packet: bool | int = True,
             acquire: tuple[int, int] | None = None) -> int:
        """Read ``nbytes`` at the device address into a column's A or X stream (``dims``: gathered,
        see ``txn.Txn.shim_bd``), ``repeat`` times in a row (``iteration``: each from a further address). On a
        packet-switched stream (``PACKET``) the fill gets the stream's packet header unless ``packet``
        is False (words that carry their own headers: control packets) or another packet id (a separate route);
        acquire (shim lock, value): the task waits for the lock."""
        assert self.op.ENDPOINTS[stream][0] == txn.MM2S, stream
        pid = self._packet.get(stream) if packet is True else None if packet is False else packet
        return self._task(stream, col, address, nbytes, dims, repeat, iteration, packet=pid, acquire=acquire)

    def drain(self, stream: str, col: int, address: int, nbytes: int, dims: tuple | None = None,
              repeat: int = 1, iteration: tuple[int, int] | None = None, *,
              release: tuple[int, int] | None = None) -> list[int]:
        """Write the next ``nbytes`` of a column's output stream to the device address (``dims``:
        scattered), issued at once; the returned one-element list holds its BD's word index. release (shim lock,
        value): added to the lock when the transfer is done."""
        assert self.op.ENDPOINTS[stream][0] == txn.S2MM, stream
        return [self._task(stream, col, address, nbytes, dims, repeat, iteration, release=release)]

    def _task(self, stream: str, col: int, address: int, nbytes: int, dims: tuple | None, repeat: int,
              iteration: tuple[int, int] | None = None, *, packet: int | None = None,
              acquire: tuple[int, int] | None = None, release: tuple[int, int] | None = None) -> int:
        """A transfer, after waiting for the channel's oldest when 4 are queued (or as many host tasks as the stream
        has BDs)."""
        direction, channel = self.op.ENDPOINTS[stream]
        key = (col, stream)
        self._controller(col, direction, channel)
        n_bd = self._bd_count.get(stream, txn.QUEUE_DEPTH)
        self._room(stream, col, lambda q: len(q) < txn.QUEUE_DEPTH and sum(e is not None for e in q) < n_bd)
        k = self._next_bd.get(key, 0)
        self._next_bd[key] = k + 1
        bd = self.op.BD_BASE[stream] + k % n_bd
        at = self.txn.shim_bd(col, bd, address=address, nbytes=nbytes, dims=dims, iteration=iteration, packet=packet,
                              acquire=acquire, release=release)
        self._queued[key].append(self.txn.n_ops)
        self.txn.shim_push(col, direction, channel, bd, repeat=repeat)
        return at

    def _room(self, stream: str, col: int, ok: object) -> None:
        """Wait for a column's oldest host tasks of a stream (and the array's tasks before them) until its queue is
        ``ok``."""
        direction, channel = self.op.ENDPOINTS[stream]
        queued = self._queued.setdefault((col, stream), [])
        while not ok(queued):
            i = next(i for i, e in enumerate(queued) if e is not None)  # (the array's tasks carry no token)
            self.txn.set_token(queued[i])
            del queued[:i + 1]
            self.txn.sync(col=col, direction=direction, channel=channel)

    def device_task(self, stream: str, col: int) -> None:
        """A task of the stream that the array pushes (into the column's queue, without a token) after the transfers
        issued so far; the queue keeps a slot for it."""
        self._room(stream, col, lambda q: len(q) < txn.QUEUE_DEPTH)
        self._queued.setdefault((col, stream), []).append(None)
        self._array_tasks = True

    def device_done(self, stream: str, col: int) -> None:
        """The array's tasks of the column's stream are complete (a sync waited for outputs that consumed them), and so
        are the tasks queued before them: no wait for those."""
        q = self._queued.get((col, stream), [])
        if None in q:
            del q[:len(q) - q[::-1].index(None)]

    def _retire(self, streams: list[str], only: tuple[int, ...] | None = None) -> None:
        """Wait for every queued task of the streams (only: of those columns): a token on each channel's last
        one, one wait per run of columns."""
        for stream in streams:
            direction, channel = self.op.ENDPOINTS[stream]
            cols = sorted(c for (c, s), q in self._queued.items() if s == stream and q and (only is None or c in only))
            for c in cols:
                last = self._queued[(c, stream)][-1]
                assert last is not None, f"stream {stream}, column {c}: no host task after the array's to wait for"
                self.txn.set_token(last)
                self._queued[(c, stream)].clear()
            start = 0
            for i in range(1, len(cols) + 1):
                if i == len(cols) or cols[i] != cols[i - 1] + 1:
                    self.txn.sync(col=cols[start], direction=direction, channel=channel, ncol=i - start)
                    start = i

    def wait_fills(self, stream: str, cols: tuple[int, ...]) -> None:
        """Wait for the columns' fills of a stream (a fill of another column may follow them into its memtile:
        a packet-switched stream's sources keep no order between them)."""
        self._retire([stream], cols)

    def sync(self, fills: bool = False) -> None:
        """Wait for the drains issued so far (a channel's last drain carries the token), and with ``fills`` for the
        fills: the transaction's end. The fills issued before are complete after the drains as well: their data
        produced these outputs."""
        drains = [s for s, (d, _) in self.op.ENDPOINTS.items() if d == txn.S2MM]
        fill_streams = [s for s, (d, _) in self.op.ENDPOINTS.items() if d == txn.MM2S and s not in self.inputs]
        self._retire([*self.inputs, *drains, *(fill_streams if fills else [])])

    def words(self) -> np.ndarray:
        assert not any(self._queued.values()), "tasks whose tokens are not waited for (sync(fills=True))"
        words = self.txn.words()
        # the queue model sees only the host's tasks: the array's own (``device_task``) retire without a trace
        txn.verify_if_enabled(words, queues=not self._array_tasks)
        return words


class OverlayDevice:
    """The compiled decode design on the NPU and its host buffers."""

    def __init__(self, op: OverlayOperator, *, device: object = None):
        """device: an ``artlite.Device`` to share (buffers of both designs, one per hardware context). An op that
        another OverlayDevice attached keeps its program: the devices' dispatches share its hardware context (a
        second context of the same design would make every switch between them a reconfiguration)."""
        self.op = op
        if op._program is None or device is not None and getattr(op._program, "_device", None) is not device:
            op.compile(device=device)
        prog = op._program
        prog._ensure_loaded()
        op.check_endpoints(np.fromfile(prog.insts_path, np.uint32))
        self.device, self._prog = prog._device, prog
        if backend.ADDRESSES_NEED_CONTEXT:
            prog._kernel()
        self.buffers: dict[str, object] = {}
        # (names that stand for other buffers: a model's selected sequence's, ``decode.model.ModelOverlay.select``)
        self.alias: dict[str, str] = {}
        self._submitted: tuple[Transaction, float] | None = None  # (``submit``: the running one, its start)
        self._prologue: Transaction | None = None
        # (what the devices of one hardware context share: ``stale``)
        self._shared: dict[str, bool] = vars(prog).setdefault("_overlay_shared", {"stale": False})
        self._turns = -1  # (``turn_foreign`` at the last dispatch)
        self._last = -IDLE_S  # (``portable.uptime`` at the last dispatch's end)

    @property
    def stale(self) -> bool:
        """Whether the array may have lost the design's state since its prologue last ran (``set_prologue``): of the
        hardware context, so of every device on it (a model's overlay and its drafter's: either one's prologue brings
        the state back for both, and whoever ran another context marks one of them)."""
        return self._shared["stale"]

    @stale.setter
    def stale(self, value: bool) -> None:
        self._shared["stale"] = value

    def set_prologue(self, words: np.ndarray) -> None:
        """A transaction run before the next dispatch whenever the array may have lost the state the design keeps on it
        beyond its configuration (code pages' stores: ``paging.Store``): another hardware context ran on the device, of
        this process (its owner then sets ``stale``) or of another (the NPU's turn says so: ``npu_turn``), or the
        device's last dispatch ended ``IDLE_S`` ago (the NPU powered down since: the context comes back with its
        configuration alone). Stale from here on."""
        self._prologue = Transaction(self, words)
        self.stale = True

    def open(self) -> None:
        """The hardware context now (else at the first dispatch): before a timed dispatch, as the first context of a
        process wakes the NPU (~40 ms)."""
        self._prog._kernel()

    def _fresh(self) -> None:
        """(In the NPU's turn, before a dispatch.)"""
        turns = _native.turn_foreign()
        if self._prologue is not None:
            idle = portable.uptime() - self._last >= IDLE_S
            if self.stale or idle or turns != self._turns:
                self.stale = False
                try:
                    for _ in range(portable.WAKE_DISPATCHES if idle else 1):
                        self._prog._kernel().dispatch(self._prologue.buffer, list(self.buffers.values()))
                except DispatchError as e:
                    self._note(e, self._prologue, what="the prologue's transaction (code pages)")
                    raise
        self._turns = turns

    def alloc(self, name: str, nbytes: int) -> np.ndarray:
        """A zeroed host buffer the NPU can read and write; returns its uint8 view. The NPU's DMA
        bypasses the CPU caches: ``flush`` what the host wrote before a dispatch reads it, ``fetch``
        what a dispatch wrote before the host reads it."""
        assert name not in self.buffers and nbytes % 4 == 0, name
        self.buffers[name] = self.device.alloc_host_buffer(nbytes)
        return self.buffers[name].map()

    def alloc_all(self, sizes: Mapping[str, int]) -> None:
        """Zeroed host buffers by name (as ``alloc``), those of ``sizes`` the device lacks, allocated together. On the
        XRT backend buffers allocated together are one XRT buffer of exactly their size, and a command lists at most 5
        (``backend``): an overlay's buffers and those of its prefill cost a command one, whatever their sizes, where a
        buffer allocated alone lies in a shared 1 GB region that is pinned whole, or in a dedicated one from 256 MB on."""
        new = {name: n for name, n in sizes.items() if name not in self.buffers}
        assert all(n % 4 == 0 for n in new.values()), new
        assert all(self.buffers[name].size >= n for name, n in sizes.items() if name not in new), sizes
        self.buffers |= zip(new, self.device.alloc_host_buffers(list(new.values())))

    def adopt(self, name: str, buffer: object) -> np.ndarray:
        """A buffer allocated elsewhere on this device (e.g. in a thread); returns its uint8 view."""
        assert name not in self.buffers, name
        self.buffers[name] = buffer
        return buffer.map()

    def named(self, name: str) -> object:
        """The buffer a name stands for (another name's buffer where ``alias`` maps it, else the name's own)."""
        return self.buffers[self.alias.get(name, name)]

    def view(self, name: str, *, own: bool = False) -> np.ndarray:
        """A buffer's bytes (``own``: of the buffer of that name itself, whatever the name stands for: ``alias``)."""
        return (self.buffers[name] if own else self.named(name)).map()

    def flush(self, name: str, offset: int = 0, nbytes: int | None = None, *, own: bool = False) -> None:
        """Host writes of a buffer range -> DDR (``own``: ``view``'s)."""
        b = self.buffers[name] if own else self.named(name)
        b.sync_to_device(b.size - offset if nbytes is None else nbytes, offset)

    def flush_lines(self, name: str, offsets: np.ndarray) -> None:
        """Host writes of a buffer's scattered bytes (byte offsets, in any order) -> DDR, line by line: per S and
        position the steps rewrite a few bytes of buffers whose whole span would take milliseconds to flush."""
        self.named(name).sync_lines(np.unique(np.asarray(offsets, np.int64).ravel() & ~63))

    def flush_at(self, name: str, lines: np.ndarray) -> None:
        """``flush_lines`` of line offsets given without repeats (int64)."""
        self.named(name).sync_lines(lines.ravel())

    def fetch(self, name: str, offset: int = 0, nbytes: int | None = None, *, own: bool = False) -> None:
        """Drop the CPU's cached copy of a buffer range the NPU wrote (``own``: ``view``'s)."""
        b = self.buffers[name] if own else self.named(name)
        b.sync_from_device(b.size - offset if nbytes is None else nbytes, offset)

    def address(self, name: str, offset: int = 0, *, own: bool = False) -> int:
        """Device address of a byte of a buffer (for ``Sequence.fill`` / ``drain``); ``own``: of the buffer of that
        name itself, whatever the name stands for (``alias``)."""
        return txn.device_address(self.buffers[name] if own else self.named(name), offset, self._prog._ctx)

    def dispatch(self, transaction: Transaction) -> None:
        """Run a transaction; blocking. The DispatchError of a transaction the driver's hang detection stopped names
        the operations where the firmware was."""
        with npu_turn():
            self._fresh()  # (_fresh reports a prologue failure under the prologue's name)
            try:
                t0 = time.perf_counter()
                self._prog._kernel().dispatch(transaction.buffer, list(self.buffers.values()),
                                              spin_after_us=transaction.spin_after_us(), spin_cpu=irq_cpu())
            except DispatchError as e:
                self._note(e, transaction)
                raise
            finally:
                self._last = portable.uptime()
        transaction.durations.append(time.perf_counter() - t0)

    def submit(self, transaction: Transaction) -> None:
        """Start a transaction and return: ``wait`` waits for it (the host prepares the next one meanwhile)."""
        assert self._submitted is None, "a transaction is running"
        with npu_turn():
            self._fresh()
            self._submitted = (transaction, time.perf_counter())
            self._prog._kernel().dispatch_async(transaction.buffer, list(self.buffers.values()))

    def wait(self) -> None:
        """Wait for the submitted transaction (``submit``), polling near its end as ``dispatch`` does."""
        transaction, t0 = self._submitted
        self._submitted = None
        try:
            self._prog._kernel().dispatch_wait(spin_after_us=transaction.spin_after_us(), spin_cpu=irq_cpu())
            transaction.durations.append(time.perf_counter() - t0)
        except DispatchError as e:
            self._note(e, transaction)
            raise
        finally:
            self._last = portable.uptime()

    def _note(self, e: DispatchError, transaction: Transaction, *, what: str = "a transaction") -> None:
        """Add where the firmware was to a DispatchError (``txn.hang_note``): in the transaction ``transaction`` (a
        step's, or the prologue's). Never raises: a note failing would mask the error."""
        try:
            e.add_note(txn.hang_note(transaction.words, e.health, what=what))
        except Exception as note_error:  # noqa: BLE001 (the dispatch's error is the one to see)
            e.add_note(f"(no transaction position: {note_error!r})")


_SPARE: list[Callable[[], bool]] = []  # (``heap_spare``)


def heap_spare(release: Callable[[], bool]) -> None:
    """Name device-heap buffers the process gives up before any kept transaction where the heap is full and cannot grow
    (Linux before 7.2: one heap of 64 MB per process): ``release()`` frees them and returns whether it freed anything
    (buffers that cost little to have again: the Library programs' pieces)."""
    _SPARE.append(release)


def _reclaim() -> bool:
    """(``_native.set_heap_reclaim``: a device-heap buffer does not fit a heap that cannot grow) Free one thing the
    process can have again: what ``heap_spare`` named, else the least recently used of all kept transactions
    (``Transactions``); whether it freed anything (called again while the buffer does not fit)."""
    if any(release() for release in _SPARE):
        return True
    kept = [t for t in Transactions.all if len(t)]
    if not kept:
        return False
    min(kept, key=Transactions.oldest).drop()
    return True


_native.set_heap_reclaim(_reclaim)


class Transactions:
    """What an owner keeps to run again, by key (a model's steps: per kind of step the object that holds its
    ``Transaction`` and what patches it): each transaction lies in a separate device-heap buffer, and a model's
    kinds of steps together take tens of MB (Qwen3.6 35B-A3B with its drafter: 52 of 0.9 MB). Where the device heap is
    full and cannot grow, the process gives them up one by one, the least recently used of all owners' first
    (``_reclaim``): a key that is no longer here is built again by its owner. Reading a key marks it used."""

    all: ClassVar[weakref.WeakSet[Transactions]] = weakref.WeakSet()
    _uses: ClassVar[Iterator[int]] = itertools.count()

    def __init__(self) -> None:
        self._kept: dict[Hashable, tuple[int, Any]] = {}  # (key -> its last use, the object; the least recent first)
        Transactions.all.add(self)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._kept

    def __len__(self) -> int:
        return len(self._kept)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(list(self._kept))

    def __getitem__(self, key: Hashable) -> Any:
        value = self._kept.pop(key)[1]
        self._kept[key] = (next(self._uses), value)
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._kept.pop(key, None)
        self._kept[key] = (next(self._uses), value)

    def oldest(self) -> int:
        """The last use of its least recently used key (not empty)."""
        return next(iter(self._kept.values()))[0]

    def drop(self) -> None:
        """Give up its least recently used key (not empty)."""
        del self._kept[next(iter(self._kept))]


class Transaction:
    """A transaction in a dedicated instruction buffer: ``words`` is the buffer's uint32 view
    (rewrite BD words there, then ``flush``); ``durations``: its last dispatches' (s)."""

    def __init__(self, dev: OverlayDevice, words: np.ndarray):
        txn.verify_if_enabled(words, queues=False)
        self.buffer = dev.device.alloc_insts_buffer(words.nbytes)
        self.words = self.buffer.map().view(np.uint32)[:words.size]
        self.words[:] = words
        self.durations: collections.deque[float] = collections.deque(maxlen=8)
        self.flush()

    def spin_after_us(self) -> int:
        """When a dispatch starts to poll for its end (``irq_cpu``): SPIN_MARGIN_US before its shortest recent
        duration (-1: no spin, before it ran or without the interrupts' CPU)."""
        if not self.durations or irq_cpu() < 0:
            return -1
        return max(0, int(min(self.durations) * 1e6) - SPIN_MARGIN_US)

    def flush(self) -> None:
        self.buffer.sync_to_device(self.words.nbytes, 0)

    def flush_lines(self, lines: np.ndarray) -> None:
        """Flush the lines (byte offsets, int64, no repeats) that a step patched: a step's control code is megabytes."""
        self.buffer.sync_lines(lines)


class WeightBuffer:
    """A weight image's buffer (a user-memory BO, ``Device.alloc_user_buffer``), allocated and packed in a thread from
    construction on: an app starts it first and hands it to its overlay (``weights=``); its tokenizer, prompt, designs
    and the overlay's other buffers meanwhile (the allocation and the C++ packer release the GIL). ``image``: an
    ``decode.image.TileImage``.
    ``threads``: the packer's (default: 3/4 of the physical cores: the 27B's pack still ends before the app's main
    thread, which all 16 slow down by ~0.1 s)."""

    def __init__(self, image: object, *, threads: int = 0):
        self.image = image
        threads = threads or max(1, (os.cpu_count() or 2) * 3 // 8)
        self._allocated = threading.Event()
        self._out: list = []
        self._thread = threading.Thread(target=self._run, args=(threads,), name="weights", daemon=True)
        self._thread.start()

    def _run(self, threads: int) -> None:
        try:
            parallel_compile.no_device()  # (planning mode: PlanStop, raised by buffer() and wait())
            dev = Device()
            buf = dev.alloc_user_buffer(self.image.nbytes)
            self._out = [dev, buf]
            self._allocated.set()
            if self.image.pack(buf.map(), threads=threads):  # (else non-temporal stores only)
                buf.sync_to_device()
        except BaseException as e:  # noqa: BLE001 (raised by buffer() and wait())
            self._out = [e]
            self._allocated.set()

    def buffer(self) -> tuple[Device, Buffer]:
        """(the device, the buffer) once allocated (its packing may still run: ``wait``)."""
        self._allocated.wait()
        if isinstance(self._out[0], BaseException):
            raise self._out[0]
        return self._out[0], self._out[1]

    def wait(self) -> None:
        """Until the image is packed."""
        self._thread.join()
        if isinstance(self._out[0], BaseException):
            raise self._out[0]
