"""Host-side runtime of the overlays, the designs that run on host-built transactions (``artlite.txn``): the decode
overlay (``decode.design`` and its models'), the vision overlay (``vision.design``).

``Sequence`` turns a list of DMA transfers (fills of a column's input streams, drains of its output streams) into
the transaction of a dispatch; ``sync`` closes a group of transfers that must complete before the next group is
issued (an output that the next phase reads back from DDR). ``OverlayDevice`` owns the compiled design's hardware
context and dispatches transactions on host buffers; ``Transaction`` keeps one in an instruction buffer of its
own, to be patched (``txn.bd_words``) and dispatched again (its last durations time a spin on the CPU of the NPU's
interrupts near its end: ``irq_cpu``). ``WeightBuffer`` packs a weight image in a thread.

The design's operator (``OverlayOperator``) names its streams: ``ENDPOINTS`` (stream -> (direction, shim
channel)), ``BD_BASE`` (stream -> the first of the shim BDs its transfers use in turn: 16 BDs per shim tile;
``BD_COUNT``: how many, default txn.QUEUE_DEPTH), ``PACKET`` (stream -> the packet id of its transfers on a
packet-switched channel), and checks with ``check_endpoints`` that aiecc gave each stream the channel it names.
"""

from __future__ import annotations

import collections
import functools
import os
import threading
import time
from typing import ClassVar, Protocol

import numpy as np

from . import backend, parallel_compile, txn
from ._native import Buffer, Device, DispatchError

SPIN_MARGIN_US = 1000  # a transaction's spin starts this long before its shortest recent duration


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


class OverlayOperator(Protocol):
    """What ``Sequence`` and ``OverlayDevice`` need of a design's operator (an ``ArtcOperator``)."""

    ENDPOINTS: ClassVar[dict[str, tuple[int, int]]]
    BD_BASE: ClassVar[dict[str, int]]

    def compile(self, device: object = None) -> object: ...

    def check_endpoints(self, insts: np.ndarray) -> None: ...


class Sequence:
    """The transaction of one dispatch, built from stream transfers.

    ``fill`` / ``drain`` return the word index of the transfer's BD in ``words()``: its length and
    address may be rewritten there before another dispatch (``txn.bd_words``)."""

    def __init__(self, op: OverlayOperator, *, rolling: bool = False, inputs: tuple[str, ...] = ()):
        """rolling: every transfer is issued at once; a channel with 4 queued tasks first waits for its oldest
        (instead of a sync every 4 transfers and fills that wait for the fourth): the transaction runs ahead of the
        array, ``sync`` waits for the drains only. A task carries a task-complete token only when a wait needs it
        (its push is patched when the wait is issued): a channel completes its tasks in order, so one wait per
        channel covers its earlier tasks too, and every waited token costs the firmware ~0.3 us (a sync that waits
        for each of a phase's 2-3 drains per column costs a phase boundary ~5 us). ``inputs`` (rolling): fill streams
        the drains of every sync depend on (a phase's inputs, not weight runs that span phases): a sync waits for
        them first, one wait per run of columns, while the phase still runs, so the next phase's fills after the
        sync find empty queues instead of each waiting for its column's oldest (Gemma's step: 1366 waits instead of
        3751, 0.2 ms less).

        A task the array pushes itself (``device_task``: control packets that write and push BDs the host does not
        use) takes a queue slot without a token: the channel's later host task's token covers it."""
        self.op = op
        self.txn = txn.Txn()
        self.rolling = rolling
        self.inputs = inputs
        self._bd_count: dict[str, int] = getattr(op, "BD_COUNT", {})
        self._packet: dict[str, int] = getattr(op, "PACKET", {})
        self._next_bd: dict[tuple[int, str], int] = {}
        self._controlled: set[tuple[int, int, int]] = set()
        self._drains: list[tuple[str, int, int, int, tuple | None, list[int]]] = []
        self._pending: dict[tuple[int, str], int] = {}  # transfers issued since the channel's last sync
        self._fill_tokens: set[tuple[int, str]] = set()  # fills whose task-complete token is not waited for yet
        # rolling: per channel the push ops not waited for (None: a task the array pushed)
        self._queued: dict[tuple[int, str], list[int | None]] = {}

    def _bd(self, stream: str, col: int) -> int:
        k = self._next_bd.get((col, stream), 0)
        self._next_bd[(col, stream)] = k + 1
        n = self._pending.get((col, stream), 0) + 1
        assert n <= txn.QUEUE_DEPTH, (f"more than {txn.QUEUE_DEPTH} transfers of stream {stream}, column {col} "
                                      f"without a sync")
        self._pending[(col, stream)] = n
        return self.op.BD_BASE[stream] + k % txn.QUEUE_DEPTH

    def _controller(self, col: int, direction: int, channel: int) -> None:
        if (col, direction, channel) not in self._controlled:
            self.txn.token_controller(col, direction, channel)
            self._controlled.add((col, direction, channel))

    def _wait_fill(self, stream: str, col: int) -> None:
        """Wait for the fill of a stream that carries a token (the fourth since the last sync): the
        channel's queue is then empty."""
        assert not self._drains, "a full fill queue while drains wait for a sync"
        self.txn.sync(col=col, direction=txn.MM2S, channel=self.op.ENDPOINTS[stream][1])
        self._fill_tokens.discard((col, stream))
        self._pending[(col, stream)] = 0

    def fill(self, stream: str, col: int, address: int, nbytes: int, dims: tuple | None = None, repeat: int = 1,
             iteration: tuple[int, int] | None = None, *, packet: bool = True,
             acquire: tuple[int, int] | None = None) -> int:
        """Read ``nbytes`` at the device address into a column's A or X stream (``dims``: gathered,
        see ``txn.Txn.shim_bd``), ``repeat`` times in a row (``iteration``: each from a further address). A
        channel queues 4 tasks: the fourth fill since the last sync carries a token, a fifth one first waits
        for it. On a packet-switched stream (``PACKET``) the fill gets the stream's packet header unless ``packet``
        is False (words that carry their own headers: control packets); acquire (shim lock, value): the task waits
        for the lock (rolling)."""
        direction, channel = self.op.ENDPOINTS[stream]
        assert direction == txn.MM2S, stream
        if self.rolling:
            return self._task(stream, col, address, nbytes, dims, repeat, iteration,
                              packet=self._packet.get(stream) if packet else None, acquire=acquire)
        assert stream not in self._packet and acquire is None, "packet-switched streams, locks: rolling sequences"
        if self._pending.get((col, stream), 0) == txn.QUEUE_DEPTH:
            self._wait_fill(stream, col)
        token = self._pending.get((col, stream), 0) == txn.QUEUE_DEPTH - 1
        if token:
            self._controller(col, direction, channel)
            self._fill_tokens.add((col, stream))
        return self.txn.shim_task(col, direction, channel, self._bd(stream, col), address=address, nbytes=nbytes,
                                  dims=dims, repeat=repeat, token=token, iteration=iteration)

    def drain(self, stream: str, col: int, address: int, nbytes: int, dims: tuple | None = None,
              repeat: int = 1, iteration: tuple[int, int] | None = None, *,
              release: tuple[int, int] | None = None) -> list[int]:
        """Write the next ``nbytes`` of a column's output stream to the device address (``dims``:
        scattered). The transfer is issued by the next ``sync`` (rolling: at once); the returned
        one-element list then holds its BD's word index. release (shim lock, value): added to the lock when the
        transfer is done (rolling)."""
        assert self.op.ENDPOINTS[stream][0] == txn.S2MM, stream
        handle: list[int] = []
        if self.rolling:
            handle.append(self._task(stream, col, address, nbytes, dims, repeat, iteration, release=release))
            return handle
        assert repeat == 1 and iteration is None and release is None, "repeated drains, locks: rolling sequences"
        self._drains.append((stream, col, address, nbytes, dims, handle))
        return handle

    def _task(self, stream: str, col: int, address: int, nbytes: int, dims: tuple | None, repeat: int,
              iteration: tuple[int, int] | None = None, *, packet: int | None = None,
              acquire: tuple[int, int] | None = None, release: tuple[int, int] | None = None) -> int:
        """Rolling: a transfer, after waiting for the channel's oldest when 4 are queued (or as many host tasks as
        the stream has BDs)."""
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
        """Rolling: wait for a column's oldest host tasks of a stream (and the array's tasks before them) until its
        queue is ``ok``."""
        direction, channel = self.op.ENDPOINTS[stream]
        queued = self._queued.setdefault((col, stream), [])
        while not ok(queued):
            i = next(i for i, e in enumerate(queued) if e is not None)  # (the array's tasks: no token of their own)
            self.txn.set_token(queued[i])
            del queued[:i + 1]
            self.txn.sync(col=col, direction=direction, channel=channel)

    def device_task(self, stream: str, col: int) -> None:
        """Rolling: a task of the stream that the array pushes (into the column's queue, without a token) after the
        transfers issued so far; the queue keeps a slot for it."""
        assert self.rolling
        self._room(stream, col, lambda q: len(q) < txn.QUEUE_DEPTH)
        self._queued[(col, stream)].append(None)

    def device_done(self, stream: str, col: int) -> None:
        """Rolling: the array's tasks of the column's stream are complete (a sync waited for outputs that consumed
        them), and so are the tasks queued before them: no wait for those."""
        q = self._queued.get((col, stream), [])
        if None in q:
            del q[:len(q) - q[::-1].index(None)]

    def _retire(self, streams: list[str]) -> None:
        """Rolling: wait for every queued task of the streams: a token on each channel's last one, one wait per run
        of columns."""
        for stream in streams:
            direction, channel = self.op.ENDPOINTS[stream]
            cols = sorted(c for (c, s), q in self._queued.items() if s == stream and q)
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

    def sync(self, fills: bool = False) -> None:
        """Issue the drains collected since the last sync and wait for them (a channel's last
        drain carries the token). The fills issued before are complete then as well: their data
        produced these outputs. Rolling: wait for the drains issued so far (and with ``fills`` for
        the fills: the transaction's end)."""
        if self.rolling:
            drains = [s for s, (d, _) in self.op.ENDPOINTS.items() if d == txn.S2MM]
            fill_streams = [s for s, (d, _) in self.op.ENDPOINTS.items() if d == txn.MM2S and s not in self.inputs]
            self._retire([*self.inputs, *drains, *(fill_streams if fills else [])])
            return
        last = {(stream, col): i for i, (stream, col, *_) in enumerate(self._drains)}
        waits: dict[int, list[int]] = {}
        for i, (stream, col, address, nbytes, dims, handle) in enumerate(self._drains):
            direction, channel = self.op.ENDPOINTS[stream]
            self._controller(col, direction, channel)
            token = last[(stream, col)] == i
            handle.append(self.txn.shim_task(col, direction, channel, self._bd(stream, col), address=address,
                                             nbytes=nbytes, token=token, dims=dims))
            if token:
                waits.setdefault(channel, []).append(col)
        for channel, cols in waits.items():  # one wait per run of adjacent columns
            cols.sort()
            start = 0
            for i in range(1, len(cols) + 1):
                if i == len(cols) or cols[i] != cols[i - 1] + 1:
                    self.txn.sync(col=cols[start], direction=txn.S2MM, channel=channel, ncol=i - start)
                    start = i
        self._drains.clear()
        for col, stream in sorted(self._fill_tokens):  # (complete by now: their tokens must not linger)
            self.txn.sync(col=col, direction=txn.MM2S, channel=self.op.ENDPOINTS[stream][1])
        self._fill_tokens.clear()
        self._pending.clear()

    def words(self) -> np.ndarray:
        assert not self._drains and not self._fill_tokens, "drains or tokens without a sync"
        assert not any(self._queued.values()), "tasks whose tokens are not waited for (sync(fills=True))"
        return self.txn.words()


class OverlayDevice:
    """The compiled decode design on the NPU and its host buffers."""

    def __init__(self, op: OverlayOperator, *, max_instruction_bytes: int = 1 << 22, device: object = None):
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
        self._insts = self.device.alloc_insts_buffer(max_instruction_bytes)
        self._loaded = np.zeros(0, np.uint32)  # (the words load() wrote)
        self.buffers: dict[str, object] = {}
        self._submitted: tuple[Transaction, float] | None = None  # (``submit``: the running one, its start)
        self._prologue: Transaction | None = None
        self.stale = False  # (``set_prologue``)

    def set_prologue(self, words: np.ndarray) -> None:
        """A transaction run before the next dispatch whenever ``stale``: state the design keeps on the array beyond its
        configuration (code pages' stores: ``paging.Store``), lost when another hardware context ran on the device (its
        owner then sets ``stale``). Stale from here on."""
        self._prologue = Transaction(self, words)
        self.stale = True

    def open(self) -> None:
        """The hardware context now (else at the first dispatch): before a timed dispatch, as the first context of a
        process wakes the NPU (~40 ms)."""
        self._prog._kernel()

    def _fresh(self) -> None:
        if self.stale and self._prologue is not None:
            self.stale = False
            self._prog._kernel().dispatch(self._prologue.buffer, list(self.buffers.values()))

    def alloc(self, name: str, nbytes: int) -> np.ndarray:
        """A zeroed host buffer the NPU can read and write; returns its uint8 view. The NPU's DMA
        bypasses the CPU caches: ``flush`` what the host wrote before a dispatch reads it, ``fetch``
        what a dispatch wrote before the host reads it."""
        assert name not in self.buffers and nbytes % 4 == 0, name
        self.buffers[name] = self.device.alloc_host_buffer(nbytes)
        return self.buffers[name].map()

    def adopt(self, name: str, buffer: object) -> np.ndarray:
        """A buffer allocated elsewhere on this device (e.g. in a thread); returns its uint8 view."""
        assert name not in self.buffers, name
        self.buffers[name] = buffer
        return buffer.map()

    def view(self, name: str) -> np.ndarray:
        return self.buffers[name].map()

    def flush(self, name: str, offset: int = 0, nbytes: int | None = None) -> None:
        """Host writes of a buffer range -> DDR."""
        b = self.buffers[name]
        b.sync_to_device(b.size - offset if nbytes is None else nbytes, offset)

    def flush_lines(self, name: str, offsets: np.ndarray) -> None:
        """Host writes of a buffer's scattered bytes (byte offsets, in any order) -> DDR, line by line: per S and
        position the steps rewrite a few bytes of buffers whose whole span would take milliseconds to flush."""
        self.buffers[name].sync_lines(np.unique(np.asarray(offsets, np.int64).ravel() & ~63))

    def flush_at(self, name: str, lines: np.ndarray) -> None:
        """``flush_lines`` of line offsets given without repeats (int64)."""
        self.buffers[name].sync_lines(lines.ravel())

    def fetch(self, name: str, offset: int = 0, nbytes: int | None = None) -> None:
        """Drop the CPU's cached copy of a buffer range the NPU wrote."""
        b = self.buffers[name]
        b.sync_from_device(b.size - offset if nbytes is None else nbytes, offset)

    def address(self, name: str, offset: int = 0) -> int:
        """Device address of a byte of a buffer (for ``Sequence.fill`` / ``drain``)."""
        return txn.device_address(self.buffers[name], offset, self._prog._ctx)

    def load(self, words: np.ndarray) -> None:
        """Make ``words`` the transaction of the following dispatches (``dispatch()``)."""
        assert words.nbytes <= self._insts.size, f"transaction of {words.nbytes} B"
        self._insts.map().view(np.uint32)[:words.size] = words
        self._insts.sync_to_device(words.nbytes, 0)
        self._loaded = self._insts.map().view(np.uint32)[:words.size]

    def dispatch(self, transaction: Transaction | None = None) -> None:
        """Run a transaction (default: the loaded one); blocking. The DispatchError of a transaction
        the driver's hang detection stopped names the operations where the firmware was."""
        try:
            self._fresh()
            if transaction is None:
                self._prog._kernel().dispatch(self._insts, list(self.buffers.values()))
                return
            t0 = time.perf_counter()
            self._prog._kernel().dispatch(transaction.buffer, list(self.buffers.values()),
                                          spin_after_us=transaction.spin_after_us(), spin_cpu=irq_cpu())
            transaction.durations.append(time.perf_counter() - t0)
        except DispatchError as e:
            self._note(e, transaction)
            raise

    def submit(self, transaction: Transaction) -> None:
        """Start a transaction and return: ``wait`` waits for it (the host prepares the next one meanwhile)."""
        assert self._submitted is None, "a transaction is running"
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

    def _note(self, e: DispatchError, transaction: Transaction | None) -> None:
        if e.health is not None:
            ops = txn.decode(self._loaded if transaction is None else transaction.words)
            i = e.health["txn_op_idx"]
            e.add_note(f"transaction operations {max(i - 1, 0)} .. {i + 1} of {len(ops)} (txn_op_idx {i}): "
                       f"{ops[max(i - 1, 0):i + 2]}")


class Transaction:
    """A transaction in an instruction buffer of its own: ``words`` is the buffer's uint32 view
    (rewrite BD words there, then ``flush``); ``durations``: its last dispatches' (s)."""

    def __init__(self, dev: OverlayDevice, words: np.ndarray):
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
    thread, which all 16 slow down by ~0.1 s), ``release``: give back the GGUF's pages of each packed tensor."""

    def __init__(self, image: object, *, threads: int = 0, release: bool = True):
        self.image = image
        threads = threads or max(1, (os.cpu_count() or 2) * 3 // 8)
        self._allocated = threading.Event()
        self._out: list = []
        self._thread = threading.Thread(target=self._run, args=(threads, release), name="weights", daemon=True)
        self._thread.start()

    def _run(self, threads: int, release: bool) -> None:
        try:
            parallel_compile.no_device()  # (planning mode: PlanStop, raised by buffer() and wait())
            dev = Device()
            buf = dev.alloc_user_buffer(self.image.nbytes)
            self._out = [dev, buf]
            self._allocated.set()
            if self.image.pack(buf.map(), threads=threads, release=release):  # (else non-temporal stores only)
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
