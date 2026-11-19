"""Several sequences of one model at once: the lanes' decode rounds joined into steps (``Rounds``), the device to one
lane alone for everything else (``alone``), a failure for all of them (``DeviceFailed``).

A model whose overlay holds N sequences (``artlite.decode.model.ModelOverlay(sequences=)``) has N lanes
(``OverlayModel.lanes``: the model object and N - 1 shallow copies of it, each with its sequence, position, counters
and planner), each driven by a thread of its own. A lane's generation round asks for its step here (``Rounds.join``)
instead of dispatching it: the requests of the lanes that generate become one step of several sequences' slots
(``artlite.decode.step.Part``), so the weights stream once for all of them, and each lane gets its rows. A slot's
numbers do not depend on what shares its step: which requests meet in a round changes times, never tokens.

Rounds. A lane that got a round's rows is expected to ask again: the next round waits for it, up to ``WAIT`` after its
rows (or the step's time over the lanes + 1: waiting d for a lane costs every other lane d, not waiting costs the late
one the rest of a step), and leaves without it then. The time counts from the rows' release, not from the round's first
request: the lanes' host work between two rounds is serialized by the interpreter lock, so their requests arrive one
after another, and a timer from the first would close every round after half of them. A lane is no longer expected once
its generation ends or is abandoned (``idle``), while it has the device alone, or after ``LATE`` requests in a row that
came later than that (a lane whose host work is slow asks when it is ready and takes the next round: the others stop
paying a wait per round for it; one late request alone changes nothing, so a hiccup does not put a lane out of step);
with no lane to wait for, a request runs at once. The thread whose request completes a round runs its dispatch and hands
the others their rows (a lane alone pays no thread hop). A round's parts are in lane order (a step patched ahead runs as
patched only for the same parts); requests of more slots than a step has wait a round, those that waited first, then
those passed over longest ago.

Rows. The overlay's logits are a view of the NPU's one buffer, and a lane reads its rows while the next round may
run: each request gets copies (a screened head's: ``Screened.take``), valid until that lane's next request.

The device alone. A prompt's chunks and steps, the state copies and every other dispatch of one sequence run with the
device held (``alone``: re-entrant per thread, the lane's sequence selected first, ``ModelOverlay.select``), in arrival
order with the rounds: a round and a holder never overlap, and a waiting holder goes before the next round is
assembled. The selection means something only there: a round's step names its sequences itself. A long holder lets
the others in between two parts of its work: whoever waits, between a prompt's chunks (``breathe``), and the lanes'
round between a chunk's pieces (``lend``, ``waiting``: the chunk then runs in short pieces,
``artlite.prefill.chunks.PEER_PIECE_S``) once the holder has had the device for as long as the round's step takes, so
a prompt and the lanes that generate beside it take turns of about the same time.

Failures. A dispatch that fails leaves the device in a state nobody knows, for every sequence (a step's rings are
marked before it ends): the error goes to every lane of the round, and every lane's calls raise ``DeviceFailed`` until
that lane starts over (``OverlayModel.recover``).
"""

import collections
import contextlib
import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from artlite import DispatchError, _native
from artlite.decode.screen import Screened
from artlite.decode.step import Part

log = logging.getLogger("artlite.lanes")

WAIT = 0.003  # seconds a round waits for an expected lane at least (after its last rows)
LATE = 2  # a lane's late requests in a row after which the rounds no longer wait for it
HISTORY = 4096  # the rounds ``Rounds.history`` keeps


class DeviceFailed(RuntimeError):
    """A dispatch failed while the device holds several sequences: its state is unknown for all of them; a lane's
    calls raise this until it starts over (``OverlayModel.recover``)."""


def alone[F: Callable[..., Any]](method: F) -> F:
    """A method of a model on its overlay that runs with the device to its lane alone (``Rounds.alone``; a model of
    one lane: as it is)."""
    @functools.wraps(method)
    def run(self: Any, *args: Any, **kw: Any) -> Any:
        rounds = self._rounds
        if rounds is None:
            return method(self, *args, **kw)
        with rounds.alone(self.seq):
            return method(self, *args, **kw)
    return run  # type: ignore[return-value]


def kept(logits: np.ndarray | Screened, a: int = 0, b: int | None = None) -> np.ndarray | Screened:
    """A step's logits of slots a .. b - 1 (default: all) as copies: valid past the overlay's next dispatch."""
    b = len(logits) if b is None else b
    return logits.take(a, b) if isinstance(logits, Screened) else np.array(logits[a:b])


@dataclass(eq=False)
class Request:
    """A lane's round: ``n`` slots of sequence ``seq`` at ``position``, their inputs ``args`` (the overlay step's
    arguments before the position, each per slot); then its result: ``rows`` (the slots' logits, copies), ``wall``
    (the dispatch's seconds), ``seconds`` (those less its one-time host work and other processes' turns: a step's
    time) and ``parts`` (the sequences of its step)."""

    seq: int
    args: tuple
    position: int
    n: int
    asked: float = 0.0
    gap: float = math.inf  # (seconds from the lane's last rows to this request)
    waited: bool = False  # (a round left it out: its slots did not fit)
    rows: Any = None
    wall: float = 0.0
    seconds: float = 0.0
    parts: int = 0
    error: BaseException | None = None
    done: bool = False


class Round(NamedTuple):
    """A round as it ran (``Rounds.history``): per part its sequence, slots and the seconds from its lane's last rows to
    its request; the lanes it waited for in vain; the seconds the free device waited for the round's requests; the
    dispatch's seconds and its one-time host work."""

    parts: tuple[tuple[int, int, float], ...]
    late: tuple[int, ...]
    idle: float
    wall: float
    once: float


_ROUND = object()  # (the round's place in the device's queue)


class Rounds:
    """The lanes of a model on ``overlay`` (an ``artlite.decode.model.ModelOverlay`` of several sequences): module
    docstring. ``join``: a lane's round; ``alone``: the device to a lane; ``idle``: a lane stops asking; ``budget``: a
    lane's slots of the next round; ``recover``: a lane starts over after a failure."""

    def __init__(self, overlay: Any):
        self.overlay = overlay
        n = overlay.sequences
        self._cv = threading.Condition(threading.Lock())
        self._pending: dict[int, Request] = {}  # (lane -> its request)
        self._expected: set[int] = set()  # (the lanes the next round waits for: they got rows and have not asked)
        self._released = [-math.inf] * n  # (per lane: when its last rows were handed out; -inf: it asked for none since)
        self._late = [0] * n  # (per lane: its requests in a row that came after the round's wait for it)
        self._passed = [-1] * n  # (per lane: the last round that left its request out)
        self._queue: collections.deque[object] = collections.deque()  # (who waits for the device, in arrival order)
        self._queued = False  # (the round is in the queue: it is while requests are pending)
        self._running = False  # (a round's dispatch runs: of the lanes ``_busy``)
        self._busy: frozenset[int] = frozenset()
        self._owner: int | None = None  # (the thread that has the device alone, and how deep)
        self._depth = 0
        self._lent = False  # (that thread lets the waiting round run now: ``lend``)
        self._failures = 0
        self._free = 0.0  # (when the device was last given back)
        self._step_s = 0.0  # (the last round's dispatch)
        self._failure: BaseException | None = None  # (the last failed dispatch's error)
        self._down: set[int] = set()  # (the lanes that have not started over since)
        self.rounds = 0
        self.history: collections.deque[Round] = collections.deque(maxlen=HISTORY)

    # ------------------------------------------------------------------
    # A lane's side
    # ------------------------------------------------------------------

    def budget(self, seq: int) -> int:
        """The slots lane ``seq`` plans its next round for: a step's over the lanes that ask for rounds, 1 at least."""
        with self._cv:
            lanes = len(self._expected | self._pending.keys() | self._busy | {seq})
        return max(self.overlay.SLOTS // lanes, 1)

    def idle(self, seq: int) -> None:
        """Lane ``seq`` asks for no round until its next request (its generation ended or was abandoned)."""
        with self._cv:
            self._quiet(seq)

    def _quiet(self, seq: int) -> None:
        """Lane ``seq`` asks for no round for now: the round that waits for it goes on."""
        self._released[seq], self._late[seq] = -math.inf, 0
        if seq in self._expected:
            self._expected.discard(seq)
            self._cv.notify_all()

    def mine(self) -> bool:
        """Whether the calling thread has the device alone."""
        return self._owner == threading.get_ident()

    def down(self, seq: int) -> bool:
        """Whether lane ``seq`` has not started over since a dispatch failed (its calls raise ``DeviceFailed``)."""
        return seq in self._down

    def join(self, seq: int, args: tuple, position: int, n: int) -> Request:
        """Lane ``seq``'s round of n slots at ``position`` (``Request``), in a step with the other lanes' rounds; blocks
        until its rows are there. ``DeviceFailed``: the step failed, or a dispatch failed before and the lane has not
        started over (``recover``)."""
        req = Request(seq, args, position, n)
        cv = self._cv
        with cv:
            self._check(seq)
            assert seq not in self._pending, f"lane {seq} has a round waiting"
            req.asked = time.perf_counter()
            req.gap = req.asked - self._released[seq]
            # (late: after the wait for it; a lane late twice in a row is not waited for until it is in time again)
            self._late[seq] = self._late[seq] + 1 if self._cap() < req.gap < math.inf else 0
            self._expected.discard(seq)
            self._pending[seq] = req
            if not self._queued:
                self._queue.append(_ROUND)
                self._queued = True
            try:
                while not req.done:
                    due = self._due()
                    if due == 0.0:
                        self._lead()
                    else:
                        cv.wait(due)
            finally:
                if not req.done:  # (the thread's own exception while it waited)
                    del self._pending[seq]
                    if not self._pending:
                        self._queue.remove(_ROUND)
                        self._queued = self._lent = False
                    cv.notify_all()
        if req.error is not None:
            raise DeviceFailed(f"a decode step failed ({type(req.error).__name__}: {req.error}): the device's "
                               f"sequences start over") from req.error
        return req

    @contextlib.contextmanager
    def alone(self, seq: int) -> Iterator[None]:
        """The device to the calling thread alone, lane ``seq``'s sequence selected (re-entrant: inside, the thread's
        own dispatches run directly); after the holders and the round that came before it. A ``DispatchError`` inside
        is every lane's failure."""
        me, cv, ov = threading.get_ident(), self._cv, self.overlay
        with cv:
            if self._owner == me:
                self._depth += 1
            else:
                self._check(seq)
                self._quiet(seq)  # (it asks for no round meanwhile)
                ticket = object()
                self._queue.append(ticket)
                try:
                    while self._running or self._owner is not None or self._queue[0] is not ticket:
                        cv.wait()
                    self._check(seq)
                finally:
                    self._queue.remove(ticket)
                    cv.notify_all()
                self._owner, self._depth = me, 1
        before = ov.seq
        try:
            ov.select(seq)
            yield
        except DispatchError as e:
            with cv:
                self._fail(e, range(ov.sequences))
            raise
        finally:
            if self._depth > 1:
                ov.select(before)
            with cv:
                self._depth -= 1
                if not self._depth:
                    self._owner, self._free = None, time.perf_counter()
                    cv.notify_all()

    def breathe(self, seq: int) -> None:
        """(by the thread that has the device alone for lane ``seq``, between two parts of its work) Whoever waits
        for the device runs first - the holders that came, the lanes' round -, then the thread has it again, its
        sequence selected. Nothing where nobody waits, or inside another holder of the thread's own."""
        cv, me = self._cv, threading.get_ident()
        with cv:
            if self._depth != 1 or not self._queue:
                return
            assert self._owner == me
            ticket = object()
            self._queue.append(ticket)
            self._owner, self._free = None, time.perf_counter()
            cv.notify_all()
            error: BaseException | None = None
            while True:  # (the thread has the device again before it goes on, an interrupt's exception too: the
                # context that holds it gives it back)
                try:
                    while self._running or self._owner is not None or (error is None and self._queue[0] is not ticket):
                        cv.wait()
                    break
                except BaseException as e:  # noqa: BLE001
                    error = error or e
            self._queue.remove(ticket)
            self._owner, self._depth = me, 1
            cv.notify_all()
            if error is not None:
                raise error
            self._check(seq)
        self.overlay.select(seq)

    def waiting(self) -> bool:
        """Whether lanes' requests wait for a round (a holder's question between two of its dispatches)."""
        return bool(self._pending)

    def lend(self) -> None:
        """(by the thread that has the device alone, between two of its dispatches, which were another hardware
        context's) The round of the requests that wait runs now, as it is, if the thread has had the device for a
        step's time since the last one; then the thread has it again. ``DeviceFailed``: the round's step failed."""
        cv = self._cv
        with cv:
            if not self._pending or time.perf_counter() - self._free < self._step_s:
                return
            assert self._owner == threading.get_ident()
            self.overlay.dev.stale = True  # (its cores' code pages come back before the step)
            self._lent, failures = True, self._failures
            cv.notify_all()
            while self._lent or self._running:
                cv.wait()
            if self._failures != failures:
                e = self._failure
                raise DeviceFailed(f"a decode step failed beside the device's holder ({type(e).__name__}: {e})") from e

    def recover(self, seq: int) -> None:
        """After a failure: lane ``seq``'s calls run again (the caller starts its sequence over)."""
        with self._cv:
            self._down.discard(seq)
            self._released[seq], self._late[seq] = -math.inf, 0

    # ------------------------------------------------------------------
    # A round
    # ------------------------------------------------------------------

    def _check(self, seq: int) -> None:
        if seq in self._down:
            e = self._failure
            raise DeviceFailed(f"a dispatch failed ({type(e).__name__}: {e}): the device's sequences start over") from e

    def _fail(self, error: BaseException, seqs: Any) -> None:
        """A dispatch failed: the rings of the sequences it ran claim rows that may not exist; every waiting request
        fails with it, and every lane's later calls until it starts over (``recover``)."""
        for s in seqs:
            self.overlay._vrings[s].clear()
        self._failure, self._down = error, set(range(self.overlay.sequences))
        self._failures += 1
        for r in self._pending.values():
            r.error, r.done = error, True
        self._pending.clear()
        if self._queued:
            self._queue.remove(_ROUND)
            self._queued = False
        self._expected.clear()
        self._cv.notify_all()

    def _cap(self) -> float:
        """The seconds a round waits for an expected lane after its last rows: while waiting costs the lanes less than
        the late one would lose (a step's rest), ``WAIT`` at least."""
        return max(WAIT, self._step_s / (len(self._expected) + len(self._pending) + 1))

    def _due(self) -> float | None:
        """Seconds until the calling lane, whose request is pending, runs the round: 0 now (the device is free, the
        round's turn has come, every expected lane asked or is late; or its holder lends it: ``lend``); None: not
        before another thread's notify."""
        if self._lent:
            return 0.0
        if self._running or self._owner is not None or self._queue[0] is not _ROUND:
            return None
        if not self._expected:
            return 0.0
        return max(max(self._released[s] for s in self._expected) + self._cap() - time.perf_counter(), 0.0)

    def _admit(self) -> list[Request]:
        """The pending requests of the next step, in lane order: those a round left out first, then those passed over
        longest ago, while their slots fit."""
        take, left = [], self.overlay.SLOTS
        for r in sorted(self._pending.values(), key=lambda r: (not r.waited, -self._passed[r.seq], r.seq)):
            if r.n <= left:
                take.append(r)
                left -= r.n
            else:
                r.waited, self._passed[r.seq] = True, self.rounds
        return sorted(take, key=lambda r: r.seq)

    def _lead(self) -> None:
        """Run the round (the calling thread holds the lock, which the dispatch gives up): the expected lanes that did
        not ask are passed over, the admitted requests get their rows or the step's error."""
        cv = self._cv
        late: tuple[int, ...] = ()
        lent, self._lent = self._lent, False
        if not lent:  # (a holder's pause: the round does not wait, and the lanes that have not asked are not late)
            late = tuple(sorted(self._expected))
            self._expected.clear()
        reqs = self._admit()
        for r in reqs:
            del self._pending[r.seq]
        self._queue.remove(_ROUND)
        if self._pending:
            self._queue.append(_ROUND)
        else:
            self._queued = False
        self._running, self._busy = True, frozenset(r.seq for r in reqs)
        start = time.perf_counter()
        idle = 0.0 if lent else start - max(self._free, min(r.asked for r in reqs))
        error: BaseException | None = None
        wall = seconds = once = 0.0
        cv.release()
        try:
            wall, seconds, once = self.dispatch(reqs)
        except BaseException as e:  # noqa: BLE001 (every lane of the round learns of it)
            error = e
        finally:
            cv.acquire()
        self._running, self._busy, self._free = False, frozenset(), time.perf_counter()
        self.rounds += 1
        if error is not None:
            for r in reqs:
                r.error, r.done = error, True
            self._fail(error, [r.seq for r in reqs])
            if not isinstance(error, Exception):  # (the thread's own: an interrupt)
                raise error
            return
        self._step_s = wall
        for r in reqs:
            r.wall, r.seconds, r.parts, r.done = wall, seconds, len(reqs), True
            self._released[r.seq] = self._free
            if self._late[r.seq] < LATE:
                self._expected.add(r.seq)
        self.history.append(Round(tuple((r.seq, r.n, r.gap) for r in reqs), late, idle, wall, once))
        if log.isEnabledFor(logging.DEBUG):
            log.debug("round %d: %s, %.2f ms%s%s", self.rounds,
                      ", ".join(f"lane {r.seq} x{r.n} (+{r.gap * 1e3:.2f} ms)" for r in reqs), wall * 1e3,
                      f", the device idle {idle * 1e3:.2f} ms" if idle > 1e-4 else "",
                      f", without lane {', '.join(map(str, late))}" if late else "")
        cv.notify_all()

    def dispatch(self, reqs: list[Request]) -> tuple[float, float, float]:
        """One step of the requests' slots, the device held by the caller: each request's ``rows``; the dispatch's
        seconds, those of a step (less its one-time host work and other processes' turns), that work's."""
        ov = self.overlay
        args = tuple(a[0] if len(a) == 1 else np.concatenate(a) for a in zip(*(r.args for r in reqs), strict=True))
        t0, w0 = time.perf_counter(), _native.turn_waited()
        logits = ov.step(*args, tuple(Part(r.seq, r.position, r.n) for r in reqs))
        wall = time.perf_counter() - t0
        once = ov.once_s
        a = 0
        for r in reqs:
            r.rows = kept(logits, a, a + r.n)
            a += r.n
        return wall, wall - (_native.turn_waited() - w0) - once, once
