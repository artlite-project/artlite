"""Several lanes of one model (models/lanes.py: ``Rounds``; ``OverlayModel``'s lanes) on a stub overlay whose step
gives each slot the row of its sequence and position, in a buffer the next step overwrites and with two dispatches at
once refused: the lanes' attributes; generations on threads of their own against each alone (their rounds' parts and
slots); a reader that stops, a lane that is always late, rows read late; a prompt between the rounds, and the rounds
between a prompt's chunks; more lanes than slots; failed dispatches; lanes with a drafter (its dispatches before a
round's step, a step's per-slot keyword, what else a lane reads of a step). No NPU."""

import itertools
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

import artlite
from artlite.decode.step import Part
from artlite.decode.vring import VRing
from models import lanes
from models.lanes import DeviceFailed, alone
from models.overlay_model import OverlayModel
from models.sampling import Sampling, argmax

VOCAB, SLOTS, P = 64, 4, 8  # (P: a prompt's tokens)
GREEDY = Sampling(greedy=True)


class Overlay:
    """An overlay of len(texts) sequences and 4 slots: a step's row of a slot at position p of sequence s has its
    largest logit at texts[s][p + 1], in one buffer every step overwrites (as the NPU's); ``ran``: each step's parts
    (a step of the selected sequence: its part); ``delay``: a dispatch's seconds; ``fail``: the index in ``ran`` of a
    step that raises, and what. ``marks``: each step's keyword ``mark`` (per slot, or None); ``last``: the last step's
    (sequence, position) per slot; ``drafts``: a drafter's dispatches (``draft``: the selected sequence, a position)
    as their index in ``ran``."""

    SLOTS = SLOTS
    once_s = 0.0

    def __init__(self, texts: list[list[int]], delay: float = 0.0):
        self.texts, self.sequences, self.seq, self.delay = texts, len(texts), 0, delay
        self._vrings = [VRing() for _ in texts]
        self.ran: list[tuple[Part, ...]] = []
        self.marks: list[list[int] | None] = []
        self.last: list[tuple[int, int]] = []
        self.drafts: list[tuple[int, int, int]] = []
        self.fail: tuple[int, Exception] | None = None
        self.dev = SimpleNamespace(stale=False)
        self._buf = np.zeros((SLOTS, VOCAB), bfloat16)
        self._busy = threading.Lock()

    def select(self, s: int) -> None:
        self.seq = s

    def passes(self, p: int, S: int) -> int:
        return 1

    def reset(self) -> None:
        self._vrings[self.seq].clear()

    def draft(self, seq: int, position: int) -> None:
        """A drafter's dispatch for sequence ``seq`` (the selected one) at ``position``."""
        assert self._busy.acquire(blocking=False), "two dispatches at once"
        try:
            assert seq == self.seq, "a drafter's dispatch on another sequence's caches"
            self.drafts.append((len(self.ran), seq, position))
            time.sleep(self.delay / 4)
        finally:
            self._busy.release()

    def step(self, x: np.ndarray, position: int | tuple[Part, ...], *, logits: bool = True,
             mark: np.ndarray | None = None) -> np.ndarray | None:
        parts = position if isinstance(position, tuple) else (Part(self.seq, position, len(x)),)
        assert self._busy.acquire(blocking=False), "two dispatches at once"
        try:
            assert sum(q.n for q in parts) == len(x) <= SLOTS and [q.seq for q in parts] == sorted({q.seq for q in parts})
            assert mark is None or len(mark) == len(x)
            self.ran.append(parts)
            self.marks.append(None if mark is None else [int(t) for t in mark])
            self.last = [(q.seq, q.position + i) for q in parts for i in range(q.n)]
            if self.fail is not None and self.fail[0] == len(self.ran) - 1:
                raise self.fail[1]
            time.sleep(self.delay)
            self._buf[:] = -8.0
            i = 0
            for q in parts:
                t = self.texts[q.seq]
                for p in range(q.position, q.position + q.n):
                    self._buf[i, t[min(p + 1, len(t) - 1)]] = 8.0
                    i += 1
            return self._buf[:i] if logits else None
        finally:
            self._busy.release()


class Prefill:
    """A prefill of chunks of 16 tokens (from a multiple of 8 on) on the stub overlay: a chunk is two pieces, each a
    dispatch of its own, in the overlay's ``ran`` as the selected sequence's part of 16 slots; between them the
    chunk lets its ``guest`` in (``artlite.operators.fused.library.Guest``: the lanes' round), if it waits."""

    group, dispatches, guest = 8, 2, None

    def __init__(self, ov: Overlay):
        self.ov = ov

    def plan(self, p0: int, n: int, *, steps: bool = True) -> list[int]:
        return [16] * (n // 16) if p0 % 8 == 0 else []

    def prepare(self, position: int, n: int) -> None:
        pass

    def run(self, x: np.ndarray, position: int, T: int) -> None:
        ov = self.ov
        for piece in range(2):
            if piece and self.guest is not None and self.guest.waiting():
                self.guest.lend()
            assert ov._busy.acquire(blocking=False), "two dispatches at once"
            try:
                ov.ran.append((Part(ov.seq, position, T),))
                time.sleep(2 * ov.delay)
            finally:
                ov._busy.release()


class Lanes(OverlayModel):
    """A model of a lane per text on the stub overlay: lane s's next token after position p is texts[s][p + 1];
    ``prefill``: prompts in chunks (``Prefill``), else as decode steps."""

    eos_token_ids = (0,)
    capacity = 256
    prefill = None

    def __init__(self, texts: list[list[int]], *, delay: float = 0.0, lookup: bool = False, prefill: bool = False):
        self.overlay = Overlay(texts, delay)
        self.lookup = lookup
        if prefill:
            self.prefill = Prefill(self.overlay)
        self.position = 0
        self._open_lanes()

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (np.asarray(tokens, np.float32).reshape(-1, 1),)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        return self._step_inputs(tokens)

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        out = np.full(VOCAB, -8.0, np.float32)
        out[self.text[self.position + n]] = 8.0
        return out

    @alone
    def reset(self) -> None:
        self.overlay.reset()
        self.position = 0

    @property
    def text(self) -> list[int]:
        return self.overlay.texts[self.seq]

    def reply(self, n: int = P, *, pause: float = 0.0, stop: int = -1) -> list[int]:
        """The lane's text after its first n tokens, generated (the prompt as decode steps): greedy; ``pause``: the
        reader's seconds per token; ``stop``: it stops reading after that many tokens."""
        out = []
        for token in self.generate(self.forward(self.text[:n]), -1, GREEDY, self.text[:n]):
            out.append(token)
            if len(out) == stop:
                break
            time.sleep(pause)
        return out


class Drafting(Lanes):
    """``Lanes`` with a drafter: a round's guesses are a dispatch each on the lane's sequence before its step (the
    lane's next tokens; ``wrong``: from that depth on another token), which a shared step runs where it has room.
    Its steps take a per-slot keyword (``mark``: the slots' tokens, of the lanes in ``marked``), and a lane reads each
    slot's (sequence, position) of its steps (``took``)."""

    drafter = True
    LANE_SHARED = (*OverlayModel.LANE_SHARED, "wrong", "marked")
    LANE_OWN = (*OverlayModel.LANE_OWN, "took", "refused")
    took: tuple[tuple[int, int], ...] = ()
    refused = 0  # (rounds whose guesses could not run)

    def __init__(self, texts: list[list[int]], *, wrong: int = SLOTS, marked: tuple[int, ...] = (), **kw: object):
        self.wrong, self.marked = wrong, marked
        super().__init__(texts, **kw)

    def _mtp(self, token: int, k: int, penalties: object) -> list[int]:
        out: list[int] = []
        with self._before() as free:
            self.refused += not free
            for i in range(k if free else 0):
                self.overlay.draft(self.seq, self.position)
                t = self.text[min(self.position + 1 + i, len(self.text) - 1)]
                out.append(t if i < self.wrong else t % (VOCAB - 1) + 1)
        self.drafted += len(out)
        self._chains = (1.0,) * len(out)
        return out

    def _mtp_offer(self, n: int) -> tuple[float, ...]:
        return tuple(self._kappa(d) for d in range(n))

    def _step_kw(self, tokens: list[int]) -> dict[str, object]:
        return {"mark": np.asarray(tokens) if self.seq in self.marked else None}

    def _tapped(self, S: int) -> tuple:
        return (np.array(self.overlay.last[:S]),)

    def _took(self, tokens: list[int], taken: tuple) -> None:
        self.took += tuple(map(tuple, taken[0].tolist()))


def texts(n: int, length: int = 72, *, copy: bool = False) -> list[list[int]]:
    """n texts of ``length`` tokens that end with the end token; ``copy``: each repeats a span of its own again and
    again (the context's drafts are right)."""
    rng = np.random.default_rng(n)
    out = []
    for _ in range(n):
        t = rng.integers(1, VOCAB, length - 1).tolist()
        if copy:
            t = (t[:12] * length)[:length - 1]
        out.append([*t, 0])
    return out


def run(*jobs: Callable[[], object]) -> list[object]:
    """Each job on a thread of its own: their results (an exception: the job's, in its place)."""
    def job(fn: Callable[[], object]) -> object:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return e
    with ThreadPoolExecutor(len(jobs)) as pool:
        return [f.result(timeout=60) for f in [pool.submit(job, fn) for fn in jobs]]


def test_lanes() -> None:
    """A lane per sequence: the model itself and copies of it that share the device's side and have a sequence,
    position, counters and planner of their own; an attribute that is neither is refused; a model of one sequence is
    its only lane and nothing joins its steps."""
    m = Lanes(texts(3))
    a, b, c = m.lanes
    assert a is m and [x.seq for x in m.lanes] == [0, 1, 2] and b.lanes == m.lanes
    assert b.overlay is a.overlay and b._pool is a._pool and b._rounds is a._rounds is not None
    assert b.planner is not a.planner and c.planner is not b.planner
    b.position, b.dispatches, b.drafted = 5, 3, 2
    assert (a.position, a.dispatches, c.dispatches, a.drafted) == (0, 0, 0, 0) and a._total("dispatches") == 3
    assert vars(m).keys() <= {*Lanes.LANE_SHARED, *Lanes.LANE_OWN}
    one = Lanes(texts(1))
    assert one.lanes == (one,) and one._rounds is None and one.budget is None
    assert one.reply() == one.text[P:] and all(len(q) == 1 for q in one.overlay.ran)

    class Odd(Lanes):
        LANE_SHARED = (*OverlayModel.LANE_SHARED, "known")

        def __init__(self) -> None:
            self.known, self.cache = 1, {}
            super().__init__(texts(2))

    with pytest.raises(TypeError, match="cache neither shared"):
        Odd()


@pytest.mark.parametrize(("n", "lookup"), [(2, False), (2, True), (3, False), (4, True)])
def test_generations(n: int, lookup: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """n lanes generating at once give each one's tokens alone (with and without the context's drafts); their rounds
    share steps, the parts in lane order within a step's slots; a step's spare slots are the lanes' drafts."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)  # (every round waits for its lanes: its parts whatever the threads' timing)
    m = Lanes(texts(n, copy=lookup), delay=0.002, lookup=lookup)
    alone_ = [lane.reply() for lane in m.lanes]  # (one after another: every round one lane's)
    assert alone_ == [lane.text[P:] for lane in m.lanes]
    rounds = m._rounds
    assert all(len(r.parts) == 1 for r in rounds.history)
    if lookup:
        assert max(r.parts[0][1] for r in rounds.history) == SLOTS and m.stats()
    k = len(rounds.history)
    for lane in m.lanes:
        lane.reset()
    assert run(*(lane.reply for lane in m.lanes)) == alone_
    hist = list(rounds.history)[k:]
    assert all([s for s, _, _ in r.parts] == sorted(s for s, _, _ in r.parts) for r in hist)
    assert all(sum(slots for _, slots, _ in r.parts) <= SLOTS for r in hist)
    # (all lanes in a round but for the first ones, the lanes' prompts ending one after another, and with drafts
    # the last ones: the lanes' texts end rounds apart)
    full = [r for r in hist if len(r.parts) == n]
    assert len(full) >= (0.6 * len(hist) if lookup else len(hist) - 2 * n), (len(full), len(hist))
    assert not any(r.late for r in hist) and max(r.idle for r in hist) < 1.0
    used = [sum(slots for _, slots, _ in r.parts) for r in full]
    if lookup:  # (the texts copy themselves: every lane has drafts to offer, and the step is full)
        assert sum(u == SLOTS for u in used) > 0.8 * len(used) and max(s for r in full for _, s, _ in r.parts) <= SLOTS + 1 - n
    else:
        assert set(used) == {n}
    assert [lane.position for lane in m.lanes] == [len(lane.text) - 1] * n


@pytest.mark.parametrize("tells", [True, False])
def test_reader_stops(tells: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane whose reader stops reading (a stop string: its generator stays suspended): told (``idle``), no round of
    the other lane waits for it; not told, one round does, once."""
    monkeypatch.setattr(lanes, "WAIT", 0.1)
    m = Lanes(texts(2), delay=0.002)
    a, b = m.lanes
    its = []

    def stopper() -> list[int]:
        it = b.generate(b.forward(b.text[:P]), -1, GREEDY, b.text[:P])
        its.append(it)  # (suspended, as a job's after its stop string)
        out = [next(it) for _ in range(5)]
        if tells:
            b.idle()
        return out

    assert run(a.reply, stopper) == [a.text[P:], b.text[P:P + 5]]
    hist = list(m._rounds.history)
    assert [r.late for r in hist if r.late] == ([] if tells else [(1,)])
    assert sum(r.idle > 0.05 for r in hist) == (0 if tells else 1)


def test_late_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane that reads its rows while later rounds run reads its own (copies: the device's buffer holds the other
    lane's by then); a lane whose reader is slow is waited for twice, then asks when it is ready, and every token is
    its text's."""
    monkeypatch.setattr(lanes, "WAIT", 0.002)
    m = Lanes(texts(2, 120), delay=0.002)
    a, b = m.lanes
    a.position = b.position = P
    rows = b._joined([b.text[P]])
    for i in range(3):
        mine = a._joined([a.text[P + i]])
        assert argmax(mine[0]) == a.text[P + i + 1]
    assert argmax(rows[0]) == b.text[P + 1] and argmax(m.overlay._buf[0]) == a.text[P + 3]
    for lane in m.lanes:
        lane.reset()
        lane.idle()
    k = len(m._rounds.history)
    assert run(a.reply, lambda: b.reply(pause=0.02, stop=6)) == [a.text[P:], b.text[P:P + 6]]
    hist = list(m._rounds.history)[k:]
    assert sum(1 in r.late for r in hist) == lanes.LATE
    assert sum(len(r.parts) == 2 for r in hist) == 5  # (each of its rounds shares the other lane's next step)


def test_prompt_between_rounds() -> None:
    """A lane's prompt while another generates: its steps run with the device alone, one after another between two
    rounds, on its own sequence (the stub refuses two dispatches at once); then both generate."""
    m = Lanes(texts(2), delay=0.002)
    a, b = m.lanes

    def late() -> list[int]:
        time.sleep(0.03)
        return b.reply(40)

    assert run(a.reply, late) == [a.text[P:], b.text[40:]]
    ran = m.overlay.ran
    prompt = [i for i, parts in enumerate(ran) if parts[0].seq == 1 and parts[0].position < 40]
    assert len(prompt) == 10 and prompt == list(range(prompt[0], prompt[0] + 10)), prompt
    assert all(len(ran[i]) == 1 for i in prompt) and ran[prompt[0] - 1][0].seq == 0  # (a's rounds ran before it)
    assert any(len(parts) == 2 for parts in ran[prompt[-1]:])


def test_rounds_between_chunks() -> None:
    """A prompt in prefill chunks while another lane generates: between two chunks, and between a chunk's two
    pieces, the other lane's round runs (one step each time: the prompt came first), and no dispatch overlaps a
    piece; the prompt's lane then generates."""
    m = Lanes(texts(2, 120), delay=0.002, prefill=True)
    a, b = m.lanes

    def late() -> list[int]:
        time.sleep(0.03)
        return b.reply(96)  # (6 chunks)

    assert run(a.reply, late) == [a.text[P:], b.text[96:]]
    ran = m.overlay.ran
    pieces = [i for i, parts in enumerate(ran) if parts[0].n == 16]
    assert [ran[i][0] for i in pieces] == [Part(1, 16 * (k // 2), 16) for k in range(12)]
    assert [j - i for i, j in itertools.pairwise(pieces)] == [2] * 11, pieces  # (a round between two pieces)
    assert all(len(ran[i + 1]) == 1 and ran[i + 1][0].seq == 0 for i in pieces[:-1])
    assert m.overlay.dev.stale  # (a chunk's other hardware context ran: the overlay's pages before its next step)


def test_spare_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three lanes on four slots: the fourth is the draft of the lane that has one (its text copies itself; the
    others' drafts, where they have any, are worth less), and every token is its lane's."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)
    plain, copies = texts(3), texts(3, copy=True)
    m = Lanes([plain[0], copies[1], plain[2]], delay=0.002, lookup=True)
    assert run(*(lane.reply for lane in m.lanes)) == [lane.text[P:] for lane in m.lanes]
    full = [r for r in m._rounds.history if len(r.parts) == 3]
    assert len(full) > 20
    second = [{s: n for s, n, _ in r.parts} for r in full]
    assert sum(q[1] == 2 for q in second) > 0.8 * len(full), [q[1] for q in second]
    assert all(sum(q.values()) <= SLOTS for q in second)
    assert m.lanes[1].accepted_ctx > 15 and m.lanes[1].dispatches < m.lanes[0].dispatches


def test_more_lanes_than_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five lanes on a step of four slots: every round leaves one lane out, each in its turn."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)
    m = Lanes(texts(5, 60), delay=0.001)
    assert run(*(lane.reply for lane in m.lanes)) == [lane.text[P:] for lane in m.lanes]
    hist = list(m._rounds.history)
    assert max(len(r.parts) for r in hist) == SLOTS
    full = [r for r in hist if len(r.parts) == SLOTS][:40]
    took = [sum(s in [q[0] for q in r.parts] for r in full) for s in range(5)]
    assert len(full) == 40 and max(took) - min(took) <= 2, took


def test_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step that fails in a round fails that round's lanes, and every lane's calls until the lane starts over
    (``recover``); a failed dispatch of a lane alone too; an error of a lane's own (a step that does not fit) is its
    own."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)
    m = Lanes(texts(3), delay=0.002)
    a, b, c = m.lanes
    c.forward(c.text[:P])
    with pytest.raises(ValueError, match="at most"):
        c.steps([1] * 5)
    assert argmax(c.steps([c.text[P]])[0]) == c.text[P + 1]
    m.overlay.fail = (len(m.overlay.ran) + 16, RuntimeError("boom"))  # (a round of a and b)
    out = run(a.reply, b.reply)
    assert all(isinstance(e, DeviceFailed) and isinstance(e.__cause__, RuntimeError) for e in out), out
    assert len(m.overlay.ran[-1]) == 2
    for call in (lambda: c.steps([1]), lambda: c.rewind(0), lambda: a._joined([1]), c.reset):
        with pytest.raises(DeviceFailed, match="boom"):
            call()
    m.overlay.fail = None
    a.recover()
    assert a.position == 0 and a.reply() == a.text[P:]  # (a lane that started over runs; the others do not yet)
    with pytest.raises(DeviceFailed, match="boom"):
        b.forward(b.text[:P])
    for lane in (a, b, c):
        lane.recover()
    assert [lane.position for lane in m.lanes] == [0, 0, 0]
    assert run(a.reply, b.reply, c.reply) == [lane.text[P:] for lane in m.lanes]
    for lane in m.lanes:
        lane.reset()
    m.overlay.fail = (len(m.overlay.ran) + 1, artlite.DispatchError("timeout"))  # (a prompt's second step)
    with pytest.raises(artlite.DispatchError):
        a.forward(a.text[:P])
    with pytest.raises(DeviceFailed, match="timeout"):
        b.forward(b.text[:P])
    m.overlay.fail = None
    for lane in (a, b):
        lane.recover()
    assert run(a.reply, b.reply) == [a.text[P:], b.text[P:]]


@pytest.mark.parametrize("n", [2, 3])
def test_drafter(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lanes with a drafter: a lane's guesses run before its round's step, with the device and on its sequence, and
    the lanes still meet in every step (no round waits in vain or leaves a lane out); a lane guesses for its share
    of the step's slots (a slot left over: the lanes' in turns) and the step is full; each lane reads its own slots
    of a step."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)
    m = Drafting(texts(n, 120), delay=0.004)
    alone_ = [lane.reply() for lane in m.lanes]
    assert alone_ == [lane.text[P:] for lane in m.lanes]
    ov, rounds = m.overlay, m._rounds
    assert max(len(q) == 1 and q[0].n for q in ov.ran) == SLOTS and m.lanes[0].accepted > 60  # (alone: 3 guesses a round)
    for lane in m.lanes:
        lane.reset()
        lane.took = ()
    k, d0 = len(rounds.history), len(ov.drafts)
    assert run(*(lane.reply for lane in m.lanes)) == alone_
    hist = list(rounds.history)[k:]
    full = [r for r in hist if len(r.parts) == n]
    assert len(full) >= len(hist) - 2 * n and not any(r.late for r in hist), (len(full), len(hist))
    assert sum(sum(slots for _, slots, _ in r.parts) == SLOTS for r in full) > 0.9 * len(full)
    drafts = ov.drafts[d0:]
    per = {}  # (per step: each lane's guesses before it)
    for i, seq, _ in drafts:
        per.setdefault(i, []).append(seq)
    shared = [q for i, q in per.items() if len(ov.ran[i]) == n]
    # (its share: two lanes one each, three the fourth slot's in turns; the first step's lanes guessed before the
    # others asked)
    if n == 2:
        assert len(shared) > 20 and sum(sorted(q) == [0, 1] for q in shared) >= len(shared) - 2, shared[:8]
    else:
        assert len(shared) > 20 and sum(len(q) == 1 for q in shared) >= len(shared) - 2, shared[:8]
        turns = [sum(q == [s] for q in shared) for s in range(n)]
        assert max(turns) - min(turns) <= 2, turns
    assert not any(lane.refused for lane in m.lanes)
    for lane in m.lanes:  # (the slots a lane read: its own, at its positions in order; then its guesses past the end)
        mine, end = [(s, p) for s, p in lane.took if p >= P], len(lane.text) - 1
        assert mine[:end - P] == [(lane.seq, p) for p in range(P, end)]
        assert all(s == lane.seq and p >= end for s, p in mine[end - P:])


def test_drafter_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lanes with a drafter, one's guesses wrong: every text is its own, whatever shares its steps."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)
    for n in (2, 3):
        m = Drafting(texts(n, 100), delay=0.002)
        b = m.lanes[1]
        vars(b)["_mtp"] = lambda token, k, penalties, mtp=b._mtp: [t % (VOCAB - 1) + 1 for t in mtp(token, k, penalties)]
        assert run(*(lane.reply for lane in m.lanes)) == [lane.text[P:] for lane in m.lanes]
        assert b.accepted == 0 and b.drafted > 20 and m.lanes[0].accepted > 20


def test_step_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step's per-slot keyword: a joint step takes it where each of its lanes has it (the slots in their order),
    a lane's own step where that lane has it."""
    monkeypatch.setattr(lanes, "WAIT", 5.0)
    for marked in ((0, 1), (0,)):
        long, short = texts(2, 96)
        m = Drafting([long, [*short[:60], 0]], delay=0.002, marked=marked)
        a, b = m.lanes
        assert run(a.reply, b.reply) == [a.text[P:], b.text[P:]]
        ov = m.overlay
        joint = [(parts, mark) for parts, mark in zip(ov.ran, ov.marks, strict=True) if len(parts) == 2]
        assert len(joint) > 20
        for parts, mark in joint:
            want = [m.lanes[q.seq].text[q.position] for q in parts for _ in range(1)]
            assert (mark is not None) == (marked == (0, 1))
            assert mark is None or [mark[0], mark[parts[0].n]] == want, (parts, mark)
        # (a round of one lane: lane 0's last ones, its text the longer)
        own = [mark for parts, mark in zip(ov.ran, ov.marks, strict=True)
               if len(parts) == 1 and parts[0].seq == 0 and parts[0].position >= P]
        assert own and all(mark is not None for mark in own)


def test_drafter_beside_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane with a drafter generates while another lane's prompt holds the device (chunks that lend it to a round's
    step): its rounds go on without guesses then, with them after."""
    m = Drafting(texts(2, 200), delay=0.002, prefill=True)
    a, b = m.lanes

    def late() -> list[int]:
        time.sleep(0.03)
        return b.reply(160)  # (10 chunks)

    assert run(lambda: a.reply(P), late) == [a.text[P:], b.text[160:]]
    ran, drafts = m.overlay.ran, m.overlay.drafts
    pieces = [i for i, parts in enumerate(ran) if parts[0].n == 16]
    between = [i for i in range(pieces[0] + 1, pieces[-1]) if ran[i][0].n != 16]
    assert len(between) >= 12 and all(len(ran[i]) == 1 and ran[i][0].seq == 0 for i in between)  # (a's rounds)
    during = [i for i, _, _ in drafts if pieces[0] < i <= pieces[-1]]
    assert len(during) <= 2, during  # (none but where the prompt was between two chunks)
    assert a.refused + sum(1 for i in between if ran[i][0].n == 1) >= len(between) - 2
    assert sum(1 for i, s, _ in drafts if i > pieces[-1] and s == 0) > 10  # (after the prompt: guesses again)
