"""Several lanes of one model (models/lanes.py: ``Rounds``; ``OverlayModel``'s lanes) on a stub overlay whose step
gives each slot the row of its sequence and position, in a buffer the next step overwrites and with two dispatches at
once refused: the lanes' attributes; generations on threads of their own against each alone (their rounds' parts and
slots); a reader that stops, a lane that is always late, rows read late; a prompt between the rounds; more lanes than
slots; failed dispatches. No NPU."""

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
    step that raises, and what."""

    SLOTS = SLOTS
    once_s = 0.0

    def __init__(self, texts: list[list[int]], delay: float = 0.0):
        self.texts, self.sequences, self.seq, self.delay = texts, len(texts), 0, delay
        self._vrings = [VRing() for _ in texts]
        self.ran: list[tuple[Part, ...]] = []
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

    def step(self, x: np.ndarray, position: int | tuple[Part, ...], *, logits: bool = True) -> np.ndarray | None:
        parts = position if isinstance(position, tuple) else (Part(self.seq, position, len(x)),)
        assert self._busy.acquire(blocking=False), "two dispatches at once"
        try:
            assert sum(q.n for q in parts) == len(x) <= SLOTS and [q.seq for q in parts] == sorted({q.seq for q in parts})
            self.ran.append(parts)
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


class Lanes(OverlayModel):
    """A model of a lane per text on the stub overlay: lane s's next token after position p is texts[s][p + 1]."""

    eos_token_ids = (0,)
    capacity = 256
    prefill = None

    def __init__(self, texts: list[list[int]], *, delay: float = 0.0, lookup: bool = False):
        self.overlay = Overlay(texts, delay)
        self.lookup = lookup
        self.position = 0
        self._open_lanes()

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (np.asarray(tokens, np.float32).reshape(-1, 1),)

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
    share steps, the parts in lane order within a step's slots, a lane's slots its share of them."""
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
    # (all lanes in a round but for the first ones, the lanes' prompts ending one after another)
    full = [r for r in hist if len(r.parts) == n]
    assert len(full) >= len(hist) - 2 * n, (len(full), len(hist))
    assert not any(r.late for r in hist) and max(r.idle for r in hist) < 1.0
    assert max(slots for r in full for _, slots, _ in r.parts) == (SLOTS // n if lookup else 1)
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
