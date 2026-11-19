"""The decode overlays' host-side step machinery (``decode/model.py``, ``decode/step.py``, ``decode/vring.py``): the
transaction cache's key, the patch-ahead state, snapshots, the AHEAD contract. No NPU."""

import itertools

import numpy as np
import pytest

from artlite.decode.model import AheadStep, HostRecords, ModelOverlay, Snapshot, StepKey, nbytes_of
from artlite.decode.step import Part, slots_of
from artlite.decode.vring import PAIR, RingStateError, VRing, substeps


def test_step_key_and_ahead_state() -> None:
    key = StepKey(3, True, (0,), ((0, 3),))
    assert key.b == 0 and key._replace(b=1) != key
    ahead = AheadStep(key, 40, (frozenset(), ()), object(), 1)
    assert ahead.matches(StepKey(3, True, (0,), ((0, 3),)), 40, (frozenset(), ()))
    assert not ahead.matches(key, 41, (frozenset(), ()))  # another position
    assert not ahead.matches(key._replace(logits=False), 40, (frozenset(), ()))  # another step
    assert not ahead.matches(key, 40, (frozenset({0}), ()))  # another ring state: patched again
    parts = (Part(0, 40, 2), Part(2, 7, 1))  # (several sequences' step: its parts in place of the position)
    joint = AheadStep(key, parts, ((frozenset(), ()),) * 2, object(), 1)
    assert joint.matches(key, (Part(0, 40, 2), Part(2, 7, 1)), ((frozenset(), ()),) * 2)
    assert not joint.matches(key, (Part(0, 40, 2), Part(1, 7, 1)), ((frozenset(), ()),) * 2)  # another sequence
    assert not joint.matches(key, 40, ((frozenset(), ()),) * 2)


def test_parts_slots() -> None:
    """A step's parts give each slot its sequence and position, the parts' first slots marked."""
    seq, pos, head = slots_of((Part(2, 30, 2), Part(0, 5, 1)))
    assert (seq, pos, head) == ([2, 2, 0], [30, 31, 5], [True, False, True])


def test_sequences_need_buffers() -> None:
    """An overlay whose model keeps nothing per sequence on the device (``SEQ``) holds one."""
    with pytest.raises(ValueError, match="holds one"):
        ModelOverlay(None, layers=None, vocab=None, weights=None, sequences=2)


def test_joint_patterns() -> None:
    """The sub-step patterns of several sequences' steps of S slots (``ModelOverlay._joint_subs``, built ahead by
    ``prepare``): every way to cut S slots into two or more runs, whatever the sequences' count from two on (a part
    that crosses a pair of records is two sub-steps, as two parts are); with those of one sequence's steps, every
    way to cut S slots."""

    class Pairs(ModelOverlay):
        SEQ = ("KV",)
        period = PAIR

        def __init__(self, sequences: int):
            self.sequences = sequences

        def subs(self, p: int, S: int) -> tuple:
            return tuple(substeps(p, S, PAIR, 4))

        def _join(self, parts: tuple[Part, ...]) -> tuple:
            out, a0 = [], 0
            for q in parts:
                out += [(a0 + a, n) for a, n in self.subs(q.position, q.n)]
                a0 += q.n
            return tuple(out)

    def cuts(S: int) -> set[tuple]:
        out = set()
        for mask in range(1 << (S - 1)):
            edges = [0, *(i + 1 for i in range(S - 1) if mask >> i & 1), S]
            out.add(tuple((a, b - a) for a, b in itertools.pairwise(edges)))
        return out

    for sequences in (2, 3, 4):
        ov = Pairs(sequences)
        for S in range(1, 5):
            joint = ov._joint_subs(S)
            assert joint == {u for u in cuts(S) if len(u) > 1}, (sequences, S)
            assert joint | {ov.subs(p, S) for p in range(PAIR)} == cuts(S)


def test_snapshot_bytes() -> None:
    """A snapshot's bytes: its rings, its open records and what the model adds (nested)."""
    rec = HostRecords(0, 9, [(32, np.zeros((2, 1, 64), np.uint8))])
    sn = Snapshot(np.zeros(100, np.uint8), VRing(), 4, (0, 9, rec), {"conv": (np.zeros((3, 8), np.uint8), 5, 2)})
    assert sn.nbytes == 100 + 128 + 24 == nbytes_of([sn])
    assert rec.cut(8).nbytes == 128  # (positions 0 .. 7 are in group 0 still)


def test_ring_check_raises() -> None:
    ring = VRing()
    ring.ran(3, 8, 4)  # (record 1: rows 8 .. 11)
    ring.check(3, 11, 12)
    with pytest.raises(RingStateError):
        ring.check(3, 12, 20)  # (a step that skipped positions 12 .. 15: not in the ring)


def test_ahead_contract() -> None:
    """A model with AHEAD patches the next step while one runs: its ``_set_position`` must take ``ahead``."""

    class Bad(ModelOverlay):
        AHEAD = True

        def _set_position(self, st: object, p: int) -> bool:
            return True

    with pytest.raises(TypeError, match="ahead"):
        Bad(None, layers=None, vocab=None, weights=None)
