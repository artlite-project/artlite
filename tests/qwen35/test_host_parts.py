"""The qwen35 host side's parts that need no model or NPU: the lazy-commit cursor, the step records' tables, the drafters'
injection bookkeeping, the drafter prefill's protocol and the checks of its sizes."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from artlite import txn
from models.qwen35 import conv_state
from models.qwen35.cursor import GdnCursor
from models.qwen35.dflash_core import MAX_CHUNK, MAX_PADDING, POS_PER_GROUP
from models.qwen35.dflash_prefill import DFlashPrefill
from models.qwen35.drafter_prefill import DrafterPrefill
from models.qwen35.drafting import DFlashDrafting, Injection
from models.qwen35.steps import PanelTable, StatePanel


def test_cursor() -> None:
    """A cursor's blocks alternate between GSTATE's halves, a step's acceptance advances it, reset keeps the commit state,
    and a copy assigned back restores every field."""
    cur = GdnCursor(gp=16, gh=1, gpos=19, committed=False)
    assert [cur.half(p) for p in (16, 23, 24, 31, 32, 40)] == [1, 1, 0, 0, 1, 0]
    cur.advance(5)  # (the step at 19 ends its block: past 24)
    assert (cur.kshift, cur.cshift, cur.gnext) == (8, 5, 24)
    cur.advance(2)
    assert (cur.kshift, cur.cshift, cur.gnext) == (0, 2, 21)
    fields = dataclasses.fields(cur)
    distinct = GdnCursor(**{f.name: (i + 1 if f.type is int else bool(i % 2)) for i, f in enumerate(fields)})
    saved = distinct.copy()
    cur.assign(distinct)
    assert cur == saved and cur is not distinct
    cur.reset()
    assert (cur.committed, cur.accepted) == (saved.committed, saved.accepted)
    assert all(getattr(cur, f.name) == f.default for f in fields if f.name not in ("committed", "accepted"))


def test_panel_table() -> None:
    """State panels as arrays: a drain's BD word index is its handle's first."""
    panels = [StatePanel(8, 100, 64, False), StatePanel([20, 21], 200, 32, True), StatePanel(40, 300, 64, False)]
    t = PanelTable.of(panels)
    assert t.at.tolist() == [8, 20, 40] and t.offset.tolist() == [100, 200, 300]
    assert t.nbytes.tolist() == [64, 32, 64] and t.drain.tolist() == [False, True, False]
    assert t.lines.tolist() == sorted(set(((4 * np.array([8, 20, 40, 10, 22, 42])) & ~63).tolist()))


def test_conv_slot() -> None:
    """The conv state's bytes fit a column's GHDR slot of the models' channels; more do not."""
    conv_state.check_slot(1024)
    with pytest.raises(AssertionError):
        conv_state.check_slot(conv_state.HSLOT)


def test_injection() -> None:
    """A pending injection loses the sibling branch's rows with the carry (``drop_carry``); without a sibling's slots it is
    kept."""
    model = SimpleNamespace(carry=(7,))
    d = DFlashDrafting(model)
    d.inj = Injection(8, 5, 100, [0, 1, 2, 3, 6])
    d.drop_carry()
    assert d.inj == Injection(8, 4, 100, [0, 1, 2, 3])
    d.inj = Injection(8, 1, 100, [6])
    d.drop_carry()
    assert d.inj is None
    d.inj = Injection(8, 3, 100)
    d.drop_carry()
    assert d.inj == Injection(8, 3, 100)


class _Minimal(DrafterPrefill):
    def rows(self) -> dict[str, int]:
        return {}

    def buffers(self) -> dict[str, object]:
        return {}

    def external(self) -> set[str]:
        return set()

    def static(self, dev: object, ov: object) -> None:
        pass

    def end_items(self, T: int, tok: object, sl: object) -> list[tuple]:
        return []

    def seconds(self, Ss: float, **constants: float) -> float:
        return 0.0


def test_drafter_prefill_defaults() -> None:
    """A drafter that adds nothing at a chunk's start, at a layer's, in the parameters or around the chunk."""
    d = _Minimal()
    assert d.start_items(256, None) == [] and d.tap_items(0, 256, None) == [] and d.params(256, 0) == {}
    assert d.before(None, 0, 256) is None and d.after(None, 0, 200, 256) is None
    with pytest.raises(TypeError):
        DrafterPrefill()


def _dflash_prefill(sizes: tuple[int, ...]) -> DFlashPrefill:
    shape = SimpleNamespace(over=MAX_CHUNK // POS_PER_GROUP)
    p = object.__new__(DFlashPrefill)
    p.d, p.pf = SimpleNamespace(shape=shape), SimpleNamespace(sizes=sizes)
    return p


def test_dflash_prefill_sizes() -> None:
    """The prefill's sizes against the DFlash2 rings: the dense models' sizes pass, a chunk past the rings' room or a gap
    of more padding rows than the ring keeps out of its window does not."""
    from models.qwen35.prefill import SIZES

    _dflash_prefill(SIZES)._check_sizes()
    with pytest.raises(txn.HardwareLimitError, match="past the DFlash2 ring"):
        _dflash_prefill((*SIZES, 4096))._check_sizes()
    with pytest.raises(txn.HardwareLimitError, match="padding rows"):
        _dflash_prefill((256, 256 + MAX_PADDING + 32))._check_sizes()
