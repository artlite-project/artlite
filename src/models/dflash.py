"""A DFlash2 drafter's rounds in a model on the NPU (qwen35's ``NpuQwen``, Flint's ``NpuFlint``; the drafter:
``artlite.decode.dflash.overlay.DFlashOverlay``): what it has seen of the model's sequence and what it still owes the
model's steps.

The drafter takes the rows of the model's steps from the taps its step left in its ACT (``Injection``), one dispatch per
round with its block of drafts, of which the round's step verifies those that pay at the model's step times (the walk's
chain probabilities, calibrated by the rounds: ``SpeculativeMixin._keep``; an MoE's slots route to more experts, so it
takes fewer); a block may carry a sibling branch (a model whose overlay ``can_branch``: steps of 8).

The model supplies: ``dflash`` (the drafter), ``overlay``, ``position``, ``capacity``, ``carry``, ``rounds``,
``eos_token_ids``, ``_drafting``, ``_dispatch``, ``stale``, ``_rows`` (tokens' embedding rows), ``_mask`` (the mask token's
row), the counters ``t_draft``, ``drafted``, ``blocks``, ``_keep`` and ``_chains`` (``SpeculativeMixin``), and
``_dflash_ahead(S, p)``: the model's step of S slots at p patched while the drafter's dispatch runs
(``ModelOverlay.patch_ahead``, and what the model's positions need first).

A model of several lanes (``OverlayModel.lanes``: a ``DFlashDrafting`` each, one drafter): a block is one dispatch
whatever number of its drafts find a slot, so it pays for a lane that runs alone (``OverlayModel._block_pays``), and
the drafter holds the records of one lane's sequence at a time (``own``: the lane that had it keeps its state, a
``DFlashOverlay.snapshot``, and gets it back at its next turn with the drafter). A lane's rows of the model's steps
wait on the host meanwhile (``wait``: their taps, copied out of each step: the next step is another lane's too) and
are injected from the drafter's staging region before its next block, prompt chunk or checkpoint (``_run``); past
``LIMIT`` rows the drafter follows the lane's sequence no longer (``lost``), until a prompt chunk of the lane writes
its records again. No sibling branches there, and no step patched ahead (a lane's next step is the lanes').
"""

import dataclasses
import time
from typing import Any

import numpy as np


@dataclasses.dataclass
class Injection:
    """The DFlash2 drafter's rows of the model's last step that it has not injected: the step's slots ``S``, the kept rows
    ``keep`` from position ``q`` on, and, when the kept rows are not the step's first slots (a won sibling's, the slots past a
    carry), their slots ``src``."""

    S: int
    keep: int
    q: int
    src: list[int] | None = None


class DFlashDrafting:
    """The DFlash2 drafter's injections and sibling branch of the model ``model`` (module docstring)."""

    BRANCH = 2  # the tokens of a sibling branch that rounds may run (``block``; 0: none)
    LIMIT = 256  # (several lanes) a lane's rows that wait for the drafter at most: 64 injections to catch up

    def __init__(self, model: Any):
        self.model = model
        self.inj: Injection | None = None
        self.sibling: list[int] | None = None  # (the round's sibling branch: ``block``)
        self.siblings = self.siblings_won = 0  # sibling branches run and kept (their first token)
        # several lanes: the taps of the sequence's rows that wait, from the position ``rows_from`` (the drafter's
        # next of this sequence); the drafter's state of the sequence while it holds another lane's; whether it no
        # longer follows the sequence
        self.rows: tuple[np.ndarray, ...] = ()
        self.rows_from = 0
        self.saved: dict[str, Any] | None = None
        self.lost = False

    def reset(self) -> None:
        """(several lanes: the drafter's records of the lane's old sequence are nobody's; its next ``own`` starts
        them over)"""
        self.inj = None
        self.rows, self.rows_from, self.saved, self.lost = (), 0, None, False
        d = self.model.dflash
        if d.owner is self:
            d.owner = None

    # ------------------------------------------------------------------
    # Several lanes (module docstring); the device held for the lane
    # ------------------------------------------------------------------

    def own(self, *, restored: bool = False) -> None:
        """The drafter's records: this lane's sequence's (``restored``: the caller restores them, a checkpoint's)."""
        m = self.model
        d = m.dflash
        cur = d.owner
        if cur is self:
            return
        if cur is not None:
            cur.saved = d.snapshot(d.next)
        if self.saved is not None and not restored:
            d.restore(self.saved)
        elif not restored:
            d.reset()
            if self.lost:  # (no records, and none until a prompt chunk: its state at the position, for a checkpoint)
                d.next = m.position
        self.saved = None
        d.owner = self

    def wait(self, p: int, rows: np.ndarray) -> None:
        """The taps of a step's rows at p (``ModelOverlay.tap_rows``) wait for the drafter, after the sequence's rows
        before p (a rejected draft's leave)."""
        if self.lost:
            return
        kept = self.rows[:max(p - self.rows_from, 0)] if self.rows else ()
        assert not self.rows or self.rows_from + len(kept) == p, (self.rows_from, len(kept), p)
        self.rows_from, self.rows = (self.rows_from if kept else p), (*kept, *rows)
        if len(self.rows) > self.LIMIT:
            self.lose()

    def lose(self) -> None:
        """The drafter follows the lane's sequence no longer (until ``found``)."""
        self.rows, self.saved, self.lost = (), None, True
        d = self.model.dflash
        if d.owner is self:
            d.owner = None

    def found(self) -> None:
        """After a prompt chunk of the lane, which holds the drafter: the chunk wrote its records, and the next rows
        follow them."""
        self.rows, self.rows_from, self.lost = (), self.model.dflash.next, False

    def _run(self, x: np.ndarray | None, p: int) -> None:
        """The sequence's rows that wait, injected (as many as a step's slots a dispatch); with ``x`` the block's rows:
        the block at p in the last dispatch."""
        m = self.model
        d, S = m.dflash, m.overlay.SLOTS
        q = self.rows_from
        rows, self.rows = self.rows[:max(m.position - q, 0)], ()
        assert not rows or q == d.next, (q, d.next)
        while len(rows) > S or (rows and x is None):
            part, rows = rows[:S], rows[S:]
            m._dispatch(lambda part=part, q=q: d.step(len(part), len(part), q, None, 0, rows=np.stack(part)))
            q += len(part)
        if x is not None:
            m._dispatch(lambda: d.step(len(rows), len(rows), q, x, p, rows=np.stack(rows) if rows else None))
        self.rows_from = d.next
        m.stale()

    def _block(self, token: int, k: int) -> list[int]:
        """``block`` of a lane: with the device for it (``OverlayModel._before``; none where it is not to be had, or
        the drafter does not follow the sequence), the lane's rows that wait first."""
        m = self.model
        d = m.dflash
        out: list[int] = []
        m._chains = ()
        with m._before() as free:
            t0 = time.perf_counter()  # (the block's time, without the wait for the device)
            if not free or self.lost:
                return out
            self.own()
            x = np.stack([m._rows([token])[0], *[m._mask] * (d.cfg.block - 1)])
            n = min(k, d.cfg.block - 1)
            self._run(x, m.position)
            out = d.selector.walk(d.logits(), d.gates(), token, d.table.ids, n, set(m.eos_token_ids), sibling=0)
            chains = d.selector.chains
            out = out[:next((i for i in range(len(out)) if not m._keep(chains[i], i)), len(out))]
            m._chains = tuple(chains[:len(out)])
        m._row_time(time.perf_counter() - t0)  # (a block's seconds: ``OverlayModel._block_pays``)
        m.t_draft += time.perf_counter() - t0
        m.drafted += len(out)
        m.blocks += 1
        return out

    # ------------------------------------------------------------------

    def after_step(self, n: int, p: int) -> None:
        """After a committed step of n tokens at p: all of them are the drafter's to inject."""
        self.inj = Injection(n, n, p)

    def inject(self) -> None:
        """Injects the drafter's rows of the model's last step while its taps are in place. With several lanes, it
        injects the lane's waiting rows instead, after the lane takes the drafter (``own``)."""
        m = self.model
        if m._rounds is not None:
            if self.rows and not self.lost:
                self.own()
                self._run(None, 0)
            return
        if self.inj is not None:
            inj = self.inj
            self.move_taps()
            m._dispatch(lambda: m.dflash.step(inj.S, inj.keep, inj.q, None, 0))
            self.inj = None
            m.stale()

    def move_taps(self) -> None:
        """A pending injection's rows from other slots than its first ones (a won sibling's, the slots past a carry): their taps
        moved to the first slots of the tap layers' x in the model's ACT, where the drafter's injection reads them."""
        m = self.model
        src = self.inj.src
        if src is None:
            return
        for L in m.dflash.cfg.taps:
            m.overlay.move_tap_slots(L, self.inj.S, src)

    def drop_carry(self) -> None:
        """The pending injection without the sibling branch's rows."""
        n = len(self.model.carry)
        if self.inj is not None and self.inj.src:
            inj = self.inj
            self.inj = Injection(inj.S, inj.keep - n, inj.q, inj.src[:-n]) if inj.keep > n else None

    def block(self, token: int, k: int) -> list[int]:
        """A DFlash2 round's drafts after ``token`` (the next position's after ``carry``, not run yet): the kept rows of the last step
        injected and the block at the position, in one dispatch; while it runs, the round's step of the carry, the token and n
        drafts patched (``steps``' rope rows, ``patch_ahead``: a block ends at an end-of-sequence token alone). In a generation's
        rounds whose step has 8 slots, the selector's sibling branch in the last drafts' places where it is worth more."""
        m = self.model
        if m._rounds is not None:
            return self._block(token, k)
        t0 = time.perf_counter()
        d = m.dflash
        c = len(m.carry)
        x = np.stack([m._rows([token])[0], *[m._mask] * (d.cfg.block - 1)])
        if self.inj is not None:
            self.move_taps()
        inj = self.inj or Injection(0, 0, 0)
        self.inj = None
        n, p = min(k, d.cfg.block - 1), m.position

        def ahead() -> None:
            if p + c + n + 1 <= m.capacity:
                m._dflash_ahead(c + n + 1, p)

        m._dispatch(lambda: d.step(inj.S, inj.keep, inj.q, x, p + c, meanwhile=ahead))
        m.stale()
        L = self.BRANCH if (m.rounds and n > self.BRANCH and c + n + 1 == m.overlay.SLOTS
                            and getattr(m.overlay, "can_branch", False)
                            and p % 8 + c + self.BRANCH <= 7) else 0  # (a branch inside its parent's 8 positions: ``QwenOverlay.step``)
        out = d.selector.walk(d.logits(), d.gates(), token, d.table.ids, n, set(m.eos_token_ids), sibling=L)
        self.sibling = d.selector.sibling if L else None
        if self.sibling is not None:  # (its step's slots all run; no calibration: a won branch's tokens are no drafts)
            out = out[:n - L]
            self.siblings += 1
            m._chains = ()
        else:  # (the drafts that pay at the model's step times: ``SpeculativeMixin._keep``, calibrated by the rounds)
            chains = d.selector.chains
            out = out[:next((i for i in range(len(out)) if not m._keep(chains[i], i)), len(out))]
            m._chains = tuple(chains[:len(out)])
        m.t_draft += time.perf_counter() - t0
        m.drafted += len(out) + len(self.sibling or ())
        m.blocks += 1
        return out

    def kept(self, p: int, accepted: int, won: int, S: int, L: int) -> None:
        """A round after its step of S slots (its sibling branch's L last): the kept drafts' rows and a kept sibling branch's
        from its slots, the carry's not, injected after the step that kept it."""
        m = self.model
        c = len(m.carry)
        src = [*range(c, accepted), *range(S - L, S - L + won)]
        self.inj = Injection(S, len(src), p + c, src if c or won else None) if m._drafting else None


__all__ = ["DFlashDrafting", "Injection"]
