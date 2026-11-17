"""A DFlash2 drafter's rounds in a model on the NPU (qwen35's ``NpuQwen``, Flint's ``NpuFlint``; the drafter:
``artlite.decode.dflash.overlay.DFlashOverlay``): what it has seen of the model's sequence and what it still owes the
model's steps.

The drafter takes the rows of the model's steps from the taps its step left in its ACT (``Injection``), one dispatch per
round with its block of drafts; a block may carry a sibling branch (a model whose overlay ``can_branch``: steps of 8).

The model supplies: ``dflash`` (the drafter), ``overlay``, ``position``, ``capacity``, ``carry``, ``rounds``,
``eos_token_ids``, ``_drafting``, ``_dispatch``, ``stale``, ``_rows`` (tokens' embedding rows), ``_mask`` (the mask token's
row), the counters ``t_draft``, ``drafted``, ``blocks``, and ``_dflash_ahead(S, p)``: the model's step of S slots at p
patched while the drafter's dispatch runs (``ModelOverlay.patch_ahead``, and what the model's positions need first).
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

    def __init__(self, model: Any):
        self.model = model
        self.inj: Injection | None = None
        self.sibling: list[int] | None = None  # (the round's sibling branch: ``block``)
        self.siblings = self.siblings_won = 0  # sibling branches run and kept (their first token)

    def reset(self) -> None:
        self.inj = None

    def after_step(self, n: int, p: int) -> None:
        """After a committed step of n tokens at p: all of them are the drafter's to inject."""
        self.inj = Injection(n, n, p)

    def inject(self) -> None:
        """The drafter's rows of the model's last step, while its taps are in place."""
        m = self.model
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
        if self.sibling is not None:
            out = out[:n - L]
            self.siblings += 1
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
