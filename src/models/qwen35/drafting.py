"""The drafters' bookkeeping of a qwen35 model on the NPU (``qwen.NpuQwen``): what each drafter has seen of the model's
sequence and what it still owes the model's steps.

``MtpDrafting``: the multi-token-prediction drafter (``draft.QwenDraft``) runs a separate step per draft. Its row at position
q takes token q and the model's output-normed hidden state of q - 1, writes its cache record at q and guesses token q + 1.
Its records follow the model's positions: a prompt chunk writes its rows' (``QwenPrefill(draft=)``); after the model's steps
their rows wait in ``pend`` (records from ``dpos`` on, at most SLOTS - 1 of them) until a drafter step runs them with the next
ones; ``h`` is the model's hidden state of the last cached position. A round (``OverlayModel.generate``): one drafter step of
the pending rows and the sampled token's (its first draft), one-row steps for the next drafts, the model's step of the token
and the drafts (``commit=False``), then the accepted tokens' rows pending (they rewrite the draft rows' records); a round of
the context's drafts leaves the token's row pending with the accepted ones'.

A model of several lanes (``NpuQwen(sequences=)``): each lane has its ``MtpDrafting`` and the drafter a cache of each
sequence (``QwenDraft``); the drafter's steps run with the device for the lane (``OverlayModel._before``: ahead of the
lanes' joint step; a round goes without them while another lane has the device alone, its rows then pending on), the
model's hidden states are the lane's rows of its last step (``NpuQwen._hid``: the next step is another lane's too), and
the prefill's carry is kept per lane.

The DFlash2 drafter's: ``models.dflash.DFlashDrafting``.
"""

import time
from typing import TYPE_CHECKING

import numpy as np

from ..sampling import max_prob

if TYPE_CHECKING:
    from .qwen import NpuQwen


class MtpDrafting:
    """The MTP drafter's rows of the model ``model`` (module docstring)."""

    def __init__(self, model: "NpuQwen"):
        self.model = model
        self.reset()

    def reset(self) -> None:
        """No drafter rows pending, nothing cached."""
        self.pend: list[tuple[np.ndarray, np.ndarray]] = []  # (embedding, h) per row
        self.dpos = 0
        self.h = np.zeros(self.model.cfg.hidden_size, np.float32)  # (zeros before position 0)
        self.carry_at = 0  # (the position after the last chunk: the prefill's carry is its)
        self.carry: np.ndarray | None = None  # (a copy of that carry: another lane's chunk leaves its own in the prefill)
        self.ran = True  # (the last round's drafter step ran: ``draft``)

    def state(self) -> dict[str, object]:
        """What a checkpoint keeps: the rows, and the prefill's carry."""
        m = self.model
        out: dict[str, object] = {"pend": list(self.pend), "dpos": self.dpos, "h": self.h, "carry_at": self.carry_at}
        if m.prefill is not None and m.prefill.dp is not None:
            out["carry"] = (m.prefill.dp.carry if self.carry is None else self.carry).copy()
        return out

    def set_state(self, state: dict[str, object]) -> None:
        m = self.model
        self.pend, self.dpos, self.h, self.carry_at = list(state["pend"]), state["dpos"], state["h"], state["carry_at"]
        if m.prefill is not None and m.prefill.dp is not None:
            m.prefill.dp.carry[:] = state["carry"]
            self.carry = state["carry"].copy()

    def hidden(self, n: int) -> np.ndarray:
        """The model's output-normed hidden states (n, hidden) fp32 of the last step's first n slots (several lanes:
        of the lane's rows of its last step)."""
        m = self.model
        if m._rounds is not None:
            return m._normed(m._hid[:n].copy())
        return m._normed(np.stack([m.overlay.hidden(-1, s) for s in range(n)]))

    def follow(self, n: int, x: np.ndarray) -> None:
        """After the model's step (its input rows ``x``): the drafter rows of its first n tokens (cached) pending."""
        hs = self.hidden(n)
        h = np.concatenate([self.h[None], hs[:-1]])
        self.pend += list(zip(x[:n], h, strict=True))
        self.h = hs[-1]
        if len(self.pend) >= self.model.overlay.SLOTS:
            with self.model._before() as free:
                if free:
                    self._flush()

    def _flush(self) -> None:
        """Records of the pending rows while a step's worth of them wait."""
        while len(self.pend) >= self.model.overlay.SLOTS:
            self.run_pending(self.model.overlay.SLOTS)

    def step(self, rows: list[tuple[np.ndarray, np.ndarray]], logits: bool) -> np.ndarray | None:
        """A drafter step of rows at ``dpos`` on (their records; the drafter's logits (n, vocab) bf16, or None)."""
        m = self.model
        assert m.drafter.seq == m.seq, "the drafter's cache: the lane's sequence's (the device held for it)"
        p, n = self.dpos, len(rows)
        q = np.arange(p, p + n)
        old = q < m.position  # (the model's positions; text from ``rope_next`` after them)
        rope = m.drafter.rope
        rope[q[old]] = m.overlay.rope[q[old]]
        rope[q[~old]] = m.rope_next + (q[~old] - m.position)[:, None]
        emb, h = (np.stack(v) for v in zip(*rows, strict=True))
        out = m.drafter.step(emb, h, p, logits=logits)
        self.dpos += n
        return out

    def run_pending(self, n: int) -> None:
        """Records of the first n pending rows."""
        rows, self.pend = self.pend[:n], self.pend[n:]
        self.step(rows, logits=False)

    def before_prefill(self) -> None:
        """Before a prompt chunk: the records of the rows before it (the chunk's carry after decode steps)."""
        m = self.model
        while self.pend:
            self.run_pending(m.overlay.SLOTS)
        if m.position and self.carry_at != m.position:
            m.prefill.dp.carry_of(self.h)
        elif m.position and self.carry is not None:  # (several lanes: another's chunk ran since)
            m.prefill.dp.carry[:] = self.carry

    def after_prefill(self, n: int) -> None:
        """After a prompt chunk of n valid tokens: its last row's hidden state is the next one's."""
        m = self.model
        x = m.prefill.hidden(n - 1)
        self.h = m._normed(x)[0]
        self.dpos = self.carry_at = m.position + n
        self.carry = m.prefill.dp.carry.copy()

    def draft(self, token: int, n: int) -> list[int]:
        """The drafter's guesses of up to n tokens after ``token`` (the token of the next position, not run yet; its
        row runs with the pending ones): the most likely one, the most likely one after it, ... while their chain's
        probability pays for the round's slot of each and the drafter's step of the next (``_worth``, ``_more``) (ends
        with an end-of-sequence token). Several lanes: with the device for the lane (``OverlayModel._before``); none
        where it is not to be had: the token's row then waits with the round's kept ones (``ran``)."""
        m = self.model
        out: list[int] = []
        chains: list[float] = []
        with m._before() as free:
            self.ran = free
            t0 = time.perf_counter()  # (the guesses' time, without the wait for the device)
            if free:
                self._flush()
                rows = [*self.pend, (m._rows([token])[0], self.h)]
                assert self.dpos + len(rows) == m.position + 1 and len(rows) <= m.overlay.SLOTS
                self.pend = []
                raw = self.step(rows, logits=n > 0)
                chain, slot = 1.0, len(rows) - 1
                while len(out) < n:
                    i, p = max_prob(raw[slot])
                    chain *= p
                    if not m._keep(chain, len(out)):
                        break
                    out.append(int(m.drafter.table.ids[i]))
                    chains.append(chain)
                    if out[-1] in m.eos_token_ids or len(out) == n or not m._more(chain, p, len(out)):
                        break
                    h = m.drafter.hidden_state(slot)
                    t = time.perf_counter()
                    raw, slot = self.step([(m._rows(out[-1:])[0], h)], logits=True), 0
                    m._row_time(time.perf_counter() - t)
        m.t_draft += time.perf_counter() - t0
        m.drafted += len(out)
        m._chains = tuple(chains)
        return out

    def kept(self, p: int, tokens: list[int], accepted: int) -> None:
        """A drafter's round after its step: the kept drafts' rows pending (the token's ran with the drafts)."""
        m = self.model
        hs = self.hidden(accepted)
        self.pend = [(m._rows([t])[0], hs[i]) for i, t in enumerate(tokens[1:accepted])]
        self.dpos, self.h = p + 1, hs[accepted - 1]


__all__ = ["MtpDrafting"]
