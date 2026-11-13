"""A generation in rounds of speculative decoding (``SpeculativeMixin``): a step of the sampled token and up to
``DRAFTS`` draft tokens, the drafts' choice (the context's: ``drafts.Lookup``, a drafter's: ``_mtp``, as the planner
decides: ``drafts.Planner``), the drafter's calibration (``_worth``, ``_keep``, ``_more``) and the counters of its
rounds. A model on a decode overlay (``overlay_model.OverlayModel``) takes it."""

import functools
import time
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from artlite import _native

from .drafts import EWMA, Lookup, Planner
from .sampling import LogitsLike, Mask, Penalties, Probs, Sampling


class SpeculativeMixin:
    """The multi-token steps of a model on its decode overlay (``overlay``: ``artlite.decode.model.ModelOverlay``).

    ``generate`` runs a generation in rounds: a step of the sampled token and up to ``DRAFTS`` draft tokens - the
    context's (``lookup``: ``drafts.Lookup``) or a drafter's (multi-token prediction: ``_mtp``), as ``planner``
    chooses -, and as long as the token the model's logits give after a token is the draft that ran behind it, the
    draft's logits are valid too. A model with a drafter or lazy states supplies the round's step (``_verify``) and
    what follows its kept tokens (``_kept``).

    The host class supplies ``steps`` (a dispatch of up to ``overlay.SLOTS`` tokens: their bf16 logits), ``cap``,
    ``position``, ``capacity``, ``softcap``, ``eos_token_ids``, and for ``forced_drafts`` ``prompt``, ``reset`` and
    ``prefill``."""

    overlay: Any
    position: int
    capacity: int
    softcap: float
    eos_token_ids: tuple[int, ...] | set[int]
    drafted: int  # draft tokens run and accepted (of both kinds: ``NpuModel``'s counters)
    accepted: int
    DRAFTS = 3  # draft tokens per step at most (the overlay's SLOTS - 1, or fewer: ``--spec-draft-n-max``)
    STEP_COST = (1.0, 1.05, 1.1, 1.15)  # a step of S = 1 .. SLOTS tokens' time relative to one's, until measured
    P_MIN = 0.5  # a drafter's chain of drafts stops below this probability until its rounds are measured (``_worth``)
    # (a drafter step of one row's seconds; per draft depth the rounds that kept a draft there and their chains'
    # probabilities there: running averages)
    _t_row = 0.0
    _kept_at: tuple[float, ...] = ()
    _chain_at: tuple[float, ...] = ()
    _chains: tuple[float, ...] = ()  # (the chain probabilities of the drafter's last drafts: ``_calibrate``)
    rounds = False  # a generation's rounds are running (a drafter may offer a sibling: ``_sibling``)
    _rounds_of: object = None  # (the running generation's: ``_abandon``)
    carry: tuple[int, ...] = ()  # tokens kept, not run yet: a won sibling, which the next round's step runs before its token
    lookup = True  # drafts from the context
    drafter: Any = None  # a multi-token-prediction drafter
    t_draft = 0.0  # seconds in the drafter's steps
    blocks = 0  # dispatches of a block drafter (its drafts of a round in one)
    drafted_ctx = accepted_ctx = 0  # the context's of them

    @functools.cached_property
    def planner(self) -> Planner:
        """(across the model's generations: its steps' times, the drafts' acceptance)"""
        return Planner(self.STEP_COST)

    @property
    def _drafting(self) -> bool:
        return self.drafter is not None and self.DRAFTS > 0

    def stats(self) -> str:
        out = []
        if mtp := self.drafted - self.drafted_ctx:
            cost = (f"{self.t_draft / self.blocks * 1e3:.1f} ms a block" if self.blocks
                    else f"{self.t_draft / mtp * 1e3:.2f} ms each")
            out.append(f"{mtp} draft tokens ({cost}), {self.accepted - self.accepted_ctx} accepted")
        if self.drafted_ctx:
            out.append(f"{self.drafted_ctx} context draft tokens, {self.accepted_ctx} accepted")
        return "; ".join(out)

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        """The drafter's guesses of up to k tokens after ``token`` (the next position's, not run yet)."""
        raise NotImplementedError

    def _crossing(self) -> tuple[bool, ...]:
        """Per step of S = 1 .. DRAFTS + 1 tokens at the position: whether it takes one more pass over the KV caches
        than at a cache group's start."""
        ov = self.overlay
        return tuple(ov.passes(self.position, S) > ov.passes(0, S) for S in range(1, self.DRAFTS + 2))

    def _worth(self, c: float, s: int, extra: float = 0.0) -> bool:
        """Whether a draft that the model keeps with probability c (its chain's: the drafter's probability of it and of
        the drafts before it) pays for a round's step of s + 1 tokens instead of s and ``extra`` seconds more (a drafter
        step to guess it): c tokens against what that time gives at the drafter's rounds' rate (``Planner.rate``;
        ``P_MIN`` until it is measured). A dense model's slots cost little: its chains go on while c is ~0.1 or more;
        an MoE's slots route to more experts: ~0.4-0.5."""
        rate = self.planner.rate()
        if rate is None:
            return c >= self.P_MIN
        plan = self.planner
        return c >= rate * (plan.step_time(s + 1) - plan.step_time(s) + extra)

    def _keep(self, chain: float, n: int) -> bool:
        """With n drafts so far, whether the drafter's next guess (chain: the probability of all n + 1) joins them (the
        chain's probability as the model keeps such chains: ``_kappa``)."""
        return self._worth(chain * self._kappa(n), n + 1)

    def _more(self, chain: float, p: float, n: int) -> bool:
        """With n drafts (their chain's probability, the last one's p), whether the drafter guesses another (its
        probability taken as the last one's: a confident chain goes on confidently, as code or a copy does; before its
        rounds are measured, always)."""
        return self.planner.rate() is None or self._worth(chain * self._kappa(n) * p, n + 1, self._t_row)

    def _kappa(self, d: int) -> float:
        """The drafts the model keeps at depth d (0: a round's first draft) per their chains' probability (the
        drafter's calibration: its guesses after its own guesses are overconfident, Qwen3.8-27B's on prose 1.0 at
        depth 0 to 0.24 at depth 6; a depth not drafted yet takes the deepest one's; 1 before its rounds)."""
        d = min(d, len(self._chain_at) - 1)
        return self._kept_at[d] / self._chain_at[d] if d >= 0 else 1.0

    def _calibrate(self, kept: int) -> None:
        """A drafter's round kept its first ``kept`` drafts (``_chains``: the probabilities the drafter gave them)."""
        k, c = list(self._kept_at), list(self._chain_at)
        for d, chain in enumerate(self._chains):
            if d == len(c):  # (a new depth: from the depth before it, as one round of its calibration and chains; the
                # first from 1: a rejected first draft must not stop the drafts that would measure it again)
                k.append(k[-1] if d else chain)
                c.append(c[-1] if d else chain)
            k[d] += EWMA * ((kept > d) - k[d])
            c[d] += EWMA * (chain - c[d])
        self._kept_at, self._chain_at = tuple(k), tuple(c)

    def _row_time(self, seconds: float) -> None:
        """A drafter step of one row took ``seconds``."""
        self._t_row = seconds if not self._t_row else self._t_row + EWMA * (seconds - self._t_row)

    def _begin(self) -> None:
        """A generation's rounds begin (after its prompt's last step)."""

    def _verify(self, tokens: list[int], mtp: bool) -> np.ndarray:
        """A round's step of the sampled token and its drafts (``mtp``: the drafter's): their logits (S, vocab)."""
        return self.steps(tokens)

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool, won: int = 0) -> None:
        """After a round's step of ``tokens`` at p: its first m tokens are the sequence's (the rejected drafts'
        positions are forgotten: later steps overwrite their cache records); ``won``: so are the first won tokens of its
        sibling branch (``_sibling``), run again by the next step (``carry``)."""
        self.position = p + m

    def _sibling(self) -> list[int] | None:
        """The round's sibling branch: tokens from one the drafter ranks second after the round's token on, which its step
        runs in slots of their own after the drafts (``_verify``), branching off them; the drafts gave them their last
        slots. None: a chain of drafts."""
        return None

    def _drop_carry(self) -> None:
        """A generation ends with a kept sibling branch not run again (``carry``): the model forgets that it was kept."""

    def _abandon(self) -> None:
        """Whatever runs next on the sequence ends a generation whose reader stopped (a stop string, a call: ``generate``'s
        finally runs later, or never): a kept sibling branch that no step ran again is forgotten (``_drop_carry``)."""
        if self.carry:
            self._drop_carry()
        self.rounds, self.carry, self._rounds_of = False, (), None

    def generate(self, logits: np.ndarray, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = (),
                 mask: Mask | None = None, probs: Probs | None = None, *, drafts: int | None = None) -> Iterator[int]:
        """(in rounds of a step of drafts each: the class docstring; ``probs``: of a round's kept tokens, their rows of
        its step; ``drafts`` below ``DRAFTS``: a drafter's caches still follow the tokens, as after a round of the
        context's drafts)"""
        if max_tokens == 0:
            return
        most = self.DRAFTS if drafts is None else max(min(drafts, self.DRAFTS), 0)  # (this generation's)
        penalties = Penalties(sampling, prompt, mask)

        def choose(row: LogitsLike, cap: float = 0.0) -> int:
            cand = penalties.top(row, sampling.k, cap)
            token = sampling.pick(*cand)
            if probs is not None:
                probs(row, token, cap, cand)
            penalties.accept(token)
            return token

        token = choose(logits)
        n = 1
        yield token
        ctx = Lookup(getattr(prompt, "ids", prompt)) if self.lookup and most > 0 else None
        if ctx is not None:
            ctx.extend([token])
        self._begin()
        plan = self.planner
        self._abandon()
        tag = object()
        self.rounds, self._rounds_of = True, tag
        try:
            while n != max_tokens and token not in self.eos_token_ids and self.position + len(self.carry) < self.capacity:
                c = len(self.carry)
                k = min(most - c, self.capacity - self.position - 1 - c, max_tokens - n - 1 if max_tokens > 0 else most)
                plan.place(self.position)
                plan.crossing = self._crossing()
                t0, w0 = time.perf_counter(), _native.turn_waited()  # (the planner's times: this process's own work)
                m, d = ctx.draft(k) if ctx is not None else (0, [])
                j = plan.choose(m, len(d), drafter=self._drafting and k > 0)
                mtp = j is None
                d = self._mtp(token, k, penalties) if mtp else d[:j]
                sib = self._sibling() if mtp else None
                p = self.position
                t1, w1 = time.perf_counter(), _native.turn_waited()
                raw = self._verify([*self.carry, token, *d], mtp)
                t2, w2 = time.perf_counter(), _native.turn_waited()
                new: list[int] = []
                for i in range(len(d) + 1):
                    new.append(choose(raw[c + i], self.softcap))
                    if i == len(d) or new[i] != d[i] or new[i] in self.eos_token_ids:
                        break
                won = 0  # (the sibling branch's tokens kept: their own logits, the token after each)
                if sib is not None and new == [sib[0]] and sib[0] not in self.eos_token_ids:
                    won = 1
                    while True:
                        new.append(choose(raw[c + len(d) + won], self.softcap))
                        if won == len(sib) or new[-1] != sib[won] or new[-1] in self.eos_token_ids:
                            break
                        won += 1
                self._kept(p, [*self.carry, token, *d], c + 1 if won else c + len(new), mtp, won)
                self.carry = tuple(new[:won])
                # (less the step's one-time host work: a prompt's states turned into the decode's, a transaction's first
                # run, up to ~200 / ~60 ms, which a running average would carry for ten steps)
                plan.ran(c + len(d) + 1 + len(sib or ()), t2 - t1 - (w2 - w1) - self.overlay.once_s)
                self.accepted += len(new) - 1
                if mtp:
                    plan.drafted(m, len(new), t2 - t0 - (w2 - w0))
                    self._calibrate(len(new) - 1)
                elif d:
                    plan.accepted(m, len(d), len(new) - 1)
                    self.drafted += len(d)
                    self.drafted_ctx += len(d)
                    self.accepted_ctx += len(new) - 1
                if ctx is not None:
                    ctx.extend(new)
                for token in new:
                    n += 1
                    yield token
        finally:
            if self._rounds_of is tag:
                self._abandon()

    def forced_drafts(self, tokens: list[int], first: int, *, turns: Sequence[int] = (), prefill: bool = True
                      ) -> Iterator[tuple[list[int], np.ndarray]]:
        """Teacher-forced drafts of the drafter (quality tooling): tokens[:first] from position 0 as prompts that start
        at 0 and at ``turns`` (a later one after the cached positions, as a chat's next turn; with ``prefill`` False all
        as decode steps), then per position of tokens[first:-1] the drafter's guesses after its token (``_mtp``, as a
        generation's round: up to ``DRAFTS``) and the model's step of the token alone (a round that keeps it): the
        guesses (of the tokens after it) and the model's logits (vocab,) fp32 of the next token."""
        assert self._drafting, "a drafter"
        self.reset()
        pf = self.prefill
        if not prefill:
            self.prefill = None
        try:
            for a, b in zip((0, *turns), (*turns, first)):
                self.prompt(tokens[a:b], logits=b == first)
        finally:
            self.prefill = pf
        penalties = Penalties(Sampling(greedy=True))
        self._begin()
        for token in tokens[first:-1]:
            p = self.position
            d = self._mtp(token, self.DRAFTS, penalties)
            logits = self.cap(self._verify([token], True)[0])
            self._kept(p, [token], 1, True)
            yield d, logits
