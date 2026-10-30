"""Draft tokens from the context for a generation's multi-token steps (prompt lookup), and the choice of a step's
drafts by the tokens per second they are expected to give.

A step of S <= SLOTS (4, or 8) tokens streams the weights once, so the tokens it checks along with the sampled one
cost little on a dense model (+2-35% for 4 on Strix Halo at 1K, more at long contexts, where the attention runs per
token) and more on an MoE (the slots route to different experts: ~+30% per slot). A draft is kept where the token sampled from the
model's logits at its position is that draft: the text is the model's, sampled as without drafts.

``Lookup``: the longest suffix of the sequence so far (the prompt and the generated tokens, at most ``NMAX``) that
occurred before proposes the tokens after its latest occurrence. Agents copy their context all the time: an edit's old
text is a span of a file they read, a rewritten file is mostly the old one, a command or a path comes again.

``Planner`` (one per model, across its generations) chooses a round's drafts: k of the context's are worth
E(m, k) = 1 + a1 + a1 a2 + ... tokens (a_i: how often the i-th draft after a match of m tokens was accepted when the
ones before it were: ``PRIOR``, then counted), against the time of a step of k + 1 tokens at the round's position (an
average per S as the steps run, before, the model's ``STEP_COST`` ratios to a measured one; a step across a KV cache
group adds a second attention pass, a measure of its own: the position decides, and such a step's time must not stand
for S's, or S looks dear and never runs again to correct it); a drafter's rounds (multi-token prediction) by their
measured tokens per second after matches of the same length (where the text repeats the context, the drafter guesses
it as well: its average rate would undersell it there), again every ``RETRY`` rounds of that length without it. The
context's drafts take a round for ``MARGIN`` times the tokens per second of the drafter's round (a context round also
leaves the drafter the rows of its kept tokens to run), from a drafter only after a match of ``DRAFTER_MATCH`` tokens
(shorter ones' estimates are too noisy to beat a drafter round by round), and for up to ``MARGIN`` times a step of
one's (as their bucket's rounds are counted, less); where nothing pays, the round is a step of one token."""

from collections.abc import Sequence

import numpy as np

NMAX = 32  # a match's tokens at most
# P(the i-th context draft is accepted | the ones before it were) after a match of m tokens, per bucket of m (1, 2, 3,
# 4, 5, 6-8, 9-16, 17 on): agent sessions' (Claude Code, Codex and opencode on Qwen3.6-35B-A3B, temperature 1)
BUCKETS = (0, 0, 1, 2, 3, 4, 5, 5, 5)  # (m <= 8: its bucket; then 6 up to 16, 7 above)
# (drafts past the third: the third's, up to an 8-slot step's 7)
PRIOR = np.pad(np.array([(0.23, 0.46, 0.55), (0.36, 0.54, 0.62), (0.44, 0.59, 0.65), (0.51, 0.64, 0.70), (0.62, 0.70, 0.77),
                         (0.72, 0.78, 0.82), (0.81, 0.82, 0.83), (0.89, 0.91, 0.92)]), ((0, 0), (0, 4)), mode="edge")
PRIOR_N = 8  # the prior's weight in rounds
EWMA = 0.2  # a new measurement's weight in a running average
WARM = 4  # a drafter's rounds before its rate competes (its drafts until then)
RETRY = 32  # rounds without the drafter after which it drafts again (its rate is measured again)
DRAFTER_MATCH = 6  # with a drafter, the context's drafts compete after matches of this many tokens (agent sessions:
# their first draft accepted 23-62% after shorter ones, 72-89% from 6 on)
MARGIN = 1.1  # the context's drafts take a round for this many times the tokens per second of a drafter's round, and of
# a step of one while their acceptance is the prior's (shrinking with the rounds counted: a bet inside the estimate's
# error loses, on prose, whose matches are rarely copies, the agent sessions' prior is high)


def bucket(m: int) -> int:
    """The acceptance bucket of a match of m >= 1 tokens."""
    return BUCKETS[m] if m < len(BUCKETS) else 6 if m <= 16 else 7


def _mtp_bucket(m: int) -> int:
    """A drafter's rate bucket of a round whose context matched m >= 0 tokens (0: none)."""
    return bucket(m) + 1 if m else 0


class Lookup:
    """A generation's sequence so far (token ids; an image's rows, negative, match nothing) and its context drafts."""

    def __init__(self, tokens: Sequence[int] | np.ndarray):
        ids = np.asarray(getattr(tokens, "ids", tokens), np.int32).reshape(-1)
        self._buf = np.empty(max(2 * len(ids), 1024), np.int32)
        self._buf[:len(ids)] = ids
        self.n = len(ids)

    def extend(self, tokens: Sequence[int]) -> None:
        n = self.n + len(tokens)
        if n > len(self._buf):
            buf = np.empty(2 * n, np.int32)
            buf[:self.n] = self._buf[:self.n]
            self._buf = buf
        self._buf[self.n:n] = tokens
        self.n = n

    def draft(self, k: int) -> tuple[int, list[int]]:
        """Up to k tokens after the latest earlier occurrence of the sequence's longest repeated suffix (none from an
        image's row on) and that suffix's length m (0: none occurred before; no drafts)."""
        seq, L = self._buf, self.n
        if L < 2 or k <= 0:
            return 0, []
        cand = np.flatnonzero(seq[:L - 1] == seq[L - 1])  # (where the suffix's occurrences end, ascending)
        if not len(cand):
            return 0, []
        m = 1
        while m < min(NMAX, L - 1):
            c = cand[cand >= m]
            longer = c[seq[c - m] == seq[L - 1 - m]]
            if not len(longer):
                break
            cand, m = longer, m + 1
        at = int(cand[-1]) + 1
        out = seq[at:min(at + k, L)]
        neg = np.flatnonzero(out < 0)
        return m, out[:neg[0] if len(neg) else len(out)].tolist()


class Planner:
    """A model's choice of each round's drafts (module docstring): ``cost``, a step of S = 1 .. SLOTS tokens' time
    relative to one's until measured."""

    def __init__(self, cost: tuple[float, ...]):
        self.cost = cost
        self.t: list[float | None] = [None] * len(cost)  # (seconds of a step of S within a cache group, a running average)
        self.again = 0.0  # (the seconds a step adds by crossing a group, a running average)
        self.crossing = (False,) * len(cost)  # (per S: whether a step of S at the round's position crosses a group)
        self.tried = np.full(PRIOR.shape, float(PRIOR_N))
        self.kept = PRIOR * PRIOR_N
        n = len(PRIOR) + 1  # (per context-match bucket, ``_mtp_bucket``: a drafter round's count, and its tokens and
        # seconds as running averages; the rounds since its last)
        self.mtp_rounds, self.mtp_tokens, self.mtp_s = np.zeros(n, np.int64), np.zeros(n), np.zeros(n)
        self.since_mtp = np.zeros(n, np.int64)
        self.rate_n, self.rate_tokens, self.rate_s = 0, 0.0, 0.0  # (every drafter round's, running averages)

    def step_time(self, S: int) -> float:
        """A step of S tokens' seconds at the round's position."""
        return self._within(S) + self.again * self.crossing[S - 1]

    def _within(self, S: int) -> float:
        """A step of S tokens' seconds within a cache group: measured, else the prior ratio to the nearest measured S
        (relative units while none is)."""
        if self.t[S - 1] is not None:
            return self.t[S - 1]
        known = [s for s in range(len(self.t)) if self.t[s] is not None]
        if not known:
            return self.cost[S - 1]
        s = min(known, key=lambda s: abs(s - (S - 1)))
        return self.t[s] * self.cost[S - 1] / self.cost[s]

    def expected(self, m: int, k: int) -> float:
        """The tokens a round of k context drafts after a match of m tokens gives (with the sampled one)."""
        a = self.kept[bucket(m)] / self.tried[bucket(m)]
        return 1.0 + float(np.cumprod(a[:k]).sum())

    def choose(self, m: int, k: int, *, drafter: bool) -> int | None:
        """The context drafts to take of k after a match of m tokens, or None: the drafter's (``drafter``: it can
        draft this round)."""
        best_k, best = 0, 0.0
        for j in range(1, k + 1):
            r = self.expected(m, j) / self.step_time(j + 1)
            if r > best:
                best_k, best = j, r
        if drafter:
            b = _mtp_bucket(m)
            if (m < DRAFTER_MATCH or self.mtp_rounds[b] < WARM or self.since_mtp[b] >= RETRY
                    or MARGIN * self.mtp_tokens[b] / self.mtp_s[b] >= best):
                return None
            self.since_mtp[b] += 1
        margin = 1 + (MARGIN - 1) * PRIOR_N / self.tried[bucket(m), 0] if m else 1.0
        return best_k if best >= margin / self.step_time(1) else 0

    def ran(self, S: int, seconds: float) -> None:
        """A step of S tokens took ``seconds``."""
        if self.crossing[S - 1]:
            if any(t is not None for t in self.t):
                x = seconds - self._within(S)
                self.again = x if not self.again else (1 - EWMA) * self.again + EWMA * x
            return
        t = self.t[S - 1]
        self.t[S - 1] = seconds if t is None else (1 - EWMA) * t + EWMA * seconds

    def accepted(self, m: int, k: int, a: int) -> None:
        """Of k context drafts after a match of m tokens the first a were accepted."""
        b = bucket(m)
        self.tried[b, :min(a + 1, k)] += 1
        self.kept[b, :a] += 1

    def rate(self) -> float | None:
        """A drafter's rounds' tokens per second (any match), None before ``WARM`` of them."""
        return self.rate_tokens / self.rate_s if self.rate_n >= WARM else None

    def drafted(self, m: int, tokens: int, seconds: float) -> None:
        """A drafter's round (the context matched m tokens) gave ``tokens`` in ``seconds`` (its drafts and the step)."""
        w = 1.0 if not self.rate_n else EWMA
        self.rate_n += 1
        self.rate_tokens = (1 - w) * self.rate_tokens + w * tokens
        self.rate_s = (1 - w) * self.rate_s + w * seconds
        b = _mtp_bucket(m)
        w = 1.0 if not self.mtp_rounds[b] else EWMA
        self.mtp_rounds[b] += 1
        self.since_mtp[b] = 0
        self.mtp_tokens[b] = (1 - w) * self.mtp_tokens[b] + w * tokens
        self.mtp_s[b] = (1 - w) * self.mtp_s[b] + w * seconds
