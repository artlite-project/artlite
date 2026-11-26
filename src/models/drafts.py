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
ones before it were: ``FIRST`` and ``PRIOR``, then counted; a1 also by how the match's earlier occurrences continue:
where they disagree, the latest one's token is right less often, after 4 tokens 0.78 where all agree with it against
0.22 where at most half do, and agent sessions replayed offline run 1-5% faster than with a1 by m alone, most where
slots are dear), against the time of a step of k + 1 tokens at the round's position (an average per S and span of
positions as the steps run, ``SPAN``, before, the model's ``STEP_COST`` ratios to a measured
one; a step across a KV cache group adds a second attention pass, a measure of its own: the position decides, and
neither a crossing step's time nor a long context's must stand for S's at another position, or S looks dear and never
runs again to correct it; the time another process's turn on the NPU took is not the step's); a drafter's rounds
(multi-token prediction) by their measured tokens per second after matches of the same length (where the text repeats
the context, the drafter guesses it as well: its average rate would undersell it there), again every ``RETRY`` rounds
of that length without it. The context's drafts take a round for ``MARGIN`` times the tokens per second of the
drafter's round (a context round also leaves the drafter the rows of its kept tokens to run), from a drafter only
after a match of ``DRAFTER_MATCH`` tokens
(shorter ones' estimates are too noisy to beat a drafter round by round), and for up to ``MARGIN`` times a step of
one's (as their bucket's rounds are counted, less); where nothing pays, the round is a step of one token. A bucket's
drafter rounds below a step of one's tokens per second stop the drafter there (a text it cannot guess: a block
drafter's dispatch costs ~10% of a round, and a round keeps nothing of it), for ``PAUSE`` rounds at first and twice as
long each time it comes back no better, up to ``RETRY``; a bucket's rate is a slow average (``EWMA_RATE``): at the
step times' weight the rate of a drafter that gives a quarter more tokens per second fell below a step of one's after
two or three rounds that kept nothing, and stops of ``RETRY`` rounds cost Qwen3.6-35B-A3B and Occamy 5 - 13% of a
generation."""

from collections.abc import Sequence

import numpy as np

from .sequence import token_ids

NMAX = 32  # a match's tokens at most
# how the earlier occurrences of a round's match continue (``Lookup.draft``): there is one; several, all with the same
# token; several that disagree, more than half of them with the latest one's token (the drafts' first); at most half
ONE, ALL, MOST, FEW = range(4)
# P(the first context draft is accepted) after a match of m tokens, per bucket of m (1, 2, 3, 4, 5, 6-8, 9-16, 17 on) and
# its occurrences' agreement (ONE, ALL, MOST, FEW); P(the i-th draft is accepted | the ones before it were) for i = 2, 3
# per bucket (drafts past the third: the third's, up to an 8-slot step's 7): agent sessions' (Claude Code, Codex and
# opencode on Qwen3.6-35B-A3B, temperature 1)
BUCKETS = (0, 0, 1, 2, 3, 4, 5, 5, 5)  # (m <= 8: its bucket; then 6 up to 16, 7 above)
FIRST = np.array([(0.25, 0.66, 0.57, 0.16), (0.36, 0.76, 0.54, 0.19), (0.43, 0.75, 0.41, 0.20), (0.51, 0.78, 0.52, 0.22),
                  (0.57, 0.80, 0.51, 0.23), (0.61, 0.82, 0.70, 0.33), (0.80, 0.92, 0.57, 0.38), (0.88, 0.94, 0.81, 0.41)])
PRIOR = np.pad(np.array([(0.46, 0.55), (0.54, 0.62), (0.59, 0.65), (0.64, 0.70), (0.70, 0.77), (0.78, 0.82), (0.82, 0.83),
                         (0.91, 0.92)]), ((0, 0), (0, 4)), mode="edge")
PRIOR_N = 8  # the priors' weight in rounds
EWMA = 0.2  # a new measurement's weight in a running average
# ... of a drafter's round in its bucket's tokens and seconds while it drafts, after its first 1 / EWMA_RATE rounds' plain
# mean: a round's tokens vary by more (1 .. S) than the edge the drafter is judged on (15 - 25% over a step of one), which
# a rate of few rounds loses to two rounds that keep nothing. A round after a stop counts with EWMA: a text the drafter
# guesses again brings it back at once (Gemma 4 E4B at 32K: its first rounds keep little, the later ones pay 14%)
EWMA_RATE = 0.05
SPAN = 2048  # the step times' spans of positions: [0, SPAN), then [SPAN 2^(i - 1), SPAN 2^i) (a step's attention runs
# per cached position: at 32K a step of 4 costs ~2x one of 1, at 1K a few percent)
WARM = 4  # a drafter's rounds before its rate competes (its drafts until then)
RETRY = 32  # rounds without the drafter after which it drafts again (its rate is measured again)
PAUSE = 4  # ... the first time its rounds fall below a step of one's rate (doubled each time it returns no better)
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
        ids = np.asarray(token_ids(tokens), np.int32).reshape(-1)
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

    def draft(self, k: int) -> tuple[int, int, list[int]]:
        """Up to k tokens after the latest earlier occurrence of the sequence's longest repeated suffix (none from an
        image's row on), that suffix's length m (0: none occurred before; no drafts) and how its occurrences continue
        (``ONE``, ``ALL``, ``MOST``, ``FEW``)."""
        seq, L = self._buf, self.n
        if L < 2 or k <= 0:
            return 0, ONE, []
        cand = np.flatnonzero(seq[:L - 1] == seq[L - 1])  # (where the suffix's occurrences end, ascending)
        if not len(cand):
            return 0, ONE, []
        m = 1
        while m < min(NMAX, L - 1):
            c = cand[cand >= m]
            longer = c[seq[c - m] == seq[L - 1 - m]]
            if not len(longer):
                break
            cand, m = longer, m + 1
        at = int(cand[-1]) + 1
        same = int((seq[cand + 1] == seq[at]).sum())
        agree = ONE if len(cand) == 1 else ALL if same == len(cand) else MOST if 2 * same > len(cand) else FEW
        out = seq[at:min(at + k, L)]
        neg = np.flatnonzero(out < 0)
        return m, agree, out[:neg[0] if len(neg) else len(out)].tolist()


class Planner:
    """A model's choice of each round's drafts (module docstring): ``cost``, a step of S = 1 .. SLOTS tokens' time
    relative to one's until measured; ``ewma_rate``: ``EWMA_RATE``'s (a model whose drafter's edge is small, an MoE's
    whose slots are dear: a faster average stops it within a few rounds of a text it cannot guess)."""

    def __init__(self, cost: tuple[float, ...], *, ewma_rate: float = EWMA_RATE):
        self.cost, self.ewma_rate = cost, ewma_rate
        # (per span of positions, ``place``: the seconds of a step of S within a cache group, and those a step adds by
        # crossing a group, running averages)
        self._t: dict[int, list[float | None]] = {}
        self._again: dict[int, float] = {}
        self.span = 0  # (the round's)
        self.crossing = (False,) * len(cost)  # (per S: whether a step of S at the round's position crosses a group)
        # (per bucket and agreement the first drafts tried and kept, per bucket the later ones': counts from the priors)
        self.tried1, self.kept1 = np.full(FIRST.shape, float(PRIOR_N)), FIRST * PRIOR_N
        self.tried, self.kept = np.full(PRIOR.shape, float(PRIOR_N)), PRIOR * PRIOR_N
        n = len(PRIOR) + 1  # (per context-match bucket, ``_mtp_bucket``: a drafter round's count, and its tokens and
        # seconds as running averages; the rounds since its last)
        self.mtp_rounds, self.mtp_tokens, self.mtp_s = np.zeros(n, np.int64), np.zeros(n), np.zeros(n)
        self.since_mtp = np.zeros(n, np.int64)
        # (the rounds a bucket's drafter stays out, and whether its last rounds were below a step of one's rate)
        self.wait, self.lost = np.full(n, RETRY, np.int64), np.zeros(n, bool)
        self.rate_n, self.rate_tokens, self.rate_s = 0, 0.0, 0.0  # (every drafter round's, running averages)

    def place(self, position: int) -> None:
        """The round's position: its span's step times (``SPAN``)."""
        self.span = (position // SPAN).bit_length()

    @property
    def t(self) -> list[float | None]:
        """The round's span's seconds of a step of S within a cache group (None: not measured there)."""
        return self._t.setdefault(self.span, [None] * len(self.cost))

    @property
    def again(self) -> float:
        """The round's span's seconds a step adds by crossing a cache group."""
        return self._again.get(self.span, 0.0)

    @again.setter
    def again(self, seconds: float) -> None:
        self._again[self.span] = seconds

    def step_time(self, S: int) -> float:
        """A step of S tokens' seconds at the round's position."""
        return self._within(S) + self.again * self.crossing[S - 1]

    def _within(self, S: int) -> float:
        """A step of S tokens' seconds within a cache group: measured in the round's span, else the prior ratio to the
        nearest S measured there; in a span of none, the nearest span's (relative units while none is)."""
        t = self.t
        if all(x is None for x in t):
            spans = [b for b, ts in self._t.items() if any(x is not None for x in ts)]
            if spans:
                t = self._t[min(spans, key=lambda b: abs(b - self.span))]
        if t[S - 1] is not None:
            return t[S - 1]
        known = [s for s in range(len(t)) if t[s] is not None]
        if not known:
            return self.cost[S - 1]
        s = min(known, key=lambda s: abs(s - (S - 1)))
        return t[s] * self.cost[S - 1] / self.cost[s]

    def chain(self, m: int, k: int, agree: int) -> np.ndarray:
        """Per draft of k context drafts after a match of m tokens whose occurrences agree as ``agree``: how often it
        is kept (it and the drafts before it accepted)."""
        b = bucket(m)
        a = np.concatenate(([self.kept1[b, agree] / self.tried1[b, agree]], self.kept[b] / self.tried[b]))
        return np.cumprod(a[:k])

    def expected(self, m: int, k: int, agree: int) -> float:
        """The tokens a round of k context drafts after a match of m tokens whose occurrences agree as ``agree`` gives
        (with the sampled one)."""
        return 1.0 + float(self.chain(m, k, agree).sum())

    def choose(self, m: int, k: int, *, agree: int, drafter: bool) -> int | None:
        """The context drafts to take of k after a match of m tokens whose occurrences agree as ``agree``, or None:
        the drafter's (``drafter``: it can draft this round)."""
        best_k, best = 0, 0.0
        for j in range(1, k + 1):
            r = self.expected(m, j, agree) / self.step_time(j + 1)
            if r > best:
                best_k, best = j, r
        if drafter:
            b = _mtp_bucket(m)
            if self.mtp_rounds[b] < WARM or self.since_mtp[b] >= self.wait[b]:
                return None
            rate = self.mtp_tokens[b] / self.mtp_s[b]
            pays = rate * self.step_time(1) >= 1
            if pays and (m < DRAFTER_MATCH or MARGIN * rate >= best):
                self.lost[b] = False
                return None
            if pays:  # (the context's drafts are worth more)
                self.wait[b], self.lost[b] = RETRY, False
            elif not self.since_mtp[b]:  # (its last round left it below a step of one: out, longer if it was out before)
                self.wait[b] = min(2 * self.wait[b], RETRY) if self.lost[b] else PAUSE
                self.lost[b] = True
            self.since_mtp[b] += 1
        margin = 1 + (MARGIN - 1) * PRIOR_N / self.tried1[bucket(m), agree] if m else 1.0
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

    def accepted(self, m: int, k: int, a: int, *, agree: int) -> None:
        """Of k >= 1 context drafts after a match of m tokens whose occurrences agree as ``agree`` the first a were
        accepted."""
        b = bucket(m)
        self.tried1[b, agree] += 1
        self.kept1[b, agree] += a > 0
        self.tried[b, :min(a, k - 1)] += 1  # (the later drafts: tried after a kept one)
        self.kept[b, :max(a - 1, 0)] += 1

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
        w = EWMA if self.lost[b] else max(self.ewma_rate, 1.0 / (self.mtp_rounds[b] + 1))
        self.mtp_rounds[b] += 1
        self.since_mtp[b] = 0
        self.mtp_tokens[b] = (1 - w) * self.mtp_tokens[b] + w * tokens
        self.mtp_s[b] = (1 - w) * self.mtp_s[b] + w * seconds
