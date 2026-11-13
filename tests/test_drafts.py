"""Context drafts (models/drafts.py: ``Lookup``, ``Planner``), the generation's rounds of drafts
(``OverlayModel.generate``) on a fake overlay model whose text is fixed and a drafter's stop rule, the drafter's
options (``models.common.add_spec_options``); on the NPU (slow) greedy replies with drafts against steps of one
token."""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

import models
from artlite.model_config import gemma_gguf, llama_gguf, minicpmv_gguf, qwen_gguf
from models import open_model
from models.common import OverlayModel, Penalties, Sampling, add_spec_options, drafter_type, spec_options
from models.drafts import DRAFTER_MATCH, NMAX, PRIOR, RETRY, WARM, Lookup, Planner, bucket

from .common import essay
from .models import require

VOCAB = 32
GREEDY = Sampling(greedy=True)


def test_lookup() -> None:
    """The longest repeated suffix, its latest occurrence, drafts up to k and the sequence's end, none from an
    image's row on; the buffer grows."""
    seq = Lookup([1, 2, 3, 4, 9, 2, 3, 5, 7, 1, 2, 3])
    assert seq.draft(3) == (3, [4, 9, 2])  # (1 2 3: at 0 .. 2, not the later 2 3 at 5 .. 6)
    seq.extend([5])
    assert seq.draft(2) == (3, [7, 1])  # (2 3 5 at 5 .. 7)
    assert Lookup([1, 2, 3]).draft(3) == (0, [])
    assert Lookup([4, 4, 4, 4]).draft(3) == (3, [4])  # (the suffix's latest earlier end: the sequence's end next)
    assert Lookup([6, 1, -1, -1, 2, 6]).draft(3) == (1, [1])
    assert Lookup([1, 2, 3, 1, 2]).draft(0) == (0, [])
    seq = Lookup(np.arange(5))
    for i in range(3000):
        seq.extend([i % 7])
    assert seq.draft(3) == (NMAX, [(2999 + 1 + j) % 7 for j in range(3)])  # (the period's next ones)


def test_planner() -> None:
    """A dense model's steps (+5% for 4 tokens) take every draft of a weak match; an MoE's (+30% per token) none of a
    weak match and all of a long one; acceptance moves the estimate; a drafter's measured rate decides against the
    context's once it has run ``WARM`` rounds after matches of the same length (of at least ``DRAFTER_MATCH``), and it
    drafts again after ``RETRY`` rounds of that length without."""
    assert [bucket(m) for m in (1, 5, 6, 8, 9, 16, 17, 99)] == [0, 4, 5, 5, 6, 6, 7, 7]
    dense, moe = Planner((1.0, 1.02, 1.035, 1.05)), Planner((1.0, 1.31, 1.60, 1.94))
    assert dense.choose(1, 3, drafter=False) == 3
    assert moe.choose(1, 3, drafter=False) == 0 and moe.choose(20, 3, drafter=False) == 3
    for S, t in enumerate((0.040, 0.054, 0.066, 0.080), 1):  # (measured: the MoE's prior ratios, in seconds)
        moe.ran(S, t)
    assert moe.choose(1, 3, drafter=False) == 0 and moe.choose(20, 3, drafter=False) == 3
    assert moe.step_time(4) == pytest.approx(0.080)
    e0 = moe.expected(5, 3)
    assert e0 == pytest.approx(1 + PRIOR[4, 0] + PRIOR[4, 0] * PRIOR[4, 1] + np.prod(PRIOR[4, :3]))
    assert moe.choose(5, 3, drafter=False) == 2
    for _ in range(50):
        moe.accepted(5, 3, 0)
    assert moe.expected(5, 3) < 1.4 < e0 and moe.choose(5, 3, drafter=False) == 0
    p = Planner((1.0, 1.05, 1.1, 1.15))
    assert p.choose(20, 3, drafter=True) is None  # (the drafter's first rounds)
    for m in (1, 3, 20):
        for _ in range(WARM):
            p.ran(4, 0.105)
            p.drafted(m, 3 if m > 1 else 2, 0.110)  # (27 tokens/s; after matches of 1: 18)
    assert p.choose(1, 3, drafter=True) is None  # (E 1.39 per 0.105 s, but short matches: the drafter's)
    assert p.choose(DRAFTER_MATCH, 3, drafter=True) is None  # (no drafter rounds after matches of 6 yet)
    assert [p.choose(20, 3, drafter=True) for _ in range(RETRY + 1)] == [3] * RETRY + [None]  # (E 3.4 per 0.105 s)
    for _ in range(RETRY):  # (the retry counts per bucket: another bucket's drafter rounds leave it due)
        p.drafted(3, 3, 0.110)
    assert p.choose(20, 3, drafter=True) is None
    p.drafted(20, 3, 0.110)
    assert p.choose(20, 3, drafter=True) == 3
    for _ in range(10):
        p.drafted(20, 4, 0.110)  # (where the context repeats, the drafter guesses it too: 36 tokens/s)
    assert p.choose(20, 3, drafter=True) is None


def test_planner_stop() -> None:
    """A text the drafter cannot guess: rounds under a step of one's tokens per second stop it in that bucket, a step of one or the
    context's drafts run instead, and every ``RETRY`` rounds it drafts again; a drafter that pays stays however few it keeps."""
    p = Planner((1.0, 1.002, 1.006, 1.008))
    p.ran(1, 0.300)
    for _ in range(WARM):
        p.drafted(0, 1, 0.330)  # (a block's 30 ms for nothing)
    assert [p.choose(0, 0, drafter=True) for _ in range(RETRY + 1)] == [0] * RETRY + [None]
    p.drafted(0, 1, 0.330)
    assert p.choose(0, 0, drafter=True) == 0
    q = Planner((1.0, 1.002, 1.006, 1.008))
    q.ran(1, 0.340)
    for _ in range(3 * WARM):
        q.drafted(0, 2, 0.353)  # (prose at 32K: ~2.4 tokens a round, against 1 token in 340 ms)
    assert q.choose(0, 0, drafter=True) is None


def test_planner_crossing() -> None:
    """A step across a cache group measures the second pass, not S: S stays as cheap as it is (a first sample of a
    crossing kept for S would keep S from running again); the rule sees the pass where the position brings it."""
    p = Planner((1.0, 1.01, 1.015, 1.02))
    p.ran(1, 0.330)
    p.crossing = (False, False, True, True)
    p.ran(4, 0.384)  # (S 4's first step)
    assert p.t[3] is None and p.again == pytest.approx(0.384 - 0.330 * 1.02)
    assert p.step_time(4) == pytest.approx(0.384) and p.step_time(2) == pytest.approx(0.3333)
    p.crossing = (False,) * 4
    assert p.step_time(4) == pytest.approx(0.3366)
    p.ran(4, 0.340)
    assert p.t[3] == 0.340
    p.crossing = (False, False, False, True)
    p.ran(4, 0.390)
    assert p.t[3] == 0.340 and p.again == pytest.approx(0.8 * (0.384 - 0.3366) + 0.2 * 0.050)
    assert not Planner((1.0, 1.1)).t[1]  # (nothing measured: no crossing's measure in relative units)
    # a long context's steps stand for no other span's: at 30K a step of 4 takes 2x one of 1; at 100 the planner drafts
    r = Planner((1.0, 1.02, 1.035, 1.05))
    r.place(30000)
    for S, t in enumerate((0.060, 0.090, 0.105, 0.120), 1):
        r.ran(S, t)
    assert r.choose(1, 3, drafter=False) == 0
    r.place(100)
    assert r.step_time(4) == 0.120  # (a span of no steps: the nearest measured one's)
    r.ran(1, 0.030)
    assert r.step_time(4) == pytest.approx(0.030 * 1.05) and r.choose(1, 3, drafter=False) == 3
    r.place(31000)
    assert r.step_time(1) == 0.060 and r.choose(1, 3, drafter=False) == 0
    q = Planner((1.0, 1.1))
    q.crossing = (False, True)
    q.ran(2, 0.3)
    assert q.again == 0.0 and q.t == [None, None]


class Text(OverlayModel):
    """A model whose next token after position q is ``text[q + 1]`` (bf16 logits from ``steps``); ``calls``: each
    step's tokens."""

    eos_token_ids = (0,)
    capacity = 256

    def __init__(self, text: list[int]):
        self.text, self.position, self.calls = text, 0, []

    def steps(self, tokens: list[int], *, logits: bool = True) -> np.ndarray:
        p = self.position
        self.calls.append(list(tokens))
        self.position += len(tokens)
        out = np.full((len(tokens), VOCAB), -8.0, np.float32)
        for i in range(len(tokens)):  # (past the text: its end)
            out[i, self.text[min(p + i + 1, len(self.text) - 1)]] = 8.0
        return out.astype(bfloat16)

    def _crossing(self) -> tuple[bool, ...]:
        return tuple((self.position % 32) + S > 32 for S in range(1, self.DRAFTS + 2))

    def first(self, n: int) -> np.ndarray:
        """The prompt's last logits after positions 0 .. n - 1 ran."""
        self.position = n
        out = np.full(VOCAB, -8.0, np.float32)
        out[self.text[n]] = 8.0
        return out


def test_generate_drafts() -> None:
    """A text that repeats its prompt's spans: the rounds take the context's drafts (fewer steps than tokens), the
    tokens are those of steps of one, each step after the kept ones (the rejected drafts' positions forgotten);
    without lookup steps of one; penalties and a draft count of 0 too."""
    rng = np.random.default_rng(1)
    a, b = rng.integers(1, VOCAB, 24).tolist(), rng.integers(1, VOCAB, 8).tolist()
    text = a + b + a[:10] + b[:5] + a + [0]
    P = len(a) + len(b)
    m = Text(text)
    out = list(m.generate(m.first(P), -1, GREEDY, text[:P]))
    assert out == text[P:]
    assert len(m.calls) < len(out) // 2 and m.drafted_ctx > m.accepted_ctx > len(out) // 2
    p = P
    for c in m.calls:  # (each step at the position after the kept tokens: they are the text's)
        assert c[0] == text[p]
        p += next((i for i in range(1, len(c)) if c[i] != text[p + i]), len(c))
    assert m.stats() == f"{m.drafted_ctx} context draft tokens, {m.accepted_ctx} accepted"
    plain = Text(text)
    plain.lookup = False
    assert list(plain.generate(plain.first(P), -1, GREEDY, text[:P])) == out and len(plain.calls) == len(out) - 1
    one = Text(text)
    one.DRAFTS = 0
    assert list(one.generate(one.first(P), 20, GREEDY, text[:P])) == out[:20] and len(one.calls) == 19
    for drafts in (0, 1, 9):  # (a generation's cap: steps of one, of up to 2 tokens; at most the model's)
        capped = Text(text)
        assert list(capped.generate(capped.first(P), -1, GREEDY, text[:P], drafts=drafts)) == out
        assert max(map(len, capped.calls)) == min(drafts, capped.DRAFTS) + 1
    s = Sampling(greedy=True, repeat_penalty=1.0001, repeat_last_n=8)
    pen = Text(text)
    assert list(pen.generate(pen.first(P), -1, s, text[:P])) == out
    short = Text(text)
    assert list(short.generate(short.first(P), 7, GREEDY, text[:P])) == out[:7] and short.position <= P + 6


def test_spec_options(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """The drafter's options (``add_spec_options``) as ``open_model``'s: none where a step takes no drafts; a drafter
    file's type by its architecture against ``--spec-type``; "none" reaches a model without a drafter, which refuses
    another."""
    p = argparse.ArgumentParser()
    add_spec_options(p)
    assert spec_options(p.parse_args([])) == {}
    f = str(tmp_path / "d.gguf")
    assert spec_options(p.parse_args(["--spec-type", "draft-mtp", "-md", f])) == {"spec_type": "draft-mtp",
                                                                                   "draft_model": f}
    assert spec_options(p.parse_args(["-md", f, "--spec-draft-n-max", "0"])) == {"spec_type": "none"}
    with pytest.raises(SystemExit):
        p.parse_args(["--spec-draft-n-max", "-1"])
    monkeypatch.setattr("models.common.GGUFFile", lambda path: SimpleNamespace(arch=path))
    assert drafter_type("auto", "dflash", ("draft-mtp", "draft-dflash"), "Q") == "draft-dflash"
    assert drafter_type("draft-mtp", "qwen35moe", ("draft-mtp",), "Q") == "draft-mtp"
    for args, error in [(("draft-mtp", "dflash", ("draft-mtp", "draft-dflash")), "a draft-dflash drafter, not draft-mtp"),
                        (("none", "qwen35moe", ("draft-mtp",)), "a draft-mtp drafter, not none"),
                        (("draft-dflash", None, ("draft-mtp",)), "G has no draft-dflash drafter"),
                        (("mtp", None, ("draft-mtp",)), "one of auto")]:
        with pytest.raises(ValueError, match=error):
            drafter_type(*args, "G")

    class Plain:
        name = "Plain"

        def __init__(self, path: str, *, kv_k: str | None = None):
            self.kv_k = kv_k

    monkeypatch.setattr(models, "architecture", lambda _: "plain")
    monkeypatch.setattr(models, "model_class", lambda _: Plain)
    assert open_model("x", kv_k="single", spec_type="none").kv_k == "single"
    with pytest.raises(ValueError, match="no option spec_type"):
        open_model("x", spec_type="draft-mtp")


def test_draft_stop() -> None:
    """A drafter's chain (``OverlayModel._keep``, ``_more``): ``P_MIN`` until ``WARM`` drafter rounds ran (the next
    drafter step always); then a draft joins where its chain's probability, calibrated (``_kappa``: per depth the kept
    drafts per their chains' probability; a new depth from the one before it), beats the rounds' rate times its slot's time, and the drafter guesses another where the
    chain times the last guess's probability beats the rate times the slot and its step: a dense model's (7.1 tokens/s,
    slots of 5 ms, draft steps of 11 ms) chains go on to ~0.04 and ~0.11, an MoE's (30 tokens/s, slots of 12 ms) stop
    at ~0.36."""
    dense = Text([1])
    assert dense._keep(0.6, 0) and not dense._keep(0.4, 0) and dense._more(0.01, 0.01, 1)
    for S, t in enumerate((0.300, 0.305, 0.310, 0.315), 1):
        dense.planner.ran(S, t)
    for _ in range(WARM):
        dense.planner.drafted(0, 2, 0.280)
    dense._row_time(0.011)
    assert dense._keep(0.05, 1) and not dense._keep(0.032, 1)
    assert dense._more(0.2, 0.9, 1) and not dense._more(0.2, 0.5, 1)
    dense._chains = (0.8, 0.4)  # (both kept: the drafter is underconfident; depth 0 from 1, depth 1 from depth 0)
    dense._calibrate(2)
    assert dense._kappa(0) == pytest.approx(0.84 / 0.8) and dense._kappa(1) == pytest.approx(0.872 / 0.72)
    assert dense._kappa(5) == dense._kappa(1) and dense._keep(0.032, 1)
    first = Text([1])
    first._chains = (0.7,)  # (a rejected first draft: the next ones still drafted)
    first._calibrate(0)
    assert first._kappa(0) == pytest.approx(0.8)
    dense._chains = (0.9, 0.6, 0.3)  # (the first kept: the deeper calibrations fall)
    dense._calibrate(1)
    assert dense._kappa(2) < dense._kappa(1) < dense._kappa(0)
    moe = Text([1])
    for S, t in enumerate((0.040, 0.054, 0.066, 0.080), 1):
        moe.planner.ran(S, t)
    for _ in range(WARM):
        moe.planner.drafted(0, 2, 2 / 30)
    assert moe._keep(0.4, 1) and not moe._keep(0.3, 1)
    dense.planner.ran(1, 0.300)
    dense.planner.again = 0.045
    keep = dense._keep(0.1, 1)
    dense.planner.crossing = (False, False, True, True)  # (a step of 3 crosses a group: 0.33 tokens of a pass)
    assert keep and not dense._keep(0.3, 1) and dense._keep(0.1, 2)


class Branchy(Text):
    """``Text`` with a drafter of k tokens a round whose first draft is wrong every third round, a sibling branch then (the
    text's next ``L`` tokens: every other time, its second token wrong) in the place of the chain's last ``L`` drafts; a
    step's branch slots predict as their positions (``branch``: its parent slot, its slots); ``drops``: carries dropped."""

    DRAFTS, L = 7, 2
    STEP_COST = (1.0,) * 8
    prefill = None

    def __init__(self, text: list[int]):
        super().__init__(text)
        self.rounds_n, self.won, self.drops, self.sib, self.branches = 0, [], 0, None, []

    @property
    def _drafting(self) -> bool:
        return True

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        q = self.position + len(self.carry)  # (the token's position)
        d = (self.text[q + 1:q + 1 + k] + [VOCAB - 1] * k)[:k]  # (past the text: wrong)
        self.rounds_n += 1
        self.sib = None
        if self.rounds_n % 3 == 0 and k == self.DRAFTS - len(self.carry) and k > self.L:
            d = [VOCAB - 1, *d[1:]]
            self.sib = (self.text[q + 1:q + 1 + self.L] + [VOCAB - 1] * self.L)[:self.L]
            if self.rounds_n % 2:
                self.sib = [self.sib[0], VOCAB - 1][:self.L]
            d = d[:k - self.L]
        return d

    def _sibling(self) -> list[int] | None:
        return self.sib

    def _verify(self, tokens: list[int], mtp: bool) -> np.ndarray:
        if not mtp or self.sib is None:  # (a round of the context's drafts: no sibling, whatever the last drafter round's)
            return self.steps(tokens)
        self.branches.append((self.position, len(self.carry), list(tokens), list(self.sib)))
        return self.steps([*tokens, *self.sib], branch=(len(self.carry), len(self.sib)))

    def steps(self, tokens: list[int], *, logits: bool = True, branch: tuple[int, int] | None = None) -> np.ndarray:
        out = super().steps(tokens, logits=logits)
        if branch is not None:  # (branch slot j at position p + c + 1 + j: the text's token after it)
            p, (c, L) = self.position - len(tokens), branch
            out = out.astype(np.float32)
            for j in range(L):
                out[len(tokens) - L + j] = -8.0
                out[len(tokens) - L + j, self.text[min(p + c + 2 + j, len(self.text) - 1)]] = 8.0
            out = out.astype(bfloat16)
        return out

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool, won: int = 0) -> None:
        super()._kept(p, tokens, m, mtp, won)
        if won:
            self.won.append(won)

    def _drop_carry(self) -> None:
        self.drops += 1


def test_sibling() -> None:
    """A drafter's sibling branch (``OverlayModel._sibling``): a round whose first draft is wrong keeps the branch's tokens
    that the model's own logits confirm, then the token after them (its slots' logits); the next step runs them again
    before its token (``carry``) and drafts a slot fewer each; the text is greedy's. A generation that ends right after a
    kept branch drops its carry, and so does whatever runs next after a reader that stopped reading (``_abandon``)."""
    rng = np.random.default_rng(3)
    text = rng.integers(1, VOCAB - 1, 120).tolist() + [0]
    m = Branchy(text)
    out = list(m.generate(m.first(10), -1, GREEDY, text[:10]))
    assert out == text[10:]
    assert m.won and set(m.won) == {1, 2} and m.drops == 0
    p = 10
    for c in m.calls:  # (each step at the text's next position, its carry first; a branch's slots after its chain)
        assert c[0] == text[p], (c, p)
        p += next((i for i in range(1, len(c)) if p + i >= len(text) or c[i] != text[p + i]), len(c))
        if p > len(text) - 2:
            break
    for pos, c, tokens, sib in m.branches:
        assert len(tokens) + len(sib) == 8 and tokens[:c + 1] == text[pos:pos + c + 1]
    assert m.accepted == len(out) - 1 - len(m.calls)  # (a round: its token and its kept drafts or branch)
    drops = 0
    for n in range(30, 120):  # (texts that end in rounds of all kinds: some right after a kept branch)
        b = Branchy(text[:n] + [0])
        assert list(b.generate(b.first(10), -1, GREEDY, text[:10])) == text[10:n] + [0]
        drops += b.drops
    assert drops, "a generation that ends with a kept branch's carry"
    left = Branchy(text)  # (a reader that stops right after a kept branch, then a rewind: the carry forgotten once)
    it = left.generate(left.first(10), -1, GREEDY, text[:10])
    while not left.carry:
        next(it)
    left.rewind(left.position)
    assert left.drops == 1 and not left.carry and not left.rounds
    it.close()
    assert left.drops == 1


def test_rewind_recurrent() -> None:
    """A recurrent model rewinds to all its cached positions or none (its states cannot go back: a generation's forced
    call opener continues where it stopped); another to a cache group."""
    for recurrent, want in ((True, (45, 0)), (False, (40, 40))):  # (without a prefill: groups of 8)
        m = Text(list(range(1, 60)))
        m.recurrent, m.prefill, m.position = recurrent, None, 45
        m.reset = lambda m=m: setattr(m, "position", 0)
        assert (m.rewind(45), m.rewind(40)) == want


# the files on the NPU (slow): a Llama, a Gemma with its drafter, qwen35 (MiniCPM-V's language model; Qwen3.5-9B: its
# steps of several tokens past 6 cache groups)
CASES = [llama_gguf("Q4_0", "1B"), gemma_gguf("Q4_0"), minicpmv_gguf(), qwen_gguf("Q4_K_M", "9B")]


@pytest.mark.slow
@pytest.mark.parametrize("path", CASES, ids=[os.path.basename(p) for p in CASES])
def test_greedy(path: str) -> None:
    """A reply that copies its prompt (an essay's paragraphs, ~300 tokens), then another turn after it: greedy with
    drafts is the text of steps of one token, bit for bit, in fewer steps; with a drafter also with the context's
    drafts preferred (the drafter's rate made low: its states follow the context's rounds)."""
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096)
    text = essay("addiction.txt")[:1200]
    ask = [{"role": "user", "content": f"Repeat this text exactly, word for word:\n\n{text}"}]
    ids = model.chat(ask)
    turn = model.chat(ask + [{"role": "assistant", "content": text}, {"role": "user", "content": "And once more."}])

    def run(drafts: int, *, context_first: bool = False) -> tuple[list[int], list[int], int, int]:
        """The reply and the next turn's (its tokens after the reply's cached positions), the reply's steps, the
        context's accepted drafts."""
        npu.reset()
        npu.DRAFTS = drafts
        if context_first:
            pl = npu.planner
            pl.mtp_rounds[:], pl.mtp_tokens[:], pl.mtp_s[:] = WARM, 1.0, 1.0
        d0, a0 = npu.dispatches, npu.accepted_ctx
        out = list(npu.generate(npu.forward(ids), 96, GREEDY, ids))
        steps = npu.dispatches - d0
        p = npu.position
        later = turn[p:] if turn[:p] == (ids + out)[:p] else None
        more = list(npu.generate(npu.forward(later), 32, GREEDY, turn)) if later else []
        return out, more, steps, npu.accepted_ctx - a0

    out, more, steps, _ = run(0)
    got = {"drafts": run(3)} | ({"context first": run(3, context_first=True)} if npu.drafter is not None else {})
    print(f"{model.name}: {len(out)} tokens in {steps} steps of one; " + "; ".join(
        f"{k}: {g[2]} steps, {g[3]} context drafts accepted" for k, g in got.items()) + f" ({npu.stats()})")
    for g in got.values():
        assert g[:2] == (out, more) and g[2] < 0.75 * steps
    assert list(got.values())[-1][3] > len(out) // 3
