"""Context drafts (models/drafts.py: ``Lookup``, ``Planner``) and the generation's rounds of drafts
(``OverlayModel.generate``) on a fake overlay model whose text is fixed; on the NPU (slow) greedy replies with drafts
against steps of one token."""

import os

import numpy as np
import pytest
import torch

from artlite.model_config import gemma_gguf, llama_gguf, minicpmv_gguf, qwen_gguf
from models import open_model
from models.common import OverlayModel, Sampling
from models.drafts import DRAFTER_MATCH, NMAX, PRIOR, RETRY, WARM, Lookup, Planner, bucket

from .common import essay

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
    assert e0 == pytest.approx(1 + PRIOR[4, 0] + PRIOR[4, 0] * PRIOR[4, 1] + np.prod(PRIOR[4]))
    assert moe.choose(5, 3, drafter=False) == 2
    for _ in range(50):
        moe.accepted(5, 3, 0)
    assert moe.expected(5, 3) < 1.4 < e0 and moe.choose(5, 3, drafter=False) == 0
    p = Planner((1.0, 1.05, 1.1, 1.15))
    assert p.choose(20, 3, drafter=True) is None  # (the drafter's first rounds)
    for m in (1, 3, 20):
        for _ in range(WARM):
            p.ran(4, 0.105)
            p.drafted(m, 3 if m > 1 else 1, 0.110)  # (27 tokens/s; after matches of 1: 9)
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


class Text(OverlayModel):
    """A model whose next token after position q is ``text[q + 1]`` (bf16 logits from ``steps``); ``calls``: each
    step's tokens."""

    eos_token_ids = (0,)
    capacity = 256

    def __init__(self, text: list[int]):
        self.text, self.position, self.calls = text, 0, []

    def steps(self, tokens: list[int], *, logits: bool = True) -> torch.Tensor:
        p = self.position
        self.calls.append(list(tokens))
        self.position += len(tokens)
        out = torch.full((len(tokens), VOCAB), -8.0)
        for i in range(len(tokens)):  # (past the text: its end)
            out[i, self.text[min(p + i + 1, len(self.text) - 1)]] = 8.0
        return out.bfloat16()

    def first(self, n: int) -> torch.Tensor:
        """The prompt's last logits after positions 0 .. n - 1 ran."""
        self.position = n
        out = torch.full((VOCAB,), -8.0)
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
    s = Sampling(greedy=True, repeat_penalty=1.0001, repeat_last_n=8)
    pen = Text(text)
    assert list(pen.generate(pen.first(P), -1, s, text[:P])) == out
    short = Text(text)
    assert list(short.generate(short.first(P), 7, GREEDY, text[:P])) == out[:7] and short.position <= P + 6



# the files on the NPU (slow): a Llama, a Gemma with its drafter, qwen35 (MiniCPM-V's language model; Qwen3.5-9B: its
# steps of several tokens past 6 cache groups)
CASES = [llama_gguf("Q4_0", "1B"), gemma_gguf("Q4_0"), minicpmv_gguf(), qwen_gguf("Q4_K_M", "9B")]


@pytest.mark.slow
@pytest.mark.parametrize("path", CASES, ids=[os.path.basename(p) for p in CASES])
def test_greedy(path: str) -> None:
    """A reply that copies its prompt (an essay's paragraphs, ~300 tokens), then another turn after it: greedy with
    drafts is the text of steps of one token, bit for bit, in fewer steps; with a drafter also with the context's
    drafts preferred (the drafter's rate made low: its states follow the context's rounds)."""
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
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
