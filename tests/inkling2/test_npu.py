"""Inkling 2 on the NPU (``models.inkling2.inkling.NpuInkling``), the whole model (UD-Q4_K_XL; the quality tests and the
drafts also UD-Q4_K_M):

- teacher-forced decode steps over an essay against llama.cpp's runs of the same GGUF file (logits,
  ``tests/goldens/text.py``): KL of the next-token distributions and argmax agreement against its decode (ubatch 1),
  next to its own prompt run against it (its graphs differ: Q8_0 activations, and the routing of near-tied experts);
- a checkpoint: the steps after a restore give the logits they gave before it, bit for bit (the convs' state and the
  sliding-window caches back with it);
- steps of S tokens against steps of one, a rollback of rejected drafts, greedy text with the context's drafts: bit for
  bit, the same text;
- greedy text after five prompts against llama.cpp's decode of the same tokens (``test_text``; the prompts through the
  prefill);
- a prompt through the prefill against the same prompt as decode steps (``test_prefill``), and the decode after it
  against llama.cpp's (``test_prefill_quality``).
"""

import numpy as np
import pytest
import torch

from artlite.decode.protocol import SLOTS
from artlite.decode.step import Part
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, inkling_gguf
from models.inkling2.inkling import Inkling, NpuInkling
from models.inkling2.overlay import InklingOverlay
from models.sampling import Sampling

from ..common import essay, essays
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from ..numerics import kl_rows

pytestmark = pytest.mark.slow  # whole GGUF models: run with --run-slow


QUANTS = ["UD-Q4_K_XL", "UD-Q4_K_M"]  # (the files the quality tests take: UD-Q4_K_M's layers 20 .. 23 Q5_K gate / up)


def _path(quant: str = "UD-Q4_K_XL") -> str:
    path = inkling_gguf(quant)
    require(path)
    return path


@pytest.mark.parametrize("quant", QUANTS)
def test_teacher_forced(quant: str) -> None:
    """(the agreement over the tokens whose top two llama.cpp's decode has at least 1 apart, as Flint's: a mixture of
    experts flips near-tied experts on either side's rounding)"""
    path = _path(quant)
    tokens = llama_tokenize(path, essay("founders.txt"))[:128]
    dec = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=1, logits="f32").logits, np.float32).copy())
    pro = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=0, logits="f32").logits, np.float32).copy())
    model = Inkling(path).load(capacity=1024)
    got = torch.from_numpy(model.forced_logits(tokens))
    kl, kl_llama = float(kl_rows(dec, got).mean()), float(kl_rows(dec, pro).mean())
    top2 = torch.topk(dec, 2).values
    sure = top2[:, 0] - top2[:, 1] >= 1
    agree = float((dec.argmax(-1) == got.argmax(-1))[sure].float().mean())
    agree_llama = float((dec.argmax(-1) == pro.argmax(-1))[sure].float().mean())
    print(f"KL {kl:.4f} (llama.cpp prompt vs decode {kl_llama:.4f}), argmax agreement {agree:.3f} ({agree_llama:.3f}) "
          f"over {int(sure.sum())} of {len(tokens)} tokens")
    assert kl < 1.5 * kl_llama and agree > agree_llama - 0.05


def test_prefill() -> None:
    """600 essay tokens through the prefill (the planner's chunks, the last token a decode step: ``NpuInkling``'s
    ``last_step``) against the same prompt as decode steps: the last token's next-token distributions (KL), the greedy
    16 tokens after it."""
    path = _path()
    host = Inkling(path)
    tokens = llama_tokenize(path, essay("founders.txt"))[:600]
    out = {}
    for prefill in (True, False):
        model = host.load(capacity=1024, prefill=prefill)
        logits = np.asarray(model.prompt(tokens, logits=True), np.float32).copy()
        out[prefill] = (logits, list(model.generate(logits, 16, Sampling(greedy=True), tokens)))
        del model
    kl = float(kl_rows(torch.from_numpy(out[False][0][None]), torch.from_numpy(out[True][0][None])).mean())
    print(f"KL {kl:.4f}; text {host.tokenizer.decode(out[True][1])!r}")
    assert kl < 0.01 and out[True][1] == out[False][1]


@pytest.mark.parametrize("quant", QUANTS)
@pytest.mark.parametrize("P", [1024, 2048 + 512])
def test_prefill_quality(P: int, quant: str) -> None:
    """An essay prompt of P tokens through the prefill (its chunks, the last token a decode step), then 64 tokens
    teacher-forced as decode steps, against llama.cpp's decode of the same tokens: KL of the next-token distributions and
    argmax agreement over the sure tokens, next to llama.cpp's own prompt run (a prompt graph over the first P tokens,
    then decode graphs) against its decode: within 2x its KL (P 1024 / 2560: 0.0018 / 0.0025 against 0.0015 / 0.0015;
    our decode steps over the prompt 0.0018 / 0.0020), the same argmax."""
    path = _path(quant)
    Q = 64
    tokens = llama_tokenize(path, essays())[:P + Q]

    def ref(**kw: object) -> torch.Tensor:
        return torch.from_numpy(np.asarray(golden(path, tokens, logits="f32", logits_from=P, **kw).logits, np.float32).copy())

    dec, pro = ref(ubatch=1), ref(ubatch=512, split=P)
    model = Inkling(path).load(capacity=4096)
    got = torch.from_numpy(model.forced_logits(tokens, P))
    kl, kl_llama = float(kl_rows(dec, got).mean()), float(kl_rows(dec, pro).mean())
    top2 = torch.topk(dec, 2).values
    sure = top2[:, 0] - top2[:, 1] >= 1
    agree = float((dec.argmax(-1) == got.argmax(-1))[sure].float().mean())
    agree_llama = float((dec.argmax(-1) == pro.argmax(-1))[sure].float().mean())
    print(f"P {P}: KL {kl:.4f} (llama.cpp prompt vs decode {kl_llama:.4f}), argmax agreement {agree:.3f} "
          f"({agree_llama:.3f}) over {int(sure.sum())} of {Q} tokens; chunks {model.prefill.plan(0, P - 1)}")
    assert kl < 2 * kl_llama and agree > agree_llama - 0.05


def test_checkpoint() -> None:
    """A checkpoint at position 37 (inside a cache group; the conv rings and history windows mid-way), 11 steps after
    it, a restore and the same steps again; then a prompt from position 0 again after a reset."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:48]
    model = Inkling(path).load(capacity=1024)
    model.prompt(tokens[:37])
    cp = model.checkpoint()
    first = [np.asarray(model.step(t), np.float32).copy() for t in tokens[37:]]
    model.restore(37, cp)
    again = [np.asarray(model.step(t), np.float32).copy() for t in tokens[37:]]
    for i, (a, b) in enumerate(zip(first, again, strict=True)):
        assert np.array_equal(a, b), i
    model.reset()
    model.prompt(tokens[:37])
    assert np.array_equal(np.asarray(model.step(tokens[37]), np.float32), first[0])


def test_steps_of_s() -> None:
    """Steps of S = 2, 3, 4 tokens give every position the logits steps of one give, bit for bit (positions 0 .. 79:
    across the first cache group, the history windows and the raw rings at every offset)."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:80]
    model = Inkling(path).load(capacity=1024)
    one = model.forced_logits(tokens, batch=1)
    for S in (2, 3, 4):
        got = model.forced_logits(tokens, batch=S)
        bad = np.flatnonzero((got != one).any(-1))
        assert not bad.size, (S, bad[:8])


def test_rollback() -> None:
    """A step of a token and drafts that are rejected, then the steps after the kept tokens (``SpeculativeMixin``'s
    round: the position back to after them), give the logits the steps that never ran the drafts give, bit for bit: the
    raw rings, the history windows' offset and the open records follow the position (the first 20 tokens a prompt in
    both: the prefill's)."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:48]
    model = Inkling(path).load(capacity=1024)
    want = model.forced_logits(tokens, 20, batch=1)
    model.reset()
    model.prompt(tokens[:20])
    got = []
    i = 20
    for keep, wrong in ((1, [7, 8, 9]), (2, [11, 12]), (3, [13]), (1, []), (4, [])):
        run = tokens[i:i + keep] + wrong
        rows = model.cap(np.asarray(model.steps(run)))
        got += list(rows[:keep])
        model.position = i + keep  # (as ``SpeculativeMixin._kept``: the rejected drafts' positions forgotten)
        i += keep
    while i < len(tokens):
        got.append(model.cap(np.asarray(model.steps([tokens[i]])))[0])
        i += 1
    got = np.stack(got)
    bad = np.flatnonzero((got != want).any(-1))
    assert not bad.size, bad[:8]


def test_ring() -> None:
    """The sliding-window caches as rings (``InklingOverlay(wrap=)``: 18 groups, 1152 positions) give the logits of
    linear caches, bit for bit, past a turn: 1300 essay tokens in steps of 4, the last 64 positions' logits (a ring's
    range in two fills, every sub-step's transfers ordered so that no host wait blocks a core: ``_attention``)."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:1300]
    rows = []
    for wrap in (18, 0):
        model = NpuInkling(GGUFFile(path), capacity=2048, wrap=wrap)
        rows.append(model.forced_logits(tokens, len(tokens) - 64, batch=4))
        del model
    bad = np.flatnonzero((rows[0] != rows[1]).any(-1))
    assert not bad.size, bad[:8]


def test_prefill_rings() -> None:
    """Prompts through the prefill on sliding-window rings (40 groups: each chunk within one turn, ``fits``; a window
    across the wrap from the guard, ``mirror``) and on two sequences (lanes: a prompt of the second between the first's
    prompt and its steps) give the decode steps after them the logits of linear caches on one sequence after the same
    chunks, bit for bit."""
    path = _path()
    g = GGUFFile(path)
    ids = llama_tokenize(path, essays())
    a, b, a2 = ids[:3000], ids[3000:4900], ids[4900:4908]  # (two prompts; the first one's tokens after the second's)
    Q, W = 16, 40
    ring = NpuInkling(g, capacity=4096, wrap=W, prefill=True, sequences=2)
    l0, l1 = ring.lanes
    chunks = []
    run = ring.prefill.run
    ring.prefill.run = lambda e, p0, T=0: (chunks.append(p0), run(e, p0, T))[1]

    def steps(lane: NpuInkling, toks: list[int]) -> np.ndarray:
        return np.concatenate([lane.cap(np.asarray(lane.steps([t]))) for t in toks])

    l0.reset()
    l0.prompt(a[:-Q])
    l1.reset()
    l1.prompt(b[:-Q])
    got_a, got_b, got_a2 = steps(l0, a[-Q:]), steps(l1, b[-Q:]), steps(l0, a2)
    assert any(p // ring.prefill.group % W == 0 for p in chunks[1:]), ("no chunk at a ring's wrap", chunks)
    del ring, l0, l1
    lin = NpuInkling(g, capacity=4096, wrap=0, prefill=True)
    fits = lin.prefill.fits
    lin.prefill.fits = lambda T, p0: fits(T, p0) and (p0 // 64) % W + -(-T // 64) <= W  # (the ring's chunks)
    want_a = lin.forced_logits(a + a2, len(a) - Q)
    want_b = lin.forced_logits(b, len(b) - Q)
    for what, x, y in (("a", np.concatenate([got_a, got_a2]), want_a), ("b", got_b, want_b)):
        bad = np.flatnonzero((x != y).any(-1))
        assert not bad.size, (what, bad[:8])


def test_sequences() -> None:
    """Four sequences on the device (``InklingOverlay(sequences=4)``): alternated steps of one sequence's 1 .. 4 slots,
    and joint steps of 2 .. 4 sequences' parts (dealt at random, one sequence joining late, a sequence's second slot
    too), give every slot the logits of its sequence alone in steps of one, bit for bit (each sequence's caches, raw
    rings, history windows: the residual stream's step per part)."""
    path = _path()
    g = GGUFFile(path)
    cfg = InklingConfig.from_gguf(g)
    N, NSEQ = 40, 4
    ids = llama_tokenize(path, essay("founders.txt"))[:NSEQ * N]
    tn = np.asarray(g.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1)

    def emb(t: list[int]) -> np.ndarray:
        x = g.rows("token_embd.weight", np.asarray(t)).astype(np.float32)
        return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + cfg.rms_norm_eps).astype(np.float32) * tn

    xs = [emb(ids[s * N:(s + 1) * N]) for s in range(NSEQ)]
    ov = InklingOverlay(g, max_positions=1024, sequences=NSEQ)
    ov.load()

    def run(x: np.ndarray, at: int | tuple[Part, ...]) -> np.ndarray:
        return np.asarray(ov.step(x[None], at, logits=True), np.float32).copy()

    def fresh() -> None:
        for s in range(NSEQ):
            ov.select(s)
            ov.reset()

    want = []
    for s in range(NSEQ):  # (alone, in steps of one, as sequence 0)
        ov.select(0)
        ov.reset()
        want.append(np.concatenate([run(xs[s][p:p + 1], p) for p in range(N)]))
    assert not np.array_equal(want[0], want[1])

    def check(got: np.ndarray, at: list[tuple[int, int]], what: str) -> None:
        bad = [(s, p) for row, (s, p) in zip(got, at, strict=True) if not np.array_equal(row, want[s][p])]
        assert not bad, f"{what}: (sequence, position) {bad[:8]} differ from the sequence alone"

    fresh()  # alternated: a step of one sequence, then of the next
    pos, i = [0] * NSEQ, 0
    while min(pos) < N:
        s = i % NSEQ
        n = min((1, 4, 3, 2)[(i // NSEQ + s) % 4], N - pos[s])
        i += 1
        if n:
            ov.select(s)
            check(run(xs[s][pos[s]:pos[s] + n], pos[s]), [(s, pos[s] + k) for k in range(n)], "alternated")
            pos[s] += n
    fresh()  # joint steps
    rng = np.random.default_rng(0)
    pos, steps, parts_seen = [0] * NSEQ, 0, set()
    while min(pos) < N:
        take = [[s, 1] for s in range(NSEQ) if pos[s] < N and not (s == 2 and steps < 6)]
        parts_seen.add(len(take))
        spare = SLOTS - len(take)
        for t in take:
            if spare and rng.random() < 0.4 and pos[t[0]] + t[1] < N:
                t[1] += 1
                spare -= 1
        rng.shuffle(take)
        parts = tuple(Part(s, pos[s], n) for s, n in take)
        got = run(np.concatenate([xs[s][pos[s]:pos[s] + n] for s, n in take]), parts)
        check(got, [(s, pos[s] + k) for s, n in take for k in range(n)], f"step {steps} {parts}")
        for s, n in take:
            pos[s] += n
        steps += 1
    assert {2, 3, 4} <= parts_seen, parts_seen


@pytest.mark.parametrize("quant", QUANTS)
def test_drafts(quant: str) -> None:
    """Greedy text with the context's drafts is the text without them, and the drafts are kept: an essay's first 300
    tokens, then its first 100 again (the model goes on repeating it, as the drafts guess)."""
    path = _path(quant)
    tokens = llama_tokenize(path, essay("founders.txt"))[:300]
    prompt = tokens + tokens[:100]
    model = Inkling(path).load(capacity=1024)
    texts = []
    for drafts in (0, 3):
        model.reset()
        logits = model.prompt(prompt, logits=True)
        model.lookup = drafts > 0
        texts.append(list(model.generate(logits, 64, Sampling(greedy=True), prompt, drafts=drafts)))
    assert texts[0] == texts[1]
    assert model.accepted_ctx > 0.5 * model.drafted_ctx > 0, (model.drafted_ctx, model.accepted_ctx)


NEAR = 1.0  # (logits: a token within this of llama.cpp's top one is a near tie, as test_teacher_forced's sure tokens)
TEXT_PROMPTS = {
    "chat": [{"role": "user", "content": "Hi! I'm planning a weekend trip to Lisbon in October. What should I see, "
                                         "and is it worth taking a day trip to Sintra?"}],
    "code": [{"role": "user", "content": "Write a Python function that merges overlapping intervals, given as a list of "
                                         "(start, end) tuples, and returns them sorted. Include a short docstring."}],
    "math": [{"role": "user", "content": "A train leaves at 9:40 and travels 270 km at an average of 90 km/h, then "
                                         "waits 25 minutes and covers another 120 km at 80 km/h. When does it arrive? "
                                         "Show the steps."}],
    "instruction": [{"role": "user", "content": (
        "You are helping me prepare a one-page brief for a city council meeting about adding protected bike lanes on "
        "Main Street. Please: (1) open with a two-sentence summary of the proposal; (2) list three benefits, each with "
        "one concrete example from another city; (3) list the two strongest objections local shop owners are likely "
        "to raise, and a fair response to each; (4) propose a six-month pilot with measurable goals; (5) end with a "
        "single question the council should vote on. Keep the tone neutral and avoid jargon, use short paragraphs, "
        "and do not invent statistics: where a number would help, say what should be measured instead.")}],
}


@pytest.mark.parametrize("quant", QUANTS)
def test_text(quant: str) -> None:
    """Greedy text of 128 tokens after five prompts (a chat, code, math, a long instruction as chat turns without
    thinking; an essay's first 400 tokens) against llama.cpp's decode of the prompt and our tokens (``goldens.text``,
    teacher-forced on ours): each of our tokens is llama.cpp's top one, or within ``NEAR`` of it (a near tie)."""
    path = _path(quant)
    host = Inkling(path)
    prompts = {k: host.chat(m) for k, m in TEXT_PROMPTS.items()}
    prompts["essay"] = llama_tokenize(path, essay("founders.txt"))[:400]
    model = host.load(capacity=1024)
    near, total = 0, 0
    for name, toks in prompts.items():
        model.reset()
        logits = model.prompt(toks, logits=True)
        out = list(model.generate(logits, 128, Sampling(greedy=True), toks))
        assert len(out) == 128 or out[-1] in host.eos_token_ids, (name, len(out))
        ref = np.asarray(golden(path, toks + out, ubatch=512, split=len(toks), logits="f32",
                                logits_from=len(toks) - 1).logits, np.float32)
        gap = ref[np.arange(len(out)), :].max(-1) - ref[np.arange(len(out)), out]
        bad = np.flatnonzero(gap >= NEAR)
        near += int(((gap > 0) & (gap < NEAR)).sum())
        total += len(out)
        print(f"{name}: {len(out)} tokens, {int((gap == 0).sum())} llama.cpp's top, {int(((gap > 0) & (gap < NEAR)).sum())} "
              f"near ties: {host.tokenizer.decode(out)[:160]!r}")
        assert not bad.size, (name, bad[:8], gap[bad[:8]])
    print(f"{total} tokens, {near} near ties")
