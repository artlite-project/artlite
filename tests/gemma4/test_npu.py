"""Gemma 4 on the NPU against its host models, on the GGUF file (models.gemma4):

- slices of layers (one sliding-window and one full-attention layer; the layers that share a KV
  cache with the ones that own it) through ``GemmaOverlay.step``, one token or four per
  dispatch, K cached as hi + lo or single blocks: logits, attention outputs, the residual and the
  caches against the overlay's arithmetic on the host (ref.GemmaNpuRef);
- the whole model, teacher-forced: KL of the next-token distributions against the fp32
  reference (gemma_cpu.GemmaCpu) on the same weights; prompt tokens run in batches without the
  layers that share a cache;
- multi-token prediction: the drafter on the NPU against the fp32 drafter; a generated text is
  the model's own (every token the argmax of the logits of its position, which agree with the
  same text's steps of one token), in fewer dispatches than tokens;
- the tokenizer and the chat template on fixed examples (ids as llama.cpp's llama-tokenize).
"""

import os

import pytest
import torch

from artlite.decode.design import SLOTS as NB
from artlite.decode.design import XB, packed_slots
from artlite.gguf_model import GGUFFile
from artlite.model_config import assistant_of, gemma_gguf
from models.common import Sampling
from models.gemma4 import attn as AT
from models.gemma4.gemma import NpuGemma
from models.gemma4.overlay import GemmaOverlay
from models.gemma4.tokenizer import GemmaTokenizer, chat_prompt

from ..common import cached_ref, model_key
from . import cpu as gemma_cpu
from . import ref

pytestmark = pytest.mark.slow  # the GGUF model + CPU references: run with --run-slow

VOCAB = 2048
PROMPT = "Explain how photosynthesis works in plants, step by step."
# a fixed continuation (teacher forcing): the CPU and NPU models see the same tokens
TEXT = ("Photosynthesis is the process by which plants, algae and some bacteria convert light energy into "
        "chemical energy. It takes place in the chloroplasts, mainly in the leaves.")
# a prompt of decode steps and prefill chunks (test_gemma_prefill)
LONG = ("Water moves through the environment in a continuous cycle. The sun heats oceans, lakes and rivers, and "
        "some of the water evaporates into the air as vapor. Plants release more vapor through tiny pores in their "
        "leaves. As warm, moist air rises, it cools, and the vapor condenses around dust and salt particles to form "
        "clouds. When the droplets in a cloud grow heavy enough, they fall as rain, snow, sleet or hail. Some of this "
        "precipitation soaks into the ground and refills underground aquifers, some runs off into streams, and some "
        "is stored for months in glaciers and snowpacks. Engineers who design dams, farmers who plan irrigation and "
        "cities that manage storm drains all depend on understanding these steps. Summarize the main stages of the "
        "cycle, and then explain what powers it.")


def _gguf() -> GGUFFile:
    path = gemma_gguf("Q4_0")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    return GGUFFile(path)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """Relative rms error (inf when got is not finite: a NaN would compare as no error)."""
    if not torch.isfinite(got.float()).all():
        return float("inf")
    return float((got.float() - want.float()).pow(2).sum().sqrt() / want.float().pow(2).sum().sqrt())


def _rel_positions(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per position of a cache (heads, positions, head_dim): the relative rms error."""
    d, w = (got.float() - want.float()).transpose(0, 1).flatten(1), want.float().transpose(0, 1).flatten(1)
    return d.norm(dim=1) / w.norm(dim=1)


def _kl(want: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    lr, lp = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    return (lr.exp() * (lr - lp)).sum(-1)


class _Cpu:
    """The fp32 reference, built (its matrices dequantized once for the module) on first use: the
    tests' reference results are cached (tests/refs), and a run that finds them all never builds it."""

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self) -> object:
        if self._model is None:
            self._model = gemma_cpu.GemmaCpu(_gguf())
            self._model.preload()
        return self._model

    def logits(self, seq: list[int], rows: slice | list[int]) -> torch.Tensor:
        """The fp32 logits of seq's positions ``rows`` (cached)."""
        path = _gguf().path
        key = (model_key(path), seq, (rows.start, rows.stop) if isinstance(rows, slice) else rows)
        return cached_ref(path, "cpu_logits", key, lambda: self.model.logits(self.model.hidden(seq))[rows],
                          sources=gemma_cpu.SOURCES)


@pytest.fixture(scope="module")
def cpu() -> _Cpu:
    return _Cpu()


@pytest.fixture(scope="module")
def npu() -> object:
    """The model on the NPU, with its drafter when the file is there."""
    return NpuGemma(_gguf(), capacity=1024)


# (the layers, the relative rms error allowed): the host model leaves out the panels' 14 bits and the 32-k sums'
# bf16 rounding, a few 0.1% per stage; a slice fed the raw embedding is not what its layers were trained on, and the
# four layers from 22 on double the difference with every layer
@pytest.mark.parametrize("kv_k", ["hilo", "single"])
@pytest.mark.parametrize("batch", [1, 3, NB], ids=["one-token", "batch3", "batch"])  # (batch 3: steps of 3 and 2)
@pytest.mark.parametrize(("layers", "tol"), [([0, 5], 0.015), ([22, 23, 24, 29], 0.04)], ids=["swa+full", "shared-kv"])
def test_gemma_overlay_layers(layers: list[int], tol: float, batch: int, kv_k: str) -> None:
    import numpy as np

    g = _gguf()
    ids = GemmaTokenizer(g).encode(chat_prompt(g, [{"role": "user", "content": PROMPT}]))[:20]
    model = ref.GemmaNpuRef(g, layers, VOCAB, kp=2 if kv_k == "hilo" else 1)
    trace: dict[str, torch.Tensor] = {}
    want = model.run(ids, trace)
    x = model.embed(ids)
    e = model.layer_rows(ids)

    ov = GemmaOverlay(g, layers=layers, vocab=VOCAB, max_positions=64, kv_k=kv_k)
    ov.load()
    worst: dict[str, float] = {}

    def keep(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
        worst[name] = max(worst.get(name, 0.0), _rel(got, want))

    for t0 in range(0, len(ids), batch):
        n = min(batch, len(ids) - t0)
        logits = ov.step(x[t0:t0 + n], e[t0:t0 + n], t0).float()
        ov.dev.fetch("ACT")
        act = ov.dev.view("ACT")
        for s in range(n):
            t = t0 + s
            keep("logits", logits[s], want[t])
            keep("r", ov.residual(s), trace["r"][t])
            for layer in layers:
                rg = ov.inp[(layer, "o")][0]
                sp = packed_slots(n)
                o = act[rg.off + XB:rg.off + XB + 2 * sp * rg.K].view(np.uint16).reshape(-1, sp, 8)[:, s].reshape(-1)
                keep(f"attn.{layer}", torch.from_numpy(o.copy()).view(torch.bfloat16), trace[f"attn.{layer}"][t])
    r0 = (len(ids) - 1) // 8 * 8  # (the last record: its values in the ring)
    for layer, cache in model.caches.items():  # the caches as the last token left them
        k, v = ov.read_kv(layer, range(len(ids)))
        worst[f"K.{layer}"], worst[f"V.{layer}"] = _rel(k, cache.K), _rel(v, cache.V)
        ov.dev.fetch("KV")
        rows = [AT.ring_rows(ov.dev.view("KV")[ov.ring(layer, h) + (r0 // 8 & 1) * ov.T:][:ov.T], ov.cfg.head_dim(layer))
                for h in range(AT.KV_HEADS)]
        worst[f"ring.{layer}"] = _rel(torch.from_numpy(np.stack(rows)[:, :len(ids) - r0]), cache.rows[:, r0:])
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < tol for v in worst.values()), worst


def test_gemma_npu_teacher_forced(cpu: _Cpu, npu: object) -> None:
    """The prompt in batches without logits, then the text one token and four tokens per dispatch."""

    g = npu.gguf
    tok = GemmaTokenizer(g)
    prompt = tok.encode(chat_prompt(g, [{"role": "user", "content": PROMPT}]))
    seq = prompt + tok.encode(TEXT)
    first = len(prompt) - 1  # the prompt's tokens before its last run without logits
    want = cpu.logits(seq, slice(first, None))
    for batch in (1, NB):
        got = npu.forced_logits(seq, first, batch=batch)
        kl = float(_kl(want, got).mean())
        agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
        print(f"{batch} per dispatch: KL(same) {kl:.5f}, argmax agreement {agree:.3f} over {len(seq) - first} tokens")
        assert kl < 0.003 and agree > 0.9, f"{batch} per dispatch: KL(same) {kl:.5f}, argmax agreement {agree:.3f}"


def test_gemma_prefill(cpu: _Cpu, npu: object) -> None:
    """A prompt through the prefill (NpuGemma.prompt: a chunk of 5 tokens at 0, decode steps to position 32, a chunk
    of the rest) leaves the KV caches the decode path leaves, at every position up to the rounding of the two paths'
    GEMM inputs (single bfp16 panels here, hi / lo there: 1-3% per position through layer 11); the logits of the text
    after it against the fp32 reference. (Deeper layers amplify that rounding at some positions of this prompt by tens:
    the fp32 model with the prefill's GEMM-input rounding alone moves position 71's layer-23 K by ~30%, and the prefill
    leaves it ~90% off the fp32 model's with or without the decode steps; the logits cover those layers.)"""

    g = npu.gguf
    tok = GemmaTokenizer(g)
    prompt = tok.encode(chat_prompt(g, [{"role": "user", "content": LONG}]))
    assert len(prompt) > 40, len(prompt)
    seq = prompt + tok.encode(TEXT)
    first = len(prompt) - 1
    layers = (0, 5, 11)  # sliding-window and full-attention layers, before the deep ones amplify a position's rounding
    npu.reset()
    for i in range(0, first, NB):  # the decode path
        npu.steps(seq[i:min(i + NB, first)], logits=False)
    want_kv = {li: npu.overlay.read_kv(li, range(first)) for li in layers}
    npu.reset()
    d0 = npu.dispatches
    npu.prompt(seq[:5])  # one chunk of 5 tokens
    npu.prompt(seq[5:first])  # decode steps to position 32, then a chunk
    assert npu.position == first and npu.dispatches - d0 < 15
    worst = {}  # (the worst position)
    for li in layers:
        for name, got, want in zip("KV", npu.overlay.read_kv(li, range(first)), want_kv[li]):
            worst[f"{name}.{li}"] = round(float(_rel_positions(got, want).max()), 4)
    got = torch.cat([gemma_cpu.softcap(npu.steps(seq[i:i + NB]).float(), npu.cfg.final_logit_softcapping)
                     for i in range(first, len(seq), NB)])
    want = cpu.logits(seq, slice(first, None))
    kl = float(_kl(want, got).mean())
    agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
    print(worst, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}")
    assert all(v < (0.03 if k[0] == "K" else 0.04) for k, v in worst.items()), worst  # (K.11 0.021, V.11 0.029)
    assert kl < 0.0006 and agree > 0.94, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}"  # (0.00020, 1.0)


def test_gemma_forward(cpu: _Cpu, npu: object) -> None:
    """NpuGemma.forward with the prompt's last token in a prefill chunk (a chunk, then a step from the prefill's last
    layer on: GemmaOverlay.step(start=)): its logits against the fp32 reference's and those of the same token as a
    whole decode step after the prompt's other tokens; the next step continues from both."""

    g = npu.gguf
    tok = GemmaTokenizer(g)
    prompt = tok.encode(chat_prompt(g, [{"role": "user", "content": LONG}]))
    npu.reset()
    assert sum(npu.prefill.plan(0, len(prompt))) >= len(prompt), "the last token in a chunk"
    nxt = tok.encode(TEXT)[0]
    want = cpu.logits(prompt + [nxt], slice(len(prompt) - 1, None))
    out = {}
    for name in ("forward", "steps"):
        npu.reset()
        if name == "forward":
            d0 = npu.dispatches
            a = npu.forward(prompt)
            assert npu.dispatches - d0 < 15
        else:
            npu.prompt(prompt[:-1])
            a = npu.step(prompt[-1])
        out[name] = torch.stack([a, npu.step(nxt)])
    kl = {k: [round(float(x), 5) for x in _kl(want, v)] for k, v in out.items()}
    between = [round(float(x), 5) for x in _kl(out["steps"], out["forward"])]
    print(f"KL(same) {kl}, forward vs steps {between}")
    assert all(x < 0.002 for v in kl.values() for x in v) and all(x < 0.002 for x in between), (kl, between)
    assert (out["forward"].argmax(-1) == want.argmax(-1)).all()


def test_gemma_draft(cpu: _Cpu, npu: object) -> None:
    """Draft steps of the drafter on the NPU (on the NPU's caches, from the NPU's hidden state)
    against the fp32 drafter on the fp32 model's."""

    if npu.drafter is None:
        pytest.skip("no drafter next to the model")
    g = npu.gguf
    tok = GemmaTokenizer(g)
    ids = tok.encode(chat_prompt(g, [{"role": "user", "content": PROMPT}])) + tok.encode(TEXT)
    positions = (1, 2, 9, 17, 32, 33, 40)  # (32 / 33: the sliding-window cache's group boundary)

    def drafts() -> list[torch.Tensor]:
        """The fp32 drafter's logits: 3 steps at each position, each continuing with the previous argmax."""
        asst = gemma_cpu.GemmaAssistant(GGUFFile(assistant_of(g.path)), cpu.model)
        state, out, done = gemma_cpu.GemmaState(), [], 0
        for p in positions:
            h = cpu.model.hidden(ids[done:p], state)[-1]
            done, t = p, ids[p]
            for _ in range(3):
                want, h = asst.step(t, h, state, p)
                out.append(want)
                t = int(want.argmax())
        return out

    wants = iter(cached_ref(g.path, "draft", (model_key(g.path), model_key(assistant_of(g.path)), ids, positions),
                            drafts, sources=gemma_cpu.SOURCES))
    npu.reset()
    kls, agree, done = [], [], 0
    for p in positions:
        npu.forward(ids[done:p])
        done = p
        t, x_h = ids[p], npu.hidden_state()
        for _ in range(3):
            want = next(wants)
            got = torch.empty(npu.drafter.vocab)  # (the drafter's logits in vocabulary order)
            got[torch.from_numpy(npu.drafter.rows)] = npu.drafter.step(npu.inputs([t])[0][0], x_h, p).float()
            kls.append(float(_kl(want, got)))
            agree.append(int(want.argmax()) == int(got.argmax()))
            t, x_h = int(want.argmax()), None  # both sides continue with the reference's draft
    kl, ag = sum(kls) / len(kls), sum(agree) / len(agree)
    print(f"{len(kls)} draft steps: KL {kl:.5f} (max {max(kls):.4f}), argmax agreement {ag:.3f}")
    assert kl < 0.02 and ag >= 0.8, f"KL {kl:.5f}, argmax agreement {ag:.3f}"


def test_gemma_mtp_generation(npu: object) -> None:
    """Greedy generation with drafts: fewer dispatches than tokens, every token the argmax of the
    logits the NPU computed for its position, and those logits the ones of the same text's steps of
    one token (no reference cache: the text depends on the NPU's near-ties; the steps' agreement
    with the fp32 model is test_gemma_npu_teacher_forced's)."""

    if npu.drafter is None:
        pytest.skip("no drafter next to the model")
    g = npu.gguf
    ids = GemmaTokenizer(g).encode(chat_prompt(g, [{"role": "user", "content": PROMPT}]))
    log: list[tuple[int, list[int], torch.Tensor]] = []
    steps = npu.steps

    def recording(tokens: list[int], *, logits: bool = True) -> torch.Tensor | None:
        p0 = npu.position
        raw = steps(tokens, logits=logits)
        if raw is not None:
            log.append((p0, list(tokens), gemma_cpu.softcap(raw.float(), npu.cfg.final_logit_softcapping)))
        return raw

    npu.reset()
    npu.steps = recording
    try:
        first = npu.forward(ids)
        log.append((len(ids) - 1, ids[-1:], first[None]))  # (the prompt's last token: through the prefill's step)
        d0 = npu.dispatches
        out = list(npu.generate(first, 48, Sampling(greedy=True)))
        dispatches = npu.dispatches - d0
    finally:
        del npu.steps
    seq = ids + out
    got: dict[int, torch.Tensor] = {}
    for p0, toks, lg in log:  # the logits after position p0 + i: valid if the text's tokens ran up to it
        for i in range(len(toks)):
            if toks[:i + 1] != seq[p0:p0 + i + 1]:
                break
            got[p0 + i] = lg[i]
    pos = list(range(len(ids) - 1, len(seq) - 1))
    mine = torch.stack([got[p] for p in pos])
    assert [int(t) for t in mine.argmax(-1)] == out, "a token is not the argmax of its position's logits"
    assert npu.position == len(seq) - 1, "the rejected drafts' positions were not forgotten"
    want = npu.forced_logits(seq[:-1], len(ids) - 1)
    kl = float(_kl(want, mine).mean())
    agree = float((want.argmax(-1) == mine.argmax(-1)).float().mean())
    print(f"{len(out)} tokens in {dispatches} dispatches; KL(same) {kl:.5f}, argmax agreement {agree:.3f}")
    assert len(out) == 48 and dispatches <= 40, f"{len(out)} tokens in {dispatches} dispatches"
    assert kl < 0.001 and agree > 0.95, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}"


def test_gemma_tokenizer() -> None:

    g = _gguf()
    tok = GemmaTokenizer(g)
    cases = {
        "Hello, world!": [9259, 236764, 1902, 236888],
        "  two  spaces\n\nand a blank line": [138, 13498, 138, 35220, 108, 624, 496, 11580, 1757],
        "naïve café — 日本語 🙂": [1789, 238527, 560, 33443, 2192, 33375, 238582, 57235],
        "def f(x):\n\treturn x**2  # square": [2063, 517, 236769, 236781, 1473, 107, 255968, 2060, 1123, 1018, 236778, 138,
                                               236865, 6281],
    }
    for text, ids in cases.items():
        assert tok.encode(text) == ids, text
        assert tok.decode(ids) == text
    prompt = chat_prompt(g, [{"role": "user", "content": "Why is the sky blue?"}])
    assert prompt == "<bos><|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n"
    assert tok.encode(prompt) == [2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105, 4368, 107]
