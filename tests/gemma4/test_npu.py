"""Gemma 4 on the NPU against its host models, on the GGUF file (models.gemma4):

- slices of layers (one sliding-window and one full-attention layer; the layers that share a KV
  cache with the ones that own it) through ``GemmaOverlay.step``, one, three or four tokens per
  dispatch, K cached as hi + lo or single blocks: logits, attention outputs, the residual and the
  caches against the overlay's arithmetic on the host (ref.GemmaNpuRef);
- four sequences on one device, alternated and in joint steps: every slot the logits and residual of its sequence
  alone, bit for bit;
- the whole model, teacher-forced: KL of the next-token distributions against the fp32
  reference (gemma_cpu.GemmaCpu) on the same weights; prompt tokens run in batches without the
  layers that share a cache;
- multi-token prediction: the drafter on the NPU against the fp32 drafter; a generated text is
  the model's own (every token the argmax of the logits of its position, which agree with the
  same text's steps of one token), in fewer dispatches than tokens; its drafts after prompts through
  the prefill as after decode steps (the tokenizer: test_tokenizer.py).
"""


import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.decode.protocol import SLOTS as NB
from artlite.decode.protocol import XB, packed_slots
from artlite.decode.step import Part
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import assistant_of, gemma_gguf
from models.gemma4 import attn as AT
from models.gemma4.gemma import Gemma, NpuGemma
from models.gemma4.host import GemmaHost
from models.gemma4.overlay import GemmaOverlay
from models.gemma4.tokenizer import GemmaTokenizer, chat_prompt
from models.sampling import Sampling

from ..common import as_torch, cached_ref, draft_gate, draft_prompts, model_key, to_torch
from ..model_files import require
from ..numerics import kl_rows, rel_finite
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
    require(path)
    return GGUFFile(path)


def _rel_positions(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per position of a cache (heads, positions, head_dim): the relative rms error."""
    got, want = as_torch(got), as_torch(want)
    d, w = (got.float() - want.float()).transpose(0, 1).flatten(1), want.float().transpose(0, 1).flatten(1)
    return d.norm(dim=1) / w.norm(dim=1)


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
@pytest.mark.parametrize(("layers", "tol"), [([0, 5], 0.015), ([22, 23, 24, 29], 0.04)], ids=["swa+full", "shared-kv"])
def test_gemma_overlay_layers(layers: list[int], tol: float, kv_k: str) -> None:
    """The text in steps of 1 token, of 3 (steps of 3 and 2) and of NB on one overlay (``reset`` between the runs);
    the worst error over the runs."""
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
        worst[name] = max(worst.get(name, 0.0), rel_finite(got, want))

    for batch in (1, 3, NB):
        ov.reset()
        for t0 in range(0, len(ids), batch):
            n = min(batch, len(ids) - t0)
            logits = to_torch(ov.step(x[t0:t0 + n], e[t0:t0 + n], t0)).float()
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
            keep(f"K.{layer}", k, cache.K)
            keep(f"V.{layer}", v, cache.V)
            ov.dev.fetch("KV")
            rows = [ring_rows(ov.dev.view("KV")[ov.ring(layer, h) + (r0 // 8 & 1) * ov.T:][:ov.T],
                              ov.cfg.head_dim(layer)) for h in range(AT.KV_HEADS)]
            keep(f"ring.{layer}", torch.from_numpy(np.stack(rows)[:, :len(ids) - r0]), cache.rows[:, r0:])
        print(f"{batch} per dispatch:", {k: round(v, 4) for k, v in worst.items()})
        assert all(v < tol for v in worst.values()), (batch, worst)


STEPS = (1, 4, 3, 2, 4, 4, 1, 3, 3, 2, 1, 4)  # (step sizes, repeated: every alignment to the records and their pairs)


@pytest.mark.parametrize("layers", [[0, 5], [22, 23, 24, 29]], ids=["swa+full", "shared-kv"])
def test_gemma_overlay_sequences(layers: list[int]) -> None:
    """Four sequences on one device (``GemmaOverlay(sequences=4)``: texts of their own, at positions of their own)
    give, slot by slot, the logits and the residual of each sequence alone in steps of one, bit for bit:

    - alternated (``select``: a step of 1 .. 4 slots of one, then of the next: no copies between them);
    - in joint steps (``step``'s parts: one sequence 5 positions behind, one joining at the 12th step, slots dealt at
      random, a sequence's second slot too: sub-steps of every pattern up to a sequence per slot in both cache
      kinds), some of them with the tokens' per-layer projections as the steps alone left them (``step(p=)``: without
      the projection's phase);

    and each sequence's caches and open records' rings are then those of the sequence alone."""
    g = _gguf()
    N, NSEQ = 44, 4  # (crosses the cache groups: 32 positions at head_dim 256, 16 at 512)
    ids = GemmaTokenizer(g).encode(LONG + " " + TEXT)[:NSEQ * N]
    assert len(ids) == NSEQ * N
    host = GemmaHost(g)
    xs = [host.embed(ids[s * N:(s + 1) * N]).astype(bfloat16).astype(np.float32) for s in range(NSEQ)]
    es = [host.layer_rows(ids[s * N:(s + 1) * N]) for s in range(NSEQ)]
    ov = GemmaOverlay(g, layers=layers, vocab=VOCAB, max_positions=64, sequences=NSEQ)
    ov.load()
    own = [L for L in layers if ov.cfg.kv_source(L) == L]

    def slots(n: int) -> list[np.ndarray]:  # (per slot of the last step of n slots: its logits and residual)
        lg = np.asarray(ov._npu_logits(n)).view(np.uint16)
        return [np.concatenate([lg[s], ov.residual(s).view(np.uint32)], dtype=np.int64) for s in range(n)]

    def caches(s: int) -> list[np.ndarray]:  # (sequence s's caches as stored, its open records' rows)
        ov.select(s)
        ov.dev.fetch("KV")
        out = [a for L in own for a in ov.read_kv(L, range(N))]
        r0 = (N - 1) // 8
        rows = N - 8 * r0
        return out + [ring_rows(ov.dev.view("KV")[ov.ring(L, h) + (r0 & 1) * ov.T:][:ov.T], ov.cfg.head_dim(L))[:rows]
                      for L in own for h in range(AT.KV_HEADS)]

    want, kept, proj = [], [], []
    for s in range(NSEQ):  # (alone, in steps of one, as sequence 0)
        ov.select(0)
        ov.reset()
        row, ps = [], []
        for p in range(N):
            ov.step(xs[s][p:p + 1], es[s][p:p + 1], p)
            row.append(slots(1)[0])
            ps.append(ov.read_p(1)[0].copy())
        want.append(row)
        proj.append(ps)
        kept.append(caches(0))
    assert not any(np.array_equal(a, b) for a, b in zip(want[0], want[1])), "the sequences' texts differ"

    def check(got: list[np.ndarray], at: list[tuple[int, int]], what: str) -> None:
        bad = [(s, p) for g_, (s, p) in zip(got, at, strict=True) if not np.array_equal(g_, want[s][p])]
        assert not bad, f"{what}: (sequence, position) {bad[:8]} differ from the sequence alone"

    def fresh() -> None:
        for s in range(NSEQ):
            ov.select(s)
            ov.reset()

    fresh()  # alternated: a step of one sequence, then of the next
    pos, i = [0] * NSEQ, 0
    while min(pos) < N:
        s = i % NSEQ
        n = min(STEPS[(i // NSEQ + s) % len(STEPS)], N - pos[s])
        i += 1
        if n:
            ov.select(s)
            ov.step(xs[s][pos[s]:pos[s] + n], es[s][pos[s]:pos[s] + n], pos[s])
            check(slots(n), [(s, pos[s] + k) for k in range(n)], "alternated")
            pos[s] += n
    for s in range(NSEQ):
        assert all(np.array_equal(a, b) for a, b in zip(caches(s), kept[s], strict=True)), f"alternated: caches of {s}"

    fresh()  # joint steps
    rng = np.random.default_rng(0)
    pos, steps, patterns, projected = [0] * NSEQ, 0, set(), set()
    while min(pos) < N:
        take = [[s, 1] for s in range(NSEQ) if pos[s] < N and not (s == 1 and steps < 5) and not (s == 2 and steps < 12)]
        spare = NB - len(take)
        for t in take:  # (a second slot of its own)
            if spare and rng.random() < 0.4 and pos[t[0]] + t[1] < N:
                t[1] += 1
                spare -= 1
        rng.shuffle(take)
        parts = tuple(Part(s, pos[s], n) for s, n in take)
        at = [(s, pos[s] + k) for s, n in take for k in range(n)]
        patterns.add(ov._join(parts))
        p = np.stack([proj[s][q] for s, q in at]) if rng.random() < 0.5 else None
        ov.step(np.concatenate([xs[s][pos[s]:pos[s] + n] for s, n in take]),
                np.concatenate([es[s][pos[s]:pos[s] + n] for s, n in take]), parts, p=p)
        projected.add(ov.ple)
        check(slots(len(at)), at, f"step {steps}{'' if p is None else ' (kept projections)'}")
        if p is None:
            assert np.array_equal(ov.read_p(len(at)), np.stack([proj[s][q] for s, q in at])), f"step {steps}: p"
        for s, n in take:
            pos[s] += n
        steps += 1
    assert projected == {True, False}
    assert all(max(len(u) for u in k) == NB for k in zip(*patterns)), "a step of a sequence per slot, in both kinds"
    for s in range(NSEQ):
        assert all(np.array_equal(a, b) for a, b in zip(caches(s), kept[s], strict=True)), f"joint steps: caches of {s}"


def test_gemma_npu_teacher_forced(cpu: _Cpu, npu: object) -> None:
    """The prompt in batches without logits, then the text one token and four tokens per dispatch."""

    g = npu.gguf
    tok = GemmaTokenizer(g)
    prompt = tok.encode(chat_prompt(g, [{"role": "user", "content": PROMPT}]))
    seq = prompt + tok.encode(TEXT)
    first = len(prompt) - 1  # the prompt's tokens before its last run without logits
    want = cpu.logits(seq, slice(first, None))
    for batch in (1, NB):
        got = torch.from_numpy(npu.forced_logits(seq, first, batch=batch))
        kl = float(kl_rows(want, got).mean())
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
    got = torch.cat([gemma_cpu.softcap(to_torch(npu.steps(seq[i:i + NB])).float(), npu.cfg.final_logit_softcapping)
                     for i in range(first, len(seq), NB)])
    want = cpu.logits(seq, slice(first, None))
    kl = float(kl_rows(want, got).mean())
    agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
    print(worst, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}")
    assert all(v < (0.03 if k[0] == "K" else 0.04) for k, v in worst.items()), worst  # (K.11 0.021, V.11 0.029)
    assert kl < 0.0006 and agree > 0.94, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}"  # (0.00020, 1.0)

    # NpuGemma.forward with the prompt's last token in a prefill chunk (a chunk, then a step from the prefill's last
    # layer on: GemmaOverlay.step(start=)): the logits of the first two text tokens' rows (0 and 1 of ``want``)
    # against the fp32 reference's and those of the same token as a whole decode step after the prompt's other tokens;
    # the next step continues from both
    assert sum(npu.prefill.plan(0, len(prompt))) >= len(prompt), "the last token in a chunk"
    nxt = seq[len(prompt)]
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
        out[name] = torch.from_numpy(np.stack([a, npu.step(nxt)]))
    kls = {k: [round(float(x), 5) for x in kl_rows(want[:2], v)] for k, v in out.items()}
    between = [round(float(x), 5) for x in kl_rows(out["steps"], out["forward"])]
    print(f"forward: KL(same) {kls}, forward vs steps {between}")
    assert all(x < 0.002 for v in kls.values() for x in v) and all(x < 0.002 for x in between), (kls, between)
    assert (out["forward"].argmax(-1) == want[:2].argmax(-1)).all()


def test_gemma_kept_projections(npu: object) -> None:
    """Steps of tokens whose per-layer projections the host kept (``NpuGemma.steps``: it writes them, the step runs
    without the projection phase) give the logits of the steps that projected them, bit for bit."""
    g = npu.gguf
    ids = GemmaTokenizer(g).encode(chat_prompt(g, [{"role": "user", "content": PROMPT}]))[:12]
    npu._p.clear()
    runs = []
    for _ in range(2):
        npu.reset()
        runs.append([to_torch(npu.steps(ids[i:i + NB]).copy()) for i in range(0, len(ids), NB)])
    assert not npu.overlay.ple
    for a, b in zip(*runs, strict=True):
        assert torch.equal(a, b)


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
            got[torch.from_numpy(npu.drafter.rows)] = to_torch(npu.drafter.step(npu.inputs([t])[0][0], x_h, p)).float()
            kls.append(float(kl_rows(want, got)))
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
            log.append((p0, list(tokens), gemma_cpu.softcap(to_torch(raw).float(), npu.cfg.final_logit_softcapping)))
        return raw

    npu.reset()
    npu.steps = recording
    try:
        first = npu.forward(ids)
        log.append((len(ids) - 1, ids[-1:], torch.from_numpy(first)[None]))  # (the prompt's last token: through the prefill's step)
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
    want = torch.from_numpy(npu.forced_logits(seq[:-1], len(ids) - 1))
    kl = float(kl_rows(want, mine).mean())
    agree = float((want.argmax(-1) == mine.argmax(-1)).float().mean())
    print(f"{len(out)} tokens in {dispatches} dispatches; KL(same) {kl:.5f}, argmax agreement {agree:.3f}")
    assert len(out) == 48 and dispatches <= 40, f"{len(out)} tokens in {dispatches} dispatches"
    assert kl < 0.001 and agree > 0.95, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}"


def test_gemma_mtp_prefill(npu: object) -> None:
    """The drafter after prompts through the prefill drafts as after the same prompts as decode steps
    (``common.draft_gate``: teacher-forced over the model's greedy replies; the drafter attends to the caches of the
    prefill's last layers): 2.571 vs 2.557 tokens per round."""

    if npu.drafter is None:
        pytest.skip("no drafter next to the model")
    got = draft_gate(npu, draft_prompts(Gemma(npu.gguf.path)))
    pre, steps = got["prefill"].mean(0), got["steps"].mean(0)
    assert pre[0] > 0.985 * steps[0] and (pre[1:] > steps[1:] - 0.02).all(), (pre, steps)
