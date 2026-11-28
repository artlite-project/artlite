"""The Spark-X decode overlay (models/spark: ``SparkOverlay``) on the NPU:

- slices of layers 0 .. 3 (three sliding-window layers and a full one) with a truncated vocabulary against the fp32 CPU
  model (``cpu.SparkCpu``) on the same weights, in steps of 1 .. 8 tokens: per position the logits and each layer's
  output residual, over 700 positions (past the window of 512: a sliding-window layer streams its cache from the
  window's first group and masks the keys before each row's window) and over 2800 (past the rings' first wrap: the
  window's groups in two fills); the files of each kind of design (official Q4_K_M and Q8_0: gate / up windows of
  8-row tiles; bartowski's Q4_K_M: gate / up blocks of two formats; bartowski's Q4_0);
- steps of S slots give each slot the logits of a step of one, bit for bit, also where a step continues from fewer
  slots than the last one ran (rejected drafts: the slots past the accepted ones hold other tokens);
- a snapshot and its restore continue a sequence as it was (the rings' windows);
- three sequences on one device, alternated and in joint steps (an attention pass per sequence), give each one's logits
  alone, bit for bit;
- the whole model, teacher-forced (slow): KL of the next-token distributions against the fp32 model next to llama.cpp's
  own (the 8-slot design's GEMV takes x as single bfp16 rows and sums its sub-blocks to bf16: N0's simulation of its
  rounding points, per layer 10-15% below the NPU's error at each point), argmax agreement.
"""

import numpy as np
import pytest
import torch

from artlite.decode.step import Part
from artlite.gguf_model import GGUFFile
from models.spark.overlay import SparkOverlay

from ..common import to_torch
from ..numerics import kl_rows, rel_finite
from . import cpu, ref

VOCAB = 512
SLICE = [0, 1, 2, 3]
STEPS = (1, 8, 3, 5, 8, 2, 7, 8, 4, 6, 1, 8)  # (step sizes, repeated: every alignment to the records and groups)
TOL = 0.03  # (relative rms error of a position's logits and residuals: the GEMV's bfp16 x, the caches' blocks)
FILES = {"1.7B": ("1.7B", "Q4_K_M", "XHToken"), "4B": ("4B", "Q4_K_M", "XHToken"), "4B-Q8_0": ("4B", "Q8_0", "XHToken"),
         "bartowski-4B": ("4B", "Q4_K_M", "bartowski"), "bartowski-4B-Q4_0": ("4B", "Q4_0", "bartowski")}


def _run(ov: SparkOverlay, x: torch.Tensor, steps: tuple[int, ...], hidden: list | None = None) -> torch.Tensor:
    """The logits of every position of x, in steps of the given sizes (repeated) from position 0; ``hidden``: gains per
    position the residual entering each later layer and the LM head."""
    out = torch.zeros(x.shape[0], ov.vocab)
    p, i = 0, 0
    while p < x.shape[0]:
        n = min(steps[i % len(steps)], x.shape[0] - p)
        i += 1
        out[p:p + n] = to_torch(ov.step(x[p:p + n].numpy(), p)).float()
        if hidden is not None:
            nxt = [*ov.layers[1:], -1]
            hidden += [np.stack([ov.hidden(L, slot=s) for L in nxt]) for s in range(n)]
        p += n
    return out


@torch.no_grad()
def _cpu(m: cpu.SparkCpu, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The CPU model's logits (positions, VOCAB) and residuals (positions, layers, hidden) after each layer."""
    st = cpu.SparkState()
    x = m.emb[torch.tensor(ids)]
    res = []
    for L in m.run_layers:
        x = m.layer(L, x, st, 0)
        res.append(x)
    return m.mm(m.rms(x, m.norm), m.head_w[:VOCAB]), torch.stack(res, 1)


@pytest.fixture(scope="module")
def slice17() -> SparkOverlay:
    """The 1.7B's slice, loaded once for the steps, snapshot and wrap tests (the positions to 3456: the rings' wrap)."""
    ov = SparkOverlay(ref.gguf(), layers=SLICE, vocab=VOCAB, max_positions=3456)
    ov.load()
    return ov


@pytest.mark.parametrize(("name", "n"), [("1.7B", 2800), ("4B", 700), ("4B-Q8_0", 300), ("bartowski-4B", 300),
                                         ("bartowski-4B-Q4_0", 300)],
                         ids=["1.7B-wrap", "4B", "4B-Q8_0", "bartowski-4B", "bartowski-4B-Q4_0"])
def test_spark_overlay_layers(name: str, n: int, request: pytest.FixtureRequest) -> None:
    g = ref.gguf(*FILES[name])
    ids = ref.corpus_ids(n, FILES[name][0])
    m = cpu.SparkCpu(g, layers=SLICE)
    want, wres = _cpu(m, ids)
    if name == "1.7B":
        ov = request.getfixturevalue("slice17")
        ov.reset()
    else:
        ov = SparkOverlay(g, layers=SLICE, vocab=VOCAB, max_positions=n + 64)
        ov.load()
    hidden: list[np.ndarray] = []
    got = _run(ov, m.emb[torch.tensor(ids)], STEPS, hidden)
    err = {"logits": [rel_finite(got[p], want[p]) for p in range(n)]}
    for i, L in enumerate(SLICE):
        err[f"res.{L}"] = [rel_finite(hidden[p][i], wres[p, i]) for p in range(n)]
    print(name, n, {k: (round(float(np.mean(v)), 4), round(max(v), 4)) for k, v in err.items()})
    assert all(np.mean(v) < TOL and max(v) < 3 * TOL for v in err.values()), err


def test_spark_overlay_steps(slice17: SparkOverlay) -> None:
    """Steps of 1 .. 8 slots give the logits of steps of one, bit for bit, past the window; so do steps that continue
    from fewer slots than the last one ran (a generation's rejected drafts: the slots past the accepted ones hold other
    tokens)."""
    g = ref.gguf()
    ids = ref.essay_ids(600)
    ov = slice17
    ov.reset()
    m = cpu.SparkCpu(g, layers=[])
    x = m.emb[torch.tensor(ids)]
    one = _run(ov, x, (1,))
    ov.reset()
    assert torch.equal(_run(ov, x, STEPS), one)
    ov.reset()
    rng = np.random.default_rng(0)
    p = 0
    while p < len(ids):  # (each step: up to 8 slots, the accepted ones the text's, then other tokens)
        n = min(int(rng.integers(1, 9)), len(ids) - p)
        acc = int(rng.integers(1, n + 1))
        xs = x[p:p + n].clone()
        xs[acc:] = m.emb[torch.from_numpy(rng.integers(0, 120000, n - acc))]
        got = to_torch(ov.step(xs.numpy(), p)).float()
        assert torch.equal(got[:acc], one[p:p + acc]), p
        p += acc


def test_spark_overlay_snapshot(slice17: SparkOverlay) -> None:
    """A snapshot at a position past the window and its restore after other steps (past the rings' wrap: their groups
    and the open records overwritten): the steps after it give the logits they gave before, bit for bit."""
    g = ref.gguf()
    ids = ref.essay_ids(740)
    ov = slice17
    ov.reset()
    emb = cpu.SparkCpu(g, layers=[]).emb
    x = emb[torch.tensor(ids)]
    _run(ov, x[:700], (8, 3))
    snap = ov.snapshot(700)
    a = [to_torch(ov.step(x[p:p + 4].numpy(), p)).float() for p in range(700, 732, 4)]
    rng = np.random.default_rng(1)
    for p in range(732, 3400, 8):  # (other tokens: the rings' groups and the open records hold others)
        ov.step(emb[torch.from_numpy(rng.integers(0, 120000, 8))].numpy(), p)
    ov.restore(snap)
    b = [to_torch(ov.step(x[p:p + 4].numpy(), p)).float() for p in range(700, 732, 4)]
    assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["4B-Q8_0", "bartowski-4B"])
def test_spark_npu_teacher_forced(name: str) -> None:
    """The whole model, steps of 8 on the chat (and of one on bartowski's) and on an essay of 768 tokens past the
    window (4B-Q8_0's), against the fp32 model: KL of the next-token distributions (the essay's rows from 512 on) within
    1.6x llama.cpp's own (its Q8 activations; module docstring) or the design's own rounding (3e-3: N0's simulation of
    its rounding points on the 4B's chats 2.5e-3, as a Q8_0 file's, where llama.cpp's is below it), and the argmax
    mostly the same."""
    spec = FILES[name]
    g = GGUFFile(ref.path(*spec))
    ov = SparkOverlay(g, max_positions=1100)
    ov.load()
    emb = cpu.SparkCpu(g, layers=[]).emb
    chat = ref.chat_ids(spec[0])
    runs = [(chat, (8,), 0)] + ([(chat, (1,), 0)] if name == "bartowski-4B" else []) \
        + ([(ref.essay_ids(768, spec[0]), (8,), 512)] if name == "4B-Q8_0" else [])
    for ids, steps, row0 in runs:
        want = ref.cpu_logits(spec, ids, from_row=row0)
        ov.reset()
        got = _run(ov, emb[torch.tensor(ids)], steps)[row0:]
        k = kl_rows(want, got)
        agree = float((want.argmax(-1) == got.argmax(-1)).double().mean())
        lk = float(kl_rows(want, ref.llama_logits(spec, ids, from_row=row0)).mean())
        print(f"{name} {len(ids)} tokens S {steps[0]}: KL {float(k.mean()):.5f} (llama.cpp {lk:.5f}) max {float(k.max()):.4f}, "
              f"argmax {agree:.4f}")
        assert float(k.mean()) < max(3e-3, 1.6 * lk) and agree > 0.95, (len(ids), steps)


@pytest.mark.parametrize("name", ["1.7B", "4B"])
def test_spark_overlay_sequences(name: str) -> None:
    """Three sequences on one device (``SparkOverlay(sequences=3)``: separate texts at separate positions,
    the first past its sliding window) give, slot by slot, the logits of each sequence alone in steps of one, bit for
    bit: alternated (``select``: a step of 1 .. 8 slots of one, then of the next) and in joint steps (``step``'s parts:
    an attention pass per sequence, its rows of o alone; slots dealt at random, a sequence's drafts too, some of them
    rejected, one sequence joining later)."""
    size, quant, org = FILES[name]
    g = ref.gguf(size, quant, org)
    lengths = (600, 180, 140)
    nseq = len(lengths)
    ids = ref.essay_ids(sum(lengths) + 1, size)[1:]
    offs = np.cumsum([0, *lengths])
    emb = cpu.SparkCpu(g, layers=[]).emb
    xs = [emb[torch.tensor(ids[offs[s]:offs[s + 1]])] for s in range(nseq)]
    ov = SparkOverlay(g, layers=SLICE, vocab=VOCAB, max_positions=704, sequences=nseq)
    ov.load()
    want = []
    for s in range(nseq):  # (alone, in steps of one, as sequence 0)
        ov.select(0)
        ov.reset()
        want.append(_run(ov, xs[s], (1,)))
    assert not torch.equal(want[0][:32], want[1][:32]), "the sequences' texts differ"

    def check(got: torch.Tensor, at: list[tuple[int, int]], what: str) -> None:
        bad = [(s, p) for row, (s, p) in zip(got, at, strict=True) if not torch.equal(row, want[s][p])]
        assert not bad, f"{what}: (sequence, position) {bad[:8]} differ from the sequence alone"

    def fresh() -> None:
        for s in range(nseq):
            ov.select(s)
            ov.reset()

    fresh()  # alternated: a step of one sequence, then of the next
    pos, i = [0] * nseq, 0
    while any(pos[s] < lengths[s] for s in range(nseq)):
        s = i % nseq
        n = min(STEPS[(i // nseq + s) % len(STEPS)], lengths[s] - pos[s])
        i += 1
        if n:
            ov.select(s)
            check(to_torch(ov.step(xs[s][pos[s]:pos[s] + n].numpy(), pos[s])).float(), [(s, pos[s] + k) for k in range(n)],
                  "alternated")
            pos[s] += n

    fresh()  # joint steps, with rejected drafts
    rng = np.random.default_rng(0)
    pos, steps, deals = [0] * nseq, 0, set()
    while any(pos[s] < lengths[s] for s in range(nseq)):
        take = [[s, 1] for s in range(nseq) if pos[s] < lengths[s] and not (s == nseq - 1 and steps < 12)]
        spare = ov.SLOTS - len(take)
        for t in take:  # (more slots for the same sequence: drafts)
            while spare and rng.random() < 0.5 and pos[t[0]] + t[1] < lengths[t[0]]:
                t[1] += 1
                spare -= 1
        parts = tuple(Part(s, pos[s], n) for s, n in take)
        deals.add(tuple(n for _, n in take))
        x, acc = [], {}
        for s, n in take:  # (its accepted slots the text's, the rejected ones other tokens)
            acc[s] = int(rng.integers(1, n + 1))
            xi = xs[s][pos[s]:pos[s] + n].clone()
            xi[acc[s]:] = emb[torch.from_numpy(rng.integers(0, 120000, n - acc[s]))]
            x.append(xi)
        got = to_torch(ov.step(torch.cat(x).numpy(), parts)).float()
        a = 0
        for s, n in take:
            check(got[a:a + acc[s]], [(s, pos[s] + k) for k in range(acc[s])], f"step {steps}")
            pos[s] += acc[s]
            a += n
        steps += 1
    assert max(len(u) for u in deals) >= 3 and len(deals) > 20, "steps of three sequences, their slots dealt in many ways"
