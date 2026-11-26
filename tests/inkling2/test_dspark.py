"""Inkling 2's DSpark drafter (``models.inkling2.dspark``) on the NPU against its torch model (``dspark_ref.Drafter``):

- the model's taps (``InklingOverlay(taps=)``: the residual entering the tap visits, the combine's second copy) against
  the CPU model's (``dspark_ref.model_taps``) over an essay's tokens;
- injections of an essay's taps (the K / V records of every layer against the reference's) and a block after them (its
  rows' normed states, ``joint_proj``'s h part, the head plane's screened logits and their argmax), the reference's
  block of 4 rows;
- greedy rounds with drafts: the same text as without them (two chat prompts, 96 tokens), drafts accepted.
"""

import dataclasses

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import inkling_dspark_of, inkling_gguf
from models.inkling2.inkling import Inkling, NpuInkling
from models.sampling import Sampling

from ..common import essay
from ..llama_cpp import llama_tokenize
from ..model_files import require
from .dspark_ref import Drafter, model_taps
from .ref import Model, State
from .test_npu import TEXT_PROMPTS

pytestmark = pytest.mark.slow  # the whole GGUF model


def _paths() -> tuple[str, str]:
    path = inkling_gguf()
    require(path)
    dpath = inkling_dspark_of(path)
    if dpath is None:
        pytest.skip(f"no DSpark drafter beside {path}")
    return path, dpath


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


@pytest.fixture(scope="module")
def run() -> tuple[NpuInkling, list[int], np.ndarray]:
    """The model with its drafter, an essay's first 44 tokens as decode steps and their taps (the model's steps')."""
    path, dpath = _paths()
    m = NpuInkling(GGUFFile(path), capacity=2048, dspark=dpath)
    toks = llama_tokenize(path, essay("founders.txt"))[:45]
    rows = []
    for i in range(0, 44, 4):
        m.steps(toks[i:i + 4], logits=True)
        rows.append(m._rows_of(4))
    return m, toks, np.concatenate(rows)


def test_taps(run: tuple[NpuInkling, list[int], np.ndarray]) -> None:
    """The first tap (visit 5's input) at the slices' level; the deeper ones and the final state through the routing's
    near ties (``test_decode.test_pass``)."""
    m, toks, taps = run
    path, _ = _paths()
    ref = model_taps(Model(GGUFFile(path), q8=False), toks[:44], State(), m.dspark.cfg.taps).numpy()
    errs = [_rel(taps[:, j], ref[:, j]) for j in range(ref.shape[1])]
    print("taps' relative rms errors", [f"{e:.2e}" for e in errs])
    assert errs[0] < 1e-2 and max(errs) < 0.2


def test_block(run: tuple[NpuInkling, list[int], np.ndarray]) -> None:
    """Injections of 44 positions in steps of 4, a block at 44: against the reference on the same taps."""
    m, toks, taps = run
    path, dpath = _paths()
    d = m.dspark
    d.reset()
    for q in range(0, 44, 4):
        d.dev.stale = True
        d.inject(taps[q:q + 4], q)
    ref = Drafter(GGUFFile(dpath), GGUFFile(path))
    ref.cfg = dataclasses.replace(ref.cfg, block=4)
    ref.inject(torch.from_numpy(taps), 0)
    for L in range(d.cfg.layers):
        k, v = d.read_kv(L, range(44))
        kr = torch.stack([ref.k[L][p] for p in range(44)]).numpy().transpose(1, 0, 2)
        vr = torch.stack([ref.v[L][p] for p in range(44)]).numpy().transpose(1, 0, 2)
        print(f"layer {L}: K {_rel(k, kr):.2e}, V {_rel(v, vr):.2e}")
        assert _rel(k, kr) < 1e-2 and _rel(v, vr) < 2e-2
    d.dev.stale = True
    d.block(m._step_inputs([toks[44]])[0][0, 0], 44)
    h, base = ref.block(toks[44], 44)
    jr = (h @ ref.w("joint_proj.weight")[:, 2 * d.cfg.rank:2 * d.cfg.rank + d.cfg.hidden].T).numpy()
    sc = d.screened().astype(np.float32)[:, :base.shape[1]]
    eh, ej, es = _rel(d.h(), h.numpy()), _rel(d.jh_rows()[1:], jr[1:]), _rel(sc[1:], base.numpy()[1:])
    print(f"h {eh:.2e}, jh {ej:.2e}, screened logits {es:.2e}")
    assert eh < 2e-2 and ej < 3e-2 and es < 0.1  # (the plane: the high nibbles of the codes)
    assert (sc[1:].argmax(1) == base.numpy()[1:].argmax(1)).all()


@pytest.mark.parametrize("name", ["code", "math"])
def test_rounds(run: tuple[NpuInkling, list[int], np.ndarray], name: str) -> None:
    """Greedy text of 96 tokens with the drafter's rounds = without drafts; drafts accepted."""
    m = run[0]
    path, _ = _paths()
    toks = Inkling(path).chat(TEXT_PROMPTS[name])
    m.lookup = False
    texts = {}
    for drafts in (0, 3):
        m.reset()
        m.drafted = m.accepted = 0
        logits = m.prompt(toks, logits=True)
        texts[drafts] = list(m.generate(logits, 96, Sampling(greedy=True), toks, drafts=drafts))
        print(f"{name}, drafts {drafts}: {m.drafted} drafted, {m.accepted} accepted")
    assert texts[3] == texts[0] and m.accepted > 0


def test_prefill() -> None:
    """A prompt through prefill chunks (the drafter's rows from the chunks' taps, ``InklingPrefill(taps=)``): its
    records against the same prompt as decode steps', greedy rounds after it = no drafts."""
    path, dpath = _paths()
    toks = llama_tokenize(path, essay("founders.txt"))[:600]
    m = NpuInkling(GGUFFile(path), capacity=2048, dspark=dpath, prefill=True)
    m.prompt(toks, logits=True)
    m._inject()
    kp = [m.dspark.read_kv(L, range(560)) for L in range(m.dspark.cfg.layers)]
    s = NpuInkling(GGUFFile(path), capacity=2048, dspark=dpath)
    s.prompt(toks, logits=True)
    s._inject()
    for L, (k, v) in enumerate(kp):
        ks, vs = s.dspark.read_kv(L, range(560))
        print(f"layer {L}: K {_rel(k, ks):.2e}, V {_rel(v, vs):.2e} (the prefill's taps against the steps')")
        assert _rel(k, ks) < 0.1 and _rel(v, vs) < 0.15  # (the prefill's GEMMs and near-tied experts: its taps' own)
    m.lookup = False
    texts = {}
    for drafts in (0, 3):
        m.reset()
        m.drafted = m.accepted = 0
        logits = m.prompt(toks, logits=True)
        texts[drafts] = list(m.generate(logits, 96, Sampling(greedy=True), toks, drafts=drafts))
        print(f"drafts {drafts}: {m.drafted} drafted, {m.accepted} accepted")
    assert texts[3] == texts[0] and m.accepted > 0
