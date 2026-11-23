"""The torch reference of Inkling 2 (``ref.py``) against llama.cpp's CPU runs of UD-Q4_K_XL (no NPU).

Slices: a visit of each kind fed llama.cpp's own inputs (``Model.walk`` driven by its visits' l_out rows: the residual,
the block states, the passes' banks and messages) agrees with llama.cpp's tensors of the visit to the activations'
Q8_0 rounding (two RMS norms' last bits flip a code now and then: Q, K, V, R ~1e-4, the visit's output ~3e-3) and routes
every token to the same experts; the final norm and the head agree to 1e-6, the embedding rows exactly.

Whole model (slow): 48 visits of a mixture of experts amplify any rounding (a token's experts differ from llama.cpp's in
a third of the late visits), and llama.cpp's own runs of a prompt in one batch and token by token differ by ~0.05 nats
a token; on three prompts (a chat prompt, code, an essay past the 1024-position window) the reference's teacher-forced
KL to those two runs stays within that floor. tau (the full layers' position factor) acts past position 8191:
a full visit's Q and R on 9000 tokens of the essays."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, Visit, inkling_gguf
from models.inkling2.inkling import Inkling

from ..common import ESSAYS_DIR, essay
from ..goldens.text import Golden, golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from ..numerics import kl_rows, rel
from . import ref

MODEL = inkling_gguf()
CHAT = ("<|message_system|><|content_text|>You are a careful assistant.<|end_message|><|message_system|><|content_text|>"
        "Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Summarize the following text in two "
        "sentences.\n\n{}<|end_message|>")
# a visit of each kind: layer 0 (no mix), a full prelude layer, the core's first (a block, its mix the prelude's), pass
# 2's first (messages and controllers), pass 2's full layer (three mixed inputs, each controlled), pass 3's middle
# block, the coda's first and last
CHECK = (0, 5, 6, 18, 23, 36, 42, 47)
NAMES = "Qcur|Kcur|Vcur|Rcur|kqv_out|attn_out|ffn_inp|ffn_moe_logits|ffn_moe_topk|ffn_out|l_out"


def t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(x, np.float32))


class Run:
    """A prompt's llama.cpp runs: ``G`` every visit's l_out and the final states and logits; ``S`` the tensors of the
    ``CHECK`` visits."""

    def __init__(self, text: str):
        self.tokens = llama_tokenize(MODEL, text)
        self.T = len(self.tokens)
        self.G: Golden = golden(MODEL, self.tokens, layers=("all", "g"), names="embd|l_out|result_norm", logits="f32")
        self.S: Golden = golden(MODEL, self.tokens, layers=tuple(VISITS[v].slot for v in CHECK), names=NAMES)

    def out(self, v: Visit) -> torch.Tensor:
        return t(self.G["l_out", v.slot]).reshape(self.T, -1)

    def walk(self, m: ref.Model, until: int | None = None) -> tuple[tuple, torch.Tensor, list[torch.Tensor]]:
        """``Model.walk`` on llama.cpp's visit outputs: the inputs (cur, blocks, messages) of visit ``until`` and the
        last output and block states."""
        args: list[tuple] = []

        def run(v: Visit, cur: torch.Tensor, blocks: list[torch.Tensor], messages: torch.Tensor | None) -> torch.Tensor:
            if v.index == until:
                args.append((cur, list(blocks), messages))
            if v.block:
                blocks.append(cur)
            return self.out(v)

        emb = ref.rms(t(self.G["embd"]), m.cfg.rms_norm_eps) * m.vec("token_embd_norm.weight")
        cur, blocks = m.walk(emb, run)
        return (args[0] if args else ()), cur, blocks


VISITS = InklingConfig.from_gguf(GGUFFile(MODEL)).visits if os.path.exists(MODEL) else ()


@pytest.fixture(scope="module")
def model() -> ref.Model:
    require(MODEL)
    torch.set_num_threads(16)
    return ref.Model(GGUFFile(MODEL))


@pytest.fixture(scope="module")
def chat() -> Run:
    require(MODEL)
    return Run(CHAT.format(essay("apple.txt")[:400]))


def test_rows(chat: Run) -> None:
    """``Inkling.rows``: llama.cpp's embedding rows."""
    m = Inkling(MODEL)
    assert np.array_equal(m.rows(chat.tokens), np.asarray(chat.G["embd"]))


@pytest.mark.parametrize("v", CHECK)
def test_visit(model: ref.Model, chat: Run, v: int) -> None:
    """A visit from llama.cpp's inputs: its tensors, its experts (where llama.cpp's 8th and 9th selection keys are more
    than 1e-4 apart: layer 0's near-ties, 1e-5 apart, flip with its input's last bits)."""
    r, vis = chat, VISITS[v]
    (cur, blocks, messages), _, _ = r.walk(model, v)
    assert len(blocks) == {0: 0, 5: 1, 6: 1, 18: 1, 23: 2, 36: 2, 42: 3, 47: 4}[v]
    assert (messages is None) == (not vis.messages)
    model.record, model.out = set(NAMES.split("|")), {}
    with torch.no_grad():
        model.visit(vis, cur, blocks, ref.VisitCache(), 0, messages)
    s = vis.slot
    for name, tol in (("Qcur", 3e-4), ("Kcur", 3e-4), ("Vcur", 3e-4), ("Rcur", 3e-4), ("attn_out", 2e-3),
                      ("ffn_inp", 1e-3), ("ffn_moe_logits", 3e-4), ("ffn_out", 1e-2), ("l_out", 1e-2)):
        if name == "attn_out" and vis.block:  # (a block's first layer: ffn_inp is its attention's output)
            continue
        got = model.out[name, s].reshape(r.S[name, s].shape)
        assert rel(got, r.S[name, s]) < tol, (name, rel(got, r.S[name, s]))
    ids = np.sort(model.out["ffn_moe_topk", s].numpy(), -1)
    want = np.sort(np.asarray(r.S["ffn_moe_topk", s]).reshape(ids.shape), -1)
    logits = np.asarray(r.S["ffn_moe_logits", s]).reshape(r.T, -1)
    keys = np.sort(1 / (1 + np.exp(-logits)) + model.vec(f"blk.{vis.layer}.exp_probs_b.bias").numpy(), -1)
    clear = keys[:, -8] - keys[:, -9] > 1e-4
    assert clear.mean() > 0.95 and np.array_equal(ids[clear], want[clear])


def test_head(model: ref.Model, chat: Run) -> None:
    """The output's depth mix and norm from llama.cpp's last visit and block states, the head's softcapped logits from
    its final states."""
    _, cur, blocks = chat.walk(model)
    assert len(blocks) == 4
    with torch.no_grad():
        assert rel(model.final(cur, blocks), chat.G["result_norm"]) < 1e-6
        logits = model.head(t(chat.G["result_norm"]))
    assert rel(logits, chat.G.logits) < 1e-6 and float(logits.abs().max()) <= 30


def _prompts() -> dict[str, tuple[list[int], int]]:
    """The whole-model prompts: tokens and the first row compared."""
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    if not names:
        pytest.skip("no essay corpus (tests/datasets/fetch_datasets.py)")
    code = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>"
            "What does this function return for n = 10?\n\n```python\ndef f(n):\n"
            "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\n"
            "Explain step by step, then give 3 tests in pytest.<|end_message|>"
            "<|message_model|><|content_text|>For n = 10 the loop runs ten times; f(10) = 55. The tests:\n\n```python\n"
            "def test_f():\n    assert f(0) == 0\n    assert f(1) == 1\n    assert f(10) == 55\n```<|end_message|>"
            "<|content_model_end_sampling|>")
    long = llama_tokenize(MODEL, essay(names[0]))[:1100]
    assert len(long) == 1100
    return {"chat": (llama_tokenize(MODEL, CHAT.format(essay("apple.txt")[:400])), 0),
            "code": (llama_tokenize(MODEL, code), 0), "essay": (long, 900)}


@pytest.mark.slow
def test_model(model: ref.Model) -> None:
    """The reference as close to llama.cpp's batched (B) and token-by-token (S) runs as they are to each other: over
    the rows of three prompts (the essay's past the window), the mean of its teacher-forced KL(B || ref) and KL(S || ref)
    within KL(B || S) (measured: 0.017 against 0.020; per prompt KL(B || ref) / KL(B || S) 0.97, 1.09, 1.0)."""
    require(MODEL)
    ref_kl, floor = [], []
    for tokens, first in _prompts().values():
        b = np.asarray(golden(MODEL, tokens, logits="f16", logits_from=first).logits, np.float64)
        s = np.asarray(golden(MODEL, tokens, ubatch=1, logits="f16", logits_from=first).logits, np.float64)
        model.record = set()
        r = model.forward(tokens).numpy()[first:].astype(np.float64)
        ref_kl.append((kl_rows(b, r) + kl_rows(s, r)) / 2)
        floor.append(kl_rows(b, s))
    ref_kl, floor = np.concatenate(ref_kl), np.concatenate(floor)
    assert ref_kl.mean() <= floor.mean(), (ref_kl.mean(), floor.mean())


@pytest.mark.slow
def test_tau(model: ref.Model) -> None:
    """Full visit 5's Q and R (times tau) and its output from llama.cpp's inputs on 9000 tokens of the essays: tau is 1
    up to position 8190 and grows after it."""
    require(MODEL)
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    if not names:
        pytest.skip("no essay corpus (tests/datasets/fetch_datasets.py)")
    tokens = llama_tokenize(MODEL, "".join(essay(n) for n in names))[:9000]
    G = golden(MODEL, tokens, ubatch=2048, layers=(4, 5, "g"), names="embd|l_out|Qcur|Rcur", logits="none")
    T, v = len(tokens), VISITS[5]
    emb = ref.rms(t(G["embd"]), model.cfg.rms_norm_eps) * model.vec("token_embd_norm.weight")
    model.record, model.out = {"Qcur", "Rcur", "l_out"}, {}
    with torch.no_grad():
        model.visit(v, t(G["l_out", 4]).reshape(T, -1), [emb], ref.VisitCache(), 0)
    tau = model.cfg.tau(range(T))
    assert tau[8190] == 1 and tau[-1] > 1.009
    for name in ("Qcur", "Rcur"):
        got, want = model.out[name, 5], t(G[name, 5]).reshape(model.out[name, 5].shape)
        assert rel(got[8191:], want[8191:]) < 3e-4, name
        assert rel(got[8191:] / torch.from_numpy(tau[8191:])[:, None, None], want[8191:]) > 3e-4, "tau acts"
    assert rel(model.out["l_out", 5], G["l_out", 5]) < 1e-2
