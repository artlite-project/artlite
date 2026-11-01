"""The DFlash2 block drafter (``qwen35.dflash``) of Qwen3.8-27B: ``ref.DFlash`` (fp32) against llama.cpp's draft-dflash, its design on
the NPU against ``ref.DFlash`` on the same taps, rounds of a generation.

- the reference: the block's tensors (llama.cpp quantizes its activations to Q8_K: 1-5% per layer), the same drafts; past the window, on
  random taps (the model does not run);
- injections of random S, with rejected rows that hold junk, past a wrap of the ring: every closed record of the window and the open rows;
- the block on an essay's taps, before the window's end and past it: the head's logits and gates (1-3%), the top 16, the drafts;
- greedy rounds: every token the argmax of the model's one-token steps but at a tie, tokens per round;
- a prompt through prefill chunks against the same prompt as decode steps: the ring's records and the open rows, the rounds after it.
"""

import os

import numpy as np
import pytest
import torch

from artlite.decode.design import XB, interleave
from artlite.decode.kv import kv_from_groups
from artlite.gguf_model import GGUFFile
from artlite.model_config import qwen_dflash_of
from models.common import Sampling
from models.qwen35.dflash import DFlashOverlay
from models.qwen35.dflash_core import HD, PAIR, ROWS_B
from models.qwen35.overlay import QwenOverlay
from models.qwen35.qwen import NpuQwen
from models.qwen35.tokenizer import QwenTokenizer

from ..common import essays
from . import ref

pytestmark = pytest.mark.slow  # the whole 27B: its taps from llama.cpp's CPU run, its head in place

NAMES = ("noise_norm", "attn_conv_in", "Qcur", "Kcur", "Vcur", "attn_conv_out", "ffn_inp", "ffn_conv_in", "ffn_out", "ffn_conv_out", "l_out")


def _files() -> tuple[str, str]:
    model = ref.size_gguf("3.8-27B")
    if not os.path.exists(model):
        pytest.skip(f"no {model}")
    path = qwen_dflash_of(model)
    if path is None:
        pytest.skip(f"no DFlash2 drafter of {model}")
    return model, path


def _tokens(model: str, n: int) -> list[int]:
    return QwenTokenizer(GGUFFile(model)).encode(essays()[:8 * n])[:n]


def _emb(d: ref.DFlash, token: int) -> np.ndarray:
    return ref.f32(d.model.rows("token_embd.weight", [token] + [d.cfg.mask_token] * (d.cfg.block - 1)).numpy())


def test_ref() -> None:
    """The block and the selector against llama.cpp."""
    model, path = _files()
    ids, P = _tokens(model, 96), 80
    names = "|".join((*NAMES, "inp_g_embeddings", "Kcur_injected", "Vcur_injected", "result_norm", "result_output"))
    g = ref.golden(model, ids, ubatch=len(ids), logits="none", dflash=path, dflash_at=P, dflash_names=names)
    taps = np.ascontiguousarray(g.dflash("taps"))
    d = ref.DFlash(path, model, 256)
    assert ref.rel_err(d.inject(taps, np.arange(P)), g.dflash("inp_g_embeddings")) < 0.02
    for L in range(d.cfg.layers):
        assert ref.rel_err(d.K[L, :P], np.asarray(g.dflash("Kcur_injected", L))) < 0.04
        assert ref.rel_err(d.V[L, :P], np.asarray(g.dflash("Vcur_injected", L))) < 0.04
    out = d.block(_emb(d, ids[P]), P)
    for L in range(d.cfg.layers):
        for n in NAMES:
            assert ref.rel_err(out[f"{n}-{L}"], np.asarray(g.dflash(n, L))) < 0.07, (n, L)
    logits = np.asarray(g.dflash("result_output"))
    assert ref.rel_err(out["logits"], logits) < 0.03
    k, T = d.cfg.top_k, d.cfg.block
    lat = np.asarray(g["dflash.lattice"])
    drafts = d.walk(lat[:, :k].astype(np.int64), lat[:, k:k + k * k].reshape(T, k, k))
    assert d.walk(*d.lattice(out["logits"], out["gate"], ids[P])) == drafts


def test_ref_window() -> None:
    """Past the window: a row attends to its last 2048 positions and the block, on random taps."""
    model, path = _files()
    P = 2100
    ids = list(range(1000, 1000 + P + 8))
    g = ref.golden(model, ids, ubatch=64, logits="none", dflash=path, dflash_at=P, dflash_names="result_norm", dflash_random=True)
    d = ref.DFlash(path, model, 2400, head=np.arange(64))
    d.inject(np.ascontiguousarray(g.dflash("taps")), np.arange(P))
    assert ref.rel_err(d.block(_emb(d, ids[P]), P)["result_norm"], np.asarray(g.dflash("result_norm"))) < 0.04


class _Npu:
    """The drafter on the model's overlay (nothing else runs the model: taps go straight into its x)."""

    def __init__(self, model: str, path: str):
        self.target = QwenOverlay(GGUFFile(model), max_positions=4096)
        self.target.load()
        self.d = DFlashOverlay(self.target, GGUFFile(path))
        self.d.load()
        self.head = self.d.head.block("lm").rows.reshape(-1).copy()

    def taps(self, rows: np.ndarray) -> None:
        """Rows (S, taps, hidden) as a step of S slots leaves them in the model's activations."""
        tgt, S = self.target, rows.shape[0]
        act = tgt.dev.view("ACT")
        for i, L in enumerate(self.d.cfg.taps):
            rg = tgt.inp[tgt._input(L)][0]
            x = interleave(rows[:, i], tgt._sp(S)).view(np.uint8).reshape(-1)
            act[rg.off + XB:rg.off + XB + x.size] = x
            tgt.dev.flush("ACT", rg.off + XB, x.size)


@pytest.fixture(scope="module")
def npu() -> _Npu:
    return _Npu(*_files())


def _inject(n: _Npu, taps: np.ndarray, rng: np.random.Generator | None, start: int = 0) -> None:
    """Rows from start on in steps of 8 of them, or (rng) of random S with rows past a random ``keep`` junk."""
    q, P = start, taps.shape[0]
    while q < P:
        S = min(8 if rng is None else int(rng.integers(1, 9)), P - q)
        keep = S if rng is None else int(rng.integers(1, S + 1))
        rows = taps[q:q + S].copy()
        rows[keep:] = 7.0
        n.taps(rows)
        n.d.step(S, keep, q, None, 0)
        q += keep


def test_injection(npu: _Npu) -> None:
    """An injection's records: the ring after a wrap, the open rows."""
    _, path = _files()
    d = npu.d
    d.reset()
    P = 2300
    taps = np.random.default_rng(2).normal(0, 1 / 16, (P, len(d.cfg.taps), d.H)).astype(np.float32)
    _inject(npu, taps, np.random.default_rng(1))
    assert d.next == P
    rf = ref.DFlash(path, ref.size_gguf("3.8-27B"), P + 8, head=np.arange(64))
    rf.inject(taps, np.arange(P))
    kv, T, G = d.dev.view("KV"), d.T, d.shape.ring
    d.dev.fetch("KV")
    ro = P >> 3
    for L, c in ((0, 0), (3, 5), (4, 7)):
        for r in range(ro - 256, ro):
            a = d._ring(L, c) + (r >> 2) % G * T
            k, v = kv_from_groups(kv[a:a + 2 * 2 * PAIR].reshape(1, 2, 2 * PAIR).copy(), HD, dmajor=True)
            at = (r & 3) * 8
            assert ref.rel_err(k[at:at + 8], rf.K[L, 8 * r:8 * r + 8, c]) < 0.03, (L, c, r)
            assert ref.rel_err(v[at:at + 8], rf.V[L, 8 * r:8 * r + 8, c]) < 0.03, (L, c, r)
        o = kv[d._open(L, c):d._open(L, c) + 4 * ROWS_B].view(np.uint16).astype(np.uint16).view(np.int16).copy()
        rows = torch.from_numpy(o).view(torch.bfloat16).float().numpy().reshape(4, 16, 8, 8)
        pos = np.arange(8 * ro, P)
        for part, x in enumerate((rf.K, rf.V)):
            got = rows[2 * part + (ro & 1)][:, pos - 8 * ro].transpose(1, 0, 2).reshape(len(pos), -1)
            assert ref.rel_err(got, x[L, pos, c]) < 0.03


@pytest.mark.parametrize("P", [80, 2300])
def test_block(npu: _Npu, P: int) -> None:
    """The block on an essay's taps: the logits and gates of the head's rows, the top 16, the drafts; the selector as ref's."""
    model, path = _files()
    ids = _tokens(model, P + 16)
    g = ref.golden(model, ids, ubatch=len(ids), logits="none", dflash=path, dflash_at=P, dflash_names="")
    taps = np.ascontiguousarray(g.dflash("taps"))
    d = npu.d
    d.reset()
    _inject(npu, taps, None)
    rf = ref.DFlash(path, model, P + 8, head=npu.head)
    rf.inject(taps, np.arange(P))
    emb = _emb(rf, ids[P])
    d.step(0, 0, P, emb, P)
    out = rf.block(emb, P)
    raw, gates = d.logits(), d.gates()
    logits = raw.float().numpy().copy()
    assert ref.rel_err(logits, out["logits"]) < 0.035 and ref.rel_err(gates, out["gate"]) < 0.05
    top = [len(set(np.argsort(-logits[s])[:16].tolist()) & set(np.argsort(-out["logits"][s])[:16].tolist())) for s in range(1, 8)]
    assert min(top) >= 14, top
    drafts = rf.walk(*rf.lattice(logits, gates, ids[P]))
    assert drafts == rf.walk(*rf.lattice(out["logits"], out["gate"], ids[P]))
    assert d.selector.walk(raw, gates, ids[P], npu.head, 7, set()) == drafts


def test_rounds() -> None:
    """Greedy rounds on prompts run as decode steps: every token the argmax of one-token steps over the same text but at a tie, code
    keeping most of a block."""
    model, path = _files()
    g = GGUFFile(model)
    tk = QwenTokenizer(g)
    m = NpuQwen(g, capacity=1024, prefill=False, dflash=path)
    m.lookup = False
    ids = tk.encode("<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number using memoization, with a docstring "
                    "and two tests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    logits = m.forward(ids)
    m.prepare(len(ids))
    out = list(m.generate(logits, 128, Sampling(greedy=True), ids))
    rounds = len(out) - 1 - m.accepted
    assert (len(out) - 1) / rounds > 5, (len(out), rounds)
    m.reset()
    d, m.DRAFTS = m.dflash, 0
    m.dflash, m.drafter = None, None
    lg = m.forward(ids)
    flips = []
    for i, t in enumerate(out):
        top = torch.topk(lg.float(), 2)
        if int(top.indices[0]) != t:
            flips.append(float(top.values[0] - top.values[1]))
        if i + 1 < len(out):
            lg = m.step(t)
    m.dflash = d
    assert not [f for f in flips if f > 0.26], flips  # (a step's numbers follow its grouping at the rounding level)


def _state(d: DFlashOverlay) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Of three KV heads the cache's closed records' K, V (positions, hd) and the open object's rows."""
    kv, T, G = d.dev.view("KV"), d.T, d.shape.ring
    d.dev.fetch("KV")
    out = []
    ro = d.next >> 3
    for L, c in ((0, 0), (2, 3), (4, 7)):
        k, v = [], []
        for r in range(max(ro - 256, 0), ro):
            a = d._ring(L, c) + (r >> 2) % G * T
            kr, vr = kv_from_groups(kv[a:a + 2 * 2 * PAIR].reshape(1, 2, 2 * PAIR).copy(), HD, dmajor=True)
            at = (r & 3) * 8
            k.append(kr[at:at + 8])
            v.append(vr[at:at + 8])
        o = kv[d._open(L, c):d._open(L, c) + 4 * ROWS_B].view(np.uint16).astype(np.uint16).view(np.int16).copy()
        out.append((np.concatenate(k), np.concatenate(v), torch.from_numpy(o).view(torch.bfloat16).float().numpy().reshape(4, 16, 8, 8)))
    return out


@pytest.mark.parametrize("n", [301, 2301])
def test_prefill(n: int) -> None:
    """The records of a prompt's chunks (``qwen35.dflash_prefill``) are the injections' of the same rows, within the two paths' rounding
    (the model's rows differ between a chunk and decode steps: K ~2%, V ~4%, and at a rare token, here 1 in ~300, by far more, as llama.cpp's own
    prompt and decode graphs do at that token: ' colloquial', 26-29%); a chunk's last record's rows open, past a ring's wrap."""
    model, path = _files()
    g = GGUFFile(model)
    ids = _tokens(model, n)
    m = NpuQwen(g, capacity=4096, prefill=True, dflash=path)
    m.lookup = False
    m.forward(ids)
    m._inject()
    d = m.dflash
    assert d.next == n
    a = _state(d)
    pf, m.prefill = m.prefill, None
    m.reset()
    m.forward(ids)
    m._inject()
    b = _state(d)
    m.prefill = pf
    r = n >> 3
    for (ka, va, oa), (kb, vb, ob) in zip(a, b, strict=True):
        ek = [ref.rel_err(x, y) for x, y in zip(ka, kb, strict=True)]
        ev = [ref.rel_err(x, y) for x, y in zip(va, vb, strict=True)]
        assert np.quantile(ek, 0.98) < 0.08 and np.quantile(ev, 0.98) < 0.15, (np.quantile(ek, 0.98), np.quantile(ev, 0.98))
        if n & 7:
            for part in (0, 2):
                assert ref.rel_err(oa[part + (r & 1)][:, :n & 7], ob[part + (r & 1)][:, :n & 7]) < 0.08
