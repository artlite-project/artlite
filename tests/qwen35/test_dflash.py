"""The DFlash2 block drafter (``qwen35.dflash``) of Qwen3.8-27B: ``ref.DFlash`` (fp32) against llama.cpp's draft-dflash, its design on
the NPU against ``ref.DFlash`` on the same taps, rounds of a generation.

- the reference: the block's tensors (llama.cpp quantizes its activations to Q8_K: 1-7% per layer, as much as two of its builds
  differ by), the same drafts but at near ties; past the window, on random taps (the model does not run);
- injections of random S, with rejected rows that hold junk, past a wrap of the ring: every closed record of the window and the open rows;
- the block on an essay's taps, before the window's end and past it: the head's logits and gates (1-3%), the top 16, the drafts;
- greedy rounds: every token the argmax of the model's one-token steps but at a tie, tokens per round; steps that branch (a sibling branch
  in their last slots) against the chains they stand for, rounds with them;
- a prompt through prefill chunks against the same prompt as decode steps: the ring's records and the open rows, the rounds after it;
- the paths of a real session: sampled rounds, context rounds between the drafter's, a second turn (decode steps and a chunk after a
  generation), chunks over the ring's wrap, a prompt ending on a record's or a chunk's edge, the prompt cache's restores.
"""

import itertools
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from artlite.decode.design import META, XB, interleave
from artlite.decode.kv import kv_from_groups
from artlite.gguf_model import GGUFFile
from artlite.model_config import qwen_dflash_of
from artlite.operators.bfp16 import block_bytes, block_values
from artlite.operators.design import COLS
from models.common import Sampling, seed
from models.qwen35.dflash import DFlashOverlay
from models.qwen35.dflash_core import HD, PAIR, POS_PER_GROUP, QC, ROWS_B
from models.qwen35.overlay import QwenOverlay
from models.qwen35.qwen import NpuQwen, Qwen
from models.server.cache import PromptCache
from models.tokenizer import BpeTokenizer

from ..common import essays, to_torch
from ..numerics import from_bf16_bits, rel_max
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
    return BpeTokenizer(GGUFFile(model)).encode(essays()[:8 * n])[:n]


def _emb(d: ref.DFlash, token: int) -> np.ndarray:
    return ref.f32(d.model.rows("token_embd.weight", [token] + [d.cfg.mask_token] * (d.cfg.block - 1)))


def test_ref() -> None:
    """The block and the selector against llama.cpp."""
    model, path = _files()
    ids, P = _tokens(model, 96), 80
    names = "|".join((*NAMES, "inp_g_embeddings", "Kcur_injected", "Vcur_injected", "result_norm", "result_output"))
    g = ref.golden(model, ids, ubatch=len(ids), logits="none", dflash=path, dflash_at=P, dflash_names=names)
    taps = np.ascontiguousarray(g.dflash("taps"))
    d = ref.DFlash(path, model, 256)
    assert rel_max(d.inject(taps, np.arange(P)), g.dflash("inp_g_embeddings")) < 0.02
    for L in range(d.cfg.layers):
        assert rel_max(d.K[L, :P], np.asarray(g.dflash("Kcur_injected", L))) < 0.04
        assert rel_max(d.V[L, :P], np.asarray(g.dflash("Vcur_injected", L))) < 0.04
    out = d.block(_emb(d, ids[P]), P)
    for L in range(d.cfg.layers):
        for n in NAMES:
            assert rel_max(out[f"{n}-{L}"], np.asarray(g.dflash(n, L))) < 0.08, (n, L)
    logits = np.asarray(g.dflash("result_output"))
    assert rel_max(out["logits"], logits) < 0.03
    k, T = d.cfg.top_k, d.cfg.block
    lat = np.asarray(g["dflash.lattice"])
    drafts = d.walk(lat[:, :k].astype(np.int64), lat[:, k:k + k * k].reshape(T, k, k))
    # llama.cpp's walk on our lattice: each of its drafts our best candidate after its predecessor, or within 1 of it
    # (a near tie: the two lattices' scores, up to ~32, are up to 0.5 apart)
    cand, score = d.lattice(out["logits"], out["gate"], ids[P])
    for i, t in enumerate(drafts, 1):
        c = 0 if i == 1 else list(cand[i - 1]).index(drafts[i - 2])
        assert t in cand[i] and score[i, c].max() - score[i, c, list(cand[i]).index(t)] < 1, (i, drafts)


def test_ref_window() -> None:
    """Past the window: a row attends to its last 2048 positions and the block, on random taps."""
    model, path = _files()
    P = 2100
    ids = list(range(1000, 1000 + P + 8))
    g = ref.golden(model, ids, ubatch=64, logits="none", dflash=path, dflash_at=P, dflash_names="result_norm", dflash_random=True)
    d = ref.DFlash(path, model, 2400, head=np.arange(64))
    d.inject(np.ascontiguousarray(g.dflash("taps")), np.arange(P))
    assert rel_max(d.block(_emb(d, ids[P]), P)["result_norm"], np.asarray(g.dflash("result_norm"))) < 0.04


def test_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The drafter of ``Qwen``'s ``spec_type`` / ``draft_model``: the DFlash2 file by default, the MTP block (the image lays
    it out), none (the image leaves it out), the file given; one asked for where it cannot run says so."""
    model, path = _files()
    monkeypatch.setattr("models.qwen35.qwen.WeightBuffer", lambda image: SimpleNamespace(image=image))  # (no packing)
    for spec, draft, dflash, mtp in [("auto", None, path, None), ("draft-mtp", None, None, ""), ("none", None, None, None),
                                     ("auto", path, path, None)]:
        q = Qwen(model, spec_type=spec, draft_model=draft)
        assert (q.dflash, q.mtp, q.weights.image.mtp_layer >= 0) == (dflash, mtp, mtp == ""), spec
    with pytest.raises(ValueError, match="images in prompts"):
        Qwen(model, mmproj="mmproj.gguf", spec_type="draft-dflash")
    with pytest.raises(ValueError, match="a draft-dflash drafter, not draft-mtp"):
        Qwen(model, spec_type="draft-mtp", draft_model=path)
    monkeypatch.setattr("models.qwen35.qwen.qwen_dflash_of", lambda _: None)
    with pytest.raises(ValueError, match="no DFlash2 drafter"):
        Qwen(model, spec_type="draft-dflash")


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


def _pan(x: np.ndarray) -> np.ndarray:
    """The values of x as bfp16 blocks of 8 along the last axis."""
    x = np.ascontiguousarray(x, np.float32)
    return np.asarray(block_values(block_bytes(x.reshape(*x.shape[:-1], -1, 8))), np.float64).reshape(x.shape)


def _bf16(b: np.ndarray) -> np.ndarray:
    return from_bf16_bits(np.ascontiguousarray(b).view(np.uint16)).astype(np.float64)


def _attention_errors(d: DFlashOverlay, rf: ref.DFlash, L: int, p: int) -> np.ndarray:
    """The block at p just ran: per column, slot and query head the relative error of the attention core's o (aout's x) against the
    attention of its own operands (the rows' raw q, K, V in ACT, the ring, the open object; K and Q as the core's panels)."""
    d.dev.fetch("ACT")
    d.dev.fetch("KV")
    act, kv, T, G, W, cfg = d.dev.view("ACT"), d.dev.view("KV"), d.T, d.shape.ring, d.shape.window, d.cfg
    ro = p >> 3
    first = max(p - W + 1, 0) // POS_PER_GROUP
    th = cfg.rope_theta(np.arange(p, p + 8)).astype(np.float64)
    wq, wk = (rf.vec(f"blk.{L}.attn_{n}_norm.weight").astype(np.float64) for n in "qk")

    def normed(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        y = x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w
        return ref.rope_partial(y.reshape(8, -1, HD), th).astype(np.float64)

    rg, errs = d.inp[(L, "aout")][0], []
    for c in range(COLS):
        raw = np.stack([_bf16(act[d._q(L, c) + s * XB + 4 * META:][:2 * (QC + 2) * HD]) for s in range(8)]).reshape(8, QC + 2, HD)
        q, kb, vb = normed(raw[:, :QC], wq), normed(raw[:, QC:QC + 1], wk)[:, 0], raw[:, QC + 1]
        closed = [kv_from_groups(kv[a:a + 2 * 2 * PAIR].reshape(1, 2, 2 * PAIR).copy(), HD, dmajor=True)
                  for a in (d._ring(L, c) + g % G * T for g in range(first, ((ro - 1) >> 2) + 1))]
        n = 8 * ro - POS_PER_GROUP * first
        Kc = np.concatenate([np.asarray(k, np.float64) for k, _ in closed])[:n]
        Vc = np.concatenate([np.asarray(v, np.float64) for _, v in closed])[:n]
        rows = _bf16(kv[d._open(L, c):d._open(L, c) + 4 * ROWS_B]).reshape(4, 16, 8, 8)
        Ko, Vo, old = np.zeros((16, HD)), np.zeros((16, HD)), p - 8 * ro
        Ko[:old] = rows[ro & 1].transpose(1, 0, 2).reshape(8, -1)[:old]
        Vo[:old] = rows[2 + (ro & 1)].transpose(1, 0, 2).reshape(8, -1)[:old]
        Ko[old:old + 8], Vo[old:old + 8] = kb, vb
        K, V = np.concatenate([Kc, _pan(Ko)]), np.concatenate([Vc, _pan(Vo.reshape(2, 8, HD).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(16, HD)])
        pos = POS_PER_GROUP * first + np.arange(K.shape[0])
        o = _bf16(act[rg.off + XB + 2 * 8 * QC * HD * c:][:2 * 8 * QC * HD]).reshape(QC, 16, 8, 8).transpose(2, 0, 1, 3).reshape(8, QC, HD)
        for s in range(8):
            sc = _pan(q[s]) @ K.T / np.sqrt(HD)
            sc[:, (pos <= p + s - W) | (pos >= p + 8)] = -np.inf
            a = np.exp(sc - sc[:, -16:].max(-1, keepdims=True))
            want = (_pan(a) @ V) / a.sum(-1, keepdims=True)
            errs += np.sqrt(((o[s] - want) ** 2).sum(-1) / (want * want).sum(-1)).tolist()
    return np.array(errs)


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
    P = 2603
    assert P > d.shape.ring * 32, "past a wrap"
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
            assert rel_max(k[at:at + 8], rf.K[L, 8 * r:8 * r + 8, c]) < 0.03, (L, c, r)
            assert rel_max(v[at:at + 8], rf.V[L, 8 * r:8 * r + 8, c]) < 0.03, (L, c, r)
        o = kv[d._open(L, c):d._open(L, c) + 4 * ROWS_B].view(np.uint16).astype(np.uint16).view(np.int16).copy()
        rows = torch.from_numpy(o).view(torch.bfloat16).float().numpy().reshape(4, 16, 8, 8)
        pos = np.arange(8 * ro, P)
        for part, x in enumerate((rf.K, rf.V)):
            got = rows[2 * part + (ro & 1)][:, pos - 8 * ro].transpose(1, 0, 2).reshape(len(pos), -1)
            assert rel_max(got, x[L, pos, c]) < 0.03


@pytest.mark.parametrize("P", [2292, 2295, 2600])
def test_attention_core(npu: _Npu, P: int) -> None:
    """The attention core against the attention of its own operands, at every layer, on random taps (random S, junk rows, past the ring's
    wrap): their logits are no check, a broad softmax over random keys follows K's 8-bit mantissas (5-60% from fp32, reproduced by those
    panels alone), but each layer's o stays within its panels' rounding."""
    _, path = _files()
    d = npu.d
    d.reset()
    taps = np.random.default_rng(3).normal(0, 1 / 16, (P, len(d.cfg.taps), d.H)).astype(np.float32)
    _inject(npu, taps, np.random.default_rng(1))
    rf = ref.DFlash(path, ref.size_gguf("3.8-27B"), 8)
    emb = _emb(rf, 1000)
    d.step(0, 0, P, emb, P)
    for L in range(d.cfg.layers):
        e = _attention_errors(d, rf, L, P)
        assert np.median(e) < 0.03 and np.quantile(e, 0.98) < 0.08 and e.max() < 0.25, (L, np.median(e), e.max())


@pytest.mark.parametrize("P", [80, 2500])
def test_block(npu: _Npu, P: int) -> None:
    """The block on an essay's taps: the logits and gates of the head's rows, the top 16, the drafts (on the oracle's lattice); the
    selector as ref's."""
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
    logits = raw.astype(np.float32)
    assert rel_max(logits, out["logits"]) < 0.035 and rel_max(gates, out["gate"]) < 0.05
    top = [len(set(np.argsort(-logits[s])[:16].tolist()) & set(np.argsort(-out["logits"][s])[:16].tolist())) for s in range(1, 8)]
    assert min(top) >= 14, top
    drafts = rf.walk(*rf.lattice(logits, gates, ids[P]))
    # (on the oracle's lattice: each draft its best candidate after its predecessor, or within 1 of it: a near tie, as test_ref's)
    cand, score = rf.lattice(out["logits"], out["gate"], ids[P])
    for i, t in enumerate(drafts, 1):
        c = 0 if i == 1 else list(cand[i - 1]).index(drafts[i - 2])
        assert t in cand[i] and score[i, c].max() - score[i, c, list(cand[i]).index(t)] < 1, (i, drafts)
    assert d.selector.walk(raw, gates, ids[P], npu.head, 7, set()) == drafts


def test_rounds() -> None:
    """Greedy rounds on prompts run as decode steps: every token the argmax of one-token steps over the same text but at a tie, code
    keeping most of a block."""
    model, path = _files()
    g = GGUFFile(model)
    tk = BpeTokenizer(g)
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
        top = torch.topk(torch.from_numpy(lg), 2)
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
    """The records of a prompt's chunks (``qwen35.dflash_prefill``) are the injections' of the same rows, within the two paths' rounding (the
    model's rows differ between a chunk and decode steps: K ~2%, V ~4%, and by far more at its unstable tokens, 1 in 300 of this text, up to 3%
    of others: tokens the layers from 20 on reshape, whose rows part ways between any two graphs, llama.cpp's own too, ' colloquial' 26-29%,
    with the GDN and attention cores within their rounding of their own operands there); a chunk's last record's rows open."""
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
        ek = [rel_max(x, y) for x, y in zip(ka, kb, strict=True)]
        ev = [rel_max(x, y) for x, y in zip(va, vb, strict=True)]
        assert np.quantile(ek, 0.98) < 0.08 and np.quantile(ev, 0.98) < 0.15, (np.quantile(ek, 0.98), np.quantile(ev, 0.98))
        if n & 7:
            for part in (0, 2):
                assert rel_max(oa[part + (r & 1)][:, :n & 7], ob[part + (r & 1)][:, :n & 7]) < 0.08


@pytest.fixture(scope="module")
def app() -> tuple[NpuQwen, BpeTokenizer]:
    """The whole path, prompts through the prefill; a test's first ``reset``."""
    model, path = _files()
    g = GGUFFile(model)
    m = NpuQwen(g, capacity=8192, prefill=True, dflash=path)
    m.prepare(1)
    return m, BpeTokenizer(g)


def _chat(tk: BpeTokenizer, text: str) -> list[int]:
    return tk.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _by_steps(m: NpuQwen, ids: list[int]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The records of ``ids`` as steps of one token."""
    pf, m.prefill = m.prefill, None
    try:
        m.reset()
        for t in ids:
            m.steps([t], logits=False)
        m._inject()
        return _state(m.dflash)
    finally:
        m.prefill = pf


def _close(a: list, b: list, n: int) -> None:
    """Records of two paths over the same n tokens: per position (the model's rows differ between the paths: K ~2%, V ~4%, and at its unstable
    tokens by far more, up to 3% of a text's: ``test_prefill``), and the open rows."""
    r = n >> 3
    for (ka, va, oa), (kb, vb, ob) in zip(a, b, strict=True):
        ek = [rel_max(x, y) for x, y in zip(ka, kb, strict=True)]
        ev = [rel_max(x, y) for x, y in zip(va, vb, strict=True)]
        assert np.quantile(ek, 0.9) < 0.08 and np.quantile(ev, 0.9) < 0.15, (np.quantile(ek, 0.9), np.quantile(ev, 0.9))
        for part in (0, 2) if n & 7 else ():
            x, y = oa[part + (r & 1)], ob[part + (r & 1)]
            e = [rel_max(x[:, j], y[:, j]) for j in range(n & 7)]
            print(f"open rows {part}: {np.round(e, 3)}")
            assert np.median(e) < 0.08, e


def test_sampled(app: tuple[NpuQwen, BpeTokenizer]) -> None:
    """Rounds of a sampled generation (temperature 0.7, top 20): a kept draft is a sample of the model's logits like any other token, so
    every token lies among the top 20 of one-token steps over the same text (a step's rounding moves ranks by a few)."""
    m, tk = app
    m.lookup = False
    ids = _chat(tk, "Write a short story about a lighthouse keeper who finds a message in a bottle.")
    s = Sampling(temp=0.7, top_k=20, top_p=0.8)
    seed(7)
    m.reset()
    a0 = m.accepted
    out = list(m.generate(m.forward(ids), 96, s, ids))
    assert m.accepted - a0 >= 8
    m.reset()
    d, DRAFTS, pf = m.dflash, m.DRAFTS, m.prefill
    m.dflash, m.DRAFTS, m.prefill = None, 0, None
    try:
        lg = m.forward(ids)
        ranks = []
        for i, t in enumerate(out):
            ranks.append(int((lg > lg[t]).sum()))
            if i + 1 < len(out):
                lg = m.step(t)
    finally:
        m.dflash, m.DRAFTS, m.prefill = d, DRAFTS, pf
    assert max(ranks) < 24, ranks


def test_branch(app: tuple[NpuQwen, BpeTokenizer]) -> None:
    """A step whose last L slots branch off slot c (``QwenOverlay.step``'s branch, inside slot c's 8 positions): from one state,
    its branch slots' logits those of a chain of the tokens to slot c and the branch, its other slots' those of the chain
    without the branch, bit for bit, all of them finite (a step after a prefill, at a block's start and at its middle: the chain
    past the block's end); then greedy rounds with sibling branches on prose: every token the argmax of one-token steps over the same text but at
    a tie."""
    m, tk = app
    ov = m.overlay
    rng = np.random.default_rng(5)
    text = _tokens(m.gguf.path, 1200)

    def step(ids: list[int], p: int, branch: tuple[int, int] | None = None) -> np.ndarray:
        ov.rope[p:p + len(ids)] = np.arange(p, p + len(ids))[:, None]
        out = np.asarray(ov.step(m._rows(ids), p, commit=False, branch=branch))
        assert np.isfinite(out.astype(np.float32)).all(), (p, branch)
        ov.accept(0)
        return out.view(np.uint16).copy()

    n = 0
    for n0 in (37, 64, 1001):
        m.reset()
        m.prompt(text[:n0])
        p = m.position
        d = [int(x) for x in rng.integers(1000, 100000, 8)]
        for c, L in ((0, 2), (1, 2), (0, 1), (1, 1), (3, 2)):
            if p % 8 + c + L > 7:
                continue
            chain = [text[n0], *d][:8 - L]
            A = step(chain + d[6:6 + L], p, (c, L))
            B = step(chain[:c + 1] + d[6:6 + L], p)
            C = step(chain, p)
            assert np.array_equal(A[8 - L:], B[c + 1:]), (n0, c, L)
            assert np.array_equal(A[:8 - L], C), (n0, c, L)
            n += 1
    assert n >= 10
    m.lookup = False
    ids = _chat(tk, "Tell me about the history of the printing press in a few paragraphs.")
    m.reset()
    s0, w0 = m.siblings, m.siblings_won
    out = list(m.generate(m.forward(ids), 160, Sampling(greedy=True), ids))
    assert m.siblings - s0 >= 10 and m.siblings_won - w0 >= 2, (m.siblings - s0, m.siblings_won - w0)
    m.reset()
    d, DRAFTS, pf = m.dflash, m.DRAFTS, m.prefill
    m.dflash, m.DRAFTS, m.prefill = None, 0, None
    try:
        lg, flips = m.forward(ids), []
        for i, t in enumerate(out):
            top = torch.topk(torch.from_numpy(lg), 2)
            if int(top.indices[0]) != t:
                flips.append(float(top.values[0] - top.values[1]))
            if i + 1 < len(out):
                lg = m.step(t)
    finally:
        m.dflash, m.DRAFTS, m.prefill = d, DRAFTS, pf
    assert not [f for f in flips if f > 0.26], flips


def test_context_rounds(app: tuple[NpuQwen, BpeTokenizer]) -> None:
    """Context rounds between the drafter's on a copy of the prompt (forced, two of three: the Planner keeps the drafter while it beats them,
    which on a copy it does): their kept rows reach the drafter's cache, the records are those of steps of one over the same tokens."""
    m, tk = app
    m.lookup = True
    plan, round_of = m.planner, itertools.count(1)

    def alternate(_m: int, k: int, *, drafter: bool) -> int | None:
        return None if drafter and not next(round_of) % 3 else k

    plan.choose = alternate
    text = essays()[:1200]
    ids = _chat(tk, f"Repeat the following text word for word, and nothing else.\n\n{text}")
    m.reset()
    m.prepare(len(ids))
    c0, a0 = m.drafted_ctx, m.accepted
    try:
        out = list(m.generate(m.forward(ids), 192, Sampling(greedy=True), ids))
    finally:
        del plan.choose
    assert m.drafted_ctx > c0, "no context rounds ran"
    assert m.accepted > a0
    m._inject()
    d = m.dflash
    assert d.next == m.position == len(ids) + len(out) - 1
    n = m.position
    a = _state(d)
    _close(a, _by_steps(m, (ids + out)[:n]), n)


def test_second_turn(app: tuple[NpuQwen, BpeTokenizer]) -> None:
    """A second turn after a generation: decode steps up to a cache group, then a chunk, the drafter's rows of all of them; the rounds
    after it draft."""
    m, tk = app
    m.lookup = False
    first = _chat(tk, "Name three primes.")
    m.reset()
    gen = list(m.generate(m.forward(first), 24, Sampling(greedy=True), first))
    assert m.position == len(first) + len(gen) - 1
    second = tk.encode(f"<|im_end|>\n<|im_start|>user\n{essays()[:2400]}\n\nSummarize it in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    second = [gen[-1], *second]
    p0, a0 = m.position, m.accepted
    lg = m.forward(second)
    n = m.position
    assert n - p0 == len(second) and n // 32 > (p0 + 32) // 32, "a chunk in the prompt"
    m._inject()
    d = m.dflash
    assert d.next == n
    a = _state(d)
    seq = first + gen[:-1] + second
    _close(a, _by_steps(m, seq), n)
    m.reset()
    lg = m.forward(first)
    assert list(m.generate(lg, 24, Sampling(greedy=True), first)) == gen
    lg = m.forward(second)
    out = list(m.generate(lg, 48, Sampling(greedy=True), seq))
    assert m.accepted - a0 >= 8, (len(out), m.accepted - a0)


def _llama_records(n: int) -> tuple[list[int], ref.DFlash]:
    model, path = _files()
    ids = _tokens(model, n + 8)
    g = ref.golden(model, ids, ubatch=256, logits="none", dflash=path, dflash_at=n, dflash_names="")
    rf = ref.DFlash(path, model, n + 8, head=np.arange(64))
    rf.inject(np.ascontiguousarray(g.dflash("taps")), np.arange(n))
    return ids, rf


def test_ring_wraps(app: tuple[NpuQwen, BpeTokenizer]) -> None:
    """A prompt of several chunks over wraps of the ring, its last one padded: every closed record of the window against the injection of
    llama.cpp's taps of the same tokens (K, V: as ``test_prefill``'s paths differ, and the rare token with them), the padding rows' records
    past the window."""
    m, _ = app
    m.lookup = False
    n = 5000
    ids, rf = _llama_records(n)
    assert m.prefill.plan(0, n)[-1] > n - sum(m.prefill.plan(0, n)[:-1]), "a padded last chunk"
    m.reset()
    m.forward(ids[:n])
    m._inject()
    d = m.dflash
    assert d.next == n and n > 2 * d.shape.ring * 32
    kv, T, G = d.dev.view("KV"), d.T, d.shape.ring
    d.dev.fetch("KV")
    ro = n >> 3
    for L, c in ((0, 0), (2, 3), (4, 7)):
        ek, ev, energy = [], [], []
        for r in range((n - d.shape.window + 7) >> 3, ro):
            a = d._ring(L, c) + (r >> 2) % G * T
            k, v = kv_from_groups(kv[a:a + 2 * 2 * PAIR].reshape(1, 2, 2 * PAIR).copy(), HD, dmajor=True)
            at = (r & 3) * 8
            for j in range(8):
                ek.append(rel_max(k[at + j], rf.K[L, 8 * r + j, c]))
                ev.append(rel_max(v[at + j], rf.V[L, 8 * r + j, c]))
            energy.append(float(np.linalg.norm(v[at:at + 8]) / np.linalg.norm(rf.V[L, 8 * r:8 * r + 8, c])))
        assert np.quantile(ek, 0.9) < 0.1 and np.quantile(ev, 0.9) < 0.2, (np.quantile(ek, 0.9), np.quantile(ev, 0.9))
        assert min(energy) > 0.5, (int(np.argmin(energy)), min(energy))  # (a padding row's record: zeros)


@pytest.mark.parametrize("n", [256, 264, 320, 325])
def test_record_edges(app: tuple[NpuQwen, BpeTokenizer], n: int) -> None:
    """A prompt ending on a chunk's end, a record's, a cache group's, and past it: the block after it has the drafts of the same taps
    run through ``ref.DFlash``."""
    m, _ = app
    m.lookup = False
    model, path = _files()
    ids = _tokens(model, n + 8)
    g = ref.golden(model, ids, ubatch=len(ids), logits="none", dflash=path, dflash_at=n, dflash_names="")
    m.reset()
    m.forward(ids[:n])
    d = m.dflash
    head = d.table.ids
    rf = ref.DFlash(path, model, n + 8, head=head)
    rf.inject(np.ascontiguousarray(g.dflash("taps"))[:n], np.arange(n))
    emb = _emb(rf, ids[n])
    m._inject()
    assert d.next == n
    d.step(0, 0, n, emb, n)
    logits, gates = d.logits().astype(np.float32), d.gates()
    out = rf.block(emb, n)
    # (gates, max-based: the block at 256 is 7.7% off the oracle through decode steps, 8.3% through the prefill (its bfp16
    # activations' taps), at 264, 320, 325 3-5%: a gate of that position's block is sensitive)
    assert rel_max(logits, out["logits"]) < 0.06 and rel_max(gates, out["gate"]) < 0.09
    top = [len(set(np.argsort(-logits[s])[:16].tolist()) & set(np.argsort(-out["logits"][s])[:16].tolist())) for s in range(1, 8)]
    assert min(top) >= 12, top


def test_prompt_cache(app: tuple[NpuQwen, BpeTokenizer]) -> None:
    """The server's prompt cache: a prompt past the ring's window, an answer far longer than the ring's slack, then a request that goes
    back to the prompt's end, in place; another sequence and the first one's records back. The block after a continuation has the logits
    and gates, bit for bit, of the same continuation run with nothing in between (but for the head's table, the rows of the tokens seen)."""
    m, tk = app
    m.lookup = False
    A = tk.encode(essays()[:12000])[:2401]
    B = tk.encode(essays()[20000:32000])[:1500]
    cont = tk.encode(" The essay argues that progress makes things more addictive.")
    d = m.dflash
    greedy = Sampling(greedy=True)

    def block() -> tuple[torch.Tensor, np.ndarray]:
        x = np.stack([m._rows([cont[-1]])[0], *[m._mask] * 7])
        m._inject()
        d.step(0, 0, m.position, x, m.position)
        fixed = np.setdiff1d(np.arange(d.vocab), d.table.rows)
        return to_torch(d.logits()[:, fixed]), d.gates().copy()

    m.reset()
    m.forward(A)
    m.prompt(cont[:-1])
    want = block()

    m.reset()
    cache = PromptCache(m)
    assert cache.prepare(A)[0] == 0
    lg = m.forward(A)
    cache.checkpoint(A, lg)
    cp = cache.active.checkpoints[len(A)]
    gen = list(m.generate(lg, 160, greedy, A))
    run = A + gen[:-1]
    assert m.position == len(run) and len(gen) > 64
    cache.ran(run)
    # in place, back over the answer
    ids = A + cont[:-1]
    kept, _, how = cache.prepare(ids)
    assert kept == len(A) and "checkpoint" in how, how
    m.prompt(ids[kept:])
    cache.ran(ids)
    got = block()
    assert torch.equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    # another sequence, then the first one's from RAM
    kept, _, _ = cache.prepare(B)
    assert kept == 0
    lg = m.forward(B)
    cache.checkpoint(B, lg)
    list(m.generate(lg, 8, greedy, B))
    cache.ran(B + [0] * (m.position - len(B)))
    ids2 = A + cont[:-1]
    kept, _, how = cache.prepare(ids2)
    assert kept == len(A), (kept, how)
    m.prompt(ids2[kept:])
    cache.ran(ids2)
    got = block()
    assert torch.equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    print(f"a checkpoint: {cp.nbytes / 2**20:.0f} MiB")
