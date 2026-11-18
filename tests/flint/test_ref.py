"""The Flint oracles of ``ref.py`` against llama.cpp's own tensors of the official int4 file (CPU, no NPU): a chat
prompt of ~120 tokens (layers 0 dense and full, 1 dense, 2 MoE and sliding-window, the head) and an essay of 700
tokens, past the 513-key window (layers 2 and 5: sliding-window and full attention).

Every block is fed llama.cpp's own input, so what differs is the block: the fp32 ops (norms, RoPE, attention with sinks,
the router's sigmoid, top 6 and weights) agree to ~1e-6, and the Q4_1 matmuls to ~1e-6 with llama.cpp's Q8_1 activations
(``ref.mm``), against ~3e-3 for plain fp32 activations. Chained blocks (two matmuls: the activation between them rounds
differently where a fp32 difference of 1e-7 flips a Q8_1 code) agree to ~1e-4, a whole layer to ~1e-3.
"""

import os

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig, flint_gguf

from ..common import ESSAYS_DIR, essay
from ..goldens.text import Golden, golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from ..moe import ref as moe_ref
from ..np_ops import f32, rms_norm
from ..numerics import rel_max
from . import ref

MODEL = flint_gguf()
NAMES = ("attn_norm|Qcur|Kcur|Vcur|kqv_out|ffn_inp|ffn_norm|ffn_up|ffn_gate|ffn_swiglu|ffn_out|ffn_moe_logits|"
         "ffn_moe_topk|ffn_moe_weights_norm|ffn_moe_swiglu|ffn_moe_weighted|ffn_moe_out|ffn_shexp|l_out|result_norm")
LAYERS = (0, 1, 2, 48)
LONG = 2
HEAD_ROWS = 4  # (the head's rows test_head compares: the last ones)


class Run:
    """A prompt's golden and the file."""

    def __init__(self, tokens: list[int], layers: tuple[int | str, ...], names: str, logits: str = "none") -> None:
        require(MODEL)
        self.tokens = np.asarray(tokens)
        self.T = len(tokens)
        self.pos = np.arange(self.T)
        self.g = GGUFFile(MODEL)
        self.cfg = FlintConfig.from_gguf(self.g)
        self.G: Golden = golden(MODEL, tokens, layers=layers, names=names, logits=logits)

    def __call__(self, name: str, layer: int | None = None) -> np.ndarray:
        return f32(self.G[name] if layer is None else self.G[name, layer])

    def vec(self, name: str) -> np.ndarray:
        return f32(self.g.tensor(name).dequantize()).reshape(-1)


@pytest.fixture(scope="module")
def chat() -> Run:
    require(MODEL)
    text = essay("apple.txt")[:500]
    prompt = ("<|system|>You are a careful assistant.<|user|>Summarize the following text in two sentences.\n\n"
              + text + "<|assistant|>")
    return Run(llama_tokenize(MODEL, prompt), (*LAYERS, "g"), NAMES, "f32")


@pytest.fixture(scope="module")
def long() -> Run:
    require(MODEL)
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt"))
    tokens = llama_tokenize(MODEL, essay(names[0]))[:700]
    assert len(tokens) == 700, "an essay of 700 tokens"
    return Run(tokens, (LONG, 5), "Qcur|Kcur|Vcur|kqv_out")


def attention_of(r: Run, L: int, *, window: int | None | str = "layer", sinks: bool = True) -> np.ndarray:
    """A layer's heads' output from llama.cpp's Q (after the norm and RoPE), K and V."""
    b = f"blk.{L}."
    sk = r.vec(b + "attn_sinks.weight") if sinks else np.full(r.cfg.num_attention_heads, -np.inf, np.float32)
    w = (r.cfg.sliding_window if r.cfg.kinds[L] == "swa" else None) if window == "layer" else window
    return ref.attention(r("Qcur.2", L), r("Kcur.2", L), r("Vcur.1", L), sk, window=w, q_pos=r.pos).reshape(r.T, -1)


@pytest.mark.parametrize("L", [0, 2])
def test_attention_ops(chat: Run, L: int) -> None:
    """The projections (Q8_1 activations), the norms and RoPE (SWA layers; NoPE: the norm only) and the attention from
    llama.cpp's inputs."""
    r, b = chat, f"blk.{L}."
    cfg, kind = r.cfg, r.cfg.kinds[L]
    T, xn = r.T, r("attn_norm", L)
    assert rel_max(rms_norm(r("ffn_inp", L), r.vec(b + "ffn_norm.weight"), cfg.rms_norm_eps), r("ffn_norm", L)) < 1e-6
    for n, key in (("q", "Qcur"), ("k", "Kcur"), ("v", "Vcur")):
        w = ref.weight(r.g, b + f"attn_{n}.weight")
        assert rel_max(ref.mm(xn, w), r(key, L)) < 3e-6, n
        assert rel_max(ref.mm(xn, w, q8=False), r(key, L)) > 3e-4, "llama.cpp rounds its activations"
    q = ref.qk_norm(r("Qcur", L).reshape(T, -1, cfg.head_dim), r.vec(b + "attn_q_norm.weight"), cfg.rms_norm_eps)
    k = ref.qk_norm(r("Kcur", L).reshape(T, -1, cfg.head_dim), r.vec(b + "attn_k_norm.weight"), cfg.rms_norm_eps)
    assert rel_max(ref.rope_neox(q, cfg, r.pos, kind), r("Qcur.2", L)) < 1e-6
    assert rel_max(ref.rope_neox(k, cfg, r.pos, kind), r("Kcur.2", L)) < 1e-6
    if kind == "full":
        assert np.array_equal(ref.rope_neox(q, cfg, r.pos, kind), q), "NoPE"
    else:
        assert rel_max(q, r("Qcur.2", L)) > 0.1, "RoPE turns the SWA layers' heads"
    assert rel_max(attention_of(r, L), r("kqv_out", L)) < 3e-6
    y = ref.mm(r("kqv_out", L), ref.weight(r.g, b + "attn_output.weight"))
    x = {0: f32(r.g.rows("token_embd.weight", r.tokens)), 2: r("l_out", 1)}.get(L)
    if x is not None:
        assert rel_max(x + y, r("ffn_inp", L)) < 3e-6


def test_dense_ffn(chat: Run) -> None:
    L = 0
    r, b = chat, f"blk.{L}."
    fn = r("ffn_norm", L)
    gate, up, down = (ref.weight(r.g, b + f"ffn_{n}.weight") for n in ("gate", "up", "down"))
    assert rel_max(ref.mm(fn, gate), r("ffn_gate", L)) < 3e-6
    assert rel_max(ref.mm(fn, up), r("ffn_up", L)) < 3e-6
    assert rel_max(ref.mm(r("ffn_swiglu", L), down), r("ffn_out", L)) < 3e-6
    assert rel_max(ref.dense_ffn(r.g, L, fn), r("ffn_out", L)) < 3e-4
    assert rel_max(r("ffn_inp", L) + r("ffn_out", L), r("l_out", L)) < 1e-6


def test_moe_block(chat: Run) -> None:
    L = 2
    r = chat
    cfg, g = r.cfg, r.g
    fn = r("ffn_norm", L)
    logits = ref.router(g, L, fn)
    assert rel_max(logits, r("ffn_moe_logits", L)) < 3e-6
    ids, w = moe_ref.moe_route(r("ffn_moe_logits", L), cfg.n_experts_used, "sigmoid")
    assert np.array_equal(ids, r("ffn_moe_topk", L).astype(int))
    assert rel_max(w, r("ffn_moe_weights_norm", L).reshape(r.T, -1)) < 1e-6
    assert np.allclose(w.sum(-1), 1, atol=1e-6)
    # the experts from llama.cpp's own SwiGLU activations (the down matrices) and from its input
    get = ref.expert_weights(g, L)
    swiglu, weighted = r("ffn_moe_swiglu", L), r("ffn_moe_weighted", L)
    for e in np.unique(ids)[:24]:
        t, i = np.nonzero(ids == e)
        down = get(int(e))[2]
        got = ref.mm(swiglu[t, i], down) * w[t, i][:, None]
        assert rel_max(got, weighted[t, i]) < 3e-6, e
    got = ref.moe_block(g, cfg, L, fn)
    assert rel_max(got["moe"], r("ffn_moe_out", L)) < 1e-3
    assert rel_max(got["shexp"], r("ffn_shexp", L)) < 1e-3
    assert rel_max(got["out"], r("ffn_out", L)) < 1e-3
    assert rel_max(r("ffn_inp", L) + r("ffn_out", L), r("l_out", L)) < 1e-6
    assert rel_max(moe_ref.moe_ffn(fn, ids, w, get, lambda x, *m: ref.ffn_q41(x, *m, q8=False)), r("ffn_moe_out", L)) > 3e-4


def test_layers(chat: Run) -> None:
    """Whole layers from llama.cpp's input rows (two Q8_1 roundings can flip: ~1e-3)."""
    r = chat
    emb = f32(r.g.rows("token_embd.weight", r.tokens))
    for L, x in ((0, emb), (1, r("l_out", 0)), (2, r("l_out", 1))):
        out = ref.layer(r.g, r.cfg, L, x, r.pos)
        assert np.array_equal(out["ffn_ids"], r("ffn_moe_topk", L).astype(int)) if L == 2 else True
        assert rel_max(out["l_out"], r("l_out", L)) < 1e-2, L
        assert rel_max(out["attn_norm"], r("attn_norm", L)) < 1e-6


def test_head(chat: Run) -> None:
    """The final norm and the LM head's last rows from layer 48's output."""
    r, n = chat, HEAD_ROWS
    hn, logits = ref.head_logits(r.g, r.cfg, r("l_out", 48)[-n:])
    assert rel_max(hn, r("result_norm")[-n:]) < 1e-6
    want = f32(r.G.logits)[-n:]
    assert np.array_equal(logits.argmax(-1), want.argmax(-1))
    assert rel_max(logits, want) < 1e-3


def test_window_and_sinks(long: Run) -> None:
    """700 tokens: the SWA layer's window of 513 keys (p - 512 .. p) and the sinks matter, the full layer has neither
    window nor RoPE."""
    r = long
    for L in (LONG, 5):
        assert rel_max(attention_of(r, L), r("kqv_out", L)) < 3e-6, L
    swa, full = (attention_of(r, LONG, window=w, sinks=s) for w, s in ((None, True), (r.cfg.sliding_window, False)))
    want = r("kqv_out", LONG)
    assert rel_max(swa[:513], want[:513]) < 3e-6, "positions below 513 see the same keys"
    assert rel_max(swa[513:], want[513:]) > 1e-4, "the window cuts keys"
    assert rel_max(full, want) > 1e-2, "the sinks take weight"
    for w in (512, 514):
        assert rel_max(attention_of(r, LONG, window=w)[513:], want[513:]) > 1e-5, f"window {w}"
    assert rel_max(attention_of(r, 5, window=r.cfg.sliding_window), r("kqv_out", 5)) > 1e-4, "layer 5 attends everything"
