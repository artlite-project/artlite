"""The multi-token-prediction drafter of Qwen3.6-35B-A3B (``qwen35.draft.QwenDraft``, the Q4_0 MTP file) on the decode
overlay of a 2-layer slice (layers 0 and 3: the model's design, its formats with the drafter's), against
``qwen35_ref.mtp_block`` on llama.cpp's hidden states of an essay (golden(mtp=): the rows' h and the drafter file's
token embeddings): per row the next draft's h and the head's logits, in steps of S rows from position 0."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_mtp_of
from artlite.operators.qwen35.draft import QwenDraft
from artlite.operators.qwen35.overlay import QwenOverlay

from . import qwen35_ref as ref
from .common import essay

MODEL = ref.qwen_gguf("Q4_K_M", "35B-A3B")
N = 100


@pytest.fixture(scope="module")
def setup() -> tuple:
    mtp = qwen_mtp_of(MODEL) if os.path.exists(MODEL) else None
    if mtp is None:
        pytest.skip("no Qwen3.6-35B-A3B model file or MTP drafter")
    tokens = ref.llama_tokenize(MODEL, essay("apple.txt"))[:N]
    G = ref.golden(MODEL, tokens, layers=(40, "g"), names="mtp_tok_embd|result_norm", logits="none", mtp=mtp)
    emb = np.asarray(G.mtp("mtp_tok_embd", 40), np.float32).reshape(N, -1)
    hr = np.asarray(G["result_norm"], np.float32).reshape(N, -1)
    h = np.concatenate([np.zeros_like(hr[:1]), hr[:-1]])
    gm = GGUFFile(mtp)
    target = QwenOverlay(GGUFFile(MODEL), layers=[0, 3], vocab=4096, max_positions=1024, formats=("Q4_0",))
    target.load()
    draft = QwenDraft(target, gm)
    draft.load()
    r = ref.mtp_block(gm, QwenConfig.from_gguf(gm), emb, h, np.arange(N), head_rows=draft.vocab)
    wr = gm.tensor("blk.40.ffn_gate_inp.weight").dequantize().numpy().reshape(gm.tensor("blk.40.ffn_gate_inp.weight").rows, -1)
    p = np.sort(ref.softmax(r["mtp_attn_post_norm"] @ wr.T), -1)[:, ::-1]
    tied = (p[:, 7] - p[:, 8]) / p[:, 7] < 0.01  # (rows whose 8th and 9th experts nearly tie: either may win)
    return draft, torch.from_numpy(emb), torch.from_numpy(h), r, tied


@pytest.mark.parametrize("S", [1, 3, 4])
def test_draft_steps(setup: tuple, S: int) -> None:
    """Rows within 1.5% rms of the oracle's h (median 0.5%) but where the router's 8th and 9th experts tie within 1%;
    the head's argmax the oracle's on those rows where its top two logits are 0.1 apart."""
    draft, emb, h, r, tied = setup
    draft.reset()
    lg, hs = [], []
    for p in range(0, N, S):
        k = min(S, N - p)
        lg.append(draft.step(emb[p:p + k], h[p:p + k], p).float().numpy().copy())
        hs += [draft.hidden_state(s).numpy() for s in range(k)]
    lg, hs = np.concatenate(lg), np.stack(hs)
    want = r["h_nextn"].astype(np.float64)
    e = np.sqrt(((hs - want) ** 2).mean(-1) / (want ** 2).mean(-1))
    assert np.median(e) < 7e-3 and e[~tied].max() < 1.5e-2, (np.median(e), e[~tied].max())
    top2 = np.sort(r["logits"], -1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0] > 0.1) & ~tied
    assert (lg.argmax(-1) == r["logits"].argmax(-1))[clear].all()
