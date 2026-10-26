"""MiniCPM-V 4.7 1B's language model on the NPU with an image (models.qwen35: a ``Sequence`` prompt): the image's
embeddings (llama.cpp's encoder on the astronaut) between two essay paragraphs at the official M-RoPE positions,
teacher-forced over a description of the image, against llama.cpp's language model on the same rows at the same
positions (``ref.seq_golden``): the prompt through the prefill in chunks of 256 (two boundaries inside the
image, then a chunk of text alone at its shifted positions) or as decode steps. The logits barely tell the image's
positions apart (llama.cpp's own answer moves by a KL of 0.005 with the image's H and W set to T), so the prefill's K
cache is checked against the decode's too: the image's slots (rotated by their own cos / sin rows) as close as the
text's."""

import os

import numpy as np
import pytest
import torch

from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, minicpmv_gguf
from models.minicpmv import host as M
from models.qwen35 import attn as AT
from models.qwen35.qwen import NpuQwen, Qwen
from models.qwen35.tokenizer import QwenTokenizer, chat_prompt

from ..common import essay
from . import ref as V
from .test_ref import IMAGE, MMPROJ, NAMES

pytestmark = pytest.mark.slow  # (the whole model)

ANSWER = ("The image shows an astronaut in a white spacesuit riding a brown horse across a grassy field. The horse is "
          "galloping, and the astronaut holds the reins with one hand. In the background there are rolling hills, a "
          "few trees and a clear blue sky with some clouds.")


def _kl(want: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    return (lw.exp() * (lw - lg)).sum(-1)


@pytest.fixture(scope="module")
def case() -> tuple:
    path = minicpmv_gguf()
    for p in (path, MMPROJ):
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")
    g = GGUFFile(path)
    tok = QwenTokenizer(g)
    gold = V.golden(path, MMPROJ, IMAGE, "<__media__>Describe the image.", names=NAMES)  # (test_ref's)
    _, grid = M.slice_image(M.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))
    chunk = gold.embd(0)  # (llama.cpp's image chunk: the tokens after <image>, through the last marker)
    embd = chunk[np.asarray(grid.layout(newline_after=False)[1:]) == M.MEDIA]
    img = grid.sequence(embd, GGUFFile(MMPROJ).rows("v.tok_embd_sep", [0, 1, 2, 3]).numpy())
    text = chat_prompt(g, _messages(None), enable_thinking=False)
    head, tail = text.split("<|image_pad|>")
    seq = tok.encode(head + "<image_id>0</image_id>") + img + tok.encode(tail.removeprefix("\n"))
    answer = tok.encode(ANSWER)
    K = len(seq)
    seq = seq + answer
    rows = np.empty((len(seq), g.tensor("token_embd.weight").cols), np.float32)
    rows[~seq.given] = g.rows("token_embd.weight", seq.ids[~seq.given]).numpy()
    rows[seq.given] = seq.rows
    want = torch.from_numpy(np.asarray(V.seq_golden(path, rows, seq.pos, K - 1).logits, np.float32)[:-1].copy())
    return path, seq, K, want


def _messages(image: str | None) -> list[dict[str, object]]:
    """The prompt: an essay paragraph, the image (its file: the app's image part), more essay and the question."""
    words = essay("founders.txt").split()
    before, after = " ".join(words[:90]), " ".join(words[90:300]) + "\n\nDescribe the image."
    img = {"type": "image"} | ({"image": image} if image else {})
    return [{"role": "user", "content": [{"type": "text", "text": before}, img, {"type": "text", "text": after}]}]


def _kl_report(name: str, want: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    kl = _kl(want, got)
    agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
    print(f"{name}: KL vs llama.cpp at the official positions {float(kl.mean()):.4f} (max {float(kl.max()):.4f}), "
          f"argmax agreement {agree:.3f} over {len(kl)} tokens")
    return float(kl.mean()), agree


def _k_cache(model: NpuQwen, n: int) -> np.ndarray:
    """K (attention layers x KV heads, n, head_dim) of the first n slots, as the cache holds them."""
    ov = model.overlay
    ov.dev.fetch("KV")
    kv = ov.dev.view("KV")
    out = []
    for o in ov.kv_off.values():
        for h in range(ov.ash.kvh):
            a = o + h * ov.ash.head_bytes
            out.append(KV.kv_from_groups(kv[a:a + ov.ash.head_bytes].reshape(-1, 3, ov.T), AT.HD)[0][:n])
    return np.stack(out)


def test_image_prompt(case: tuple) -> None:
    path, seq, K, want = case
    i_img = int(np.flatnonzero(seq.ids == M.IMAGE)[0])
    i_end = int(np.flatnonzero(seq.ids == M.SLICE_END)[-1])
    assert i_img < 256 < i_end < 512 < K - 3, (i_img, i_end, K)  # (the chunks the docstring names)
    caches = {}
    for prefill in (True, False):
        model = NpuQwen(GGUFFile(path), capacity=1024, prefill=prefill, sizes=(256,))
        got = [model.forward(seq[:K])]
        caches[prefill] = _k_cache(model, K - K % 32)
        got = torch.stack(got + [model.step(int(t)) for t in seq.ids[K:-1]])
        kl, agree = _kl_report("prefill" if prefill else "steps", want, got)
        assert model.rope_next == seq.next - 1  # (the text positions continued through the answer's steps)
        assert kl < 0.008 and agree > 0.92
        del model
    err = lambda a, b: float(np.linalg.norm(caches[True][:, a:b] - caches[False][:, a:b])
                             / np.linalg.norm(caches[False][:, a:b]))
    e_img, e_text = err(i_img, i_end + 1), err(512, K - K % 32)
    print(f"K cache, prefill vs decode steps: the image's slots {e_img:.4f}, the text's after it {e_text:.4f}")
    assert e_img < 1.5 * e_text + 2e-3


def test_app_image_prompt(case: tuple) -> None:
    """The app's path (``Qwen(mmproj=)``: the chat prompt with the NPU encoder's embeddings) teacher-forced over the
    description, against llama.cpp's own encoder and language model: the prompt's tokens and positions the
    reference's, the KL within ~2x of the language model's alone (``test_image_prompt``: llama.cpp's embeddings; the
    encoder's own share: V3's gate); the image encoded again after the language model's dispatches (another hardware
    context ran in between) bit-identical."""
    path, seq, K, want = case
    app = Qwen(path, mmproj=MMPROJ)
    im = M.load_image(IMAGE)
    first, _ = app.vision.embeddings(im)
    prompt = app.chat(_messages(IMAGE))
    assert np.array_equal(prompt.ids, seq.ids[:K]) and np.array_equal(prompt.pos, seq.pos[:K])
    assert np.array_equal(prompt.given, seq.given[:K])
    model = app.load(capacity=1024)
    got = [model.forward(prompt)]
    got = torch.stack(got + [model.step(int(t)) for t in seq.ids[K:-1]])
    kl, agree = _kl_report("the NPU encoder, prefill", want, got)
    assert kl < 0.016 and agree > 0.9
    app.vision._kept.clear()
    again, _ = app.vision.embeddings(im)
    assert np.array_equal(first, again)
