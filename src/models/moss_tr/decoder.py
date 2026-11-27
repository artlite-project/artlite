"""MOSS-Transcribe's decoder on the NPU: the checkpoint's Qwen3-0.6B-shaped language model as ``models.llama``'s overlay
(28 layers, 16 query / 8 KV heads of 128, a tied output weight), and its prompt - the chat template's text with the VQ
adaptor's rows where the audio span's pads are.

The prompt takes embedding rows as ``models.lfm.lfm``'s does: a ``models.sequence.Sequence``'s given rows stand for
their tokens' ids, so the audio tower's 375 rows reach the language model through the prefill's chunks and the decode
steps alike. The reference: the checkpoint's decoder at bf16, whose output weight is the tokens' embedding
(``hf_model.HfModel``'s ``output.weight``).

The reference is transformers' Qwen3 in float32 on the same prompt and tokens, teacher-forced: a peaked position keeps
its argmax and its KL, a flat one may drift and flip where the top two are closer than the noise - the decoder's bf16
activations move the tail, and the reference's own bf16 logits carry a KL floor of 2.9e-4 as well, so a flat position's
drift is not the prefill's or the cache's arithmetic. ``a8_gate`` is that criterion; the audio tower's real rows pass
it too. The measured KLs and the stages' times are the model page's (``docs/models/moss.md``).
"""

import numpy as np

from artlite.hf_model import HfModel
from models.llama.llama import NpuLlama
from models.sequence import Sequence
from models.templating import render_chat
from models.tokenizer import BpeTokenizer

from .host import splice_audio


class Decoder(NpuLlama):
    """The decoder on the NPU with a prompt's embedding rows (``NpuLlama``'s steps and chunks)."""

    def rows_of(self, tokens: list[int] | Sequence) -> np.ndarray:
        """The tokens' embedding rows (n, hidden) fp32: their ids' rows, a sequence's given rows."""
        if not isinstance(tokens, Sequence):
            return self.gguf.rows("token_embd.weight", tokens)
        x = np.empty((len(tokens), self.cfg.hidden_size), np.float32)
        t = ~tokens.given
        if t.any():
            x[t] = self.gguf.rows("token_embd.weight", tokens.ids[t])
        if tokens.given.any():
            x[tokens.given] = tokens.rows
        return x

    def _step_inputs(self, tokens: list[int] | Sequence) -> tuple:
        return (self.rows_of(tokens),)

    def _chunk_inputs(self, tokens: list[int] | Sequence, *, first: bool) -> tuple:
        return (self.rows_of(tokens),)


def decoder(gguf: HfModel, *, capacity: int = 2048, kv_k: str = "single", prefill: bool = True) -> Decoder:
    """The checkpoint's decoder on the NPU (``capacity`` positions of KV cache: the decoder's 67.7 KB a position)."""
    return Decoder(gguf, capacity=capacity, kv_k=kv_k, prefill=prefill)


def a8_gate(got: np.ndarray, ref: np.ndarray, *, h_max: float = 0.5, kl_max: float = 1e-3) -> dict:
    """The A8 criterion for a bf16 decoder against a float32 reference: the teacher-forced logits ``got`` / ``ref``
    (positions, vocabulary). A *peaked* position - the reference's distribution's entropy within ``h_max`` nats, i.e.
    one token holding the mass - must reproduce its argmax and keep the KL (``got`` || ``ref``) within ``kl_max``; a
    flat one may drift (the decoder's bf16 activations move the tail, as the reference's own bf16 logits do) and its
    argmax may flip where the top two are closer than the noise. The discriminator is the shape, not the top two's gap:
    a flat distribution's top two can stand apart while its tail carries the same mass. Returns the summary (``ok``:
    every peaked position inside both)."""
    a, b = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"logits {a.shape} against {b.shape}: (positions, vocabulary)")
    p = np.exp(a - a.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    q = np.exp(b - b.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    ent = -np.sum(q * np.log(q + 1e-300), 1)
    kl = np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300)), 1)
    peaked = ent <= h_max
    same = a.argmax(1) == b.argmax(1)
    flips = int(np.count_nonzero(~same & peaked))
    return {"positions": len(a), "peaked": int(peaked.sum()), "flips": flips,
            "worst_kl_peaked": float(kl[peaked].max()) if peaked.any() else 0.0,
            "ok": bool(flips == 0 and (not peaked.any() or kl[peaked].max() <= kl_max))}


def digits(tok: BpeTokenizer) -> dict[str, int]:
    """The time markers' digit tokens' ids (the processor requires one token each)."""
    out = {d: tok.encode_plain(d) for d in "0123456789"}
    assert all(len(v) == 1 for v in out.values()), out
    return {d: int(v[0]) for d, v in out.items()}


def prompt(gguf: HfModel, tok: BpeTokenizer, rows: np.ndarray, messages: list[dict[str, object]],
           *, count: int | None = None) -> Sequence:
    """The conversation's prompt: the chat template's text (its one ``<|audio_pad|>``) with the adaptor's ``rows``
    (``count`` of them, the rows' count by default) at the pads' positions."""
    return splice_audio(render_chat(gguf, messages), rows, tok.encode, digits=digits(tok), count=count)
