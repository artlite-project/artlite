"""Gemma 4 (E4B) inference on the NPU from a GGUF file (Q4_0 layers, Q8_0 embedding).

A decode dispatch runs up to four consecutive tokens on one stream of the weights
(``models.gemma4.overlay.GemmaOverlay``, the decode overlay): every layer's projections and the LM head on
the main and helper cores, the attention and the KV cache on the attention cores. A prompt runs in chunks of up to
2048 tokens through the prefill (``models.gemma4.prefill.GemmaPrefill``: GEMMs on the same weight image,
its caches the decode's), only the layers that own a KV cache (24 of E4B's 42: the others read the caches of layers
22 / 23); its last token runs as a decode step (the logits). The host reads what enters them - the tokens' embedding
rows and per-layer embedding rows (a decode step's per-layer inputs: the NPU projects the embeddings, ``GemmaOverlay``;
a prefill chunk's per-layer inputs, ``per_layer_model_proj`` of the embedding plus the per-layer embedding row, on a
worker thread while the previous chunk runs) - and samples from the logits.

Multi-token prediction (``NpuGemma.generate``, with the model's ``gemma4-assistant`` drafter, ``mtp-<model file>`` next
to it: ``models.gemma4.draft.GemmaDraft``): a dispatch streams the weights once for up to four tokens, so the
sampled token takes three draft tokens along - the drafter's guesses of what follows, a dispatch of the same design each
(~1 ms) - and every draft that the model's own choice at its position confirms is a token without a dispatch of its
own. The text is the model's: a draft is only kept where the token sampled from the model's logits is that draft.

Memory: the weights the NPU streams (the layers and the LM head: 3.0 GB of E4B's Q4_0 file) are
re-laid-out into a device buffer at load, at their quantization's size; nothing is written to
disk. The per-layer token embedding (1.6 GB) is never loaded: its rows are read from the mapped
file, as is the token embedding's row.
"""

import functools
import os
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from artlite import parallel_compile
from artlite.decode.design import SLOTS as BATCH  # tokens per dispatch
from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig, assistant_of

from ..common import Checkpoint, Mask, Model, OverlayModel, Penalties, Sampling
from .draft import GemmaDraft, output_norm
from .host import GemmaHost
from .overlay import GemmaOverlay
from .prefill import GemmaPrefill
from .tokenizer import GemmaTokenizer, chat_prompt


class NpuGemma(OverlayModel):
    """The model on the NPU and the host side of a token.

    ``capacity``: the positions the caches hold (48.4 KB per position: the 20 sliding-window layers' linear caches and
    the 4 full layers'; ``kv_k`` "single": 32.3 KB); ``prefill``: prompts in chunks through the prefill (else as decode
    steps); ``drafter``: the multi-token-prediction drafter's GGUF file ("": ``mtp-<model file>`` next to the model,
    when it is there; None: none); ``kv_k``: the caches' K precision ("hilo": bfp16 hi + lo blocks, "single": one block
    per 8 dims)."""

    DRAFTS = BATCH - 1  # draft tokens per dispatch
    P_CACHE = 1024  # tokens whose per-layer projections are kept (21 KB each: ``steps``)

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, threads: int = 8, prefill: bool = True,
                 drafter: str | None = "", kv_k: str = "hilo"):
        self.gguf = gguf
        self.host = GemmaHost(gguf, threads=threads)
        self._p: OrderedDict[int, np.ndarray] = OrderedDict()  # (token -> its per-layer projection, bf16 bits)
        self.cfg = self.host.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.softcap = self.cfg.final_logit_softcapping
        if capacity > self.cfg.max_position_embeddings:
            raise ValueError(f"capacity {capacity} exceeds the model's context of {self.cfg.max_position_embeddings}")
        self.capacity = capacity
        self.overlay = GemmaOverlay(gguf, max_positions=capacity + BATCH, kv_k=kv_k)
        self.prefill = GemmaPrefill(self.overlay) if prefill else None
        path = assistant_of(gguf.path) if drafter == "" else drafter
        self.drafter: GemmaDraft | None = None
        if path and (drafter or os.path.exists(path)):
            self.drafter = GemmaDraft(self.overlay, GGUFFile(path))
        parallel_compile.compile_all([self.overlay.op] + ([self.prefill.lib] if prefill else []))
        self.overlay.load(threads=threads)
        if self.drafter is not None:
            self.drafter.load(threads=threads)
            self.drafter.gguf.release()
        if self.prefill is not None:
            self.prefill.bind()
            self.prefill.lib._kernel()  # (its HW context before timing, as the decode's)
        for S in range(1, BATCH + 1):  # (the steps' transactions; with drafts every S has logits)
            self.overlay.prepare(S, logits=S == 1 or self.drafter is not None)
            self.overlay.prepare(S, logits=False)
        if self.prefill is not None:  # (a prompt's last token after its chunk)
            self.overlay.prepare(1, logits=True, start=self.prefill.layers[-1])
        if self.cfg.per_layer_dim and self.prefill is not None:  # (the prefill's per-layer projection, a bf16 copy:
            # before timing)
            self.host.per_layer_inputs([0], self.host.embed([0]), bf16=True)
        gguf.release()
        self.position = 0
        self.t_draft = 0.0
        self.drafted = self.accepted = 0

    def describe(self) -> str:
        d = f", drafter {os.path.basename(self.drafter.gguf.path)}" if self.drafter is not None else ""
        return (f"{self.overlay.image.streamed_bytes() / 1e9:.2f} GB of weights, {self.overlay.sizes['KV'] / 1e6:.0f} MB "
                f"of KV cache ({self.capacity} positions){d}")

    def stats(self) -> str:
        if not self.drafted:
            return ""
        return (f"{self.drafted} draft tokens ({self.t_draft / self.drafted * 1e3:.2f} ms each), {self.accepted} "
                f"accepted")

    def reset(self) -> None:
        """Forget the cached positions."""
        self.position = 0

    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        """(the drafter's steps patch their caches' fills again)"""
        super().restore(position, checkpoint)
        if self.drafter is not None:
            self.drafter.reset()

    def inputs(self, tokens: list[int]) -> tuple[torch.Tensor, np.ndarray | None]:
        """What a decode step needs of its tokens: their scaled embeddings (n, hidden) as bf16 values, and their
        per-layer token rows (n, layers, per_layer_dim; ``GemmaHost.layer_rows``) or None."""
        e = self.host.layer_rows(tokens) if self.cfg.per_layer_dim else None
        return self.host.embed(tokens).bfloat16().float(), e

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return self.inputs(tokens)

    def steps(self, tokens: list[int], *, logits: bool = True, **step: object) -> torch.Tensor | None:
        """(a step of tokens whose per-layer projections are kept writes them instead of projecting the embeddings,
        ``GemmaOverlay.step(p=)``: 0.6 ms less; one that projects keeps its tokens')"""
        if not self.cfg.per_layer_dim:
            return super().steps(tokens, logits=logits, **step)
        kept = all(t in self._p for t in tokens)
        out = super().steps(tokens, logits=logits, p=np.stack([self._p[t] for t in tokens]) if kept else None, **step)
        t0 = time.perf_counter()
        if kept:
            for t in tokens:
                self._p.move_to_end(t)
        else:
            for t, row in zip(tokens, self.overlay.read_p(len(tokens)), strict=True):
                self._p[t] = row.copy()
            while len(self._p) > self.P_CACHE:
                self._p.popitem(last=False)
        self.t_host += time.perf_counter() - t0
        return out

    def chunk_inputs(self, tokens: list[int], x: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A prefill chunk's inputs: the scaled embeddings (n, hidden) and the per-layer inputs (n, the prefill's layers,
        dim), bf16 values (x: the embeddings, computed already)."""
        x = self.host.embed(tokens) if x is None else x
        pli = self.host.per_layer_inputs(tokens, x, bf16=True, layers=len(self.prefill.layers))
        return x.bfloat16(), None if pli is None else pli.bfloat16()

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        """(the first chunk: its per-layer inputs on the worker thread while its first operators run)"""
        if not first:
            return self.chunk_inputs(tokens)
        x = self.host.embed(tokens)
        return x.bfloat16(), self._pool.submit(lambda: self.chunk_inputs(tokens, x)[1])

    def _chunk_logits(self, token: int, n: int) -> torch.Tensor:
        """(a step from the prefill's last layer on - the layers that share a KV cache, the LM head: 19 of E4B's 42
        layers, ~25 ms less than a decode step -, its input the chunk's residual stream)"""
        t0 = time.perf_counter()
        x = self.prefill.residual(n - 1)
        emb, e = self.inputs([token])
        self.t_host += time.perf_counter() - t0
        return self._dispatch(lambda: self.overlay.step(x, e, self.position + n - 1, start=self.prefill.layers[-1],
                                                        emb=emb)[0])

    # ------------------------------------------------------------------
    # Multi-token prediction
    # ------------------------------------------------------------------

    def hidden_state(self, slot: int | None = None) -> torch.Tensor:
        """The output-normed hidden state (hidden,) bf16 of slot ``slot`` of the last step (it computed logits; default:
        its last token's, ``slot``): what a draft continues from."""
        slot = self.slot if slot is None else slot
        return output_norm(self.overlay.residual(slot), self.host.mat("output_norm.weight"), self.cfg.rms_norm_eps)

    @functools.cached_property
    def _draft_where(self) -> np.ndarray:
        """The drafter's logit of each token (the inverse of its ``rows``)."""
        rows = self.drafter.rows
        out = np.empty_like(rows)
        out[rows] = np.arange(len(rows), dtype=rows.dtype)
        return out

    def draft(self, token: int, h: torch.Tensor, n: int, penalties: Penalties | None = None) -> list[int]:
        """The drafter's guesses of the (up to n) tokens after ``token`` - the token of the next position, not run yet;
        h: the hidden state of the position before it -: its most likely one, the most likely one after that, ...
        (ends with an end-of-sequence token); ``penalties``: the generation's (up to ``token``), which the guesses
        get too."""
        t0 = time.perf_counter()
        out: list[int] = []
        recent = penalties.copy() if penalties is not None and penalties.active else None
        while len(out) < n and token not in self.eos_token_ids:
            x = self.host.embed([token])[0]
            raw = self.drafter.step(x, h if not out else None, self.position)
            if recent is None:
                token = self.drafter.token(raw)
            else:
                token = int(recent.top(raw, 1, self.softcap, rows=self.drafter.rows, where=self._draft_where)[1][0])
                recent.accept(token)
            out.append(token)  # (its per-layer inputs: with the verifying step's others, in one pass)
        self.t_draft += time.perf_counter() - t0
        self.drafted += len(out)
        return out

    def generate(self, logits: torch.Tensor, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = (),
                 mask: Mask | None = None) -> Iterator[int]:
        """With the drafter: multi-token prediction. The sampled token's dispatch takes up to ``DRAFTS`` draft tokens
        along; the model's logits after a token give its own next token (sampled), and as long as that is the draft that
        ran behind it, the draft's logits are valid too. The rejected drafts' positions are forgotten (the next steps
        overwrite their cache records)."""
        if self.drafter is None:
            yield from super().generate(logits, max_tokens, sampling, prompt, mask)
            return
        if max_tokens == 0:
            return
        penalties = Penalties(sampling, prompt, mask)
        token = sampling.pick(*penalties.top(logits, sampling.k))
        penalties.accept(token)
        h = self.hidden_state()
        n = 1
        yield token
        while n != max_tokens and token not in self.eos_token_ids and self.position < self.capacity:
            k = min(self.DRAFTS, self.capacity - self.position - 1, max_tokens - n - 1 if max_tokens > 0 else BATCH)
            d = self.draft(token, h, k, penalties) if k > 0 else []
            raw = self.steps([token, *d])
            new: list[int] = []
            for i in range(len(d) + 1):
                new.append(sampling.pick(*penalties.top(raw[i], sampling.k, self.softcap)))
                penalties.accept(new[i])
                if i == len(d) or new[i] != d[i] or new[i] in self.eos_token_ids:
                    break
            self.position -= len(d) + 1 - len(new)
            self.accepted += len(new) - 1
            h = self.hidden_state(len(new) - 1)
            for token in new:
                n += 1
                yield token


class Gemma(Model):
    """Gemma 4 from a GGUF file: the architecture, tokenizer and chat template from its metadata. ``kv_k``: the KV
    caches' K precision ("hilo": bfp16 hi + lo blocks, "single": one block per 8 dims)."""

    name = "Gemma 4"
    thinks = True
    reasoning = ("<|channel>thought", "<channel|>")  # (a thought is the "thought" channel)

    def __init__(self, path: str, *, kv_k: str = "hilo"):
        if kv_k not in ("hilo", "single"):
            raise ValueError(f"kv_k: {kv_k!r} (one of hilo, single)")
        self.kv_k = kv_k
        self.gguf = GGUFFile(path)
        self.tokenizer = GemmaTokenizer(self.gguf)
        self.eos_token_ids = GemmaConfig.from_gguf(self.gguf).eos_token_ids

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_bos=True)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False) -> list[int]:
        return self.tokenizer.encode(chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools,
                                                 preserve_thinking=preserve))

    def sampling(self, *, think: bool = False) -> Sampling:
        return Sampling(temp=1.0, top_k=64, top_p=0.95)  # (the model card's)

    def load(self, *, capacity: int, prefill: bool = True) -> NpuGemma:
        return NpuGemma(self.gguf, capacity=capacity, prefill=prefill, kv_k=self.kv_k)
