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

Multi-token prediction (``OverlayModel.generate``, with the model's ``gemma4-assistant`` drafter, ``mtp-<model file>``
next to it: ``models.gemma4.draft.GemmaDraft``): a dispatch streams the weights once for up to four tokens, so the
sampled token takes three draft tokens along - the drafter's guesses of what follows, a dispatch of the same design each
(~1 ms), or the context's (``models.drafts``) - and every draft that the model's own choice at its position confirms is
a token without a dispatch of its own. The text is the model's: a draft is only kept where the token sampled from the
model's logits is that draft.

Several sequences (``NpuGemma(sequences=N)``): the device holds a KV cache per sequence and the model has a lane for
each (``OverlayModel.lanes``); the lanes that generate share their decode steps, a step's slots dealt to them
(``models.lanes``: one stream of the weights for all of them). The drafter guesses for one lane at a time, on that
lane's caches, before the lanes' joint step (``Rounds.before``), for the lane's share of the step's slots; the step
runs the drafts it has room for, those kept most often first.

Memory: the weights the NPU streams (the layers and the LM head: 3.0 GB of E4B's Q4_0 file) are
re-laid-out into a device buffer at load, at their quantization's size; nothing is written to
disk. The per-layer token embedding (1.6 GB) is never loaded: its rows are read from the mapped
file, as is the token embedding's row.
"""

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.kv import kv_precision
from artlite.decode.protocol import SLOTS as BATCH  # tokens per dispatch
from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig, assistant_of

from ..lanes import alone
from ..model import Model
from ..npu_model import Checkpoint
from ..overlay_model import OverlayModel
from ..sampling import Penalties, Sampling
from ..spec_options import drafter_type
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
    per 8 dims); ``sequences``: the sequences the device holds, each with KV caches of ``capacity`` positions
    (``lanes``: the model of each; their generations share decode steps)."""

    STEP_COST = (1.0, 1.03, 1.07, 1.09)  # (E4B Q4_0 at 1K)
    P_CACHE = 1024  # tokens whose per-layer projections are kept (21 KB each: ``steps``)
    LANE_SHARED = (*OverlayModel.LANE_SHARED, "gguf", "host", "capacity", "_p", "_p_lock")
    LANE_OWN = (*OverlayModel.LANE_OWN, "_h", "_resid")
    _h: np.ndarray | None = None  # the hidden state a draft continues from (``hidden_state``)
    # (a drafter: the residual streams (S, hidden) after the last layer of the lane's last step with logits)
    _resid: np.ndarray | None = None

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, threads: int = 8, prefill: bool = True,
                 drafter: str | None = "", kv_k: str = "hilo", sequences: int = 1):
        self.gguf = gguf
        self.host = GemmaHost(gguf, threads=threads)
        # (token -> its per-layer projection, bf16 bits: every lane's; a lane's lookup and another's eviction: the lock)
        self._p: OrderedDict[int, np.ndarray] = OrderedDict()
        self._p_lock = threading.Lock()
        self.cfg = self.host.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.softcap = self.cfg.final_logit_softcapping
        if capacity > self.cfg.max_position_embeddings:
            raise ValueError(f"capacity {capacity} exceeds the model's context of {self.cfg.max_position_embeddings}")
        self.capacity = capacity
        self.overlay = GemmaOverlay(gguf, max_positions=capacity + BATCH, kv_k=kv_k, sequences=sequences)
        self.prefill = GemmaPrefill(self.overlay) if prefill else None
        path = assistant_of(gguf.path) if drafter == "" else drafter
        self.drafter: GemmaDraft | None = None  # (OverlayModel's)
        if path and (drafter or os.path.exists(path)):
            self.drafter = GemmaDraft(self.overlay, GGUFFile(path))
        self._open(threads=threads)
        gguf.release()
        self.position = 0
        self._open_lanes()

    def _open(self, *, threads: int) -> None:
        """(and the prefill's hardware context before timing, as the decode's)"""
        super()._open(threads=threads)
        if self.prefill is not None:
            self.prefill.lib._kernel()

    def _also(self) -> dict[str, int]:
        """(and the drafter's buffers)"""
        return super()._also() | (self.drafter.sizes if self.drafter is not None else {})

    def _load_drafters(self, threads: int) -> None:
        if self.drafter is not None:
            self.drafter.load(threads=threads)
            self.drafter.gguf.release()

    def _prepare_steps(self) -> None:
        """(with drafts every S has logits; a prompt's last token after its chunk: a step from the prefill's last layer)"""
        for S in range(1, BATCH + 1):
            self.overlay.prepare(S, logits=True)
            self.overlay.prepare(S, logits=False)
        if self.prefill is not None:
            self.overlay.prepare(1, logits=True, start=self.prefill.layers[-1])

    def describe(self) -> str:
        ov = self.overlay
        d = f", drafter {os.path.basename(self.drafter.gguf.path)}" if self.drafter is not None else ""
        kv = f"{ov.sizes['KV'] / 1e6:.0f} MB of KV cache"
        return (f"{ov.image.streamed_bytes() / 1e9:.2f} GB of weights, "
                f"{kv if ov.sequences == 1 else f'{ov.sequences} sequences of {kv} each'} "
                f"({self.capacity} positions){d}")

    def reset(self) -> None:
        """Forget the cached positions."""
        self.position = 0

    @alone
    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        """(the drafter's steps patch their caches' fills again)"""
        super().restore(position, checkpoint)
        if self.drafter is not None:
            self.drafter.reset()

    def inputs(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
        """What a decode step needs of its tokens: their scaled embeddings (n, hidden) as bf16 values, and their
        per-layer token rows (n, layers, per_layer_dim; ``GemmaHost.layer_rows``) or None."""
        e = self.host.layer_rows(tokens) if self.cfg.per_layer_dim else None
        return self.host.embed(tokens).astype(bfloat16).astype(np.float32), e

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return self.inputs(tokens)

    def _kept_p(self, tokens: list[int]) -> np.ndarray | None:
        """The tokens' per-layer projections (n, the overlay's layers, per_layer_dim) bf16 bits where every one of them
        is kept (``_p``), else None."""
        with self._p_lock:
            if not all(t in self._p for t in tokens):
                return None
            for t in tokens:
                self._p.move_to_end(t)
            return np.stack([self._p[t] for t in tokens])

    def _keep_p(self, tokens: list[int], rows: np.ndarray) -> None:
        """Keep the tokens' per-layer projections as a step left them (``GemmaOverlay.read_p``)."""
        with self._p_lock:
            for t, row in zip(tokens, rows, strict=True):
                self._p[t] = row.copy()
            while len(self._p) > self.P_CACHE:
                self._p.popitem(last=False)

    @alone
    def steps(self, tokens: list[int], *, logits: bool = True, **step: object) -> np.ndarray | None:
        """(a step of tokens whose per-layer projections are kept writes them instead of projecting the embeddings,
        ``GemmaOverlay.step(p=)``: 0.6 ms less; one that projects keeps its tokens'; with a drafter, a step with
        logits keeps its residual streams: ``hidden_state``)"""
        p = self._kept_p(tokens) if self.cfg.per_layer_dim else None
        out = super().steps(tokens, logits=logits, **({"p": p} if p is not None else {}), **step)
        t0 = time.perf_counter()
        ov, n = self.overlay, len(tokens)
        self._took(tokens, (ov.read_p(n) if self.cfg.per_layer_dim and ov.ple else None,
                            ov.residuals(n) if logits and self.drafter is not None else None))
        self.t_host += time.perf_counter() - t0
        return out

    def _step_kw(self, tokens: list[int]) -> dict[str, object]:
        return {"p": self._kept_p(tokens)} if self.cfg.per_layer_dim else {}

    def _tapped(self, S: int) -> tuple:
        """(the slots' per-layer projections where the step projected them, bf16 bits; with a drafter their residual
        streams after the last layer)"""
        ov = self.overlay
        return (ov.read_p(S) if self.cfg.per_layer_dim and ov.ple else None,
                ov.residuals(S) if self.drafter is not None else None)

    def _took(self, tokens: list[int], taken: tuple) -> None:
        p, resid = taken
        if p is not None:
            self._keep_p(tokens, p)
        if resid is not None:
            self._resid = resid

    def chunk_inputs(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
        """A prefill chunk's inputs: the scaled embeddings (n, hidden) and the per-layer token rows (n, the prefill's
        layers, dim; ``GemmaHost.layer_rows``), bf16 values."""
        return self.host.embed(tokens).astype(bfloat16), self._layer_rows(tokens)

    def _layer_rows(self, tokens: list[int]) -> np.ndarray | None:
        if not self.cfg.per_layer_dim:
            return None
        return self.host.layer_rows(tokens, len(self.prefill.layers)).astype(bfloat16)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        """(the first chunk: its per-layer token rows on the worker thread while its first operators run)"""
        if not first:
            return self.chunk_inputs(tokens)
        return self.host.embed(tokens).astype(bfloat16), self._pool.submit(self._layer_rows, tokens)

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        """(a step from the prefill's last layer on - the layers that share a KV cache, the LM head: 19 of E4B's 42
        layers, ~25 ms less than a decode step -, its input the chunk's residual stream)"""
        t0 = time.perf_counter()
        x = self.prefill.residual(n - 1)
        emb, e = self.inputs([token])
        self.t_host += time.perf_counter() - t0
        out = self._dispatch(lambda: self.overlay.step(x, e, self.position + n - 1, start=self.prefill.layers[-1],
                                                       emb=emb)[0])
        if self.drafter is not None:
            self._resid = self.overlay.residuals(1)
        return out

    def outputs(self, tokens: list[int], at: Sequence[int] | None = None, *, norm: bool = True,
                progress: Callable[[int], None] | None = None, sets: Sequence[int] | None = None) -> np.ndarray:
        """(none: a chunk runs the layers that share a KV cache for its last token alone, ``_chunk_logits``)"""
        raise NotImplementedError("Gemma 4 has no readouts")

    # ------------------------------------------------------------------
    # Multi-token prediction
    # ------------------------------------------------------------------

    def hidden_state(self, slot: int | None = None) -> np.ndarray:
        """The output-normed hidden state (hidden,) bf16 of slot ``slot`` of the lane's last step with logits (default:
        its last token's, ``slot``): what a draft continues from."""
        slot = self.slot if slot is None else slot
        r = self._resid[slot] if self._resid is not None else self.overlay.residual(slot)
        return output_norm(r, self.host.mat("output_norm.weight"), self.cfg.rms_norm_eps)

    def draft(self, token: int, h: np.ndarray, n: int, penalties: Penalties | None = None) -> list[int]:
        """The drafter's guesses of the (up to n) tokens after ``token`` - the token of the next position, not run yet;
        h: the hidden state of the position before it -: its most likely one, the most likely one after that, ...
        (ends with an end-of-sequence token); ``penalties``: the generation's (up to ``token``), which the guesses
        get too. Several lanes: with the device for the lane's sequence (``_before``; none where it is not to be
        had)."""
        out: list[int] = []
        recent = penalties.copy() if penalties is not None and (penalties.active or penalties.mask is not None) \
            else None  # (the generation's penalties and its mask, forked: the guesses follow the grammar)
        with self._before() as free:
            t0 = time.perf_counter()  # (the guesses' time, without the wait for the device)
            while free and len(out) < n and token not in self.eos_token_ids:
                x = self.host.embed([token])[0]
                raw = self.drafter.step(x, h if not out else None, self.position)
                if recent is None:
                    token = self.drafter.token(raw)
                else:  # (a grammar's mask: the best of the few best guesses it takes, each checked, no whole mask)
                    grammar, recent.mask = recent.mask, None
                    ids = recent.top(raw, GUESSES if grammar is not None else 1, self.softcap, rows=self.drafter.rows,
                                     where=self.drafter.where)[1]
                    recent.mask = grammar
                    j = grammar.first_allowed(ids) if grammar is not None else 0
                    if j < 0:
                        break
                    token = int(ids[j])
                    recent.accept(token)
                out.append(token)  # (its per-layer inputs: with the verifying step's others, in one pass)
        self.t_draft += time.perf_counter() - t0
        self.drafted += len(out)
        self._chains = (1.0,) * len(out)  # (how often a draft of each depth is kept: ``_calibrate``, ``_mtp_offer``)
        return out

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        return self.draft(token, self._h, k, penalties)

    def _mtp_offer(self, n: int) -> tuple[float, ...]:
        """(a draft is a guess from the hidden state alone, nothing follows it: a shared step runs those it has room
        for; a draft of depth d is kept as often as the rounds' drafts of that depth were: ``_calibrate``)"""
        return tuple(self._kappa(d) for d in range(n))

    def _begin(self) -> None:
        """(the hidden state a draft continues from: the prompt's last token's)"""
        if self.drafter is not None:
            self._h = self.hidden_state()

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool, won: int = 0) -> None:
        super()._kept(p, tokens, m, mtp, won)
        if self.drafter is not None:
            self._h = self.hidden_state(m - 1)


GUESSES = 8  # a drafter's best guesses a grammar's mask checks in order (``NpuGemma.draft``)


class Gemma(Model):
    """Gemma 4 from a GGUF file: the architecture, tokenizer and chat template from its metadata. ``kv_k``: the KV
    caches' K precision ("hilo": bfp16 hi + lo blocks, "single": one block per 8 dims); ``spec_type``, ``draft_model``:
    the drafter (``spec_options.drafter_type``): auto, ``mtp-<model file>`` next to it when it is there; none; draft-mtp: that
    file, which must exist; ``draft_model``: a ``gemma4-assistant`` file."""

    name = "Gemma 4"
    parallel = True
    thinks = True
    reasoning = ("<|channel>thought", "<channel|>")  # (a thought is the "thought" channel)

    def __init__(self, path: str, *, kv_k: str = "hilo", spec_type: str = "auto", draft_model: str | None = None):
        kv_precision(kv_k)  # (a K precision the overlay takes, before the file opens)
        self.kv_k = kv_k
        kind = drafter_type(spec_type, draft_model, ("draft-mtp",), self.name)
        if kind == "draft-mtp" and draft_model is None and not os.path.exists(assistant_of(path)):
            raise ValueError(f"{path}: no MTP drafter ({os.path.basename(assistant_of(path))} next to it)")
        self.drafter = None if kind == "none" else draft_model or ""  # (``NpuGemma``'s)
        self.gguf = GGUFFile(path)
        self.tokenizer = GemmaTokenizer(self.gguf)
        self.eos_token_ids = GemmaConfig.from_gguf(self.gguf).eos_token_ids

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_bos=True)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools,
                                                 preserve_thinking=preserve, template=self.chat_template,
                                                 **(variables or {})))

    def sampling(self, *, think: bool = False) -> Sampling:
        return Sampling(temp=1.0, top_k=64, top_p=0.95)  # (the model card's)

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> NpuGemma:
        """(``sequences``: ``NpuGemma``'s)"""
        return NpuGemma(self.gguf, capacity=capacity, prefill=prefill, drafter=self.drafter, kv_k=self.kv_k,
                        sequences=sequences)
