"""Thinking Machines' Inkling 2 from a GGUF file (``artlite.model_config.InklingConfig``: 26B weights in 24 layers, a
mixture of 128 experts of which 8 run per token, the middle 12 layers looped 3 times: 48 visits a token).

The chat template (the file's): a conversation is blocks ``<|message_{user,model,system,tool}|>[name]<|content_*|>
.. <|end_message|>``, a model turn ends with ``<|content_model_end_sampling|>``; a system block "Thinking effort level:
x" (``reasoning_effort``: none 0 .. max 0.99, or a number; the template's default 0.9) before the first message that is
not a system one; tools as one JSON line in a ``tool_declare`` block, calls ``<|message_model|>NAME
<|content_invoke_tool_json|>{"name": .., "args": {..}}``; an image part as ``<|content_image|><|unused_200054|>``.
The template opens no turn for the reply: the model writes its own blocks (``<|message_model|><|content_thinking|>``
.. then ``<|message_model|><|content_text|>`` ..), and its turn ends at ``<|content_model_end_sampling|>``.

A decode dispatch runs up to 4 consecutive tokens through the 48 visits and the LM head on one stream of the weights
(``overlay.InklingOverlay``): the projections, the depth mixes, the loop's controllers and the experts on the main cores
and their helpers, the residual stream's state array in their memory, the attention, its KV caches and the K / V convs
on the attention cores; the head screened (``decode.screen``: the host computes its largest logits exactly). Prompts run
in prefill chunks (``prefill.InklingPrefill``; its sliding-window caches a ring that its largest chunk fits in,
``InklingPrefill.ring``), a prompt's last token as a decode step
(the head's); without the prefill as decode steps. The convs' state holds the last positions only: a sequence goes back
to a checkpoint (``recurrent``)."""

import time
from collections.abc import Sequence

import numpy as np

from artlite.decode.kv import kv_precision
from artlite.decode.protocol import SLOTS
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig, InklingDSparkConfig, inkling_dspark_of

from ..lanes import alone
from ..model import Model
from ..npu_model import Checkpoint
from ..overlay_model import OverlayModel
from ..sampling import TOP_K_ALL, Penalties, Sampling
from ..spec_options import drafter_type
from ..tokenizer import BpeTokenizer, chat_prompt
from .dspark import DSparkOverlay, MarkovChain
from .overlay import InklingOverlay
from .prefill import InklingPrefill


class NpuInkling(OverlayModel):
    """The model on the NPU (module docstring). ``capacity``: the positions the KV caches hold; ``kv_k``: their K
    precision ("hilo", "single"); ``wrap``: the sliding-window caches' ring (``InklingOverlay``); ``sequences``: the
    sequences the device holds (``artlite-server -np``: lanes whose steps share a dispatch), each with its caches and
    convs' state; ``prefill``: prompts in prefill chunks (``wrap`` by default ``InklingPrefill.ring``); ``dspark``: a DSpark
    drafter's file (``dspark.DSparkOverlay``): a generation's rounds of its drafts.

    The drafter's caches follow the model's positions: each step's taps (``InklingOverlay.tap_rows``, and the final
    normed state, the screen's input) wait on the host (``_pend``, per position) until the drafter's next dispatch, its
    injections before a round's block; a round's step keeps its kept rows' taps (``_kept``). A round: the injections,
    the block at the position, the Markov chain's drafts on the host (``dspark.MarkovChain``), the step of the token and
    the drafts the planner keeps (``SpeculativeMixin``)."""

    recurrent = True  # (the K / V convs' rings and the post convs' history windows: the last positions only)
    last_step = True  # (the prefill has no LM head)
    LANE_SHARED = (*OverlayModel.LANE_SHARED, "gguf", "_tn", "dspark", "chain")
    LANE_OWN = (*OverlayModel.LANE_OWN, "_pend", "_shadow", "_hist")
    STEP_COST = (1.0, 1.51, 1.87, 2.11)  # (steps of S on essay tokens, S3's: each slot's own experts)
    PLANNER = {"ewma_rate": 0.2}  # noqa: RUF012  (the drafter's edge is small: ``drafts.Planner``)

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "hilo", threads: int = 0,
                 wrap: int | None = None, sequences: int = 1, prefill: bool = False, taps: tuple[int, ...] = (),
                 dspark: str | None = None):
        self.gguf = gguf
        if prefill and wrap is None:  # (a ring a chunk of the largest size fits in, from its multiples)
            wrap = InklingPrefill.ring(capacity)
        dg = GGUFFile(dspark) if dspark else None
        if dg is not None:
            taps = InklingDSparkConfig.from_gguf(dg).taps[:-1]
        self.overlay = InklingOverlay(gguf, max_positions=capacity, kv_k=kv_k, wrap=wrap, sequences=sequences, taps=taps)
        self.dspark = DSparkOverlay(self.overlay, dg) if dg is not None else None
        self.drafter = self.dspark
        self.chain: MarkovChain | None = None
        self._pend: dict[int, np.ndarray] = {}  # (position -> its taps (taps, hidden) fp32: ``_rows_of``)
        # (the rounds' drafts before ``_keep`` cut them: (the first one's position, drafts, chains), checked against the
        # sequence's tokens as they come, ``_kept``: the drafter's calibration of every draft, not only the run ones)
        self._shadow: list[tuple[int, list[int], list[float]]] = []
        self._hist: dict[int, int] = {}  # (the sequence's tokens of the positions a shadow reads)
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.softcap = self.cfg.softcap
        self.prefill = InklingPrefill(self.overlay, taps=taps if dg is not None else ()) if prefill else None
        self._tn = np.asarray(gguf.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1)
        self._open(threads=threads)
        self._w_out = np.asarray(gguf.tensor("output_norm.weight").dequantize(), np.float32).reshape(-1)
        gguf.release()
        self.position = 0
        self._open_lanes()

    def _load_drafters(self, threads: int) -> None:
        if self.dspark is not None:
            d = self.dspark
            d.load(threads=threads)
            self.chain = MarkovChain(d.gguf, d.cfg, d.screen, d.fb)
            self.chain.resident()

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov, d = self.overlay, self.dspark
        kv = f"{ov.sizes['KV'] / 1e6:.0f} MB of KV cache and conv state"
        out = (f"{ov.image.nbytes / 1e9:.2f} GB of weights ({ov.image.streamed_bytes() / 1e9:.2f} GB dense), "
               f"{kv if ov.sequences == 1 else f'{ov.sequences} sequences of {kv} each'} ({self.capacity} positions, K "
               f"{'hi + lo' if ov.kp == 2 else 'single'})")
        if d is not None:
            out += f", DSpark drafter ({d.image.nbytes / 1e9:.2f} GB, {d.sizes['KV'] / 1e6:.0f} MB of records)"
        return out

    @alone
    def reset(self) -> None:
        """Forget the cached positions (and the convs' state)."""
        self._abandon()
        self.overlay.reset()
        if self.dspark is not None:
            self.dspark.reset()
        self._pend, self._shadow, self._hist = {}, [], {}
        self.position = 0

    def _overlays(self) -> list[InklingOverlay]:
        return [self.overlay] + ([self.dspark] if self.dspark is not None else [])

    # ------------------------------------------------------------------
    # The drafter's rows
    # ------------------------------------------------------------------

    def _rows_of(self, S: int) -> np.ndarray:
        """The last step's taps (S, taps, hidden) fp32: the tap visits' residuals, then the final normed states."""
        ov = self.overlay
        x = ov._lm_x(S)
        fin = np.stack([ov.screen.input(x[s]) for s in range(S)])
        return np.concatenate([ov.tap_rows(S), fin[:, None]], 1)

    @alone
    def steps(self, tokens: list[int], *, logits: bool = True, **step: object) -> np.ndarray | None:
        """(the step's taps wait for the drafter: ``_pend``)"""
        p = self.position
        out = super().steps(tokens, logits=logits, **step)
        if self.dspark is not None:
            self.dspark.dev.stale = True  # (another hardware context ran: the drafter's code pages again)
            if self.DRAFTS > 0:
                for i, row in enumerate(self._rows_of(len(tokens))):
                    self._pend[p + i] = row
        return out

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool, won: int = 0) -> None:
        """(the rejected drafts' taps are forgotten; the rounds' drafts checked against the kept tokens: ``_shadow``)"""
        super()._kept(p, tokens, m, mtp, won)
        for q in [q for q in self._pend if q >= self.position]:
            del self._pend[q]
        if not self._shadow:
            return
        for i in range(m):
            self._hist[p + i] = tokens[i]
        left = []
        for q, drafts, chains in self._shadow:
            k = 0
            while k < len(drafts) and self._hist.get(q + k, -1) == drafts[k]:
                k += 1
            if k == len(drafts) or q + k < self.position:  # (decided: all kept, or a known token differs)
                self._chains = tuple(chains)
                super()._calibrate(k)
            else:
                left.append((q, drafts, chains))
        self._shadow = left
        lo = min((q for q, _, _ in left), default=self.position)
        self._hist = {q: t for q, t in self._hist.items() if q >= lo}

    def _begin(self) -> None:
        """(a generation's rounds: no drafts of an earlier one to check)"""
        super()._begin()
        self._shadow, self._hist = [], {}

    def _calibrate(self, kept: int) -> None:
        """(a round's: from its drafts before ``_keep`` cut them, as the tokens come, ``_kept``)"""

    def _inject(self, end: int | None = None) -> None:
        """The waiting rows before ``end`` (default: the position) into the drafter's caches (dispatches of up to 4
        positions from its next one; the last row stays for f_rnn)."""
        d, end = self.dspark, self.position if end is None else end
        for q in [q for q in self._pend if q < d.next and q != end - 1]:  # (injected)
            del self._pend[q]
        if d.next >= end or d.next not in self._pend:
            return
        while d.next < end and d.next in self._pend:
            q, n = d.next, 0
            while n < SLOTS and q + n in self._pend and q + n < end:
                n += 1
            rows = np.stack([self._pend[q + i] for i in range(n)])
            self._dispatch(lambda rows=rows, q=q: d.inject(rows, q))
            for i in range(n):
                if q + i != end - 1:
                    del self._pend[q + i]
        self.stale()

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        """A round's drafts after ``token`` (the next position's): the waiting rows injected, the block at the position,
        the Markov chain's drafts (up to k)."""
        t0 = time.perf_counter()
        d, P = self.dspark, self.position
        self._inject()
        last = self._pend.get(P - 1)
        if last is None or d.next != P:  # (the drafter does not follow the sequence: no drafts)
            self._chains = ()
            return []
        f_rnn = d.mixes(last[None])[-1, 0]
        self._dispatch(lambda: d.block(self._step_inputs([token])[0][0, 0], P))
        self.stale()
        out = self.chain.walk(d.screened(), d.h(), d.jh_rows(), f_rnn, token, min(k, SLOTS - 1), set(self.eos_token_ids))
        chains = list(self.chain.chains)  # (the drafts that pay at the model's step times: ``SpeculativeMixin._keep``;
        # an MoE's slots route to experts of their own)
        if out:
            self._shadow.append((P + 1, list(out), chains))
        out = out[:next((i for i in range(len(out)) if not self._keep(chains[i], i)), len(out))]
        self._chains = tuple(chains[:len(out)])
        self._row_time(time.perf_counter() - t0)
        self.t_draft += time.perf_counter() - t0
        self.drafted += len(out)
        self.blocks += 1
        return out

    @alone
    def checkpoint(self) -> Checkpoint:
        """(the drafter's waiting rows in first)"""
        self._abandon()
        if self.dspark is not None:
            self._inject()
        return super().checkpoint()

    def _state(self) -> dict[str, object]:
        return {"pend": dict(self._pend), "next": self.dspark.next} if self.dspark is not None else {}

    def _set_state(self, state: dict[str, object]) -> None:
        if self.dspark is not None:
            self._pend, self.dspark.next = dict(state["pend"]), state["next"]
            self._shadow, self._hist = [], {}

    @alone
    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        if checkpoint is None and self.dspark is not None:
            self.dspark.reset()
            self._pend, self._shadow, self._hist = {}, [], {}
        super().restore(position, checkpoint)

    def _step_inputs(self, tokens: list[int]) -> tuple:
        """(the first visit's input: the embedding rows RMS-normed with ``token_embd_norm``, as one state)"""
        x = self.gguf.rows("token_embd.weight", np.asarray(tokens)).astype(np.float32)
        x = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + self.cfg.rms_norm_eps).astype(np.float32)
        return ((x * self._tn)[None],)

    def _prefill_run(self, args: tuple, T: int, n: int, sets: Sequence[int] | None = None) -> None:
        """(a drafter's: the chunk's taps wait for it, injected after the chunk)"""
        p0 = self.position
        super()._prefill_run(args, T, n, sets)
        if self.dspark is not None:
            self.dspark.dev.stale = True
        if self.dspark is not None and self.DRAFTS > 0 and self.prefill.taps_out is not None:
            for i, row in enumerate(self.prefill.taps_out):
                self._pend[p0 + i] = row
            self._inject(p0 + n)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        """(the chunk's normed embeddings (n, hidden): the first visit's input)"""
        return (self._step_inputs(tokens)[0][0],)


class Inkling(Model):
    """An Inkling 2 GGUF file (either layout, ``InklingConfig``): the architecture, tokenizer and chat template from its
    metadata (module docstring)."""

    name = "Inkling 2"
    thinks = True
    reasoning = ("<|content_thinking|>", "<|end_message|>")
    thought_first = True

    def __init__(self, path: str, *, kv_k: str = "hilo", spec_type: str = "auto", draft_model: str | None = None):
        kv_precision(kv_k)  # (a K precision the overlay takes, before the file opens)
        self.gguf = GGUFFile(path)
        self.cfg = InklingConfig.from_gguf(self.gguf)
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.kv_k = kv_k
        kind = drafter_type(spec_type, draft_model, ("draft-dspark",), self.name)
        self.dspark = (draft_model or inkling_dspark_of(path)) if kind in ("auto", "draft-dspark") else None
        if kind == "draft-dspark" and self.dspark is None:
            raise ValueError(f"{path}: no DSpark drafter (a dspark-*.gguf beside it)")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_text(self, messages: list[dict[str, object]], *, think: bool = False,
                  tools: list[dict[str, object]] | None = None, preserve: bool = False,
                  variables: dict[str, object] | None = None) -> str:
        """The chat template's text of a conversation (``think``: the template's reasoning effort, the given
        ``reasoning_effort`` or its default; else "none", unless ``variables`` give one; ``preserve``: the earlier
        replies keep their ``reasoning_content``)."""
        if not preserve:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        variables = dict(variables or {})
        if not think:
            variables.setdefault("reasoning_effort", "none")
        return chat_prompt(self.gguf, messages, tools=tools, template=self.chat_template, **variables)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(self.chat_text(messages, think=think, tools=tools, preserve=preserve,
                                                    variables=variables))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*: temperature 1, top-k 0 (all), top-p 1, min-p 0)."""
        s = {"temp": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0} | self.cfg.sampling
        k = int(s["top_k"])
        return Sampling(temp=s["temp"], top_k=k if k > 0 else TOP_K_ALL, top_p=s["top_p"], min_p=s["min_p"])

    def rows(self, tokens: Sequence[int]) -> np.ndarray:
        """The tokens' embedding rows (n, hidden) fp32, from the mapped file (before ``token_embd_norm``, which a
        text token's row takes and an image's does not)."""
        return self.gguf.rows("token_embd.weight", np.asarray(tokens))

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> NpuInkling:
        """(``sequences``: ``NpuInkling``'s)"""
        return NpuInkling(self.gguf, capacity=capacity, kv_k=self.kv_k, sequences=sequences, prefill=prefill,
                          dspark=self.dspark if sequences == 1 else None)


__all__ = ["Inkling", "NpuInkling"]
