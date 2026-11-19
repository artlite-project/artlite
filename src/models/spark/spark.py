"""XHToken Spark-X2.5 / X2.6 from a GGUF file (``artlite.model_config.SparkConfig``): 4B (36 layers) and 1.7B (28),
three sliding-window layers per full one, head_dim 256, a sigmoid output gate per head, GeGLU; XHToken's Q4_K_M
(Q6_K / Q8_0 in the sensitive tensors, a Q8_0 head) and Q8_0 files, the community's quantizations. A DSpark drafter
(``model_config.spark_dspark_of``: the model's ``<repository>-DSpark-GGUF``) drafts 7 tokens a pass.

The chat template (the file's): every turn ``<｜start▁of▁sentence｜>`` + its role tag (``<|System|>``, ``<|User|>``,
``<|Bot|>``, ``<|Tool|>``) .. ``<｜end▁of▁sentence｜>``, a default system prompt (with its ``thinking_effort`` sentence when
thinking); a reply after ``<|Bot|><think>`` (thinking) or ``<|Bot|></think>``; an earlier reply keeps its thought where
its message has ``reasoning_content`` (``preserve``, else dropped); tools as JSON lines in ``<tools>``, calls in GLM-4.5's
syntax (``server.tools``); ``reasoning_effort`` "high" (the template's default) or "low" (``Spark.efforts``: the levels
clients send mapped to those two).

Several sequences (``NpuSpark(sequences=N)``): the device holds the caches of each and the model has a lane for each
(``OverlayModel.lanes``); the lanes that generate share their decode steps, a step's 8 slots dealt to them
(``models.lanes``: one stream of the weights for all of them, an attention pass per sequence). Without the DSpark
drafter, whose records follow one sequence."""

import logging
import time
from collections.abc import Iterator, Sequence

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import SparkConfig, spark_dspark_of

from ..dspark import MarkovChain
from ..lanes import alone
from ..model import Model
from ..npu_model import Checkpoint
from ..overlay_model import OverlayModel
from ..sampling import TOP_K_ALL, Mask, Penalties, Probs, Sampling
from ..sequence import token_ids
from ..spec_options import drafter_type
from ..tokenizer import BpeTokenizer, chat_prompt
from .dspark import DSparkOverlay, needs
from .overlay import SparkOverlay
from .prefill import SIZES, SparkPrefill

log = logging.getLogger("artlite")


class NpuSpark(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the full layers' caches hold; ``prefill``: prompts in chunks
    (else as decode steps), of the sizes ``sizes``; ``dspark``: a DSpark drafter's file: a generation's rounds of its
    drafts (``dspark.DSparkOverlay``, the Markov chain on the host: ``models.dspark``); ``sequences``: the sequences
    the device holds, each with caches of ``capacity`` positions (``lanes``: the model of each; their generations share
    decode steps; without a drafter).

    The drafter's records follow the model's: a prompt's chunks write them (``SparkPrefill(dspark=)``), after each
    committed step its rows wait (``_inj``: the step's slots and first position; their taps in the model's ACT) until the
    drafter's next dispatch, the round's block's or one of their own before the model's next dispatch. A round's step of
    the token and 7 drafts injects all 8 rows; the kept ones are the sequence's, the next block writes over the others."""

    recurrent = True  # (the sliding-window layers' rings and the drafter's hold a window: a position comes back from a
    # checkpoint, ``SparkOverlay.snapshot``)
    STEP_COST = (1.0, 1.002, 1.006, 1.008, 1.014, 1.018, 1.02, 1.02)
    LANE_SHARED = (*OverlayModel.LANE_SHARED, "gguf", "dspark", "chain")
    LANE_OWN = (*OverlayModel.LANE_OWN, "_inj", "_end")
    _inj: tuple[int, int] | None = None  # (the last step's slots and first position: its rows wait for the drafter)
    _end = 0  # (the running prompt's end)

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, threads: int = 0, prefill: bool = True,
                 sizes: tuple[int, ...] = SIZES, dspark: str | None = None, sequences: int = 1):
        if dspark and sequences > 1:
            raise ValueError("several sequences: without the DSpark drafter (its records follow one sequence)")
        self.gguf = gguf
        dg = GGUFFile(dspark) if dspark else None
        self.overlay = SparkOverlay(gguf, max_positions=capacity, sequences=sequences,
                                    **(needs(dg) if dg is not None else {}))
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.dspark = DSparkOverlay(self.overlay, dg) if dg is not None else None
        self.chain = MarkovChain(dg) if dg is not None else None
        self.drafter = self.dspark
        self.DRAFTS = self.overlay.SLOTS - 1
        self.prefill = SparkPrefill(self.overlay, sizes=sizes, dspark=self.dspark) if prefill else None
        self._open(threads=threads)
        self._w_out = self.overlay._f32("output_norm.weight")
        gguf.release()
        self.position = 0
        self._open_lanes()

    def _load_drafters(self, threads: int) -> None:
        if self.dspark is not None:
            self.dspark.load(threads=threads)
            self.chain.resident()
            self._mask = self._rows([self.dspark.cfg.mask_token])[0]

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov, d = self.overlay, self.dspark
        kv = f"{ov.sizes['KV'] / 1e6:.0f} MB of KV cache"
        out = (f"{ov.image.nbytes / 1e9:.2f} GB of weights, "
               f"{kv if ov.sequences == 1 else f'{ov.sequences} sequences of {kv} each'} ({self.capacity} positions)")
        if d is not None:
            out += f", DSpark drafter ({d.image.nbytes / 1e9:.2f} GB, {d.sizes['KV'] / 1e6:.0f} MB of records)"
        return out

    @alone
    def reset(self) -> None:
        """Forget the cached positions."""
        self._abandon()
        self.overlay.reset()
        if self.dspark is not None:
            self.dspark.reset()
        self._inj = None
        self.position = 0

    def _rows(self, tokens: list[int]) -> np.ndarray:
        """The tokens' embeddings (n, hidden) fp32."""
        return self.gguf.rows("token_embd.weight", tokens)

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (self._rows(tokens),)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        return (self._rows(tokens),)

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))

    @alone
    def steps(self, tokens: list[int], *, logits: bool = True, **step: object) -> np.ndarray | None:
        """(the drafter's rows of the last step injected first; this step's after it)"""
        p = self.position
        self._inject()
        out = super().steps(tokens, logits=logits, **step)
        if self._drafting:
            self._inj = (len(tokens), p)
        return out

    @alone
    def prompt(self, tokens: list[int], **kw: object) -> np.ndarray | None:
        """(the prompt's end: the drafter's rows its chunks write, ``_prefill_run``)"""
        self._end = self.position + len(tokens)
        return super().prompt(tokens, **kw)

    def _prefill_run(self, args: tuple, T: int, n: int, sets: Sequence[int] | None = None) -> None:
        """(the drafter's rows of the last step first; the chunk's of the positions a block's window after the prompt
        reads)"""
        self._inject()
        rows_from = self._end - self.dspark.cfg.window - 32 if self._drafting else 1 << 62
        self.prefill.run(*args, self.position, T, rows_from=rows_from)
        self.stale()

    def _overlays(self) -> list[SparkOverlay]:
        return [self.overlay] + ([self.dspark] if self.dspark is not None else [])

    @alone
    def checkpoint(self) -> Checkpoint:
        """(the drafter's rows of the last step in first: their taps do not outlive the model's next step)"""
        self._abandon()
        self._inject()
        return super().checkpoint()

    @alone
    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        self._inj = None
        super().restore(position, checkpoint)

    def _inject(self) -> None:
        """The drafter's rows of the model's last step, while its taps are in place."""
        if self._inj is not None:
            S, q = self._inj
            self._inj = None
            self._dispatch(lambda: self.dspark.run(S, q, None, 0))

    # ------------------------------------------------------------------
    # Drafts
    # ------------------------------------------------------------------

    def generate(self, logits: np.ndarray, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = (),
                 mask: Mask | None = None, probs: Probs | None = None, *, drafts: int | None = None) -> Iterator[int]:
        """(the drafter's head table follows the prompt and each token: ``HeadTable.see``)"""
        see = self.dspark.table.see if self._drafting else None
        if see is not None:
            see(token_ids(prompt))
        for token in super().generate(logits, max_tokens, sampling, prompt, mask, probs, drafts=drafts):
            if see is not None:
                see((token,))
            yield token

    def forced_drafts(self, tokens: list[int], first: int, *, turns: Sequence[int] = (), prefill: bool = True
                      ) -> Iterator[tuple[list[int], np.ndarray]]:
        """(the drafter's table sees all the tokens first: runs over the same tokens draft from the same rows)"""
        if self._drafting:
            self.dspark.table.see(tokens)
        return super().forced_drafts(tokens, first, turns=turns, prefill=prefill)

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        """A round's drafts after ``token`` (the next position's): the last step's rows injected and the block at the
        position in one dispatch, the Markov chain's drafts (up to k)."""
        t0 = time.perf_counter()
        d = self.dspark
        S, q = self._inj or (0, 0)
        self._inj = None
        x = d.block_rows(self._rows([token])[0], self._mask)
        self._dispatch(lambda: d.run(S, q, x, self.position))
        fork = getattr(penalties.mask, "fork", None)  # (a grammar's: the chain's drafts follow it)
        out, conf = self.chain.walk(d.logits(), d.table.ids, d.h(), token, min(k, d.B), set(self.eos_token_ids),
                                    fork() if fork is not None else None)
        self._chains = tuple(np.cumprod(conf).tolist())
        self.t_draft += time.perf_counter() - t0
        self.drafted += len(out)
        self.blocks += 1
        return out


class Spark(Model):
    """A spark2_5 GGUF file: the architecture, tokenizer and chat template from its metadata (module docstring); a raw
    text gets no BOS (the file's ``add_bos_token``, as llama.cpp). ``spec_type`` / ``draft_model``: its DSpark drafter
    (``spec_options``: auto, the one found by name; none)."""

    name = "Spark-X"
    parallel = True
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<｜start▁of▁sentence｜>", "<｜start▁of▁sentence｜><|User|>"
    efforts = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "high"}  # noqa: RUF012

    def __init__(self, path: str, *, spec_type: str = "auto", draft_model: str | None = None):
        self.gguf = GGUFFile(path)
        self.cfg = SparkConfig.from_gguf(self.gguf)
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        kind = drafter_type(spec_type, draft_model, ("draft-dspark",), self.name)
        self.dspark = (draft_model or spark_dspark_of(path)) if kind in ("auto", "draft-dspark") else None
        if kind == "draft-dspark" and self.dspark is None:
            raise ValueError(f"{path}: no DSpark drafter (a <repository>-DSpark-GGUF) in the models directory")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_text(self, messages: list[dict[str, object]], *, think: bool = False,
                  tools: list[dict[str, object]] | None = None, preserve: bool = False,
                  variables: dict[str, object] | None = None) -> str:
        """The chat template's text of a conversation up to the assistant's turn (``preserve``: the earlier replies keep
        their ``reasoning_content``)."""
        if not preserve:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        return chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools, template=self.chat_template,
                           **(variables or {}))

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(self.chat_text(messages, think=think, tools=tools, preserve=preserve,
                                                    variables=variables))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*: temperature 1, top-p 0.95, top-k off)."""
        s = {"temp": 1.0, "top_k": -1, "top_p": 0.95} | self.cfg.sampling
        k = int(s["top_k"])
        return Sampling(temp=s["temp"], top_k=k if k > 0 else TOP_K_ALL, top_p=s.get("top_p", 1.0),
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> NpuSpark:
        """(``sequences``: ``NpuSpark``'s; several: without the DSpark drafter)"""
        if self.dspark is not None and sequences > 1:
            log.warning("%s with %d sequences runs without its DSpark drafter (the drafter's records follow one "
                        "sequence)", self.name, sequences)
        return NpuSpark(self.gguf, capacity=capacity, prefill=prefill, dspark=self.dspark if sequences == 1 else None,
                        sequences=sequences)
