"""Recursive's Flint from a GGUF file (``artlite.model_config.FlintConfig``): 40B parameters in 49 layers, a mixture of
256 experts of which 6 run per token (~2.5B active) and a shared expert, sliding-window layers of 513 keys and full ones
without positional encoding, learned attention sinks. The official file (``model_config.flint_gguf()``: the model's
trained INT4 codes in groups of 64, stored as GGUF Q4_1 pairs) runs as Q4_1_64 tiles, its F32 router as BFP16 planes
(``flint.image``).

A decode dispatch runs up to 4 consecutive tokens through every layer and the LM head on one stream of the weights
(``flint.overlay.FlintOverlay``): the projections and the experts on the main cores and their helpers, the routing on the
helpers (the experts' tiles pushed by the columns themselves: ``moe.route``), the attention and its KV caches on the
attention cores. Prompts run in chunks of up to 8192 tokens through the prefill (``flint.prefill.FlintPrefill``:
layer-major from one library ELF on the same weight image, the host routing each MoE layer's tokens between its pieces),
which leaves the caches the decode continues from. The host reads the tokens' embedding rows from the mapped file and
samples from the logits.

With the DFlash2 drafter (incoai's Flint-DFlash2, ``model_config.flint_dflash_of``; ``artlite.decode.dflash``) a
round's drafts come from one dispatch of the drafter's design in a hardware context of its own: it takes the model's last
step's rows from the taps at layers 6, 16, 27, 37, 47 and guesses a block, whose first 3 drafts the model's next step
verifies with the token (steps of 4 slots; each slot routes its own experts, ~+33% of a step: a round verifies the drafts
that pay for their slots, ``models.dflash``, and the rounds run while they beat steps of one, ``models.drafts.Planner``).
Its head is the first 33280 rows of the model's LM head (the LM rows laid out with it: ``FlintImage(head=)``).

The chat template (the file's): GLM-style turns ``<|system|>``, ``<|user|>``, ``<|assistant|>``, ``<|observation|>`` (tool
results); the model opens its own ``<think>`` (the prompt never does); tools as JSON in ``<tools>``, calls in GLM-4.5's
syntax (``server.tools``). A generation ends at ``<|end_of_text|>``, ``<|user|>`` or ``<|observation|>``.

Several sequences (``NpuFlint(sequences=N)``): the device holds a KV cache per sequence and the model has a lane for
each (``OverlayModel.lanes``); the lanes that generate share their decode steps, a step's slots dealt to them
(``models.lanes``: the dense weights stream once, an MoE layer runs the union of the slots' experts). The DFlash2
drafter drafts for a lane where its block pays (one that runs alone), its records one lane's sequence's at a time
(``models.dflash``).

The KV caches hold K as hi + lo bfp16 blocks by default (``kv_k``; "single": one block per 8 dims, a third fewer bytes,
but +0.013 nats a token in teacher-forced text, worse on every one of six essays): per position and layer 4 KV heads x
432 B, 84.7 KB a position over the 49 layers (the sliding-window layers' caches linear in the positions)."""

import itertools
import os
from collections.abc import Iterator
from collections.abc import Sequence as Seq

import numpy as np

from artlite.decode.dflash.overlay import DFlashOverlay
from artlite.decode.head import HeadTable
from artlite.decode.kv import kv_precision
from artlite.gguf_model import GGUFFile
from artlite.hf_model import open_weights
from artlite.model_config import FlintConfig, flint_dflash_of
from artlite.overlay import WeightBuffer

from ..dflash import DFlashDrafting
from ..lanes import alone
from ..model import Model
from ..npu_model import Checkpoint
from ..overlay_model import OverlayModel
from ..sampling import TOP_K_ALL, Mask, Penalties, Probs, Sampling
from ..sequence import token_ids
from ..spec_options import drafter_type
from ..tokenizer import BpeTokenizer, chat_prompt
from .image import FlintImage
from .overlay import FlintOverlay
from .prefill import SIZES, FlintPrefill


class NpuFlint(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the KV caches hold; ``kv_k``: their K precision ("hilo",
    "single"); ``prefill``: prompts in chunks through the prefill (else as decode steps), of the sizes ``sizes``;
    ``weights``: the weight image's buffer, allocated ahead (``WeightBuffer``; with a drafter: laid out with its head);
    ``dflash``: a DFlash2 drafter's file (module docstring); ``wrap``: the sliding-window caches' ring (``FlintOverlay``);
    ``sequences``: the sequences the device holds, each with KV caches of ``capacity`` positions (``lanes``: the model of
    each; their generations share decode steps; the drafter drafts for a lane where its block pays, which is for one
    that runs alone: ``models.dflash``)."""

    STEP_COST = (1.0, 1.32, 1.65, 1.98)  # (steps of 1 .. 4 slots at 1K: 31.4 ms for one slot; each slot's experts)
    BLOCK_MARGIN = 1.0  # (a block drafter's: the DFlash2 drafter's blocks by a rule of their own, ``_block_pays``)
    drafter = None
    LANE_SHARED = (*OverlayModel.LANE_SHARED, "gguf", "dflash", "_mask")
    LANE_OWN = (*OverlayModel.LANE_OWN, "flash")
    flash: DFlashDrafting | None = None  # (the drafter's rounds of the lane's sequence: ``models.dflash``)

    @property
    def recurrent(self) -> bool:
        """(sliding-window caches as rings, ``FlintOverlay``'s past ``WRAP`` groups: the sequence goes back only to a
        checkpoint, which holds their windows)"""
        return self.overlay.wraps

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "hilo", threads: int = 0,
                 prefill: bool = True, sizes: tuple[int, ...] = SIZES, weights: WeightBuffer | None = None,
                 dflash: str | None = None, wrap: int | None = None, sequences: int = 1):
        self.gguf = gguf
        self.overlay = FlintOverlay(gguf, max_positions=capacity, kv_k=kv_k, weights=weights, head=dflash is not None,
                                    wrap=wrap, sequences=sequences)
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        # (the drafter's rings hold a chunk's writes past them and its padding rows: the prefill's largest chunk, gap)
        gap = max(b - a for a, b in itertools.pairwise((0, *sizes)))
        self.dflash = DFlashOverlay(self.overlay, GGUFFile(dflash), max_chunk=sizes[-1], max_padding=gap) if dflash else None
        self.flash = DFlashDrafting(self) if self.dflash is not None else None  # (its rounds: ``models.dflash``)
        self.prefill = FlintPrefill(self.overlay, sizes=sizes, dflash=self.dflash) if prefill else None
        self._open(threads=threads)
        self._w_out = gguf.tensor("output_norm.weight").dequantize().reshape(-1)
        gguf.release()
        self.position = 0
        self._open_lanes()

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov, f = self.overlay, self.dflash
        d = "" if f is None else (f", DFlash2 drafter {os.path.basename(f.gguf.path)} ({f.image.nbytes / 1e9:.2f} GB, "
                                  f"{f.sizes['KV'] / 1e6:.0f} MB of records)")
        kv = f"{ov.sizes['KV'] / 1e6:.0f} MB of KV cache"
        return (f"{ov.image.nbytes / 1e9:.2f} GB of weights ({ov.image.streamed_bytes() / 1e9:.2f} GB dense), "
                f"{kv if ov.sequences == 1 else f'{ov.sequences} sequences of {kv} each'} ({self.capacity} positions, K "
                f"{'hi + lo' if ov.kp == 2 else 'single'}){d}")

    def _load_drafters(self, threads: int) -> None:
        if self.dflash is not None:
            self.dflash.load(threads=threads)
            self._mask = self._rows([self.dflash.cfg.mask_token])[0]

    def _open_lanes(self) -> None:
        """(the drafter's rounds: a lane's own)"""
        super()._open_lanes()
        for lane in self.lanes[1:]:
            lane.flash = DFlashDrafting(lane) if self.dflash is not None else None

    @property
    def _follows(self) -> bool:
        """Whether the drafter's records follow the model's steps (``_drafting``: whether it may draft now)."""
        return self.dflash is not None and self.DRAFTS > 0

    def _block_pays(self, k: int) -> bool:
        """(a block for a lane that runs alone: in a round of several each lane's block would take the drafter's
        records from the other's sequence first, ``DFlashDrafting.own``: 11 MB each way at 1K positions, more than
        the block; measured with two requests: 39 tokens per second with blocks, 43.5 without)"""
        return self._rounds.asking(self.seq) == 1

    @property
    def _drafting(self) -> bool:
        """(several lanes: not while another lane has the device alone, nor where the drafter lost the lane's
        sequence)"""
        return self._follows and (self._rounds is None or not (self._rounds.held() or self.flash.lost))

    @property
    def _table(self) -> HeadTable:
        """The drafter's head."""
        return self.dflash.table

    @alone
    def prepare(self, prompt_len: int) -> None:
        """(with drafts: the rounds' steps' transactions and the drafter's, else built at their first use)"""
        super().prepare(prompt_len)
        for S in range(1, self.DRAFTS + 2) if self.lookup or self._follows else ():
            self.overlay.prepare(S)
            if self.dflash is not None:
                self.dflash.prepare(S)
        if self.dflash is not None:
            self.dflash.dev.open()  # (its hardware context took the array: the model's code pages load again)
            self.dflash.prepare(0)
            self.stale()

    @alone
    def reset(self) -> None:
        """Forget the cached positions."""
        self._abandon()
        self.overlay.reset()
        if self.dflash is not None:
            if self._rounds is None:
                self.dflash.reset()
            self.flash.reset()
        self.position = 0

    def _overlays(self) -> list[FlintOverlay | DFlashOverlay]:
        return [self.overlay] + ([self.dflash] if self.dflash is not None else [])

    @alone
    def checkpoint(self) -> Checkpoint:
        """(the drafter's rows of the last step in first: their taps do not outlive the model's next step; several
        lanes: the drafter the lane's own)"""
        self._abandon()
        if self.flash is not None and self._rounds is not None:
            self.flash.own()
        self._inject()
        return super().checkpoint()

    @alone
    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        """(several lanes: the drafter the lane's own for a checkpoint's records; without one it follows the sequence
        no longer, until the prompt's chunk)"""
        fd = self.flash if self._rounds is not None else None
        if fd is not None and checkpoint is not None:
            fd.own(restored=True)
        super().restore(position, checkpoint)
        if fd is not None:
            if checkpoint is None and position:
                fd.lose()
            else:
                fd.rows, fd.rows_from = (), position

    def _state(self) -> dict[str, object]:
        """(several lanes: whether the drafter follows the sequence)"""
        return {"lost": self.flash.lost} if self.flash is not None and self._rounds is not None else {}

    def _set_state(self, state: dict[str, object]) -> None:
        if self.flash is not None and self._rounds is not None:
            self.flash.lost = bool(state.get("lost"))

    def _rows(self, tokens: list[int]) -> np.ndarray:
        """The tokens' embedding rows (n, hidden) fp32."""
        return self.gguf.rows("token_embd.weight", tokens)

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (self._rows(tokens),)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        return (self._rows(tokens),)

    # ------------------------------------------------------------------
    # The drafter's rounds (``models.dflash``)
    # ------------------------------------------------------------------

    def steps(self, tokens: list[int], *, logits: bool = True, commit: bool = True) -> np.ndarray | None:
        """(``commit``: the drafter takes the step's rows, else the round's ``_kept`` does)"""
        p, n = self.position, len(tokens)
        self._inject()
        out = super().steps(tokens, logits=logits)
        if self._follows and self._rounds is not None:  # (while the device is the lane's)
            self.flash.wait(p, self.overlay.tap_rows(self.dflash.cfg.taps, n))
        elif commit and self._follows:
            self.flash.after_step(n, p)
        return out

    def _tapped(self, S: int) -> tuple:
        """(the drafter's taps of the step's rows)"""
        return (self.overlay.tap_rows(self.dflash.cfg.taps, S),) if self.dflash is not None else ()

    def _took(self, tokens: list[int], taken: tuple) -> None:
        if taken and self._follows:
            self.flash.wait(self.position - len(tokens), taken[0])

    def _inject(self) -> None:
        """The drafter's rows of the model's last step, while its taps are in place."""
        if self.flash is not None:
            self.flash.inject()

    def _drop_carry(self) -> None:
        if self.flash is not None:
            self.flash.drop_carry()

    def _dflash_ahead(self, S: int, p: int) -> None:
        self.overlay.patch_ahead(S, p)

    def _prefill_run(self, args: tuple, T: int, n: int, sets: Seq[int] | None = None) -> None:
        """(the drafter's rows of the last step first; several lanes: the drafter the lane's own, the chunk writes its
        records)"""
        fd = self.flash if self._rounds is not None else None
        if fd is not None:
            fd.own()
        self._inject()
        super()._prefill_run(args, T, n, sets)
        if fd is not None:
            fd.found()

    def generate(self, logits: np.ndarray, max_tokens: int, sampling: Sampling, prompt: Seq[int] = (),
                 mask: Mask | None = None, probs: Probs | None = None, *, drafts: int | None = None) -> Iterator[int]:
        """(the drafter's table follows the prompt and each token: ``HeadTable.see``)"""
        see = self._table.see if self._follows else None
        if see is not None:
            see(token_ids(prompt))
        for token in super().generate(logits, max_tokens, sampling, prompt, mask, probs, drafts=drafts):
            if see is not None:
                see((token,))
            yield token

    def forced_drafts(self, tokens: list[int], first: int, *, turns: Seq[int] = (), prefill: bool = True
                      ) -> Iterator[tuple[list[int], np.ndarray]]:
        """(the drafter's table sees all the tokens first)"""
        if self._follows:
            self._table.see(tokens)
        return super().forced_drafts(tokens, first, turns=turns, prefill=prefill)

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        return self.flash.block(token, k)

    def _mtp_offer(self, n: int) -> tuple[float, ...]:
        """(a block's drafts: a shared step runs those it has room for, each kept as often as its chain's probability
        says, ``_kappa``)"""
        return tuple(c * self._kappa(i) for i, c in enumerate(self._chains[:n]))

    def _verify(self, tokens: list[int], mtp: bool) -> np.ndarray:
        """(a round's step commits none of its rows to the drafter: ``_kept`` gives it the kept ones; without a
        drafter: the round's step, shared with the other lanes' where the model has lanes)"""
        if self.flash is None or self._rounds is not None:
            return self._joined(tokens)
        return self.steps(tokens, commit=len(tokens) == 1 and not mtp)

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool, won: int = 0) -> None:
        """(a drafter's round: its kept rows pending for the drafter, injected after the next step's start; several
        lanes: the step's rows wait already, ``DFlashDrafting.wait``)"""
        if self._rounds is not None:
            self.position = p + m
            return
        if len(tokens) == 1 and not mtp:  # (a committed step: its row pending, ``steps``)
            return
        self.position = p + m
        if self.flash is not None:
            self.flash.kept(p, m, 0, len(tokens), 0)

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))


class Flint(Model):
    """A flint GGUF file: the architecture, tokenizer and chat template from its metadata (module docstring). Opening the
    file allocates the weight image's buffer and packs it in a thread (``WeightBuffer``) while the tokenizer and the
    designs load. ``kv_k``: the KV caches' K precision ("hilo", the default; "single"); ``spec_type`` / ``draft_model``:
    its DFlash2 drafter (``spec_options.drafter_type``: auto, the file ``model_config.flint_dflash_of`` finds, when
    there is one; none; draft-dflash: it, which must exist; ``draft_model``: that file)."""

    name = "Flint"
    parallel = True
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    user_turn = "<|user|>"  # (where the first user message begins: a prompt cache's mark once the rings wrap)

    def __init__(self, path: str, *, kv_k: str = "hilo", spec_type: str = "auto", draft_model: str | None = None):
        kv_precision(kv_k)  # (a K precision the overlay takes, before the file opens)
        kind = drafter_type(spec_type, draft_model, ("draft-dflash",), self.name)
        self.dflash = (draft_model or flint_dflash_of(path)) if kind in ("auto", "draft-dflash") else None
        if kind == "draft-dflash" and self.dflash is None:
            raise ValueError(f"{path}: no DFlash2 drafter (a Flint-DFlash2-GGUF of Q4_K_M) in the models directory")
        self.gguf = open_weights(path)
        self.cfg = FlintConfig.from_gguf(self.gguf)
        self.weights = WeightBuffer(FlintImage(self.gguf, self.cfg, head=self.dflash is not None))
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.kv_k = kv_k

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_text(self, messages: list[dict[str, object]], *, think: bool = False,
                  tools: list[dict[str, object]] | None = None, preserve: bool = False,
                  variables: dict[str, object] | None = None) -> str:
        """The chat template's text of a conversation up to the assistant's turn (``preserve``: the earlier replies keep
        their ``reasoning_content``; the template has no thinking switch: the model thinks when it opens ``<think>``)."""
        if not preserve:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        return chat_prompt(self.gguf, messages, tools=tools, template=self.chat_template, **(variables or {}))

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(self.chat_text(messages, think=think, tools=tools, preserve=preserve,
                                                    variables=variables))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*: temperature 1, top-p 0.95, top-k 20)."""
        s = {"temp": 1.0, "top_k": 20, "top_p": 0.95} | self.cfg.sampling
        k = int(s["top_k"])
        return Sampling(temp=s["temp"], top_k=k if k > 0 else TOP_K_ALL, top_p=s.get("top_p", 1.0),
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> NpuFlint:
        """(``sequences``: ``NpuFlint``'s)"""
        return NpuFlint(self.gguf, capacity=capacity, kv_k=self.kv_k, prefill=prefill, weights=self.weights,
                        dflash=self.dflash, sequences=sequences)


__all__ = ["Flint", "NpuFlint"]
