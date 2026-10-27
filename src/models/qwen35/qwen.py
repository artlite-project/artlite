"""Qwen3.5 / Qwen3.6 (GGUF architectures ``qwen35``: Gated DeltaNet and full-attention layers, and ``qwen35moe``: the
same with a mixture of experts as the FFN) on the NPU.

A dispatch runs up to 4 tokens through every layer and the LM head on one stream of the weights
(``models.qwen35.overlay.QwenOverlay``): the projections on the main cores and their helpers (the MMUL
against exact-integer weight panels), the Gated DeltaNet steps (conv states and fp32 states in DDR) on the GDN cores,
the full attention and its KV cache on the attention cores. The host reads the tokens' embedding rows from the mapped
file and samples from the logits.

A prompt is token ids or a ``sequence.Sequence`` (token ids and embedding rows with M-RoPE positions: MiniCPM-V's
images): each cache slot's position (``QwenOverlay.rope``) comes from it, the text after it from its ``next`` on.

A prompt runs in chunks of up to 2048 tokens through the prefill (``models.qwen35.prefill.QwenPrefill``:
layer-major, control code assembled per chunk from one library ELF; the GEMMs read the same weight image, the chunk's K /
V, GDN states and conv states land in the decode's buffers), each chunk in pieces of a few layers (dispatches of one HW
context); its last token's logits come from the LM head alone (``QwenOverlay.logits_of``). Tokens where a chunk cannot
start (a position inside a 32-position cache group, after a generated reply), and a prompt's last 4 or fewer, run as
decode steps of up to 4 tokens. An MoE model's chunk routes its tokens on the host between the router and the experts
(``artlite.moe.prefill``).

Multi-token steps (``OverlayModel.generate``): the sampled token's step takes up to three draft tokens along - the
context's (``models.drafts``), or with a qwen35moe model's MTP drafter (``model_config.qwen_mtp_of``:
``models.qwen35.draft.QwenDraft``) its guesses of what follows, a dispatch of the same design each (2.4-3.4 ms on the
35B-A3B), while their chain's probability stays at least ``P_MIN`` - and a draft is kept where the token sampled from
the model's logits at its position is that draft. A step of drafts leaves the GDN states uncommitted: the next one
replays the accepted tokens.

Memory: the weights the NPU streams (the layers and the LM head: 15.95 GB of the 27B's 16.5 GB file) are
re-laid-out into a device buffer at load, at their quantization's size; nothing is written to disk. Load (27B, the
file in the page cache: 1.1 s): the buffer allocated and the image packed (C++) in a thread while the tokenizer,
the designs (found through the render manifest) and the overlay's other buffers load.
"""

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from artlite.decode.design import SLOTS
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_mtp_of
from artlite.overlay import WeightBuffer

from ..common import Model, OverlayModel, Penalties, Sampling, max_prob
from ..minicpmv.host import image_prompt
from ..minicpmv.npu import PROJECTORS, NpuVision
from ..sequence import Sequence
from .draft import QwenDraft, draft_formats
from .image import QwenImage
from .overlay import QwenOverlay
from .prefill import QwenPrefill
from .tokenizer import QwenTokenizer, chat_prompt

# the model card's sampling (thinking / instruct) when the file names none (general.sampling.*); MiniCPM-V 4.7's (one
# set): the language model of openbmb's files, marked by the M-RoPE order of its vision
# (``QwenConfig.rope_section_order`` "zyxt": llama.cpp's mark of MiniCPM-V 4.7; its general.name is generic)
SAMPLING = {True: {"temp": 1.0, "top_k": 20, "top_p": 0.95}, False: {"temp": 0.7, "top_k": 20, "top_p": 0.8}}
MINICPMV_SAMPLING = {"temp": 0.7, "top_k": 100, "top_p": 0.95, "penalty_repeat": 1.03}


class NpuQwen(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the full-attention layers' caches hold; ``prefill``: prompts
    in chunks through the prefill ELFs (else as decode steps), of the sizes ``sizes`` (default ``QwenPrefill``'s);
    ``drafter``: a qwen35moe model's multi-token-prediction drafter file ("": ``model_config.qwen_mtp_of``, when there
    is one; None: none)."""

    P_MIN = 0.5  # a round's drafts stop where the probability of their chain (the drafter's) falls below it
    recurrent = True  # (the GDN layers' states hold the whole sequence)

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, threads: int = 0, prefill: bool = True,
                 sizes: tuple[int, ...] | None = None, weights: WeightBuffer | None = None, drafter: str | None = ""):
        """weights: the weight image's buffer, allocated ahead (``WeightBuffer``)."""
        self.gguf = gguf
        cfg = QwenConfig.from_gguf(gguf)
        path = qwen_mtp_of(gguf.path) if drafter == "" and cfg.moe else drafter
        mtp = GGUFFile(path) if path else None
        self.overlay = QwenOverlay(gguf, max_positions=capacity, weights=weights,
                                   formats=draft_formats(mtp) if mtp is not None else ())
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.drafter = QwenDraft(self.overlay, mtp) if mtp is not None else None  # (OverlayModel's)
        if cfg.moe:
            self.STEP_COST = (1.0, 1.31, 1.60, 1.94)  # (35B-A3B at 1K: the slots route to different experts)
        self.prefill = QwenPrefill(self.overlay, sizes=sizes, draft=self.drafter) if prefill else None
        with ThreadPoolExecutor(1) as pool:
            build = pool.submit(self.prefill.compile) if prefill else None  # (while the weights are packed)
            self.overlay.load(threads=threads)
            if self.drafter is not None:
                self.drafter.load(threads=threads)
                mtp.release()
            if build is not None:
                build.result()  # (its error, if any)
                self.prefill.bind()
        self.overlay.ready()  # (the weights' thread packs meanwhile)
        gguf.release()
        self._w_out = torch.from_numpy(self.overlay._f32("output_norm.weight"))
        self.position = 0
        self.rope_next = 0  # (the M-RoPE position of the next text token)
        self._prompting = False
        self._records()

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def prepare(self, prompt_len: int) -> None:
        """(also, with drafts: the rounds' step transactions, the model's of 1 .. DRAFTS + 1 tokens with logits and the
        drafter's: 1.3 s on the 35B-A3B; else, and without ``prepare``, each is built at its first use)"""
        super().prepare(prompt_len)
        for S in range(1, self.DRAFTS + 2) if self.lookup or self._drafting else ():
            self.overlay.prepare(S)
            for logits in (True, False) if self._drafting else ():
                self.drafter.prepare(S, logits=logits)

    def describe(self) -> str:
        ov = self.overlay
        d = f", drafter {os.path.basename(self.drafter.gguf.path)}" if self.drafter is not None else ""
        return (f"{ov.image.nbytes / 1e9:.2f} GB of weights, {ov.sizes['KV'] / 1e6:.0f} MB of KV cache "
                f"({self.capacity} positions), {ov.sizes['GSTATE'] / 1e6:.0f} MB of GDN states{d}")

    def reset(self) -> None:
        """A new sequence: no cached positions, the GDN states zeroed."""
        self.overlay.reset()
        if self.drafter is not None:
            self.drafter.reset()
        self.position = self.rope_next = 0
        self._records()

    def stale(self) -> None:
        super().stale()
        if self.drafter is not None:
            self.drafter.dev.stale = True

    def rewind(self, n: int) -> int:
        """(all cached positions or none: the GDN layers' states hold the whole sequence and cannot go back; a
        ``checkpoint`` keeps a position)"""
        if n >= self.position:
            return self.position
        self.reset()
        return 0

    def _overlays(self) -> list[QwenOverlay]:
        return [self.overlay] + ([self.drafter] if self.drafter is not None else [])

    def _state(self) -> dict[str, object]:
        """(the text's next M-RoPE position; with the drafter its rows and the prefill's carry)"""
        out: dict[str, object] = {"rope_next": self.rope_next}
        if self.drafter is not None:
            out |= {"pend": list(self._pend), "dpos": self._dpos, "h": self._h, "carry_at": self._carry_at}
            if self.prefill is not None:
                out["carry"] = self.prefill.dp.carry.copy()
        return out

    def _set_state(self, state: dict[str, object]) -> None:
        self.rope_next = state["rope_next"]
        if self.drafter is not None:
            self._pend, self._dpos, self._h, self._carry_at = list(state["pend"]), state["dpos"], state["h"], \
                state["carry_at"]
            if self.prefill is not None:
                self.prefill.dp.carry[:] = state["carry"]

    def _rows(self, tokens: list[int] | Sequence) -> torch.Tensor:
        """The tokens' embeddings (n, hidden) fp32: their ids' rows, a sequence's given rows."""
        if not isinstance(tokens, Sequence):
            return self.gguf.rows("token_embd.weight", tokens)
        x = torch.empty(len(tokens), self.cfg.hidden_size)
        t = ~tokens.given
        if t.any():
            x[torch.from_numpy(t)] = self.gguf.rows("token_embd.weight", tokens.ids[t])
        if tokens.given.any():
            x[torch.from_numpy(tokens.given)] = torch.from_numpy(tokens.rows)
        return x

    def _chunk_inputs(self, tokens: list[int] | Sequence, *, first: bool) -> tuple:
        return (self._rows(tokens),)

    def prompt(self, tokens: list[int] | Sequence, *, logits: bool = False,
               progress: Callable[[int], None] | None = None) -> torch.Tensor | None:
        """(a sequence's positions from ``rope_next``)"""
        p, n = self.position, len(tokens)
        if p + n > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        seq = tokens if isinstance(tokens, Sequence) else Sequence.text(tokens)
        self.overlay.rope[p:p + n] = self.rope_next + seq.pos
        self._prompting = True
        try:
            return super().prompt(tokens, logits=logits, progress=progress)
        finally:
            self._prompting = False
            done = self.position - p  # (a prompt that ``progress`` stopped: the text after its last cached token)
            self.rope_next = (self.rope_next + seq.next if done == n else
                              int(self.overlay.rope[self.position - 1].max()) + 1 if done else self.rope_next)

    def steps(self, tokens: list[int], *, logits: bool = True, commit: bool = True) -> torch.Tensor | None:
        """(tokens after a prompt: text from ``rope_next``; ``commit``: ``QwenOverlay.step``'s, else the caller
        ``accept``s)"""
        p, n = self.position, len(tokens)
        if self._prompting:
            out = super().steps(tokens, logits=logits)
        else:
            if p + n <= self.capacity:
                self.overlay.rope[p:p + n] = self.rope_next + np.arange(n)[:, None]
            out = super().steps(tokens, logits=logits, commit=commit)
            self.rope_next += n
        if commit and self._drafting:
            self._follow(n)
        return out

    def _chunk(self, left: int, position: int | None = None) -> int:
        """(after a generation's step: a decode step first, which replays its accepted tokens into the GDN states)"""
        if self.overlay.nrep and position in (None, self.position):
            return 0
        return super()._chunk(left, position)

    def _chunk_logits(self, token: int, n: int) -> torch.Tensor:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))

    # ------------------------------------------------------------------
    # Multi-token prediction
    # ------------------------------------------------------------------
    #
    # The drafter's row at position q takes token q and the model's output-normed hidden state of q - 1, writes its
    # cache record at q and guesses token q + 1. Its records follow the model's positions: a prompt chunk writes its
    # rows' (``QwenPrefill(draft=)``); after the model's steps their rows wait in ``_pend`` (records from ``_dpos`` on,
    # at most SLOTS - 1 of them) until a drafter step runs them with the next ones; ``_h`` is the model's hidden state
    # of the last cached position. A drafter's round (``OverlayModel.generate``): one drafter step of the pending rows
    # and the sampled token's (its first draft), one-row steps for the next drafts, the model's step of the token and
    # the drafts (``commit=False``), then the accepted tokens' rows pending (they rewrite the draft rows' records); a
    # round of the context's drafts leaves the token's row pending with the accepted ones'.

    def _records(self) -> None:
        """No drafter rows pending, nothing cached."""
        self._pend: list[tuple[torch.Tensor, torch.Tensor]] = []  # (embedding, h) per row
        self._dpos = 0
        self._h = torch.zeros(self.cfg.hidden_size)  # (zeros before position 0)
        self._carry_at = 0  # (the position after the last chunk: the prefill's carry is its)

    def _hidden(self, n: int) -> torch.Tensor:
        """The model's output-normed hidden states (n, hidden) fp32 of the last step's first n slots."""
        x = torch.stack([self.overlay.hidden(-1, s) for s in range(n)])
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.cfg.rms_norm_eps) * self._w_out

    def _step_inputs(self, tokens: list[int] | Sequence) -> tuple:
        self._x = self._rows(tokens)
        return (self._x,)

    def _follow(self, n: int) -> None:
        """After the model's step: the drafter rows of its first n tokens (cached) pending."""
        hs = self._hidden(n)
        h = torch.cat([self._h[None], hs[:-1]])
        self._pend += list(zip(self._x[:n], h, strict=True))
        self._h = hs[-1]
        while len(self._pend) >= SLOTS:
            self._run_pending(SLOTS)

    def _draft_step(self, rows: list[tuple[torch.Tensor, torch.Tensor]], logits: bool) -> torch.Tensor | None:
        """A drafter step of rows at ``_dpos`` on (their records; the drafter's logits (n, vocab) bf16, or None)."""
        p, n = self._dpos, len(rows)
        q = np.arange(p, p + n)
        old = q < self.position  # (the model's positions; text from ``rope_next`` after them)
        rope = self.drafter.rope
        rope[q[old]] = self.overlay.rope[q[old]]
        rope[q[~old]] = self.rope_next + (q[~old] - self.position)[:, None]
        emb, h = (torch.stack(v) for v in zip(*rows, strict=True))
        out = self.drafter.step(emb, h, p, logits=logits)
        self._dpos += n
        return out

    def _run_pending(self, n: int) -> None:
        """Records of the first n pending rows."""
        rows, self._pend = self._pend[:n], self._pend[n:]
        self._draft_step(rows, logits=False)

    def _prefill_run(self, args: tuple, T: int, n: int) -> None:
        """(the drafter's records of the rows before the chunk first; the chunk's carry after decode steps)"""
        if self._drafting:
            while self._pend:
                self._run_pending(SLOTS)
            if self.position and self._carry_at != self.position:
                self.prefill.dp.carry_of(self._h)
        super()._prefill_run(args, T, n)
        if self._drafting:
            x = self.prefill.hidden(n - 1)
            self._h = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.cfg.rms_norm_eps) * self._w_out)[0]
            self._dpos = self._carry_at = self.position + n

    def draft(self, token: int, n: int) -> list[int]:
        """The drafter's guesses of up to n tokens after ``token`` (the token of the next position, not run yet; its
        row runs with the pending ones): the most likely one, the most likely one after it, ... while their chain's
        probability stays at least ``P_MIN`` (ends with an end-of-sequence token)."""
        t0 = time.perf_counter()
        rows = [*self._pend, (self._rows([token])[0], self._h)]
        assert self._dpos + len(rows) == self.position + 1 and len(rows) <= SLOTS
        self._pend = []
        raw = self._draft_step(rows, logits=n > 0)
        out: list[int] = []
        chain, slot = 1.0, len(rows) - 1
        while len(out) < n:
            i, p = max_prob(raw[slot])
            chain *= p
            if chain < self.P_MIN:
                break
            out.append(i)
            if out[-1] in self.eos_token_ids or len(out) == n:
                break
            h = self.drafter.hidden_state(slot)
            raw, slot = self._draft_step([(self._rows(out[-1:])[0], h)], logits=True), 0
        self.t_draft += time.perf_counter() - t0
        self.drafted += len(out)
        return out

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        return self.draft(token, k)

    def _verify(self, tokens: list[int], mtp: bool) -> torch.Tensor:
        """(a step of drafts, or a drafter's round, commits no GDN state: ``_kept`` accepts its kept tokens, which the
        next step replays, ``QwenOverlay.accept``)"""
        return self.steps(tokens, commit=len(tokens) == 1 and not mtp)

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool) -> None:
        """(a drafter's round: its kept drafts' rows pending, the token's ran with the drafts; another lazy step: its
        kept tokens' rows pending)"""
        if len(tokens) == 1 and not mtp:  # (a committed step: its row pending, ``steps``)
            return
        self.overlay.accept(m)
        self.position = p + m
        self.rope_next -= len(tokens) - m
        if mtp:
            hs = self._hidden(m)
            self._pend = [(self._rows([t])[0], hs[i]) for i, t in enumerate(tokens[1:m])]
            self._dpos, self._h = p + 1, hs[m - 1]
        elif self._drafting:
            self._follow(m)


class Qwen(Model):
    """A qwen35 model from a GGUF file (Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7's language model): the architecture,
    tokenizer and chat template from its metadata. Opening the file allocates the weight image's buffer and packs it in
    a thread (``WeightBuffer``) while the tokenizer and the designs load. ``mmproj``: MiniCPM-V's vision encoder file
    (``models.minicpmv.npu.NpuVision``: images in prompts; ``max_slices``: an image's slices at most, default the
    file's)."""

    name = "Qwen3.5"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<|im_start|>", "<|im_start|>user"
    projectors = PROJECTORS

    def __init__(self, path: str, *, mmproj: str | None = None, max_slices: int | None = None):
        self.gguf = GGUFFile(path)
        self.weights = WeightBuffer(QwenImage(self.gguf))
        self.tokenizer = QwenTokenizer(self.gguf)
        self.cfg = QwenConfig.from_gguf(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.minicpmv = self.cfg.rope_section_order == "zyxt"
        if self.minicpmv:
            self.name = "MiniCPM-V 4.7"
        self._npu: NpuQwen | None = None
        if mmproj is not None:
            if not self.minicpmv:
                raise ValueError(f"{path}: an mmproj file goes with MiniCPM-V 4.7's language model")
            self.vision = NpuVision(mmproj, max_slices=max_slices)
            if self.vision.clip.projection_dim != self.cfg.hidden_size:
                raise ValueError(f"{mmproj}: embeddings of {self.vision.clip.projection_dim}, the language model's "
                                 f"{self.cfg.hidden_size}")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False) -> list[int] | Sequence:
        text = chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools, preserve_thinking=preserve)
        images = [p["image"] for m in messages if isinstance(m.get("content"), list) for p in m["content"]
                  if isinstance(p, dict) and p.get("type") == "image"]
        if not images:
            return self.tokenizer.encode(text)
        if self.vision is None:
            raise ValueError(f"{self.name}: images need its vision encoder (an mmproj file)")
        seq = image_prompt(text, images, self.tokenizer.encode, self.vision.sequence)
        if self._npu is not None:  # (the encoder's hardware context ran: ``OverlayDevice.set_prologue``)
            self._npu.stale()
        return seq

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*), else the model card's."""
        s = (MINICPMV_SAMPLING if self.minicpmv else SAMPLING[think]) | self.cfg.sampling
        return Sampling(temp=s["temp"], top_k=int(s["top_k"]), top_p=s["top_p"],
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))

    def load(self, *, capacity: int, prefill: bool = True) -> NpuQwen:
        self._npu = NpuQwen(self.gguf, capacity=capacity, prefill=prefill, weights=self.weights)
        return self._npu
