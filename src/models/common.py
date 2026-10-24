"""What the models share: the host side of a model file (``Model``: tokenizer, chat template, default sampling), the
generation loop of a model on the NPU (``NpuModel``), a model on a decode overlay with a chunk prefill
(``OverlayModel``: its steps, prompts and accounting), sampling, chat templates, incremental detokenization and the
default KV cache capacity."""

import collections
import copy
import functools
import math
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeVar

import jinja2.sandbox
import numpy as np
import torch

R = TypeVar("R")
MEDIA_MARKER = "<__media__>"  # where an image goes in a prompt's text (llama.cpp's mtmd marker)


@dataclass(frozen=True)
class Sampling:
    """How the next token is chosen: the logits penalized as llama.cpp's penalties sampler does (``Penalties``: the
    tokens among the last ``repeat_last_n`` of the prompt and the generated ones), among the ``top_k`` largest of them,
    softmax at temperature ``temp``, then the smallest prefix of them reaching probability ``top_p``, of those the ones
    with at least ``min_p`` times the largest probability; the argmax when ``greedy`` or ``temp`` <= 0."""

    temp: float = 1.0
    top_k: int = 64
    top_p: float = 1.0
    greedy: bool = False
    min_p: float = 0.0
    repeat_penalty: float = 1.0  # a recent token's logit divided by it (multiplied when <= 0); 1: none
    repeat_last_n: int = 64  # the recent tokens (0: no penalties)
    presence_penalty: float = 0.0  # subtracted from a recent token's logit
    frequency_penalty: float = 0.0  # subtracted from a recent token's logit per occurrence

    def __post_init__(self) -> None:
        if not 0 < self.repeat_penalty < math.inf:
            raise ValueError(f"repeat_penalty {self.repeat_penalty}: greater than 0 (1: none)")
        if self.repeat_last_n < 0:
            raise ValueError(f"repeat_last_n {self.repeat_last_n}: at least 0 (0: no penalties)")

    @property
    def k(self) -> int:
        """The candidates a choice needs (1: the argmax)."""
        return 1 if self.greedy or self.temp <= 0 else self.top_k

    @property
    def penalized(self) -> bool:
        """Whether the penalties change logits."""
        return self.repeat_last_n > 0 and (self.repeat_penalty != 1.0 or self.presence_penalty != 0.0
                                           or self.frequency_penalty != 0.0)

    def pick(self, values: torch.Tensor, ids: torch.Tensor) -> int:
        """The next token from the ``k`` largest logits (values descending) and their token ids."""
        if self.k == 1:
            return int(ids[0])
        p = torch.softmax(values.float() / self.temp, -1)
        if self.top_p < 1.0 or self.min_p > 0.0:
            keep = (torch.cumsum(p, -1) - p < self.top_p) & (p >= self.min_p * p[0])  # (top_p: the smallest prefix)
            keep[0] = True  # (the most likely token stays)
            p = torch.where(keep, p, torch.zeros_like(p))
        return int(ids[torch.multinomial(p / p.sum(), 1)])


def topk_bf16(x: torch.Tensor, k: int, exclude: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The k largest of bf16 values x (a vocabulary's logits) as fp32, descending, and their indices (as torch.topk of
    x.float(), 3-10x faster): the values' bits as int16 with the negative ones' magnitude bits flipped compare as the
    values do; k = 1: their argmax, else the k blocks of 256 with the largest maxima hold the k largest. ``exclude``:
    indices that rank last."""
    b = x.view(torch.int16).numpy()
    key = b ^ ((b >> 15) & 0x7FFF)
    if exclude is not None:
        key[exclude.numpy()] = np.iinfo(np.int16).min
    if k == 1:
        i = int(np.argmax(key))
        return x[i:i + 1].float(), torch.tensor([i])
    n = key.size // 256
    blocks = key[:256 * n].reshape(n, 256).max(1)
    top = np.argpartition(blocks, -k)[-k:] if n > k else np.arange(n)
    idx = np.concatenate([(256 * top[:, None] + np.arange(256)).reshape(-1), np.arange(256 * n, key.size)])
    sel = key[idx]
    j = np.argpartition(sel, -k)[-k:]
    i = torch.from_numpy(idx[j[np.argsort(-sel[j].astype(np.int32), kind="stable")]])
    return x[i].float(), i


class Penalties:
    """llama.cpp's penalties sampler over a generation: the tokens among the last ``repeat_last_n`` of the prompt
    (``prompt``: its ids; an image's rows, negative, are none) and of the generated ones (``accept``) have their logits
    divided by ``repeat_penalty`` (multiplied when <= 0), less ``presence_penalty`` and ``frequency_penalty`` times
    their count. Nothing when the sampling has no penalties (``Sampling.penalized``)."""

    def __init__(self, sampling: Sampling, prompt: Sequence[int] = ()):
        self.sampling = sampling
        self.n = sampling.repeat_last_n if sampling.penalized else 0
        self.last: collections.deque[int] = collections.deque()
        self.counts: collections.Counter[int] = collections.Counter()
        tail: list[int] = []
        for t in reversed(getattr(prompt, "ids", prompt)):
            if len(tail) == self.n:
                break
            if t >= 0:
                tail.append(int(t))
        for t in reversed(tail):
            self.accept(t)

    @property
    def active(self) -> bool:
        return self.n > 0

    def copy(self) -> "Penalties":
        """Another window of the same tokens (a draft's)."""
        out = copy.copy(self)
        out.last, out.counts = self.last.copy(), self.counts.copy()
        return out

    def accept(self, token: int) -> None:
        """The sequence's next token."""
        if not self.n:
            return
        self.last.append(token)
        self.counts[token] += 1
        if len(self.last) > self.n:
            old = self.last.popleft()
            self.counts[old] -= 1
            if not self.counts[old]:
                del self.counts[old]

    def top(self, logits: torch.Tensor, k: int, cap: Callable[[torch.Tensor], torch.Tensor] | None = None, *,
            rows: torch.Tensor | None = None, where: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` largest penalized logits (fp32, descending) and their token ids, of a token's logits (vocab,), fp32
        or the overlay's bf16; ``cap``: the model's logits of those (monotonic: applied to the candidates alone);
        ``rows``, ``where``: the token of each logit and the logit of each token (logits in another order: a drafter's).
        The candidates: the recent tokens and the ``k`` largest logits of the others."""
        at = None
        if self.counts:
            ids = torch.tensor(list(self.counts))
            at = ids if where is None else where[ids]
        if logits.dtype == torch.bfloat16:
            v, i = topk_bf16(logits, k, exclude=at)
        else:
            x = logits.float()
            if at is not None:
                x = x.clone()
                x[at] = -math.inf
            v, i = torch.topk(x, k)
        if cap is not None:
            v = cap(v)
        if rows is not None:
            i = rows[i]
        if at is None:
            return v, i
        if k > logits.numel() - len(ids):  # (fewer other tokens than k: recent ones among them, unpenalized)
            keep = ~torch.isin(i, ids)
            v, i = v[keep], i[keep]
        x = logits[at].float()
        x = (x if cap is None else cap(x)).numpy()
        s, count = self.sampling, np.fromiter(self.counts.values(), np.float32, len(ids))
        x = np.where(x <= 0, x * s.repeat_penalty, x / s.repeat_penalty) - s.presence_penalty \
            - count * s.frequency_penalty
        v, i = torch.cat([v, torch.from_numpy(x.astype(np.float32))]), torch.cat([i, ids.to(i.dtype)])
        v, j = torch.topk(v, min(k, v.numel()))
        return v, i[j]


def auto_capacity(tokens: int) -> int:
    """KV cache positions for a session of ``tokens`` (prompt and generated): a power of two from 4096 (one decode
    design per size), with room for a prefill's padded last chunk."""
    need, cap = max(tokens, 1) + 256, 4096
    while cap < need:
        cap *= 2
    return cap


def _raise(message: str) -> None:
    raise jinja2.TemplateError(message)


def chat_template(template: str, messages: list[dict[str, object]], **variables: object) -> str:
    """The prompt text of a conversation rendered with a model's chat template (a GGUF file's
    ``tokenizer.chat_template``, Jinja2 in a sandbox) up to the assistant's turn; ``variables``: the template's own
    (``enable_thinking``, ``bos_token``, ``date_string``, ...)."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().astimezone().strftime(fmt)
    return env.from_string(template).render(messages=messages, add_generation_prompt=True, tools=None, **variables)


class Detokenizer(Protocol):
    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes: ...


class StreamDecoder:
    """Incremental detokenizer: ``feed`` returns the text completed by a token (a multi-byte character split over
    byte tokens appears once its last byte arrives)."""

    def __init__(self, tokenizer: Detokenizer, *, special: bool = False):
        self.tokenizer = tokenizer
        self.special = special
        self._pending = b""

    def feed(self, token: int) -> str:
        data = self._pending + self.tokenizer.decode_bytes([token], special=self.special)
        try:
            text = data.decode("utf-8")
            self._pending = b""
        except UnicodeDecodeError as e:
            text = data[:e.start].decode("utf-8", errors="replace")
            self._pending = data[e.start:]
            if len(self._pending) > 4:  # not an incomplete character: invalid bytes
                text += self._pending.decode("utf-8", errors="replace")
                self._pending = b""
        return text


class NpuModel:
    """A model on the NPU. ``forward`` runs a prompt after the cached positions and returns its last token's logits,
    ``step`` / ``top_k`` run one token; ``generate`` samples from them. ``t_npu``: seconds in dispatches, ``t_host``:
    host work on their inputs, ``dispatches``: their count."""

    capacity: int  # KV cache positions
    position: int  # cached positions
    eos_token_ids: tuple[int, ...] | set[int]
    t_npu = t_host = 0.0
    dispatches = 0

    def prepare(self, prompt_len: int) -> None:
        """Compile and open what a prompt of ``prompt_len`` tokens needs (before it is timed)."""

    def describe(self) -> str:
        """What the model holds in memory (weights, caches)."""
        raise NotImplementedError

    def stats(self) -> str:
        """More of its generation's statistics for the report (e.g. draft tokens), or ""."""
        return ""

    def reset(self) -> None:
        """Forget the cached positions: a new sequence."""
        raise NotImplementedError

    def rewind(self, n: int) -> int:
        """Keep at most the first ``n`` cached positions (a prompt cache: a new prompt continues from the prefix it
        shares with the cached tokens) and return how many it kept; this default keeps none (``reset``)."""
        self.reset()
        return 0

    def forward(self, tokens: list[int], *, progress: Callable[[int], None] | None = None) -> torch.Tensor:
        """Run tokens after the cached positions: the logits (vocab,) fp32 of the last. ``progress``: called with the
        number of tokens done as they run (an exception it raises stops the prompt between dispatches, the positions
        run so far cached)."""
        raise NotImplementedError

    def step(self, token: int) -> torch.Tensor:
        """Run ``token`` at the next position: the logits (vocab,) of the next token."""
        raise NotImplementedError

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``token`` at the next position: the ``k`` largest logits of the next token (fp32, descending) and their
        token ids (``penalties``: of the penalized logits)."""
        logits = self.step(token).float()
        return torch.topk(logits, k) if penalties is None else penalties.top(logits, k)

    def generate(self, logits: torch.Tensor, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = ()
                 ) -> Iterator[int]:
        """Sample up to ``max_tokens`` tokens (< 0: no limit) after the cached positions; ``logits``: of the last
        cached one; ``prompt``: the prompt's tokens (the penalties' first). Ends after an end-of-sequence token or when
        the cache is full."""
        if max_tokens == 0:
            return
        penalties = Penalties(sampling, prompt)
        token = sampling.pick(*penalties.top(logits, sampling.k))
        penalties.accept(token)
        n = 1
        yield token
        while n != max_tokens and token not in self.eos_token_ids and self.position < self.capacity:
            token = sampling.pick(*self.top_k(token, sampling.k, penalties))
            penalties.accept(token)
            n += 1
            yield token


class OverlayModel(NpuModel):
    """A model on its decode overlay (``overlay``: an ``artlite.operators.overlay.model.ModelOverlay``) with a chunk
    prefill (``prefill``: an ``artlite.operators.prefill.chunks.ChunkPrefill``, or None: prompts as decode steps).

    ``steps`` runs up to ``overlay.SLOTS`` tokens per dispatch; ``prompt`` runs a prompt's tokens in prefill chunks
    where one can start (a whole cache group: the prefill's ``plan``), as decode steps before them and after, each
    chunk's inputs prepared on a worker thread while the previous chunk runs; ``forward`` is a prompt with its last
    token's logits (a decode step's, or of a chunk's last token: ``_chunk_logits``). A model supplies a step's inputs
    (``_step_inputs``: the overlay step's arguments before the position), a chunk's (``_chunk_inputs``: the prefill
    run's), the logits of a chunk's last token and its final logits of the overlay's (``cap``: monotonic)."""

    overlay: Any
    prefill: Any
    slot = 0  # the slot of the last token whose logits a dispatch returned (a step's last; 0 after a chunk's)

    def cap(self, logits: torch.Tensor) -> torch.Tensor:
        """The model's logits of the overlay's (fp32): a soft cap, or none."""
        return logits

    @functools.cached_property
    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(1)  # (a prompt chunk's inputs while the previous chunk runs)

    def prepare(self, prompt_len: int) -> None:
        """(the overlay's and the prefill's hardware contexts, the prefill's piece BOs and its chunks' items: else the
        first prompt of a process pays them, ~50 ms, most of it the first context waking the NPU)"""
        self.overlay.dev.open()
        if self.prefill is not None:
            self.prefill.prepare(self.position, prompt_len)

    def _step_inputs(self, tokens: list[int]) -> tuple:
        """The overlay step's arguments before the position for the tokens (their embeddings, ...)."""
        raise NotImplementedError

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        """The prefill run's arguments before the position for a chunk's tokens (``first``: the prompt's first chunk,
        whose inputs nothing prepared ahead)."""
        raise NotImplementedError

    def _chunk_logits(self, token: int, n: int) -> torch.Tensor:
        """The overlay's logits (vocab,) of the last chunk's last token (``token``, the chunk's n-th; positions up to
        it cached), in the dispatches of ``_dispatch``."""
        raise NotImplementedError

    def _dispatch(self, run: Callable[[], R]) -> R:
        """A dispatch (``run``), timed and counted."""
        t0 = time.perf_counter()
        out = run()
        self.t_npu += time.perf_counter() - t0
        self.dispatches += 1
        return out

    def steps(self, tokens: list[int], *, logits: bool = True, **step: Any) -> torch.Tensor | None:
        """Run up to ``overlay.SLOTS`` tokens at the next positions in one dispatch: the overlay's logits (n, vocab)
        bf16 after each of them (a view of the NPU's buffer, valid until the next dispatch), or None (``logits=False``:
        prompt tokens); ``step``: the overlay step's other keywords."""
        if self.position + len(tokens) > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        t0 = time.perf_counter()
        args = self._step_inputs(tokens)
        self.t_host += time.perf_counter() - t0
        out = self._dispatch(lambda: self.overlay.step(*args, self.position, logits=logits, **step))
        self.position += len(tokens)
        if logits:
            self.slot = len(tokens) - 1
        return out

    def step(self, token: int, *, logits: bool = True) -> torch.Tensor | None:
        """Run ``token`` at the next position: the logits (vocab,) fp32 of the next token, or None (``logits=False``)."""
        out = self.steps([token], logits=logits)
        return None if out is None else self.cap(out[0].float())

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(``cap`` is monotonic: applied to the candidates alone)"""
        raw = self.steps([token])[0]
        if penalties is not None:
            return penalties.top(raw, k, self.cap)
        v, i = topk_bf16(raw, k)
        return self.cap(v), i

    def stale(self) -> None:
        """Another hardware context ran on the device (a prefill chunk, an image encoder): the overlay's state on the
        array comes back before its next dispatch (``OverlayDevice.set_prologue``)."""
        self.overlay.dev.stale = True

    def _prefill_run(self, args: tuple, T: int, n: int) -> None:
        """A prompt chunk of n tokens at the next position through the prefill's chunk of T (its inputs ``args``)."""
        self.prefill.run(*args, self.position, T)
        self.stale()

    def _chunk(self, left: int, position: int | None = None) -> int:
        """The size of the next prefill chunk at ``position`` (default: the next) when ``left`` tokens remain (0: the
        next token runs as a decode step)."""
        p = self.position if position is None else position
        plan = self.prefill.plan(p, left) if self.prefill is not None else []
        return plan[0] if plan else 0

    def rewind(self, n: int) -> int:
        """(down to a whole cache group of the prefill, or of 8 positions without one: a step at a multiple of 8 opens
        a new V record, so the ring holds what it reads; a prompt from a group runs in chunks at once, while one inside
        a group runs as decode steps up to the next)"""
        keep = min(n, self.position)
        self.position = keep - keep % (self.prefill.group if self.prefill is not None else 8)
        return self.position

    def prompt(self, tokens: list[int], *, logits: bool = False, progress: Callable[[int], None] | None = None
               ) -> torch.Tensor | None:
        """Run tokens after the cached positions (prefill chunks, else decode steps); ``logits``: return the last
        token's (vocab,) fp32; ``progress``: as ``forward``'s, after each chunk and each group of steps."""
        if self.position + len(tokens) > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        out, i, nxt = None, 0, None
        while i < len(tokens):
            T = self._chunk(len(tokens) - i)
            if not T:  # (decode steps up to the next position where a chunk can start)
                n = 1
                while n < min(self.overlay.SLOTS, len(tokens) - i) and \
                        not self._chunk(len(tokens) - i - n, self.position + n):
                    n += 1
                last = logits and i + n == len(tokens)
                o = self.steps(tokens[i:i + n], logits=last)
                out = self.cap(o[-1].float()) if last else None
                i += n
                if progress is not None:
                    progress(i)
                continue
            n = min(T, len(tokens) - i)
            t0 = time.perf_counter()
            args = nxt.result() if nxt is not None else self._chunk_inputs(tokens[i:i + n], first=True)
            t1 = time.perf_counter()
            m = self._chunk(len(tokens) - i - n, self.position + n)  # (the next chunk: its inputs meanwhile)
            nxt = self._pool.submit(lambda t: self._chunk_inputs(t, first=False),
                                    tokens[i + n:i + n + m]) if m else None
            self._prefill_run(args, T, n)
            self.t_host += t1 - t0
            self.t_npu += time.perf_counter() - t1
            self.dispatches += self.prefill.dispatches
            if logits and i + n == len(tokens):
                out = self.cap(self._chunk_logits(tokens[-1], n).float())
                self.slot = 0
            self.position += n
            i += n
            if progress is not None:
                progress(i)
        return out

    def forward(self, tokens: list[int], *, progress: Callable[[int], None] | None = None) -> torch.Tensor:
        """Run the tokens after the cached positions: the logits (vocab,) of the last."""
        return self.prompt(tokens, logits=True, progress=progress)

    def forced_logits(self, tokens: list[int], first: int = 0, *, batch: int = 1) -> torch.Tensor:
        """Teacher-forced logits (len(tokens) - first, vocab) fp32 of tokens[first:], from position 0, ``batch`` tokens
        per dispatch (quality tooling)."""
        self.reset()
        self.prompt(tokens[:first])
        return torch.cat([self.cap(self.steps(tokens[i:i + batch]).float()) for i in range(first, len(tokens), batch)])


class Model:
    """A model file opened on the host: its tokenizer, chat template and default sampling; ``load`` puts it on the NPU.
    One subclass per architecture (``models.ARCHITECTURES``); constructor keywords are its options."""

    name: str  # for messages: "Llama 3", ...
    vision: Any = None  # the image encoder (an option of the model's: its mmproj file), or None: text only
    projectors: tuple[str, ...] = ()  # the projector types (``clip.projector_type``) of the mmproj files it takes
    thinks = False  # the chat template has a thinking mode
    reasoning: tuple[str, str] | None = None  # the text that opens and closes a thought (the model writes them)
    tokenizer: Detokenizer
    eos_token_ids: tuple[int, ...] | set[int]

    def encode(self, text: str) -> list[int]:
        """Token ids of a raw text (no chat template), after the model's BOS if it has one."""
        raise NotImplementedError

    def chat(self, messages: list[dict[str, object]], *, think: bool = False) -> list[int]:
        """Token ids of a conversation through the chat template, up to the assistant's turn (``think``: let it
        think first; otherwise an empty thought where the template has one). A message's content may be a list of
        parts: ``{"type": "text", "text": ...}``, ``{"type": "image", "image": a file, its bytes or a PIL image}``
        (a model with ``vision``: its prompt then a sequence of ids and embedding rows; the encoder runs on the NPU)."""
        raise NotImplementedError

    def sampling(self, *, think: bool = False) -> Sampling:
        """The model's default sampling (in the thinking mode)."""
        raise NotImplementedError

    def load(self, *, capacity: int, prefill: bool = True) -> NpuModel:
        """The model on the NPU with a KV cache of ``capacity`` positions; ``prefill``: prompts through the prefill
        (else as decode steps). NotImplementedError: a file whose shapes or formats the NPU path does not handle."""
        raise NotImplementedError

    def prompt(self, text: str, *, system: str | None = None, raw: bool = False, think: bool = False,
               images: list[object] = ()) -> list[int]:
        """Token ids of a user message after an optional system message, through the chat template (``raw``: the text
        alone); ``images`` (files or their bytes) where the text has ``MEDIA_MARKER``, else before it."""
        if images and self.vision is None:
            raise ValueError(f"{self.name}: images need a vision encoder (an mmproj file)")
        if raw:
            if images:
                raise ValueError("images go through the chat template (not a raw prompt)")
            return self.encode(text)
        if think and not self.thinks:
            raise ValueError(f"{self.name} has no thinking mode")
        content: str | list[dict[str, object]] = text
        if images:
            parts = text.split(MEDIA_MARKER) if MEDIA_MARKER in text else ["", text]
            if len(parts) != len(images) + 1:
                raise ValueError(f"{len(images)} image(s) for {len(parts) - 1} {MEDIA_MARKER} marker(s) in the prompt")
            content = [{"type": "text", "text": parts[0]}] if parts[0] else []
            for im, t in zip(images, parts[1:], strict=True):
                content += [{"type": "image", "image": im}] + ([{"type": "text", "text": t}] if t else [])
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.chat(messages, think=think)
