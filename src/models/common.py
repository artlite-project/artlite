"""What the models share: the host side of a model file (``Model``: tokenizer, chat template, default sampling), the
generation loop of a model on the NPU (``NpuModel``; a sequence's state as host copies, ``Checkpoint`` and the KV
records), a model on a decode overlay with a chunk prefill (``OverlayModel``: its steps, prompts and accounting),
sampling, chat templates, incremental detokenization and the default KV cache capacity."""

import collections
import copy
import functools
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import jinja2.ext
import jinja2.sandbox
import numpy as np
import torch

from artlite import _native, backend, portable

from .drafts import EWMA, Lookup, Planner

if TYPE_CHECKING:
    from artlite.decode.model import Records

R = TypeVar("R")
MEDIA_MARKER = "<__media__>"  # where an image goes in a prompt's text (llama.cpp's mtmd marker)
LogitsLike = torch.Tensor | np.ndarray  # a vocabulary's logits: bf16 (an overlay's) or fp32
_rng = random.Random()  # (sampling's draws: ``seed``)


def seed(n: int | None = None) -> None:
    """Seed sampling's draws (None: from the operating system's randomness)."""
    _rng.seed(n)


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

    def pick(self, values: np.ndarray, ids: np.ndarray) -> int:
        """The next token from the ``k`` largest logits (values descending, fp32) and their token ids
        (``_native.pick``; the draw: ``seed``)."""
        if self.k == 1:
            return int(ids[0])
        return int(ids[_native.pick(values, self.temp, self.top_p, self.min_p, _rng.random())])


def top_logits(logits: LogitsLike, k: int, cap: float = 0.0, *, rows: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """The k largest of a vocabulary's logits after a soft cap (``cap`` > 0: cap tanh(l / cap)), fp32, descending
    (equal ones: the lower index first), and their token ids (int32; ``rows``: the token of each logit, else its index):
    one pass over the logits (``_native.top_logits``)."""
    return _native.top_logits(logits, k, cap, rows=rows)


def argmax(logits: LogitsLike) -> int:
    """The index of the largest logit (the first of equal ones)."""
    return int(_native.top_logits(logits, 1)[1][0])


def max_prob(logits: LogitsLike) -> tuple[int, float]:
    """The largest of bf16 logits (its index: the first of equal ones) and its probability (softmax)."""
    return _native.max_prob(logits)


class Mask(Protocol):
    """A constraint on a generation's tokens (a tool call's name: ``server.tools.NameMask``)."""

    def allowed(self) -> np.ndarray | None:
        """The tokens the next one may be (sorted ids), or None: any."""
        ...

    def accept(self, token: int) -> None:
        """The sequence's next token."""
        ...


class Penalties:
    """llama.cpp's penalties sampler over a generation: the tokens among the last ``repeat_last_n`` of the prompt
    (``prompt``: its ids; an image's rows, negative, are none) and of the generated ones (``accept``) have their logits
    divided by ``repeat_penalty`` (multiplied when <= 0), less ``presence_penalty`` and ``frequency_penalty`` times
    their count. Nothing when the sampling has no penalties (``Sampling.penalized``). ``mask``: the candidates are only
    the tokens it allows (it follows the accepted tokens too)."""

    def __init__(self, sampling: Sampling, prompt: Sequence[int] = (), mask: Mask | None = None):
        self.sampling = sampling
        self.n = sampling.repeat_last_n if sampling.penalized else 0
        self.last: collections.deque[int] = collections.deque()
        self.counts: collections.Counter[int] = collections.Counter()
        self.mask: Mask | None = None
        tail: list[int] = []
        for t in reversed(getattr(prompt, "ids", prompt)):
            if len(tail) == self.n:
                break
            if t >= 0:
                tail.append(int(t))
        for t in reversed(tail):
            self.accept(t)
        self.mask = mask

    @property
    def active(self) -> bool:
        return self.n > 0

    def copy(self) -> "Penalties":
        """Another window of the same tokens (a draft's: without the mask)."""
        out = copy.copy(self)
        out.last, out.counts, out.mask = self.last.copy(), self.counts.copy(), None
        return out

    def accept(self, token: int) -> None:
        """The sequence's next token."""
        if self.mask is not None:
            self.mask.accept(token)
        if not self.n:
            return
        self.last.append(token)
        self.counts[token] += 1
        if len(self.last) > self.n:
            old = self.last.popleft()
            self.counts[old] -= 1
            if not self.counts[old]:
                del self.counts[old]

    def top(self, logits: LogitsLike, k: int, cap: float = 0.0, *, rows: np.ndarray | None = None,
            where: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` largest penalized logits (fp32, descending) and their token ids, of a token's logits (vocab,), fp32
        or the overlay's bf16 (``top_logits``; ``cap``: the model's soft cap); ``rows``, ``where``: the token of each
        logit and the logit of each token (logits in another order: a drafter's). The candidates: the recent tokens
        and the ``k`` largest logits of the others; with the mask, the tokens it allows (logits in the vocabulary's
        order)."""
        allowed = self.mask.allowed() if self.mask is not None else None
        if allowed is not None:
            return self._top_of(logits, k, cap, allowed)
        if not self.counts:
            return top_logits(logits, k, cap, rows=rows)
        ids = np.fromiter(self.counts, np.int32, len(self.counts))
        s, count = self.sampling, np.fromiter(self.counts.values(), np.float32, len(ids))
        return _native.top_logits(logits, k, cap, ids if where is None else where[ids], count, s.repeat_penalty,
                                  s.presence_penalty, s.frequency_penalty, rows)

    def _top_of(self, logits: LogitsLike, k: int, cap: float, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` largest penalized logits of the ``allowed`` tokens (sorted ids) and their ids."""
        sub = logits[torch.from_numpy(allowed.astype(np.int64))] if isinstance(logits, torch.Tensor) else \
            logits[allowed]
        k = min(k, len(allowed))
        if not self.counts:
            return top_logits(sub, k, cap, rows=allowed)
        ids = np.fromiter(self.counts, np.int32, len(self.counts))
        count = np.fromiter(self.counts.values(), np.float32, len(ids))
        at = np.minimum(np.searchsorted(allowed, ids), len(allowed) - 1)
        hit = allowed[at] == ids
        s = self.sampling
        return _native.top_logits(sub, k, cap, at[hit].astype(np.int32), count[hit], s.repeat_penalty,
                                  s.presence_penalty, s.frequency_penalty, allowed)


def auto_capacity(tokens: int) -> int:
    """KV cache positions for a session of ``tokens`` (prompt and generated): a power of two from 4096 (one decode
    design per size), with room for a prefill's padded last chunk."""
    need, cap = max(tokens, 1) + 256, 4096
    while cap < need:
        cap *= 2
    return cap


def stale_toolchain() -> str | None:
    """What differs when the installed AIE toolchain (mlir_aie, llvm_aie) is not the one the checkout's pyproject.toml
    pins (a ``git pull`` without updating the venv: designs compile with another toolchain, and a measurement is not
    this commit's), else None. Without a checkout (an installed wheel): the pins of the installed artlite."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "pyproject.toml")
    try:
        with open(path, "rb") as f:
            deps = tomllib.load(f)["project"]["dependencies"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        deps = importlib.metadata.requires("artlite") or []
    stale = []
    for dep in deps:
        if pin := re.match(r"(mlir_aie|llvm_aie)\s*==\s*([^\s;]+)", dep.strip()):
            try:
                have = importlib.metadata.version(pin[1])
            except importlib.metadata.PackageNotFoundError:
                have = "not installed"
            if have != pin[2]:
                stale.append(f"{pin[1]} {have} (pyproject.toml: {pin[2]})")
    if not stale:
        return None
    return (f"the installed AIE toolchain is not this checkout's: {', '.join(stale)}; update the venv: "
            "uv sync --extra examples, then uv pip install --no-build-isolation --no-deps -ve . (docs/install.md)")


def stale_install() -> str | None:
    """What of the install is not this checkout's (``stale_toolchain``, ``backend.stale_native``), else None."""
    return "; ".join(s for s in (stale_toolchain(), backend.stale_native()) if s) or None


def xrt_smi(*args: str) -> str | None:
    """Windows: the output of ``xrt-smi`` with ``args`` (the driver's copy, else XRT's); None without one, or when it
    fails."""
    smi = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "AMD", "xrt-smi.exe")
    smi = smi if os.path.exists(smi) else backend.xrt_tool("xrt-smi")
    try:
        return subprocess.run([smi, *args], capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, TypeError, subprocess.SubprocessError):
        return None


def windows_power_mode() -> str:
    """Windows: the NPU's power mode as ``xrt-smi`` tells it (``default``, ``performance``, ...; '' unknown)."""
    mode = re.search(r"Power Mode\s*:\s*(\w+)", xrt_smi("examine", "-r", "platform") or "")
    return mode[1].lower() if mode else ""


def performance_mode(report: Callable[[str], None]) -> None:
    """Windows: raise the NPU's power mode from ``default`` to ``performance`` (``xrt-smi configure --pmode``, as
    FastFlowLM does at its start), in a thread while the model loads; ``report``: told the change. A mode set by hand
    stays (``turbo`` among them). Not on Linux: its driver takes a power mode from root only, and in ``default`` it runs
    a context at the top DPM level."""
    if not portable.WINDOWS:
        return

    def run() -> None:  # (no xrt-smi, or it failed: the mode stays)
        if windows_power_mode() == "default" and xrt_smi("configure", "--pmode", "performance") is not None:
            report("NPU power mode: default -> performance (xrt-smi)")

    threading.Thread(target=run, name="power-mode", daemon=True).start()


def _raise(message: str) -> None:
    raise jinja2.TemplateError(message)


def _tojson(value: object, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    """The templates' ``tojson`` as transformers and llama.cpp render it: the keys in their order, no HTML escapes
    (Jinja2's own sorts the keys and escapes ``<>&'``)."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def chat_template(template: str, messages: list[dict[str, object]], *, tools: list[dict[str, object]] | None = None,
                  **variables: object) -> str:
    """The prompt text of a conversation rendered with a model's chat template (a GGUF file's
    ``tokenizer.chat_template``, Jinja2 in a sandbox, as transformers renders it) up to the assistant's turn;
    ``tools``: the functions the model may call (OpenAI's ``{"type": "function", "function": {...}}``; a tool call's
    ``arguments`` in the messages are objects); ``variables``: the template's own (``enable_thinking``, ``bos_token``,
    ``date_string``, ...)."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().astimezone().strftime(fmt)
    return env.from_string(template).render(messages=messages, add_generation_prompt=True, tools=tools or None,
                                            **variables)


class Detokenizer(Protocol):
    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes: ...


class Tokenizer(Detokenizer, Protocol):
    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` (special tokens written in it become their ids; no BOS)."""
        ...


class StreamDecoder:
    """Incremental detokenizer: ``feed`` returns the text completed by a token (a multi-byte character split over
    byte tokens appears once its last byte arrives; ``pending``: its bytes so far)."""

    def __init__(self, tokenizer: Detokenizer, *, special: bool = False):
        self.tokenizer = tokenizer
        self.special = special
        self.pending = b""

    def feed(self, token: int) -> str:
        data = self.pending + self.tokenizer.decode_bytes([token], special=self.special)
        try:
            text = data.decode("utf-8")
            self.pending = b""
        except UnicodeDecodeError as e:
            text = data[:e.start].decode("utf-8", errors="replace")
            self.pending = data[e.start:]
            if len(self.pending) > 4:  # not an incomplete character: invalid bytes
                text += self.pending.decode("utf-8", errors="replace")
                self.pending = b""
        return text


def nbytes(x: object) -> int:
    """Bytes of the tensors and of the objects with ``nbytes`` (arrays, ``Records``) in nested lists, tuples and
    dicts."""
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(nbytes(v) for v in x)
    n = getattr(x, "nbytes", 0)
    return n if isinstance(n, int) else 0


@dataclass(eq=False)
class Checkpoint:
    """A sequence's state at ``position`` as host copies (``NpuModel.checkpoint``): what continuing from there needs
    besides the KV caches' records of the positions before it; ``nbytes`` of host memory."""

    position: int
    state: Any
    nbytes: int


class NpuModel:
    """A model on the NPU. ``forward`` runs a prompt after the cached positions and returns its last token's logits,
    ``step`` / ``top_k`` run one token; ``generate`` samples from them. ``t_npu``: seconds in dispatches, ``t_host``:
    host work on their inputs, ``dispatches``: their count.

    A sequence's state as host copies (a prompt cache's): ``checkpoint`` / ``restore`` (the state at a position besides
    the KV records), ``read_records`` / ``write_records`` (the records of positions)."""

    capacity: int  # KV cache positions
    position: int  # cached positions
    eos_token_ids: tuple[int, ...] | set[int]
    t_npu = t_host = 0.0
    dispatches = 0
    recurrent = False  # positions are kept only where a checkpoint was taken (states that hold the whole sequence)
    group = 1  # ``rewind`` and a ``restore`` without a checkpoint land on its multiples

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

    def prompt(self, tokens: list[int], *, logits: bool = False, progress: Callable[[int], None] | None = None
               ) -> torch.Tensor | None:
        """Run tokens after the cached positions (``forward``); ``logits``: return the last one's."""
        out = self.forward(tokens, progress=progress)
        return out if logits else None

    def checkpoint(self) -> Checkpoint:
        """The state at the current position that continuing from it needs besides the KV caches' records of the
        positions before it (any position: inside a cache group, after a step whose drafts were rejected)."""
        raise NotImplementedError

    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        """Continue from ``position``: the caches hold the records of the positions before it (this sequence's, or
        ``write_records``'), ``checkpoint`` (taken there) the rest of the state; without one, a position a rewind
        lands on (a multiple of ``group``; of a recurrent model: 0, a new sequence)."""
        raise NotImplementedError

    def read_records(self, start: int, end: int) -> "list[Records]":
        """Host copies of the KV caches' records of positions start .. end - 1 (whole groups; one per cache: the
        model's, a drafter's)."""
        raise NotImplementedError

    def write_records(self, records: "list[Records]", start: int, end: int) -> None:
        """The caches' records of positions start .. end - 1 from ``read_records``' copies (which hold them)."""
        raise NotImplementedError

    def step(self, token: int) -> torch.Tensor:
        """Run ``token`` at the next position: the logits (vocab,) of the next token."""
        raise NotImplementedError

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run ``token`` at the next position: the ``k`` largest logits of the next token (fp32, descending) and their
        token ids (``penalties``: of the penalized logits)."""
        logits = self.step(token)
        return top_logits(logits, k) if penalties is None else penalties.top(logits, k)

    def generate(self, logits: torch.Tensor, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = (),
                 mask: Mask | None = None) -> Iterator[int]:
        """Sample up to ``max_tokens`` tokens (< 0: no limit) after the cached positions; ``logits``: of the last
        cached one; ``prompt``: the prompt's tokens (the penalties' first); ``mask``: the tokens each may be (told
        each token as it is sampled). Ends after an end-of-sequence token or when the cache is full."""
        if max_tokens == 0:
            return
        penalties = Penalties(sampling, prompt, mask)
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
    """A model on its decode overlay (``overlay``: an ``artlite.decode.model.ModelOverlay``) with a chunk
    prefill (``prefill``: an ``artlite.prefill.chunks.ChunkPrefill``, or None: prompts as decode steps).

    ``steps`` runs up to ``overlay.SLOTS`` tokens per dispatch; ``prompt`` runs a prompt's tokens in prefill chunks
    where one can start (a whole cache group: the prefill's ``plan``), as decode steps before them and after, each
    chunk's inputs prepared on a worker thread while the previous chunk runs; ``forward`` is a prompt with its last
    token's logits (a decode step's, or of a chunk's last token: ``_chunk_logits``). A model supplies a step's inputs
    (``_step_inputs``: the overlay step's arguments before the position), a chunk's (``_chunk_inputs``: the prefill
    run's), the logits of a chunk's last token and the soft cap of its final logits (``softcap``).

    ``generate`` runs a generation in rounds: a step of the sampled token and up to ``DRAFTS`` draft tokens - the
    context's (``lookup``: ``drafts.Lookup``) or a drafter's (multi-token prediction: ``_mtp``), as ``planner``
    chooses -, and as long as the token the model's logits give after a token is the draft that ran behind it, the
    draft's logits are valid too. A model with a drafter or lazy states supplies the round's step (``_verify``) and
    what follows its kept tokens (``_kept``)."""

    overlay: Any
    prefill: Any
    slot = 0  # the slot of the last token whose logits a dispatch returned (a step's last; 0 after a chunk's)
    softcap = 0.0  # the model's logits of the overlay's l: softcap tanh(l / softcap) (0: l)
    DRAFTS = 3  # draft tokens per step at most (the overlay's SLOTS - 1; 0: steps of one token)
    STEP_COST = (1.0, 1.05, 1.1, 1.15)  # a step of S = 1 .. SLOTS tokens' time relative to one's, until measured
    P_MIN = 0.5  # a drafter's chain of drafts stops below this probability until its rounds are measured (``_worth``)
    # (a drafter step of one row's seconds; per draft depth the rounds that kept a draft there and their chains'
    # probabilities there: running averages)
    _t_row = 0.0
    _kept_at: tuple[float, ...] = ()
    _chain_at: tuple[float, ...] = ()
    _chains: tuple[float, ...] = ()  # (the chain probabilities of the drafter's last drafts: ``_calibrate``)
    lookup = True  # drafts from the context
    drafter: Any = None  # a multi-token-prediction drafter
    t_draft = 0.0  # seconds in the drafter's steps
    drafted = accepted = 0  # draft tokens run and accepted (of both kinds)
    drafted_ctx = accepted_ctx = 0  # the context's of them

    @functools.cached_property
    def planner(self) -> Planner:
        """(across the model's generations: its steps' times, the drafts' acceptance)"""
        return Planner(self.STEP_COST)

    @property
    def _drafting(self) -> bool:
        return self.drafter is not None and self.DRAFTS > 0

    def stats(self) -> str:
        out = []
        if mtp := self.drafted - self.drafted_ctx:
            out.append(f"{mtp} draft tokens ({self.t_draft / mtp * 1e3:.2f} ms each), "
                       f"{self.accepted - self.accepted_ctx} accepted")
        if self.drafted_ctx:
            out.append(f"{self.drafted_ctx} context draft tokens, {self.accepted_ctx} accepted")
        return "; ".join(out)

    def cap(self, logits: torch.Tensor) -> torch.Tensor:
        """The model's logits of the overlay's (fp32): the soft cap, if any."""
        return torch.tanh(logits / self.softcap) * self.softcap if self.softcap else logits

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

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(from the overlay's bf16 logits: the soft cap on the candidates alone)"""
        raw = self.steps([token])[0]
        return top_logits(raw, k, self.softcap) if penalties is None else penalties.top(raw, k, self.softcap)

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

    @property
    def group(self) -> int:
        """(a whole cache group of the prefill, or 8 positions without one: a step at a multiple of 8 opens a new V
        record, so the ring holds what it reads; a prompt from a group runs in chunks at once, while one inside a
        group runs as decode steps up to the next)"""
        return self.prefill.group if self.prefill is not None else 8

    def rewind(self, n: int) -> int:
        """(down to a multiple of ``group``)"""
        keep = min(n, self.position)
        self.position = keep - keep % self.group
        return self.position

    def _overlays(self) -> list[Any]:
        """The overlays that hold the sequence's caches and states (the model's, a drafter's)."""
        return [self.overlay]

    def _state(self) -> dict[str, Any]:
        """The model's own part of a checkpoint (its overlays' are theirs: ``ModelOverlay.snapshot``)."""
        return {}

    def _set_state(self, state: dict[str, Any]) -> None:
        """Back to ``_state``'s."""

    def checkpoint(self) -> Checkpoint:
        state = ([ov.snapshot(self.position) for ov in self._overlays()], self._state())
        return Checkpoint(self.position, state, nbytes(state))

    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        if checkpoint is None:
            if position % self.group or (self.recurrent and position):
                raise ValueError(f"position {position}: a restore without a checkpoint keeps a multiple of "
                                 f"{self.group}{' (0: the model is recurrent)' if self.recurrent else ''}")
            if self.recurrent:
                self.reset()
            self.position = position
            return
        assert checkpoint.position == position, (checkpoint.position, position)
        snapshots, state = checkpoint.state
        for ov, snap in zip(self._overlays(), snapshots, strict=True):
            ov.restore(snap)
        self._set_state(state)
        self.position = position

    def read_records(self, start: int, end: int) -> "list[Records]":
        return [ov.read_records(start, end) for ov in self._overlays()]

    def write_records(self, records: "list[Records]", start: int, end: int) -> None:
        for ov, r in zip(self._overlays(), records, strict=True):
            ov.write_records(r, start, end)

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

    def _mtp(self, token: int, k: int, penalties: Penalties) -> list[int]:
        """The drafter's guesses of up to k tokens after ``token`` (the next position's, not run yet)."""
        raise NotImplementedError

    def _crossing(self) -> tuple[bool, ...]:
        """Per step of S = 1 .. DRAFTS + 1 tokens at the position: whether it takes one more pass over the KV caches
        than at a cache group's start."""
        ov = self.overlay
        return tuple(ov.passes(self.position, S) > ov.passes(0, S) for S in range(1, self.DRAFTS + 2))

    def _worth(self, c: float, s: int, extra: float = 0.0) -> bool:
        """Whether a draft that the model keeps with probability c (its chain's: the drafter's probability of it and of
        the drafts before it) pays for a round's step of s + 1 tokens instead of s and ``extra`` seconds more (a drafter
        step to guess it): c tokens against what that time gives at the drafter's rounds' rate (``Planner.rate``;
        ``P_MIN`` until it is measured). A dense model's slots cost little: its chains go on while c is ~0.1 or more;
        an MoE's slots route to more experts: ~0.4-0.5."""
        rate = self.planner.rate()
        if rate is None:
            return c >= self.P_MIN
        plan = self.planner
        return c >= rate * (plan.step_time(s + 1) - plan.step_time(s) + extra)

    def _keep(self, chain: float, n: int) -> bool:
        """With n drafts so far, whether the drafter's next guess (chain: the probability of all n + 1) joins them (the
        chain's probability as the model keeps such chains: ``_kappa``)."""
        return self._worth(chain * self._kappa(n), n + 1)

    def _more(self, chain: float, p: float, n: int) -> bool:
        """With n drafts (their chain's probability, the last one's p), whether the drafter guesses another (its
        probability taken as the last one's: a confident chain goes on confidently, as code or a copy does; before its
        rounds are measured, always)."""
        return self.planner.rate() is None or self._worth(chain * self._kappa(n) * p, n + 1, self._t_row)

    def _kappa(self, d: int) -> float:
        """The drafts the model keeps at depth d (0: a round's first draft) per their chains' probability (the
        drafter's calibration: its guesses after its own guesses are overconfident, Qwen3.8-27B's on prose 1.0 at
        depth 0 to 0.24 at depth 6; a depth not drafted yet takes the deepest one's; 1 before its rounds)."""
        d = min(d, len(self._chain_at) - 1)
        return self._kept_at[d] / self._chain_at[d] if d >= 0 else 1.0

    def _calibrate(self, kept: int) -> None:
        """A drafter's round kept its first ``kept`` drafts (``_chains``: the probabilities the drafter gave them)."""
        k, c = list(self._kept_at), list(self._chain_at)
        for d, chain in enumerate(self._chains):
            if d == len(c):  # (a new depth: from the depth before it, as one round of its calibration and chains; the
                # first from 1: a rejected first draft must not stop the drafts that would measure it again)
                k.append(k[-1] if d else chain)
                c.append(c[-1] if d else chain)
            k[d] += EWMA * ((kept > d) - k[d])
            c[d] += EWMA * (chain - c[d])
        self._kept_at, self._chain_at = tuple(k), tuple(c)

    def _row_time(self, seconds: float) -> None:
        """A drafter step of one row took ``seconds``."""
        self._t_row = seconds if not self._t_row else self._t_row + EWMA * (seconds - self._t_row)

    def _begin(self) -> None:
        """A generation's rounds begin (after its prompt's last step)."""

    def _verify(self, tokens: list[int], mtp: bool) -> torch.Tensor:
        """A round's step of the sampled token and its drafts (``mtp``: the drafter's): their logits (S, vocab)."""
        return self.steps(tokens)

    def _kept(self, p: int, tokens: list[int], m: int, mtp: bool) -> None:
        """After a round's step of ``tokens`` at p: its first m tokens are the sequence's (the rejected drafts'
        positions are forgotten: later steps overwrite their cache records)."""
        self.position = p + m

    def generate(self, logits: torch.Tensor, max_tokens: int, sampling: Sampling, prompt: Sequence[int] = (),
                 mask: Mask | None = None) -> Iterator[int]:
        """(in rounds of a step of drafts each: the class docstring)"""
        if max_tokens == 0:
            return
        penalties = Penalties(sampling, prompt, mask)
        token = sampling.pick(*penalties.top(logits, sampling.k))
        penalties.accept(token)
        n = 1
        yield token
        ctx = Lookup(getattr(prompt, "ids", prompt)) if self.lookup and self.DRAFTS > 0 else None
        if ctx is not None:
            ctx.extend([token])
        self._begin()
        plan = self.planner
        while n != max_tokens and token not in self.eos_token_ids and self.position < self.capacity:
            k = min(self.DRAFTS, self.capacity - self.position - 1,
                    max_tokens - n - 1 if max_tokens > 0 else self.DRAFTS)
            plan.crossing = self._crossing()
            t0 = time.perf_counter()
            m, d = ctx.draft(k) if ctx is not None else (0, [])
            j = plan.choose(m, len(d), drafter=self._drafting and k > 0)
            mtp = j is None
            d = self._mtp(token, k, penalties) if mtp else d[:j]
            p = self.position
            t1 = time.perf_counter()
            raw = self._verify([token, *d], mtp)
            t2 = time.perf_counter()
            new: list[int] = []
            for i in range(len(d) + 1):
                new.append(sampling.pick(*penalties.top(raw[i], sampling.k, self.softcap)))
                penalties.accept(new[i])
                if i == len(d) or new[i] != d[i] or new[i] in self.eos_token_ids:
                    break
            self._kept(p, [token, *d], len(new), mtp)
            plan.ran(len(d) + 1, t2 - t1)
            self.accepted += len(new) - 1
            if mtp:
                plan.drafted(m, len(new), t2 - t0)
                self._calibrate(len(new) - 1)
            elif d:
                plan.accepted(m, len(d), len(new) - 1)
                self.drafted += len(d)
                self.drafted_ctx += len(d)
                self.accepted_ctx += len(new) - 1
            if ctx is not None:
                ctx.extend(new)
            for token in new:
                n += 1
                yield token

    def forced_logits(self, tokens: list[int], first: int = 0, *, batch: int = 1) -> torch.Tensor:
        """Teacher-forced logits (len(tokens) - first, vocab) fp32 of tokens[first:], from position 0, ``batch`` tokens
        per dispatch (quality tooling)."""
        self.reset()
        self.prompt(tokens[:first])
        return torch.cat([self.cap(self.steps(tokens[i:i + batch]).float()) for i in range(first, len(tokens), batch)])

    def forced_drafts(self, tokens: list[int], first: int, *, turns: Sequence[int] = (), prefill: bool = True
                      ) -> Iterator[tuple[list[int], torch.Tensor]]:
        """Teacher-forced drafts of the drafter (quality tooling): tokens[:first] from position 0 as prompts that start
        at 0 and at ``turns`` (a later one after the cached positions, as a chat's next turn; with ``prefill`` False all
        as decode steps), then per position of tokens[first:-1] the drafter's guesses after its token (``_mtp``, as a
        generation's round: up to ``DRAFTS``) and the model's step of the token alone (a round that keeps it): the
        guesses (of the tokens after it) and the model's logits (vocab,) fp32 of the next token."""
        assert self._drafting, "a drafter"
        self.reset()
        pf = self.prefill
        if not prefill:
            self.prefill = None
        try:
            for a, b in zip((0, *turns), (*turns, first)):
                self.prompt(tokens[a:b], logits=b == first)
        finally:
            self.prefill = pf
        penalties = Penalties(Sampling(greedy=True))
        self._begin()
        for token in tokens[first:-1]:
            p = self.position
            d = self._mtp(token, self.DRAFTS, penalties)
            logits = self.cap(self._verify([token], True)[0].float())
            self._kept(p, [token], 1, True)
            yield d, logits


class Model:
    """A model file opened on the host: its tokenizer, chat template and default sampling; ``load`` puts it on the NPU.
    One subclass per architecture (``models.ARCHITECTURES``); constructor keywords are its options."""

    name: str  # for messages: "Llama 3", ...
    vision: Any = None  # the image encoder (an option of the model's: its mmproj file), or None: text only
    projectors: tuple[str, ...] = ()  # the projector types (``clip.projector_type``) of the mmproj files it takes
    thinks = False  # the chat template has a thinking mode
    reasoning: tuple[str, str] | None = None  # the text that opens and closes a thought (the model writes them)
    thought_first = False  # a thought only begins an answer (its markers later in the answer are text)
    turn: str | None = None  # the text that begins every message in the chat template (a prompt cache's marks)
    user_turn: str | None = None  # the text that begins a user message
    tokenizer: Tokenizer
    eos_token_ids: tuple[int, ...] | set[int]

    def encode(self, text: str) -> list[int]:
        """Token ids of a raw text (no chat template), after the model's BOS if it has one."""
        raise NotImplementedError

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False) -> list[int]:
        """Token ids of a conversation through the chat template, up to the assistant's turn (``think``: let it
        think first; otherwise an empty thought where the template has one; ``tools``: the functions it may call,
        ``chat_template``; ``preserve``: the earlier answers keep their thoughts, the template's ``preserve_thinking``,
        where it has one). A message's content may be a list of parts: ``{"type": "text", "text": ...}``,
        ``{"type": "image", "image": a file, its bytes or a PIL image}`` (a model with ``vision``: its prompt then a
        sequence of ids and embedding rows; the encoder runs on the NPU)."""
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
