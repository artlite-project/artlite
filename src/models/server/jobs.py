"""The protocol between a request and the engine's worker: the work queued (``Job``: a generation, ``Call``: other NPU
work between the jobs), the events a job sends back (``Emit``: ``Progress``, ``Delta``, ``Done``, ``Failed`` carry their
payloads), and llama-server's records of a generation (``prob_entry``, ``timings``)."""

import concurrent.futures
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

import numpy as np

from artlite import _native

from ..detokenizer import Detokenizer
from ..sampling import LogitsLike, Sampling, logit_bits, top_logits
from .tools import Calls

Emit = Callable[[str, dict[str, Any]], None]  # (event, data) -> None: a job's events to its request


class Progress(TypedDict):
    """The payload of a ``progress`` event (``prompt_progress``)."""

    total: int
    cache: int
    processed: int
    time_ms: float


class Delta(TypedDict):
    """The payload of a ``delta`` event; ``probs`` only with ``Job.n_probs``."""

    parts: list[tuple[str, Any]]
    timings: dict[str, Any]
    probs: NotRequired[list[dict[str, Any]]]


class Done(TypedDict):
    """The payload of a ``done`` event."""

    finish_reason: str
    stop_word: str | None
    timings: dict[str, Any]
    n_prompt: int
    n_cached: int
    n_calls: int
    probs: list[dict[str, Any]]


class Failed(TypedDict):
    """The payload of an ``error`` event."""

    message: str



class Cancelled(Exception):
    """A job was cancelled: its client went away, the UI's Stop, or the server's shutdown."""


@dataclass(eq=False)
class Job:
    """A generation for the worker: the prompt's tokens and how to continue them; ``emit`` sends its events (called
    on the worker thread):

    - ``progress`` (``Progress``): the prompt's ``prompt_progress`` (``total``, ``cache``, ``processed``,
      ``time_ms``), when ``progress`` is set: at the start and after each chunk;
    - ``delta`` (``Delta``): ``parts``, the new output as (kind, value) pairs: kind ``content`` or
      ``reasoning_content`` with text, ``tool_call`` with a ``tools.ToolCall`` (``tools``: a call's start and its
      arguments' fragments), the ``timings`` so far, and with ``n_probs``: ``probs``, llama-server's entries of the
      sampled tokens since the last delta (``prob_entry``);
    - ``done`` (``Done``): ``finish_reason`` (``stop``: an end-of-sequence token or a stop string, ``length``:
      ``max_tokens`` or a full cache, ``tool_calls``: the answer has complete calls), ``stop_word``, ``timings``,
      ``n_prompt`` (the prompt's tokens), ``n_cached`` (positions cached after it), ``n_calls`` (the complete tool
      calls), and ``probs``: the entries no delta carried;
    - ``error`` (``Failed``): ``message``, a failure in the model (the job ends);
    - ``cancelled``: the job ended early (``cancel`` set)."""

    ids: list[int]  # (or a sequence of ids and embedding rows: an image's, ``Model.chat``)
    emit: Emit
    sampling: Sampling
    max_tokens: int = -1  # < 0: until an end-of-sequence token or a full cache
    drafts: int | None = None  # draft tokens per step at most (None: the model's, ``NpuModel.DRAFTS``)
    seed: int = -1  # < 0: random
    stop: list[str] = field(default_factory=list)
    reasoning: bool = False  # split the thoughts off (the model's ``reasoning`` markers)
    inside: bool = False  # the prompt ends inside a thought (it opened one)
    think: bool = False  # the model thinks first (it opens its thought itself unless the prompt did)
    tools: Calls | None = None  # parse the answer's tool calls (and force one: ``Calls.force``)
    cache_prompt: bool = True  # continue from the cached prefix
    answer_mark: bool = False  # the next prompt re-renders this answer's opener (its template drops earlier thoughts)
    progress: bool = False  # send prompt_progress events
    n_probs: int = 0  # per sampled token its probability and the n_probs most likely ones' (0: none; ``prob_entry``)
    post_sampling: bool = False  # the probabilities the sampler drew from (its candidates), not the model's
    conv_id: str | None = None  # the UI's conversation (X-Conversation-Id), for its Stop
    name: str = ""  # for the log
    cancel: threading.Event = field(default_factory=threading.Event)
    state: str = "queued"  # queued, prompt, generating, done
    kept: int = 0  # the prompt's cached tokens (the prefix it continued from)
    n_gen: int = 0  # tokens generated so far


@dataclass(eq=False)
class Call:
    """Work for the worker between jobs (``Engine.run``): an image prompt's encoder, the NPU dispatches of another
    hardware context."""

    fn: Callable[[], Any]
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    cancel: threading.Event = field(default_factory=threading.Event)


def _text(b: bytes) -> str:
    """A token's text: its bytes but an unfinished character at the end (llama-server's ``validate_utf8``)."""
    for k in range(4):
        try:
            return b[:len(b) - k].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def prob_entry(tokenizer: Detokenizer, row: LogitsLike, token: int, cap: float, cand: tuple[np.ndarray, np.ndarray],
               *, n: int, sampling: Sampling | None = None) -> dict[str, Any]:
    """llama-server's ``probs`` entry of a sampled token: its id, text and bytes, its log-probability under the model's
    logits ``row`` (soft cap ``cap``) and the ``n`` most likely tokens' (``top_logprobs``); with ``sampling``
    (``post_sampling_probs``) the probabilities ``sampling`` drew from among its candidates ``cand`` (``prob``,
    ``top_probs``)."""

    def tok(t: int, key: str, v: float) -> dict[str, Any]:
        b = tokenizer.decode_bytes([t])
        return {"id": t, "token": _text(b), "bytes": list(b), key: v}

    if sampling is not None:
        q, ids = sampling.probs(cand[0]), cand[1]
        top = [tok(int(i), "prob", float(p)) for i, p in zip(ids[:n], q[:n], strict=True) if p > 0]
        return {**tok(token, "prob", float(q[ids == token].sum())), "top_probs": top}
    lse = _native.logsumexp(logit_bits(row), cap)
    v = float(row[token])
    own = (math.tanh(v / cap) * cap if cap else v) - lse
    values, ids = top_logits(row, n, cap)
    return {**tok(token, "logprob", own),
            "top_logprobs": [tok(int(i), "logprob", float(x) - lse) for x, i in zip(values, ids, strict=True)]}


def timings(*, cache_n: int, prompt_n: int, prompt_s: float, predicted_n: int, predicted_s: float,
            drafts: tuple[int, int] = (0, 0)) -> dict[str, Any]:
    """llama-server's ``timings`` of a generation; ``drafts``: (draft tokens, accepted ones) of multi-token
    prediction."""
    t = {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": prompt_s * 1e3,
         "prompt_per_token_ms": prompt_s * 1e3 / max(prompt_n, 1),
         "prompt_per_second": prompt_n / prompt_s if prompt_s > 0 else 0.0,
         "predicted_n": predicted_n, "predicted_ms": predicted_s * 1e3,
         "predicted_per_token_ms": predicted_s * 1e3 / max(predicted_n, 1),
         "predicted_per_second": predicted_n / predicted_s if predicted_s > 0 else 0.0}
    if drafts[0]:
        t |= {"draft_n": drafts[0], "draft_n_accepted": drafts[1]}
    return t
