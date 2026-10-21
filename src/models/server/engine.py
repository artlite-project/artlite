"""The server's NPU side: one worker thread owns the model and runs the requests' generations one at a time, in
arrival order (``Engine``); the prompt cache across requests; the generated text's stop strings (``StopMatcher``) and
its split into thoughts and answer (``ReasoningSplitter``); the timings of a generation as llama-server reports them."""

import collections
import concurrent.futures
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ..common import Model, NpuModel, Sampling, StreamDecoder

log = logging.getLogger("artlite.server")

Emit = Callable[[str, dict[str, Any]], None]  # (event, data) -> None: a job's events to its request


class Cancelled(Exception):
    """A job was cancelled: its client went away, the UI's Stop, or the server's shutdown."""


@dataclass(eq=False)
class Job:
    """A generation for the worker: the prompt's tokens and how to continue them; ``emit`` sends its events (called
    on the worker thread):

    - ``progress``: the prompt's ``prompt_progress`` (``total``, ``cache``, ``processed``, ``time_ms``), when
      ``progress`` is set: at the start and after each chunk;
    - ``delta``: ``parts``, the new text as (kind, text) pairs, kind ``content`` or ``reasoning_content``, and the
      ``timings`` so far;
    - ``done``: ``finish_reason`` (``stop``: an end-of-sequence token or a stop string, ``length``: ``max_tokens`` or a
      full cache), ``stop_word``, ``timings``, ``n_prompt`` (the prompt's tokens) and ``n_cached`` (positions cached
      after it);
    - ``error``: ``message``, a failure in the model (the job ends);
    - ``cancelled``: the job ended early (``cancel`` set)."""

    ids: list[int]  # (or a sequence of ids and embedding rows: an image's, ``Model.chat``)
    emit: Emit
    sampling: Sampling
    max_tokens: int = -1  # < 0: until an end-of-sequence token or a full cache
    seed: int = -1  # < 0: random
    stop: list[str] = field(default_factory=list)
    reasoning: bool = False  # split the thoughts off (the model's ``reasoning`` markers)
    inside: bool = False  # the prompt ends inside a thought (it opened one)
    cache_prompt: bool = True  # continue from the cached prefix
    progress: bool = False  # send prompt_progress events
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
    conv_id: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


def common_prefix(a: list[int], b: list[int]) -> int:
    """The length of the longest common prefix of two prompts' tokens (ids; a sequence's given rows by their bytes)."""
    a, b = (getattr(x, "cache_keys", lambda x=x: x)() for x in (a, b))
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


def _held(text: str, words: list[str]) -> int:
    """The length of the longest end of ``text`` that begins one of ``words`` (without being all of it)."""
    best = 0
    for w in words:
        for k in range(min(len(w) - 1, len(text)), best, -1):
            if text.endswith(w[:k]):
                best = k
                break
    return best


class StopMatcher:
    """Stop strings in a stream of text: ``feed`` returns the text that may be sent and whether a stop string ended it
    (the text is cut before the stop string, ``word``). Text that may begin a stop string is held back until what
    follows decides."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.word: str | None = None
        self._held = ""

    def feed(self, text: str) -> tuple[str, bool]:
        if not self.stops:
            return text, False
        s = self._held + text
        hits = [(i, w) for w in self.stops if (i := s.find(w)) >= 0]
        if hits:
            i, self.word = min(hits)
            self._held = ""
            return s[:i], True
        k = _held(s, self.stops)
        self._held = s[len(s) - k:]
        return s[:len(s) - k], False

    def flush(self) -> str:
        """The text held back, at the end of the generation."""
        out, self._held = self._held, ""
        return out


class ReasoningSplitter:
    """A thinking model's generated text split into its thoughts (``reasoning_content``) and its answer (``content``) by
    its thought markers (``Model.reasoning``): ``feed`` returns the (kind, text) parts a text completes. The markers
    are dropped, and so is the white space around them; text that may begin a marker and trailing white space are held
    back until what follows decides. A close marker outside a thought is dropped (Gemma may end an answer with one).
    ``inside``: the text starts inside a thought (the prompt opened it)."""

    def __init__(self, markers: tuple[str, str], *, inside: bool = False):
        self.open, self.close = markers
        self.inside = inside
        self._held = ""
        self._lead = True  # (a part begins: its leading white space is dropped)

    @property
    def kind(self) -> str:
        return "reasoning_content" if self.inside else "content"

    def _part(self, out: list[tuple[str, str]], text: str) -> None:
        if self._lead:
            text = text.lstrip()
            self._lead = not text
        if text:
            out.append((self.kind, text))

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        s = self._held + text
        while True:
            marks = [self.close] if self.inside else [self.open, self.close]
            hits = [(i, m) for m in marks if (i := s.find(m)) >= 0]
            if not hits:
                break
            i, m = min(hits)
            self._part(out, s[:i].rstrip())
            s = s[i + len(m):]
            self.inside = m == self.open
            self._lead = True
        k = _held(s, marks)
        body = s[:len(s) - k]
        text = body.rstrip()
        self._held = body[len(text):] + s[len(s) - k:]
        self._part(out, text)
        return out

    def flush(self) -> list[tuple[str, str]]:
        """The text held back, at the end of the generation."""
        out: list[tuple[str, str]] = []
        self._part(out, self._held.rstrip())
        self._held = ""
        return out


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


class Engine:
    """The worker thread that owns the model: it opens and loads it (``load``: the ``Model`` and its ``NpuModel``),
    runs one token through it (``warmup``), then runs the jobs one at a time in arrival order.

    The prompt cache: ``cached`` holds the tokens whose K / V the caches hold (the last sampled token never ran). A job
    keeps the prefix its prompt shares with them (``NpuModel.rewind``: a cache group's multiple, or for Qwen only an
    exact continuation) and runs the rest; a prompt that is all cached runs its last token again (``forward`` needs
    one). ``on_ready``: called on the worker once the model is loaded (after the warmup); ``on_exit``: when the worker
    ends on its own (the load failed)."""

    def __init__(self, load: Callable[[], tuple[Model, NpuModel]], *, warmup: bool = True,
                 on_ready: Callable[[], None] | None = None, on_exit: Callable[[], None] | None = None):
        self.model: Model | None = None
        self.npu: NpuModel | None = None
        self.error: str | None = None  # why the load failed
        self.cached: list[int] = []
        self.current: Job | None = None
        self._load, self._warmup, self._on_ready, self._on_exit = load, warmup, on_ready, on_exit
        self._queue: collections.deque[Job] = collections.deque()
        self._cv = threading.Condition()
        self._stopping = False
        self._dirty = False  # (a failure: the next job resets the caches)
        self._ready = threading.Event()
        self._ids = itertools.count(1)
        self._thread = threading.Thread(target=self._run, name="artlite-npu")

    @property
    def ready(self) -> bool:
        """The model is loaded: jobs run."""
        return self._ready.is_set()

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    def submit(self, job: Job | Call) -> None:
        """Queue a job (after the ones waiting)."""
        with self._cv:
            if self._stopping:
                raise RuntimeError("the server is shutting down")
            if isinstance(job, Job):
                job.name = job.name or f"request {next(self._ids)}"
            self._queue.append(job)
            self._cv.notify()

    def run(self, fn: Callable[[], Any]) -> concurrent.futures.Future:
        """Run fn on the worker after the queued jobs (NPU work outside a generation); its result or exception."""
        call = Call(fn)
        self.submit(call)
        return call.future

    def waiting(self) -> int:
        """Jobs in the queue (not running)."""
        return len(self._queue)

    def cancel(self, conv_id: str) -> int:
        """Cancel the jobs of a conversation (running or queued); returns how many."""
        with self._cv:
            jobs = [j for j in (self.current, *self._queue) if j is not None and j.conv_id == conv_id]
        for j in jobs:
            j.cancel.set()
        return len(jobs)

    def stop(self) -> None:
        """Shut down: cancel the running and the queued jobs, end the worker after the running one's dispatch."""
        with self._cv:
            self._stopping = True
            for j in (self.current, *self._queue):
                if j is not None:
                    j.cancel.set()
            self._cv.notify()

    def join(self, timeout: float | None = None) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout)

    # ------------------------------------------------------------------
    # The worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self.model, self.npu = self._load()
            if self._warmup and not self._stopping:
                self._warm()
        except Exception as e:  # (any failure of the load: reported, the server exits)
            log.exception("loading the model failed")
            self.error = f"{type(e).__name__}: {e}"
            if self._on_exit is not None:
                self._on_exit()
            return
        self._ready.set()
        if self._on_ready is not None:
            self._on_ready()
        while True:
            with self._cv:
                while not self._queue and not self._stopping:
                    self._cv.wait()
                if not self._queue:
                    return
                job = self._queue.popleft()
                if not isinstance(job, Call):
                    self.current = job
            if isinstance(job, Call):
                try:
                    if job.cancel.is_set():
                        raise RuntimeError("the server is shutting down")
                    job.future.set_result(job.fn())
                except Exception as e:  # noqa: BLE001 (the caller's: its future)
                    job.future.set_exception(e)
                continue
            try:
                self._process(job)
            except Cancelled:
                log.info("%s: cancelled after %d of %d prompt tokens, %d generated", job.name,
                         max(min(self.npu.position, len(job.ids)) - job.kept, 0), len(job.ids) - job.kept, job.n_gen)
                job.emit("cancelled", {})
            except Exception as e:  # (a failure in the model: the job's error; the caches start over)
                log.exception("%s failed", job.name)
                self.cached, self._dirty = [], True
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                job.state = "done"
                with self._cv:
                    self.current = None

    def _warm(self) -> None:
        """A short prompt and one step (as llama-server's warmup): the first request's dispatches find their HW
        contexts and transactions ready (the prompt's tokens through the prefill: a few, Qwen's chunks take 3 at
        least)."""
        t0 = time.perf_counter()
        npu = self.npu
        ids = self.model.encode("The quick brown fox jumps over the lazy dog.")
        npu.reset()
        logits = npu.forward(ids)
        npu.top_k(int(logits.argmax()), 1)
        npu.reset()
        log.info("warmup: %.2f s", time.perf_counter() - t0)

    def _process(self, job: Job) -> None:
        """Run a job: its end is an event (``done``, ``cancelled``: ``Cancelled`` raised, ``error``: any other
        exception), and ``cached`` the tokens at the cached positions."""
        if job.cancel.is_set():
            raise Cancelled
        npu, model = self.npu, self.model
        t0 = time.perf_counter()
        if job.seed >= 0:
            torch.manual_seed(job.seed % 2**64)
        else:
            torch.seed()
        ids = job.ids
        n = min(common_prefix(self.cached, ids), len(ids) - 1) if job.cache_prompt else 0
        kept = npu.rewind(n) if n > 0 else 0
        if kept == 0 and (npu.position or self._dirty):
            npu.reset()
        self.cached, self._dirty, job.kept = ids[:kept], False, kept
        drafted0, accepted0 = getattr(npu, "drafted", 0), getattr(npu, "accepted", 0)
        d0 = npu.dispatches

        def progress(i: int) -> None:
            if job.progress:
                job.emit("progress", {"total": len(ids), "cache": kept, "processed": kept + i,
                                      "time_ms": (time.perf_counter() - t0) * 1e3})
            if job.cancel.is_set():
                raise Cancelled

        job.state = "prompt"
        gen: list[int] = []
        try:
            progress(0)
            logits = npu.forward(ids[kept:], progress=progress)
            t1 = time.perf_counter()
            d1, t_npu1, t_host1 = npu.dispatches, npu.t_npu, npu.t_host
            job.state = "generating"

            def times(t: float) -> dict[str, Any]:
                return timings(cache_n=kept, prompt_n=len(ids) - kept, prompt_s=t1 - t0, predicted_n=len(gen),
                               predicted_s=t - t1, drafts=(getattr(npu, "drafted", 0) - drafted0,
                                                           getattr(npu, "accepted", 0) - accepted0))

            text = StreamDecoder(model.tokenizer)
            stop = StopMatcher(job.stop)
            split = ReasoningSplitter(model.reasoning, inside=job.inside) \
                if job.reasoning and model.reasoning is not None else None

            def send(s: str) -> None:
                parts = split.feed(s) if split is not None else [("content", s)] if s else []
                if parts:
                    job.emit("delta", {"parts": parts, "timings": times(time.perf_counter())})

            finish = "length"
            for token in npu.generate(logits, job.max_tokens, job.sampling, ids):
                gen.append(token)
                job.n_gen = len(gen)
                if token in model.eos_token_ids:
                    finish = "stop"
                    break
                s, hit = stop.feed(text.feed(token))
                send(s)
                if hit:
                    finish = "stop"
                    break
                if job.cancel.is_set():
                    raise Cancelled
            if stop.word is None:
                send(stop.flush())
            if split is not None and (parts := split.flush()):
                job.emit("delta", {"parts": parts, "timings": times(time.perf_counter())})
        finally:
            self.cached = (ids + gen)[:npu.position]
        t2 = time.perf_counter()
        t = times(t2)
        job.emit("done", {"finish_reason": finish, "stop_word": stop.word, "timings": t, "n_prompt": len(ids),
                          "n_cached": len(self.cached)})
        # (artlite-cli's report: the generation's dispatches, the time of each, the host's inputs)
        passes = max(npu.dispatches - d1, 1)
        host = f", {(npu.t_host - t_host1) / passes * 1e3:.1f} ms of host inputs each" if npu.t_host > t_host1 else ""
        drafts = f"; {t['draft_n']} draft tokens, {t['draft_n_accepted']} accepted" if "draft_n" in t else ""
        log.info(f"{job.name}: {kept} cached tokens; prompt: {len(ids) - kept} tokens in {t1 - t0:.2f} s "
                 f"({t['prompt_per_second']:.1f} tokens/s, {d1 - d0} dispatches); generation: {len(gen)} tokens in "
                 f"{t2 - t1:.2f} s ({t['predicted_per_second']:.2f} tokens/s): {passes} dispatches of "
                 f"{(npu.t_npu - t_npu1) / passes * 1e3:.1f} ms{host}{drafts}; finish: {finish}")
