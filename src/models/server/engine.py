"""The server's NPU side: one worker thread owns the model and runs the requests' generations one at a time, in
arrival order (``Engine``), each from what the prompt cache keeps of its prompt (``cache.PromptCache``); the generated
text's stop strings (``StopMatcher``), its split into thoughts and answer (``ReasoningSplitter``) and the answer's tool
calls (``tools.Parser``; their names kept the functions' by ``tools.NameMask`` as the tokens are sampled); the timings
of a generation as llama-server reports them."""

import collections
import concurrent.futures
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from artlite import portable

from ..common import Model, NpuModel, Sampling, StreamDecoder, argmax, seed
from .cache import PromptCache
from .tools import Calls, held

log = logging.getLogger("artlite.server")
CACHE_RAM = 8192 << 20  # the prompt cache's host memory by default at most (llama-server's --cache-ram)

Emit = Callable[[str, dict[str, Any]], None]  # (event, data) -> None: a job's events to its request


class Cancelled(Exception):
    """A job was cancelled: its client went away, the UI's Stop, or the server's shutdown."""


@dataclass(eq=False)
class Job:
    """A generation for the worker: the prompt's tokens and how to continue them; ``emit`` sends its events (called
    on the worker thread):

    - ``progress``: the prompt's ``prompt_progress`` (``total``, ``cache``, ``processed``, ``time_ms``), when
      ``progress`` is set: at the start and after each chunk;
    - ``delta``: ``parts``, the new output as (kind, value) pairs: kind ``content`` or ``reasoning_content`` with
      text, ``tool_call`` with a ``tools.ToolCall`` (``tools``: a call's start and its arguments' fragments), and the
      ``timings`` so far;
    - ``done``: ``finish_reason`` (``stop``: an end-of-sequence token or a stop string, ``length``: ``max_tokens`` or a
      full cache, ``tool_calls``: the answer has complete calls), ``stop_word``, ``timings``, ``n_prompt`` (the
      prompt's tokens), ``n_cached`` (positions cached after it) and ``n_calls`` (the complete tool calls);
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
    think: bool = False  # the model thinks first (it opens its thought itself unless the prompt did)
    tools: Calls | None = None  # parse the answer's tool calls (and force one: ``Calls.force``)
    cache_prompt: bool = True  # continue from the cached prefix
    answer_mark: bool = False  # the next prompt re-renders this answer's opener (its template drops earlier thoughts)
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
        k = held(s, self.stops)
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
    ``inside``: the text starts inside a thought (the prompt opened it); ``ends``: text that ends a thought and begins
    the answer (kept: a Qwen tool call without the close marker); ``once``: a thought only begins the text (Qwen's),
    markers after it are answer. ``answering``: the answer has begun (a thought ended, or answer text came)."""

    def __init__(self, markers: tuple[str, str], *, inside: bool = False, ends: tuple[str, ...] = (),
                 once: bool = False):
        self.open, self.close = markers
        self.inside = inside
        self.ends, self.once = ends, once
        self._held = ""
        self._lead = True  # (a part begins: its leading white space is dropped)
        self._over = False  # (once: the thought is over, or the text began outside one)
        self.answering = False

    @property
    def kind(self) -> str:
        return "reasoning_content" if self.inside else "content"

    def _part(self, out: list[tuple[str, str]], text: str) -> None:
        if self._lead:
            text = text.lstrip()
            self._lead = not text
        if text:
            out.append((self.kind, text))
            if not self.inside:
                self._over = self.once
                self.answering = True

    def _marks(self) -> list[str]:
        if self.inside:
            return [self.close, *self.ends]
        return [] if self._over else [self.open, self.close]

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        s = self._held + text
        while True:
            marks = self._marks()
            hits = [(i, m) for m in marks if (i := s.find(m)) >= 0]
            if not hits:
                break
            i, m = min(hits)
            self._part(out, s[:i].rstrip())
            s = s[i:] if m in self.ends and self.inside else s[i + len(m):]
            self.answering = self.answering or self.inside
            self.inside = m == self.open
            self._over = self.once and not self.inside
            self._lead = True
        k = held(s, marks)
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

    The prompt cache (``cache.PromptCache``: ``cache_ram`` bytes of host memory - None: ``CACHE_RAM``, at most half of
    what is still available once the model is loaded (its weights and caches are host memory too) -, ``checkpoints``
    per sequence): a job continues from what it keeps of its prompt (the device's sequence, a checkpoint, a sequence
    saved in RAM) and runs the rest, in parts for a recurrent model's checkpoints (``PromptCache.marks``), with a
    checkpoint at its end; ``cached``: the tokens whose K / V the caches hold (the last sampled token never ran).
    ``on_ready``: called on the worker once the model is loaded (after the warmup); ``on_exit``: when the worker ends
    on its own (the load failed)."""

    def __init__(self, load: Callable[[], tuple[Model, NpuModel]], *, warmup: bool = True,
                 on_ready: Callable[[], None] | None = None, on_exit: Callable[[], None] | None = None,
                 cache_ram: int | None = None, checkpoints: int = 32):
        self.model: Model | None = None
        self.npu: NpuModel | None = None
        self.cache: PromptCache | None = None
        self.error: str | None = None  # why the load failed
        self.current: Job | None = None
        self._load, self._warmup, self._on_ready, self._on_exit = load, warmup, on_ready, on_exit
        self._cache_ram, self._checkpoints = cache_ram, checkpoints
        self._queue: collections.deque[Job] = collections.deque()
        self._cv = threading.Condition()
        self._stopping = False
        self._dirty = False  # (a failure: the next job resets the caches)
        self._ready = threading.Event()
        self._ids = itertools.count(1)
        self._thread = threading.Thread(target=self._run, name="artlite-npu")

    @property
    def cached(self) -> list[int]:
        """The tokens at the cached positions."""
        return self.cache.tokens if self.cache is not None else []

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
            turns = [self.model.tokenizer.encode(t) if t else None for t in (self.model.turn, self.model.user_turn)]
            ram = self._cache_ram
            if ram is None:
                free = portable.available_memory()
                ram = CACHE_RAM if free is None else min(CACHE_RAM, free // 2)
                log.info("prompt cache: up to %.1f GiB of host RAM (--cache-ram)", ram / 2**30)
            self.cache = PromptCache(self.npu, ram=ram, checkpoints=self._checkpoints, turn=turns[0],
                                     user_turn=turns[1])
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
                self.cache.clear()
                self._dirty = True
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                job.state = "done"
                with self._cv:
                    self.current = None

    def _warm(self) -> None:
        """A short prompt and one step (as llama-server's warmup): the first request's dispatches find their HW
        contexts and transactions ready (the prompt's tokens through the prefill: a few, Qwen's chunks take 3 at
        least; ``NpuModel.prepare``: the steps of drafts)."""
        t0 = time.perf_counter()
        npu = self.npu
        ids = self.model.encode("The quick brown fox jumps over the lazy dog.")
        npu.reset()
        npu.prepare(len(ids))
        logits = npu.forward(ids)
        npu.top_k(argmax(logits), 1)
        npu.reset()
        log.info("warmup: %.2f s", time.perf_counter() - t0)

    def _process(self, job: Job) -> None:
        """Run a job: its end is an event (``done``, ``cancelled``: ``Cancelled`` raised, ``error``: any other
        exception), and ``cached`` the tokens at the cached positions."""
        if job.cancel.is_set():
            raise Cancelled
        npu, model = self.npu, self.model
        t0 = time.perf_counter()
        seed(job.seed if job.seed >= 0 else None)
        ids, cache = job.ids, self.cache
        if self._dirty:
            npu.reset()
            self._dirty = False
        kept, last, how = cache.prepare(ids, enabled=job.cache_prompt)
        job.kept = kept
        drafted0, accepted0 = getattr(npu, "drafted", 0), getattr(npu, "accepted", 0)
        d0 = npu.dispatches

        def progress(i: int) -> None:
            if job.progress:
                job.emit("progress", {"total": len(ids), "cache": kept, "processed": min(kept + i, len(ids)),
                                      "time_ms": (time.perf_counter() - t0) * 1e3})
            if job.cancel.is_set():
                raise Cancelled

        tools = job.tools
        force = tools.force if tools is not None else None
        # (a forced call: its opener ends the prompt, or follows the thought where one may come first)
        thought = job.reasoning and model.reasoning is not None and (job.inside or job.think)
        early = model.tokenizer.encode(force) if force and not thought else []
        late = (tools.syntax.gap if job.inside else "") + force if force and thought else ""
        mask = tools.mask(model.tokenizer) if tools is not None else None  # (the calls' names: the functions')
        gen: list[int] = []
        it = None  # (the generation's tokens)

        def forced(tokens: list[int]) -> None:
            gen.extend(tokens)
            for t in tokens if mask is not None else ():
                mask.accept(t)

        job.state = "prompt"
        try:
            progress(0)
            at = kept
            for m in cache.marks(ids, kept, answer=job.answer_mark):  # (the prompt in parts: a checkpoint after each)
                npu.prompt(ids[at:m], progress=lambda i, b=at - kept: progress(b + i))
                at = m
                cache.checkpoint(ids[:m])
            if last is not None and not early:  # (all of it cached: a checkpoint of its end, with its logits)
                logits = last
            else:
                logits = npu.forward(ids[at:] + early, progress=lambda i, b=at - kept: progress(b + i))
                cache.checkpoint(ids + early, logits)
            forced(early)
            t1 = time.perf_counter()
            d1, t_npu1, t_host1 = npu.dispatches, npu.t_npu, npu.t_host
            job.state = "generating"

            def times(t: float) -> dict[str, Any]:
                return timings(cache_n=kept, prompt_n=len(ids) - kept, prompt_s=t1 - t0, predicted_n=len(gen),
                               predicted_s=t - t1, drafts=(getattr(npu, "drafted", 0) - drafted0,
                                                           getattr(npu, "accepted", 0) - accepted0))

            text = StreamDecoder(model.tokenizer)
            stop = StopMatcher(job.stop)
            split = ReasoningSplitter(model.reasoning, inside=job.inside, once=model.thought_first,
                                      ends=tools.syntax.ends if tools is not None else ()) \
                if job.reasoning and model.reasoning is not None else None
            calls = tools.parser() if tools is not None else None

            def parse(s: str) -> list[tuple[str, Any]]:
                parts = split.feed(s) if split is not None else [("content", s)] if s else []
                return calls.parts(parts) if calls is not None else parts

            def send(parts: list[tuple[str, Any]]) -> None:
                if parts:
                    job.emit("delta", {"parts": parts, "timings": times(time.perf_counter())})

            send(parse(force) if early else [])
            finish = "length"
            while True:
                left, forcing = job.max_tokens - len(gen) if job.max_tokens >= 0 else -1, False
                it = npu.generate(logits, left, job.sampling, ids + gen, mask)
                for token in it:
                    gen.append(token)
                    job.n_gen = len(gen)
                    if token in model.eos_token_ids:
                        finish = "stop"
                        break
                    s, hit = stop.feed(text.feed(token))
                    send(parse(s))
                    if hit:
                        finish = "stop"
                        break
                    if job.cancel.is_set():
                        raise Cancelled
                    # (the answer began and no call did: the forced call's opener follows)
                    forcing = bool(late) and split is not None and split.answering and not calls.started
                    # the answer is complete, or the opener goes next: end where every token but the last ran (a
                    # step of drafts runs several; the parser drops what follows a call when parallel is off)
                    if (forcing or (calls is not None and calls.done)) and npu.position + 1 <= len(ids) + len(gen):
                        break
                if not forcing or finish == "stop":
                    break
                n = npu.rewind(npu.position)  # (the generation stopped where the next request would continue)
                opener = model.tokenizer.encode(late)
                logits = npu.forward((ids + gen)[n:] + opener)
                forced(opener)
                send(parse(late))
                late = ""
            if stop.word is None:
                send(parse(stop.flush()))
            tail = split.flush() if split is not None else []
            send(calls.parts(tail) + calls.flush() if calls is not None else tail)
            if calls is not None and calls.calls and not calls.calling:
                finish = "tool_calls"
        finally:
            rest: list[int] = []  # (a step of drafts ran tokens past a stop string or a cancel: the device holds them)
            while it is not None and len(ids) + len(gen) + len(rest) < npu.position:
                if (t := next(it, None)) is None:
                    break
                rest.append(t)
            cache.ran((ids + gen + rest)[:npu.position])
        t2 = time.perf_counter()
        t = times(t2)
        job.emit("done", {"finish_reason": finish, "stop_word": stop.word, "timings": t, "n_prompt": len(ids),
                          "n_cached": len(self.cached), "n_calls": calls.calls if calls is not None else 0})
        # (artlite-cli's report: the generation's dispatches, the time of each, the host's inputs)
        passes = max(npu.dispatches - d1, 1)
        host = f", {(npu.t_host - t_host1) / passes * 1e3:.1f} ms of host inputs each" if npu.t_host > t_host1 else ""
        drafts = f"; {t['draft_n']} draft tokens, {t['draft_n_accepted']} accepted" if "draft_n" in t else ""
        how = f" ({how})" if how else ""
        log.info(f"{job.name}: {kept} cached tokens{how}; prompt: {len(ids) - kept} tokens in {t1 - t0:.2f} s "
                 f"({t['prompt_per_second']:.1f} tokens/s, {d1 - d0} dispatches); generation: {len(gen)} tokens in "
                 f"{t2 - t1:.2f} s ({t['predicted_per_second']:.2f} tokens/s): {passes} dispatches of "
                 f"{(npu.t_npu - t_npu1) / passes * 1e3:.1f} ms{host}{drafts}; finish: {finish}")
