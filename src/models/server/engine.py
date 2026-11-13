"""The server's NPU side: one worker thread owns the model and runs the requests' generations one at a time, in
arrival order (``Engine``; ``jobs``: its queue's work and the events back), each from what the prompt cache
(``cache.PromptCache``) keeps of its prompt (``generation.Generation``: the stop strings and the thought split of
``textflow``, the answer's tool calls of ``tools``)."""

import collections
import concurrent.futures
import itertools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from artlite import portable

from ..common import Model, NpuModel, argmax
from ..sequence import Sequence
from .cache import PromptCache, common_prefix
from .generation import Generation
from .jobs import Call, Cancelled, Job

log = logging.getLogger("artlite.server")
CACHE_RAM = 8192 << 20  # the prompt cache's host memory by default at most (llama-server's --cache-ram)


class Engine:
    """The worker thread that owns the model: it opens and loads it (``load``: the ``Model`` and its ``NpuModel``),
    runs one token through it (``warmup``), then runs the jobs one at a time in arrival order.

    The prompt cache (``cache.PromptCache``: ``cache_ram`` bytes of host memory - None: ``CACHE_RAM``, at most half of
    what is still available once the model is loaded (its weights and caches are host memory too), over ``share``
    (a router's models: each server's part); a non-causal model's none -, ``checkpoints``
    per sequence): a job continues from what it keeps of its prompt (the device's sequence, a checkpoint, a sequence
    saved in RAM) and runs the rest, in parts for a recurrent model's checkpoints (``PromptCache.marks``), with a
    checkpoint at its end; ``cached``: the tokens whose K / V the caches hold (the last sampled token never ran).
    ``on_ready``: called on the worker once the model is loaded (after the warmup); ``on_exit``: when the worker ends
    on its own (the load failed)."""

    def __init__(self, load: Callable[[], tuple[Model, NpuModel]], *, warmup: bool = True,
                 on_ready: Callable[[], None] | None = None, on_exit: Callable[[], None] | None = None,
                 cache_ram: int | None = None, share: int = 1, checkpoints: int = 32):
        self.model: Model | None = None
        self.npu: NpuModel | None = None
        self.cache: PromptCache | None = None
        self.error: str | None = None  # why the load failed
        self.current: Job | None = None
        self._load, self._warmup, self._on_ready, self._on_exit = load, warmup, on_ready, on_exit
        self._cache_ram, self._share, self._checkpoints = cache_ram, max(share, 1), checkpoints
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

    def live(self, fn: Callable[[], Any]) -> concurrent.futures.Future:
        """Run fn on the worker after the queued jobs: a live stream's step (``models.lfm3a.live.LiveAsr``, its
        ``sync`` this engine's ``hold`` and ``ran``); a failure: the caches start over."""
        def run() -> Any:
            try:
                return fn()
            except Exception:  # (the device's state is unknown)
                self.recover()
                raise
        return self.run(run)

    def begin(self) -> None:
        """(on the worker) Before work on the device: it starts over when a failure left its state unknown."""
        if self._dirty:
            self.npu.reset()
            self._dirty = False

    def recover(self) -> None:
        """(on the worker) After a failure, or before work that leaves the device's sequence unknown to the prompt cache
        (speech output: ``models.lfm3a.speech``): the cache keeps what RAM holds of the current one, the next work
        resets the device (``begin``)."""
        self.cache.clear()
        self._dirty = True

    def device_taken(self) -> None:
        """(on the worker) Before work that leaves the device's sequence unknown to the prompt cache (speech output:
        ``models.lfm3a.speech``)."""
        self.recover()

    def hold(self, ids: Sequence) -> int:
        """(on the worker) A live stream's text sequence and its next chunk's tokens: the device where they continue
        from (``PromptCache.prepare``: as the last chunk left it, else, when requests ran between the stream's chunks,
        what the prompt cache keeps), the tokens it holds."""
        self.begin()
        return self.cache.prepare(ids)[0]

    def ran(self, seq: Sequence) -> None:
        """(on the worker) After a live stream's chunk: the device holds ``seq`` (saved by the prompt cache when a
        request displaces it, as a conversation)."""
        self.cache.ran(seq)

    def readouts(self, inputs: list[list[int]], rows: list[np.ndarray], *, cache_prompt: bool = True,
                 name: str = "readout", sets: list[tuple[int, ...] | None] | None = None) -> concurrent.futures.Future:
        """Output rows of prompts (``NpuModel.outputs``, output-normed) on the worker after the queued jobs: per input
        its tokens' rows at ``rows`` (indices). One input runs after what the prompt cache keeps of it before its first
        row; several run after a prefix they share (``_start``: run once, through the prompt cache) as
        ``NpuModel.readouts`` runs them (packed chunks where the model packs, else one after another); a non-causal
        model's from position 0, ``sets`` its inputs' attention (``NpuModel.outputs``). The result: a list of
        (len(rows), hidden) fp32."""
        return self.run(lambda: self._readouts(inputs, rows, cache_prompt, name, sets))

    def _readouts(self, inputs: list[list[int]], rows: list[np.ndarray], cache_prompt: bool, name: str,
                  sets: list[tuple[int, ...] | None] | None = None) -> list[np.ndarray]:
        npu, cache = self.npu, self.cache
        cache_prompt = cache_prompt and npu.causal  # (a non-causal model reads every prompt whole)
        t0, ran = time.perf_counter(), 0
        try:
            self.begin()
            if sets is not None:  # (a non-causal model's sequences of their own attention: each whole, from position 0)
                assert not npu.causal
                cache.prepare([], enabled=False)
                out = npu.readouts(inputs, rows, sets=sets)
                cache.ran([])
                ran = sum(map(len, inputs))
            elif len(inputs) == 1:
                (ids,), (at,) = inputs, rows
                k, _, _ = cache.prepare(ids[:int(at.min())], enabled=cache_prompt)
                out = [npu.outputs(ids[k:], at - k)]
                cache.ran(ids)
                ran = len(ids) - k
            else:
                p = self._start(inputs, rows) if cache_prompt else 0
                prefix = inputs[0][:p]
                k, _, _ = cache.prepare(prefix, enabled=cache_prompt)
                # (the prefix's rest not cached: a lead the first packed chunk may run, else a readout's chunks first)
                lead = prefix[k:] if 0 < p - k == npu.lead_tokens and not npu.recurrent else []
                if k < p and not lead:
                    npu.outputs(prefix[k:], [-1])  # (in chunks: as a readout runs)
                if p and not lead:
                    cache.checkpoint(prefix)
                out = npu.readouts([ids[p:] for ids in inputs], [at - p for at in rows], lead=lead or None)
                if lead:
                    cache.checkpoint(prefix)
                cache.ran(prefix)
                ran = p - k + sum(len(ids) - p for ids in inputs)
        except Exception:  # (the device's state is unknown: the caches start over)
            self.recover()
            raise
        n = sum(map(len, inputs))
        log.info(f"{name}: {len(inputs)} input{'s' * (len(inputs) != 1)}, {ran} tokens ({n - ran} cached) in "
                 f"{time.perf_counter() - t0:.3f} s")
        return out

    def _start(self, inputs: list[list[int]], rows: list[np.ndarray]) -> int:
        """Where several prompts split into a prefix they share, run once, and their rests: the last cache group's start
        before their first difference and every prompt's first row, if the estimates (``NpuModel.readout_seconds``,
        ``prompt_seconds``) say the prefix and the rests after it take less than 0.9 of the prompts whole, else 0."""
        npu = self.npu
        p = min(min(common_prefix(inputs[0], ids) for ids in inputs[1:]), min(int(at.min()) for at in rows))
        p -= p % npu.group
        if p <= 0:
            return 0
        whole = npu.readout_seconds([len(ids) for ids in inputs])
        rest = npu.readout_seconds([len(ids) - p for ids in inputs], p)
        head = npu.prompt_seconds(p)
        if whole is None or rest is None or head is None:
            return 0
        return p if head + rest < 0.9 * whole else 0

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
            if not self.npu.causal:  # (it reads every prompt whole: nothing continues from a saved sequence)
                ram = 0
            elif ram is None:
                free = portable.available_memory()
                ram = (CACHE_RAM if free is None else min(CACHE_RAM, free // 2)) // self._share
                log.info("prompt cache: up to %.1f GiB of host RAM (--cache-ram%s)", ram / 2**30,
                         f", over --models-max {self._share}" if self._share > 1 else "")
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
                Generation(self, job).run()
            except Cancelled:
                log.info("%s: cancelled after %d of %d prompt tokens, %d generated", job.name,
                         max(min(self.npu.position, len(job.ids)) - job.kept, 0), len(job.ids) - job.kept, job.n_gen)
                job.emit("cancelled", {})
            except Exception as e:  # (a failure in the model: the job's error; the caches start over)
                log.exception("%s failed", job.name)
                self.recover()
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                job.state = "done"
                with self._cv:
                    self.current = None

    def _warm(self) -> None:
        """A short prompt and one step (as llama-server's warmup): the first request's dispatches find their HW
        contexts and transactions ready (the prompt's tokens through the prefill: a few, Qwen's chunks take 3 at
        least; ``NpuModel.prepare``: the steps of drafts); a non-causal model's: a readout (it runs no steps)."""
        t0 = time.perf_counter()
        npu = self.npu
        ids = self.model.encode("The quick brown fox jumps over the lazy dog.")
        npu.reset()
        if npu.causal:
            npu.prepare(len(ids))
            logits = npu.forward(ids)
            npu.top_k(argmax(logits), 1)
        else:
            npu.outputs(ids, [-1])
        npu.reset()
        log.info("warmup: %.2f s", time.perf_counter() - t0)
