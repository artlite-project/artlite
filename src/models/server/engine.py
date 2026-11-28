"""The server's NPU side: the worker threads that own the model, one per sequence it holds (``Lane``: one, or ``-np`` of
a model with lanes, ``OverlayModel.lanes``), each running the requests' generations one at a time (``Engine``;
``jobs``: its queue's work and the events back), each from what its lane's prompt cache (``cache.PromptCache``) keeps
of its prompt (``generation.Generation``: the stop strings and the thought split of ``textflow``, the answer's tool
calls of ``tools``).

Several lanes: the jobs go to the idle lanes in arrival order, each to the lane its request names (``id_slot``), else to
the one whose prompt cache keeps the most of its prompt where that is ``SIMILAR`` of it at least, else to the least
recently used (llama-server's choice of a slot); the lanes' generations share their decode steps (``models.lanes``).
What is no job (a ``Call``: readouts, answers, an image's encoder, a live stream's step) runs on lane 0 with the device
held, as a prompt's chunks hold it: the other lanes' steps wait for it."""

import collections
import concurrent.futures
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from artlite import portable

from .. import structured
from ..model import Model
from ..npu_model import NpuModel
from ..sampling import argmax
from ..sequence import Sequence
from . import answers
from .cache import PromptCache, common_prefix
from .generation import Generation
from .jobs import Call, Cancelled, Failed, Job

log = logging.getLogger("artlite.server")
CACHE_RAM = 8192 << 20  # the prompt cache's host memory by default at most (llama-server's --cache-ram)
SIMILAR = 0.5  # a job goes to the idle lane that keeps the most of its prompt where that is this share of it at least
# (llama-server's --slot-prompt-similarity), else to the least recently used


@dataclass(eq=False)
class Lane:
    """A sequence of the model as the engine serves it: its model (``npu``: the model's lane, ``NpuModel.lanes``), its
    prompt cache, the job it runs."""

    index: int
    npu: NpuModel
    cache: PromptCache
    thread: threading.Thread | None = None
    current: Job | None = None
    busy: bool = False  # (it runs a job or a call)
    dirty: bool = False  # (a failure: its next work resets its caches)
    used: int = 0  # (the engine's count of work done when its last ended: the least recently used lane's is lowest)


class Engine:
    """The worker threads that own the model. The first opens and loads it (``load``: the ``Model`` and its
    ``NpuModel``), runs one token through it (``warmup``; it also builds structured output's vocabulary aside:
    ``vocab``), then each lane's worker (``lanes``: the model's, ``NpuModel.lanes``) runs jobs one at a time, the
    queue in arrival order (module docstring). ``npu``, ``cache``, ``cached``: lane 0's (the calls' and a model of one
    sequence's).

    A lane's prompt cache (``cache.PromptCache``: ``cache_ram`` bytes of host memory - None: ``CACHE_RAM``, at most half
    of what is still available once the model is loaded (its weights and caches are host memory too), over ``share``
    (a router's models: each server's part) and over the lanes; a non-causal model's none -, ``checkpoints``
    per sequence): a job continues from what it keeps of its prompt (the device's sequence, a checkpoint, a sequence
    saved in RAM) and runs the rest, in parts for a recurrent model's checkpoints (``PromptCache.marks``), with a
    checkpoint at its end; ``cached``: the tokens whose K / V the caches hold (the last sampled token never ran).
    ``on_ready``: called on the worker once the model is loaded (after the warmup); ``on_exit``: when the worker ends
    without being stopped (the load failed)."""

    def __init__(self, load: Callable[[], tuple[Model, NpuModel]], *, warmup: bool = True,
                 on_ready: Callable[[], None] | None = None, on_exit: Callable[[], None] | None = None,
                 cache_ram: int | None = None, share: int = 1, checkpoints: int = 32):
        self.model: Model | None = None
        self.npu: NpuModel | None = None
        self.lanes: list[Lane] = []
        self.error: str | None = None  # why the load failed
        self._load, self._warmup, self._on_ready, self._on_exit = load, warmup, on_ready, on_exit
        self._cache_ram, self._share, self._checkpoints = cache_ram, max(share, 1), checkpoints
        self._queue: collections.deque[Job | Call] = collections.deque()
        self._cv = threading.Condition()
        self._stopping = False
        self._ready = threading.Event()
        self._ids = itertools.count(1)
        self._done = itertools.count(1)  # (``Lane.used``)
        self._thread = threading.Thread(target=self._run, name="artlite-npu")

    @property
    def cache(self) -> PromptCache | None:
        """Lane 0's prompt cache."""
        return self.lanes[0].cache if self.lanes else None

    @property
    def current(self) -> Job | None:
        """A job that runs (the lowest lane's), or None."""
        return next((lane.current for lane in self.lanes if lane.current is not None), None)

    @property
    def cached(self) -> list[int]:
        """The tokens at lane 0's cached positions."""
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
                job.arrived = time.perf_counter()
                job.beside = sum(lane.current is not None for lane in self.lanes) + \
                    sum(isinstance(j, Job) for j in self._queue)
            self._queue.append(job)
            self._cv.notify_all()

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

    def begin(self, lane: Lane | None = None) -> None:
        """(on a lane's worker; default: lane 0's) Before work on the device: the lane's sequence starts over when a
        failure left its state unknown (its own work's, or a dispatch that failed under another lane's:
        ``NpuModel.failed``)."""
        lane = lane or self.lanes[0]
        if lane.npu.failed and not lane.dirty:
            self.recover(lane)
        if lane.dirty:
            lane.npu.recover()
            lane.dirty = False

    def recover(self, lane: Lane | None = None) -> None:
        """(on a lane's worker; default: lane 0's) After a failure, or before work that leaves the device's sequence
        unknown to the prompt cache (speech output: ``models.lfm3a.speech``): the lane's cache keeps what RAM holds of
        the current one, its next work resets its sequence (``begin``)."""
        lane = lane or self.lanes[0]
        lane.cache.clear()
        lane.dirty = True

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
            if sets is not None:  # (a non-causal model's sequences, each attending within itself, whole from position 0)
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

    def answers(self, ids: list[int] | Sequence, texts: list[bytes], *, cache_prompt: bool = True,
                name: str = "decision") -> concurrent.futures.Future:
        """The log-probabilities of a prompt's answers (``texts``: each answer's bytes, ``server.answers.score``), on the
        worker after the queued jobs: the prompt through the prompt cache, the answers scored from its end."""
        return self.run(lambda: self._answers(ids, texts, cache_prompt, name))

    def _answers(self, ids: list[int] | Sequence, texts: list[bytes], cache_prompt: bool, name: str) -> list[float]:
        npu, cache = self.npu, self.cache
        t0 = time.perf_counter()
        v = self.vocab()
        # (the answers' common text with the prompt, as its tokens but the last, which may merge with what follows)
        common = next((i for i in range(min(map(len, texts))) if len({t[i] for t in texts}) > 1), min(map(len, texts)))
        lead = self.model.tokenizer.encode_plain(texts[0][:common].decode("utf-8", "ignore"))[:-1]
        cut = len(b"".join(v.pieces[t] for t in lead))
        ids, texts = ids + lead, [t[cut:] for t in texts]
        try:
            self.begin()
            k, last, _ = cache.prepare(ids, enabled=cache_prompt)
            logits = last if last is not None else npu.forward(ids[k:])
            cache.checkpoint(ids, logits)
            d0 = npu.dispatches
            out = answers.score(npu, logits, texts, v)
            steps = npu.dispatches - d0
            cache.ran(ids)
        except Exception:  # (the device's state is unknown: the caches start over)
            self.recover()
            raise
        log.info(f"{name}: {len(texts)} answers after {len(ids)} tokens ({k} cached), {steps} steps, in "
                 f"{time.perf_counter() - t0:.3f} s")
        return out

    def vocab(self) -> structured.Vocab:
        """The model's tokens for structured output (``structured.vocab``: built once, in the background after the
        load; a request that comes first waits for it). ValueError: a tokenizer without them."""
        tokenizer = self.model.tokenizer
        if not hasattr(tokenizer, "vocabulary"):
            raise ValueError("structured output is not supported by this model's tokenizer")
        cfg = getattr(self.npu, "cfg", None)
        return structured.vocab(tokenizer, int(getattr(cfg, "vocab_size", 0)), sorted(self.model.eos_token_ids))

    def waiting(self) -> int:
        """Jobs in the queue (not running)."""
        return len(self._queue)

    def job_of(self, cid: str) -> Job | None:
        """The running or queued job of a chat completion (its id, ``Job.cid``), or None."""
        with self._cv:
            return next((j for j in (*(lane.current for lane in self.lanes), *self._queue)
                         if isinstance(j, Job) and cid and j.cid == cid), None)

    def cancel(self, conv_id: str) -> int:
        """Cancel the jobs of a conversation (running or queued); returns how many."""
        with self._cv:
            jobs = [j for j in (*(lane.current for lane in self.lanes), *self._queue)
                    if isinstance(j, Job) and j.conv_id == conv_id]
        for j in jobs:
            j.cancel.set()
        return len(jobs)

    def stop(self) -> None:
        """Shut down: cancel the running and the queued jobs, end the workers after the running ones' dispatches."""
        with self._cv:
            self._stopping = True
            for j in (*(lane.current for lane in self.lanes), *self._queue):
                if j is not None:
                    j.cancel.set()
            self._cv.notify_all()

    def join(self, timeout: float | None = None) -> None:
        for thread in (self._thread, *(lane.thread for lane in self.lanes)):
            if thread is not None and thread.is_alive():
                thread.join(timeout)

    # ------------------------------------------------------------------
    # The worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self.model, self.npu = self._load()
            if self._warmup and not self._stopping:
                self._warm()
            turns = [self.model.tokenizer.encode(t) if t else None for t in (self.model.turn, self.model.user_turn)]
            npus = self.npu.lanes
            ram = self._cache_ram
            if not self.npu.causal:  # (it reads every prompt whole: nothing continues from a saved sequence)
                ram = 0
            elif ram is None:
                free = portable.available_memory()
                ram = (CACHE_RAM if free is None else min(CACHE_RAM, free // 2)) // self._share
                log.info("prompt cache: up to %.1f GiB of host RAM (--cache-ram%s%s)", ram / 2**30,
                         f", over --models-max {self._share}" if self._share > 1 else "",
                         f"; a lane's: 1 / {len(npus)} of it" if len(npus) > 1 else "")
            lanes = [Lane(i, npu, PromptCache(npu, ram=ram // len(npus) if ram > 0 else ram, turn=turns[0],
                                              checkpoints=self._checkpoints, user_turn=turns[1]))
                     for i, npu in enumerate(npus)]
        except Exception as e:  # (any failure of the load: reported, the server exits)
            log.exception("loading the model failed")
            self.error = f"{type(e).__name__}: {e}"
            if self._on_exit is not None:
                self._on_exit()
            return
        with self._cv:
            self.lanes = lanes
        self._ready.set()
        if self._on_ready is not None:
            self._on_ready()
        if self._warmup and self.npu.causal and self.npu.generates and hasattr(self.model.tokenizer, "vocabulary"):
            threading.Thread(target=self.vocab, name="artlite-vocab", daemon=True).start()  # (structured output's)
        for lane in lanes[1:]:
            lane.thread = threading.Thread(target=self._serve, args=(lane,), name=f"artlite-npu-{lane.index}")
            lane.thread.start()
        self._serve(lanes[0])

    def _serve(self, lane: Lane) -> None:
        """A lane's worker: the queue's work that goes to it (``_take``), one at a time, until the shutdown."""
        while True:
            with self._cv:
                while (work := self._take(lane)) is None:
                    if self._stopping:
                        return
                    self._cv.wait()
                lane.busy = True
                if isinstance(work, Job):
                    lane.current, work.lane, work.waited = work, lane.index, time.perf_counter() - work.arrived
            try:
                if isinstance(work, Call):
                    self._call(work)
                else:
                    self._job(lane, work)
            finally:
                with self._cv:
                    lane.busy, lane.current, lane.used = False, None, next(self._done)
                    self._cv.notify_all()

    def _take(self, lane: Lane) -> Job | Call | None:
        """(with the lock) The work ``lane`` runs now, taken from the queue: in arrival order each piece goes to a
        free lane - a call to lane 0, a job to ``_lane_of``'s (at the shutdown, when they are cancelled, to any) -,
        and the first that goes to this one is it."""
        free = {x.index for x in self.lanes if not x.busy}
        for i, work in enumerate(self._queue):
            if not free:
                break
            if isinstance(work, Call):
                to = 0 if 0 in free else None
            else:
                to = lane.index if self._stopping else self._lane_of(work, free)
            if to == lane.index:
                del self._queue[i]
                return work
            free.discard(to)
        return None

    def _lane_of(self, job: Job, free: set[int]) -> int | None:
        """The lane of ``free`` a job goes to (module docstring), or None: the lane it names is not among them."""
        if job.lane is not None:
            return job.lane if job.lane in free else None
        if len(free) > 1 and job.cache_prompt:
            keeps = {i: self._keeps(job, self.lanes[i]) for i in free}
            best = max(free, key=lambda i: (keeps[i], -i))
            if keeps[best] >= SIMILAR * len(job.ids) and keeps[best]:
                return best
        return min(free, key=lambda i: (self.lanes[i].used, i))

    @staticmethod
    def _keeps(job: Job, lane: Lane) -> int:
        """The positions of its prompt a job would keep on an idle lane (``PromptCache.keeps``; kept with the job
        while the lane runs nothing)."""
        got = job.keeps.get(lane.index)
        if got is None or got[0] != lane.used:
            got = job.keeps[lane.index] = (lane.used, lane.cache.keeps(job.ids))
        return got[1]

    def _call(self, call: Call) -> None:
        """A call on lane 0's worker, the device held: what it runs on the NPU (lane 0's sequence, other hardware
        contexts: an encoder's) comes between two steps of the other lanes."""
        try:
            if call.cancel.is_set():
                raise RuntimeError("the server is shutting down")
            self.begin()
            with self.npu.device():
                call.future.set_result(call.fn())
        except Exception as e:  # noqa: BLE001 (the caller's: its future)
            call.future.set_exception(e)

    def _job(self, lane: Lane, job: Job) -> None:
        try:
            Generation(self, job, lane).run()
        except Cancelled:
            log.info("%s: cancelled after %d of %d prompt tokens, %d generated", job.name,
                     max(min(lane.npu.position, len(job.ids)) - job.kept, 0), len(job.ids) - job.kept, job.n_gen)
            job.emit("cancelled", {})
        except Exception as e:  # (a failure in the model: the job's error; the lane's caches start over)
            log.exception("%s failed", job.name)
            self.recover(lane)
            job.emit("error", Failed(message=f"{type(e).__name__}: {e}"))
        finally:
            job.state = "done"

    def _warm(self) -> None:
        """A short prompt and one step (as llama-server's warmup): the first request's dispatches find their HW
        contexts and transactions ready (the prompt's tokens through the prefill: a few, Qwen's chunks take 3 at
        least; ``NpuModel.prepare``: the steps of drafts); a non-causal model's and a readout model's: a readout (they
        run no steps)."""
        t0 = time.perf_counter()
        npu = self.npu
        ids = self.model.encode("The quick brown fox jumps over the lazy dog.")
        npu.reset()
        if npu.causal and npu.generates:
            npu.prepare(len(ids))
            logits = npu.forward(ids)
            npu.top_k(argmax(logits), 1)
        else:
            npu.outputs(ids, [-1])
        npu.reset()
        log.info("warmup: %.2f s", time.perf_counter() - t0)
