"""One job's generation on the engine's worker (``Generation``): the prompt from what the prompt cache keeps (in parts
for a recurrent model's checkpoints), then the tokens, each through the stop strings, the thought split and the tool
call parser into the job's events, and the prompt cache told what the device holds at the end.

A forced tool call (``Calls.force``) puts its opener at the prompt's end, or - after a thought, where one may come
first - after the answer began and no call did: the generation ends where every token but the last ran, the device
rewinds to there and runs the opener."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..common import StreamDecoder, seed
from .jobs import Cancelled, Delta, Done, Job, Progress, prob_entry, timings
from .textflow import ReasoningSplitter, StopMatcher

if TYPE_CHECKING:
    from .engine import Engine

log = logging.getLogger("artlite.server")


class Generation:
    """A job run on ``engine``'s worker: ``run``; its end is an event (``done``; ``Cancelled`` raised: the caller's
    ``cancelled``, any other exception: its ``error``)."""

    def __init__(self, engine: Engine, job: Job):
        self.engine, self.job = engine, job
        self.npu, self.model, self.cache = engine.npu, engine.model, engine.cache
        self.ids = job.ids
        self.gen: list[int] = []  # (the generation's tokens)
        self.finish = "length"
        self.it: Any = None  # (the token iterator of the running step)
        self.t0 = self.t1 = 0.0  # (the start; the prompt's end)
        self.kept = 0  # (the prompt's tokens that were cached)
        self.drafted0 = self.accepted0 = self.d0 = 0
        self.logits: Any = None
        # a forced call: its opener ends the prompt (``early``), or follows the thought where one may come first
        # (``late``)
        self.force: str | None = None
        self.early: list[int] = []
        self.late = ""
        self.mask: Any = None
        self.stop = StopMatcher(job.stop)
        self.split: ReasoningSplitter | None = None
        self.calls: Any = None
        self.text: StreamDecoder | None = None
        self.entries: list[dict[str, Any]] = []  # (llama-server's probs: one per sampled token)
        self.sent = self.sampled = 0
        self.probs: Callable[..., None] | None = None

    # ------------------------------------------------------------------
    # The steps
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the job: the prompt, the tokens, the cache told what the device holds, the ``done`` event."""
        job = self.job
        if job.cancel.is_set():
            raise Cancelled
        self.t0 = time.perf_counter()
        seed(job.seed if job.seed >= 0 else None)
        self.engine.begin()
        self.kept, last, how = self.cache.prepare(self.ids, enabled=job.cache_prompt)
        job.kept = self.kept
        npu = self.npu
        self.drafted0, self.accepted0 = getattr(npu, "drafted", 0), getattr(npu, "accepted", 0)
        self.d0 = npu.dispatches
        self._forced_call()
        job.state = "prompt"
        try:
            self._prompt(last)
            d1, t_npu1, t_host1 = npu.dispatches, npu.t_npu, npu.t_host
            job.state = "generating"
            self._filters()
            self._generate()
        finally:
            self._settle()
        self._report(how, d1, t_npu1, t_host1)

    def _forced_call(self) -> None:
        """The forced call's opener (``Calls.force``): where it goes, and the mask of the calls' names."""
        job, model = self.job, self.model
        tools = job.tools
        self.force = tools.force if tools is not None else None
        thought = job.reasoning and model.reasoning is not None and (job.inside or job.think)
        self.early = model.tokenizer.encode(self.force) if self.force and not thought else []
        self.late = (tools.syntax.gap if job.inside else "") + self.force if self.force and thought else ""
        self.mask = tools.mask(model.tokenizer) if tools is not None else None  # (the calls' names: the functions')

    def _progress(self, i: int) -> None:
        job = self.job
        if job.progress:
            data: Progress = {"total": len(self.ids), "cache": self.kept,
                              "processed": min(self.kept + i, len(self.ids)),
                              "time_ms": (time.perf_counter() - self.t0) * 1e3}
            job.emit("progress", data)
        if job.cancel.is_set():
            raise Cancelled

    def _forced(self, tokens: list[int]) -> None:
        """Tokens the generation did not sample (a forced opener): they go into the generation and the mask."""
        self.gen.extend(tokens)
        for t in tokens if self.mask is not None else ():
            self.mask.accept(t)

    def _prompt(self, last: Any) -> None:
        """The prompt through the NPU (in parts: a checkpoint after each), the logits at its end (a forced opener
        after it)."""
        ids, kept, npu, cache = self.ids, self.kept, self.npu, self.cache
        self._progress(0)
        at = kept
        for m in cache.marks(ids, kept, answer=self.job.answer_mark):  # (the prompt in parts: a checkpoint after each)
            npu.prompt(ids[at:m], progress=lambda i, b=at - kept: self._progress(b + i))
            at = m
            cache.checkpoint(ids[:m])
        if last is not None and not self.early:  # (all of it cached: a checkpoint of its end, with its logits)
            self.logits = last
        else:
            self.logits = npu.forward(ids[at:] + self.early, progress=lambda i, b=at - kept: self._progress(b + i))
            cache.checkpoint(ids + self.early, self.logits)
        self._forced(self.early)
        self.t1 = time.perf_counter()

    def _times(self, t: float) -> dict[str, Any]:
        npu = self.npu
        return timings(cache_n=self.kept, prompt_n=len(self.ids) - self.kept, prompt_s=self.t1 - self.t0,
                       predicted_n=len(self.gen), predicted_s=t - self.t1,
                       drafts=(getattr(npu, "drafted", 0) - self.drafted0,
                               getattr(npu, "accepted", 0) - self.accepted0))

    def _filters(self) -> None:
        """The generated text's filters: the stop strings, the thought split, the tool call parser, and the
        probabilities' entries."""
        job, model = self.job, self.model
        tools = job.tools
        self.text = StreamDecoder(model.tokenizer)
        self.split = ReasoningSplitter(model.reasoning, inside=job.inside, once=model.thought_first,
                                       ends=tools.syntax.ends if tools is not None else ()) \
            if job.reasoning and model.reasoning is not None else None
        self.calls = tools.parser() if tools is not None else None
        post = job.sampling if job.post_sampling else None
        if job.n_probs > 0:
            self.probs = lambda row, t, cap, cand: self.entries.append(
                prob_entry(model.tokenizer, row, t, cap, cand, n=job.n_probs, sampling=post))

    def _parse(self, s: str) -> list[tuple[str, Any]]:
        split, calls = self.split, self.calls
        parts = split.feed(s) if split is not None else [("content", s)] if s else []
        return calls.parts(parts) if calls is not None else parts

    def _send(self, parts: list[tuple[str, Any]]) -> None:
        if parts:
            data: Delta = {"parts": parts, "timings": self._times(time.perf_counter())}
            if self.probs is not None:  # (the sampled tokens up to here, whose text this delta holds)
                data["probs"], self.sent = self.entries[self.sent:self.sampled], self.sampled
            self.job.emit("delta", data)

    def _generate(self) -> None:
        """The tokens, in steps: a step that ends where a forced opener goes is followed by the opener."""
        npu, model, ids, gen = self.npu, self.model, self.ids, self.gen
        logits = self.logits
        self._send(self._parse(self.force) if self.early else [])
        while True:
            forcing = self._tokens(logits)
            if not forcing or self.finish == "stop":
                break
            n = npu.rewind(npu.position)  # (the generation stopped where the next request would continue)
            opener = model.tokenizer.encode(self.late)
            logits = npu.forward((ids + gen)[n:] + opener)
            self._forced(opener)
            self._send(self._parse(self.late))
            self.late = ""
        if self.stop.word is None:
            self._send(self._parse(self.stop.flush()))
        calls = self.calls
        tail = self.split.flush() if self.split is not None else []
        self._send(calls.parts(tail) + calls.flush() if calls is not None else tail)
        if calls is not None and calls.calls and not calls.calling:
            self.finish = "tool_calls"

    def _tokens(self, logits: Any) -> bool:
        """One step of the generation from ``logits``: tokens until the end, a stop, a cancel, or where a forced opener
        goes; whether the opener goes next."""
        job, npu, model, ids, gen, calls = self.job, self.npu, self.model, self.ids, self.gen, self.calls
        left, forcing = job.max_tokens - len(gen) if job.max_tokens >= 0 else -1, False
        self.it = npu.generate(logits, left, job.sampling, ids + gen, self.mask, self.probs, drafts=job.drafts)
        for token in self.it:
            gen.append(token)
            self.sampled += 1
            job.n_gen = len(gen)
            if token in model.eos_token_ids:
                self.finish = "stop"
                break
            s, hit = self.stop.feed(self.text.feed(token))
            self._send(self._parse(s))
            if hit:
                self.finish = "stop"
                break
            if job.cancel.is_set():
                raise Cancelled
            # (the answer began and no call did: the forced call's opener follows)
            forcing = bool(self.late) and self.split is not None and self.split.answering and not calls.started
            # the answer is complete, or the opener goes next: end where every token but the last ran (a
            # step of drafts runs several; the parser drops what follows a call when parallel is off)
            if (forcing or (calls is not None and calls.done)) and npu.position + 1 <= len(ids) + len(gen):
                break
        return forcing

    def _settle(self) -> None:
        """The prompt cache told what the device holds: a step of drafts ran tokens past a stop string or a cancel."""
        npu, ids, gen = self.npu, self.ids, self.gen
        rest: list[int] = []
        while self.it is not None and len(ids) + len(gen) + len(rest) < npu.position:
            if (t := next(self.it, None)) is None:
                break
            rest.append(t)
        self.cache.ran((ids + gen + rest)[:npu.position])

    def _report(self, how: str, d1: int, t_npu1: float, t_host1: float) -> None:
        """The ``done`` event and the log line (artlite-cli's report: the generation's dispatches, the time of each, the
        host's inputs)."""
        job, npu, ids, gen, kept, calls = self.job, self.npu, self.ids, self.gen, self.kept, self.calls
        t2 = time.perf_counter()
        t = self._times(t2)
        done: Done = {"finish_reason": self.finish, "stop_word": self.stop.word, "timings": t, "n_prompt": len(ids),
                      "n_cached": len(self.engine.cached), "n_calls": calls.calls if calls is not None else 0,
                      "probs": self.entries[self.sent:self.sampled]}
        job.emit("done", done)
        passes = max(npu.dispatches - d1, 1)
        host = f", {(npu.t_host - t_host1) / passes * 1e3:.1f} ms of host inputs each" if npu.t_host > t_host1 else ""
        drafts = f"; {t['draft_n']} draft tokens, {t['draft_n_accepted']} accepted" if "draft_n" in t else ""
        how = f" ({how})" if how else ""
        t1 = self.t1
        log.info(f"{job.name}: {kept} cached tokens{how}; prompt: {len(ids) - kept} tokens in {t1 - self.t0:.2f} s "
                 f"({t['prompt_per_second']:.1f} tokens/s, {d1 - self.d0} dispatches); generation: {len(gen)} tokens "
                 f"in {t2 - t1:.2f} s ({t['predicted_per_second']:.2f} tokens/s): {passes} dispatches of "
                 f"{(npu.t_npu - t_npu1) / passes * 1e3:.1f} ms{host}{drafts}; finish: {self.finish}")
