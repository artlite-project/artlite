"""What the model server's routes share (``ApiContext``): the engine, the settings and the model file's (``Options``,
``ModelInfo``), the request rules - sampling and what a request must satisfy, the prompt's fit, how a request becomes
a job - and the state of the whole server: the fields already warned about, the stored Responses, the spoken
replies, the model's tool vocabulary. The prompts of chat requests: ``prompt.PromptBuilder``."""

import asyncio
import dataclasses
import logging
from typing import Any

from ..sampling import Sampling
from . import agents
from .engine import Engine
from .errors import ApiError, bad_request, not_supported, unavailable
from .jobs import Job
from .prompt import PromptBuilder, system_first, takes_parts
from .schemas import ChatRequest, GenerationRequest
from .settings import PENALTIES, ModelInfo, Options
from .speech import Replies
from .tools import Calls, Vocabulary, syntax_of

log = logging.getLogger("artlite.server")

TOP_K_ALL = 1000  # the candidates of top_k <= 0 (llama.cpp: all tokens; past these the probabilities are negligible)
SAMPLERS = ["penalties", "top_k", "temperature", "top_p", "min_p"]  # in the order Sampling applies them
# sampler fields the NPU path does not have, with their neutral values (others are ignored with a log line)
NEUTRAL: dict[str, object] = {
    "typical_p": 1.0, "typ_p": 1.0, "dry_multiplier": 0.0, "xtc_probability": 0.0, "mirostat": 0,
    "dynatemp_range": 0.0, "top_n_sigma": -1.0, "ignore_eos": False, "logit_bias": None, "grammar": None,
    "json_schema": None, "response_format": None, "thinking_budget_tokens": None, "speculative.n_min": 0,
    "speculative.p_min": 0.0,
}
# llama-server's other sampler settings, as /props reports them (the parameters of samplers that are off)
OTHER_PARAMS = {"dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_n_sigma": -1.0, "xtc_probability": 0.0,
                "xtc_threshold": 0.1, "typical_p": 1.0, "dry_multiplier": 0.0, "dry_base": 1.75,
                "dry_allowed_length": 2, "dry_penalty_last_n": -1, "mirostat": 0, "mirostat_tau": 5.0,
                "mirostat_eta": 0.1, "n_keep": 0, "ignore_eos": False, "n_probs": 0, "min_keep": 0,
                "post_sampling_probs": False, "backend_sampling": False, "lora": []}


class ApiContext:
    """The state of one model's server that its routes share: ``engine``, ``options``, ``info``, the template's
    properties (``syntax``: its tool calls, None: it takes no tools; ``typed``: content as text parts; ``first``: a
    system message only first), ``store`` (the Responses API's, for ``previous_response_id``), ``replies`` (the spoken
    replies: their audio ids in later turns) and ``vocab`` (the model's tokens for the calls' names, built on the first
    request with tools)."""

    def __init__(self, engine: Engine, options: Options, info: ModelInfo):
        self.engine, self.options, self.info = engine, options, info
        self.warned: set[tuple[str, str]] = set()
        self.syntax = syntax_of(info.chat_template)
        self.typed = takes_parts(info.chat_template)
        self.first = system_first(info.chat_template)
        self.store = agents.Store()
        self.replies = Replies()
        self.vocab: list[Vocabulary] = []
        self.vocab_lock = asyncio.Lock()
        self.prompts = PromptBuilder(self)

    def warn_once(self, key: tuple[str, str], message: str, *args: object) -> None:
        """Log a line the first time ``key`` comes up."""
        if key not in self.warned:
            self.warned.add(key)
            log.info(message, *args)

    def ignored(self, route: str) -> agents.Note:
        """A field of ``route``'s requests that the server ignores, logged once."""

        def note(name: str) -> None:
            self.warn_once((route, name), "%s: %s is not supported: ignored", route, name)

        return note

    # ------------------------------------------------------------------
    # Settings of a request
    # ------------------------------------------------------------------

    def default_think(self) -> bool:
        return self.options.reasoning != "off" and self.engine.model.thinks  # (on, auto: when the model can)

    def sampling(self, think: bool) -> Sampling:
        """The defaults of a request: the model's, then the command line's."""
        options = self.options
        s = self.engine.model.sampling(think=think)
        given = {"temp": options.temp, "top_k": options.top_k, "top_p": options.top_p, "min_p": options.min_p,
                 **{k: getattr(options, k) for k in PENALTIES}}
        return dataclasses.replace(s, **{k: v for k, v in given.items() if v is not None})

    def params(self, s: Sampling, drafts: int | None = None) -> dict[str, Any]:
        """llama-server's generation settings (/props, /slots; ``drafts``: a job's draft tokens per step at most)."""
        options, npu = self.options, self.engine.npu
        most = npu.DRAFTS if drafts is None else min(drafts, npu.DRAFTS)
        return {"seed": options.seed, "temperature": 0.0 if s.greedy else s.temp, "top_k": s.top_k, "top_p": s.top_p,
                "min_p": s.min_p, **{k: getattr(s, k) for k in PENALTIES}, **OTHER_PARAMS,
                "max_tokens": options.n_predict, "n_predict": options.n_predict,
                "stream": False, "reasoning_format": options.reasoning_format, "samplers": SAMPLERS,
                "speculative.n_max": most, "timings_per_token": False}

    def check(self, req: GenerationRequest) -> None:
        """What every generation request must satisfy; fields the NPU path ignores are logged."""
        if self.info.decision is not None:  # (as llama-server: its output modalities are only "decisions")
            raise not_supported("This model is a decision model without text generation, use /v1/systemone")
        if req.n not in (None, 1):
            raise bad_request("n > 1 is not supported: one sequence at a time")
        if req.model and req.model != self.options.alias:  # (as llama-server: any)
            self.warn_once(("model", req.model), "requests for model %r: served by %r", req.model, self.options.alias)
        for name, value in (req.model_extra or {}).items():
            neutral = NEUTRAL.get(name, value)
            if name in NEUTRAL and value not in (neutral, None, [], {}, ""):
                self.warn_once((name, repr(value)), "%s = %r is not supported: ignored", name, value)

    def request_sampling(self, req: GenerationRequest, think: bool) -> Sampling:
        s = self.sampling(think)
        given = {"temp": req.temperature, "top_k": req.top_k, "top_p": req.top_p, "min_p": req.min_p,
                 **{k: getattr(req, k) for k in PENALTIES}}
        try:
            s = dataclasses.replace(s, **{k: v for k, v in given.items() if v is not None})
        except ValueError as e:  # (a penalty out of range)
            raise bad_request(str(e)) from None
        if req.temperature is not None and req.temperature > 0:
            s = dataclasses.replace(s, greedy=False)
        return dataclasses.replace(s, top_k=min(s.top_k, TOP_K_ALL) if s.top_k > 0 else TOP_K_ALL)

    @staticmethod
    def stops(req: GenerationRequest) -> list[str]:
        return [req.stop] if isinstance(req.stop, str) else list(req.stop or [])

    def ping_of(self, req: GenerationRequest) -> float:
        return self.options.sse_ping_interval if req.sse_ping_interval is None else req.sse_ping_interval

    def fits(self, ids: list[int], api: str = "chat") -> None:
        """A prompt that fits the KV cache (and its first token); an overflow in the words of ``api``'s clients."""
        capacity = self.engine.npu.capacity
        if not ids:
            raise bad_request("the prompt is empty")
        if len(ids) + 1 > capacity:
            if api == "anthropic":
                raise bad_request(agents.overflow(len(ids), capacity, api))
            if api == "responses":
                raise ApiError(400, "invalid_request_error", agents.overflow(len(ids), capacity, api),
                               code="context_length_exceeded", param="input")
            raise ApiError(400, "exceed_context_size_error", f"request ({len(ids)} tokens) exceeds the available "
                           f"context size ({capacity} tokens), try increasing it", n_prompt_tokens=len(ids),
                           n_ctx=capacity)

    def valid(self, ids: list[int]) -> list[int]:
        """Token ids within the vocabulary."""
        bad = [t for t in ids if not 0 <= t < self.info.n_vocab]
        if bad:
            raise bad_request(f"token id {bad[0]} out of range (the vocabulary has {self.info.n_vocab})")
        return ids

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def job(self, req: GenerationRequest, ids: list[int], *, think: bool, reasoning: bool = False,
            max_tokens: int | None = None, conv_id: str | None = None, tools: Calls | None = None,
            answer_mark: bool = False) -> tuple[Job, asyncio.Queue[tuple[str, dict[str, Any]]]]:
        """A job for the prompt and its queue of events (filled from the worker thread; ``answer_mark``:
        ``Job.answer_mark``)."""
        options = self.options
        loop, events = asyncio.get_running_loop(), asyncio.Queue()

        def emit(kind: str, data: dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(events.put_nowait, (kind, data))
            except RuntimeError:  # (the event loop closed: the server is gone)
                pass

        n = max_tokens if max_tokens is not None else req.max_tokens if req.max_tokens is not None else \
            req.n_predict if req.n_predict is not None else options.n_predict
        if isinstance(req, ChatRequest) and not req.logprobs and req.top_logprobs is not None:
            raise bad_request("top_logprobs requires logprobs to be set to true")
        n_probs = (req.top_logprobs if req.top_logprobs is not None else 20) if getattr(req, "logprobs", None) is True \
            else req.n_probs if req.n_probs is not None else int(getattr(req, "logprobs", None) or 0)
        if n_probs < 0:
            raise bad_request("n_probs (top_logprobs, logprobs) must be >= 0")
        if req.spec_n_max is not None and req.spec_n_max < 0:
            raise bad_request("speculative.n_max must be >= 0")
        j = Job(ids=ids, emit=emit, sampling=self.request_sampling(req, think), max_tokens=n, drafts=req.spec_n_max,
                n_probs=n_probs, post_sampling=bool(req.post_sampling_probs),
                seed=req.seed if req.seed is not None else options.seed, stop=self.stops(req), reasoning=reasoning,
                inside=reasoning and self.prompts.inside(ids), think=think,
                cache_prompt=options.cache_prompt if req.cache_prompt is None else req.cache_prompt,
                progress=bool(req.return_progress), conv_id=conv_id, tools=tools, answer_mark=answer_mark)
        try:
            self.engine.submit(j)
        except RuntimeError as e:
            raise unavailable(str(e)) from None
        return j, events

    async def start(self, req: ChatRequest, *, api: str = "chat", conv_id: str | None = None
                    ) -> tuple[Job, asyncio.Queue[tuple[str, dict[str, Any]]]]:
        """The job of a chat request (of ``api``: its overflow's words) and its queue of events."""
        prompts, options = self.prompts, self.options
        self.check(req)
        if req.continue_final_message:
            raise bad_request("continue_final_message is not supported")
        if "audio" in ((req.model_extra or {}).get("modalities") or []):
            raise not_supported("audio output is not supported by artlite-server")
        fns, calls = prompts.calls_of(req)
        if calls is not None:
            calls = dataclasses.replace(calls, vocab=await prompts.vocabulary())
        ids, think = await prompts.render(req.messages, req.chat_template_kwargs, effort=req.reasoning_effort,
                                          tools=fns)
        self.fits(ids, api)
        reasoning = (req.reasoning_format or options.reasoning_format) != "none"
        # (a template that drops the earlier answers' thoughts renders this answer's opener otherwise next time)
        keeps = prompts.template_options(req.chat_template_kwargs)[1] and "preserve_thinking" in self.info.chat_template
        return self.job(req, ids, think=think, reasoning=reasoning, max_tokens=req.max_completion_tokens,
                        conv_id=conv_id, tools=calls, answer_mark=not keeps)
