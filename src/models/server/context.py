"""What the model server's routes share (``ApiContext``): the engine, the settings and the model file's (``Options``,
``ModelInfo``), the request rules - sampling and what a request must satisfy, the prompt's fit, how a request becomes
a job - and the state of the whole server: the fields already warned about, the stored Responses, the spoken
replies. The prompts of chat requests: ``prompt.PromptBuilder``."""

import asyncio
import dataclasses
import functools
import logging
from typing import Any

from starlette.concurrency import run_in_threadpool

from .. import structured
from ..sampling import TOP_K_ALL, Sampling
from ..sequence import token_ids
from ..templating import effort_levels, thought_end
from . import agents, call_grammar
from .constraints import constraint_of
from .engine import Engine
from .errors import ApiError, bad_request, not_supported, unavailable
from .jobs import Job
from .prompt import PromptBuilder, system_first, takes_parts
from .schemas import ChatRequest, GenerationRequest, Message
from .settings import PENALTIES, ModelInfo, Options
from .speech import Replies
from .tools import Calls, syntax_of

log = logging.getLogger("artlite.server")

SAMPLERS = ["penalties", "top_k", "temperature", "top_p", "min_p"]  # in the order Sampling applies them
# sampler fields the NPU path does not have, with their neutral values (others are ignored with a log line)
NEUTRAL: dict[str, object] = {
    "typical_p": 1.0, "typ_p": 1.0, "dry_multiplier": 0.0, "xtc_probability": 0.0, "mirostat": 0,
    "dynatemp_range": 0.0, "top_n_sigma": -1.0, "ignore_eos": False, "logit_bias": None,
    "speculative.n_min": 0, "speculative.p_min": 0.0, "guided_decoding_backend": None,
}
# llama-server's other sampler settings, as /props reports them (the parameters of samplers that are off)
OTHER_PARAMS = {"dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_n_sigma": -1.0, "xtc_probability": 0.0,
                "xtc_threshold": 0.1, "typical_p": 1.0, "dry_multiplier": 0.0, "dry_base": 1.75,
                "dry_allowed_length": 2, "dry_penalty_last_n": -1, "mirostat": 0, "mirostat_tau": 5.0,
                "mirostat_eta": 0.1, "n_keep": 0, "ignore_eos": False, "n_probs": 0, "min_keep": 0,
                "post_sampling_probs": False, "backend_sampling": False, "lora": []}


# a thought's tokens at most per ``reasoning_effort`` level (chat's; Responses' ``reasoning.effort``, Anthropic's
# ``output_config.effort``) where the template renders no levels; the others (none: no thought; high, xhigh, max):
# the server's budget
EFFORT_BUDGETS = {"minimal": 0, "low": 1024, "medium": 4096}


class ApiContext:
    """The state of one model's server that its routes share: ``engine``, ``options``, ``info``, the template's
    properties (``syntax``: its tool calls, None: it takes no tools; ``typed``: content as text parts; ``first``: a
    system message only first; ``efforts``: the ``reasoning_effort`` levels it renders distinctly (``none`` among them
    where it renders one of its own: Inkling's effort 0); ``thought_end``: what
    it writes after a thought, ``templating.thought_end``), ``store`` (the Responses API's, for
    ``previous_response_id``), ``replies`` (the spoken replies: their audio ids in later turns)."""

    def __init__(self, engine: Engine, options: Options, info: ModelInfo):
        self.engine, self.options, self.info = engine, options, info
        self.warned: set[tuple[str, str]] = set()
        self.syntax = syntax_of(info.chat_template)
        self.typed = takes_parts(info.chat_template)
        self.first = system_first(info.chat_template)
        self.efforts = effort_levels(info.chat_template) if info.chat_template else []
        self.store = agents.Store()
        self.replies = Replies()
        self.prompts = PromptBuilder(self)

    @functools.cached_property
    def thought_end(self) -> str:
        """What the template writes after a thought (``templating.thought_end``; once the model is loaded)."""
        reasoning = self.engine.model.reasoning
        if reasoning is None:
            return ""
        return thought_end(self.info.chat_template, reasoning[1]) if self.info.chat_template else reasoning[1]

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
        if not self.info.generates:  # (an embedding or rerank model: its NPU image holds no LM head)
            raise not_supported("This model is an embedding / rerank model without text generation, use its routes "
                                "(--embedding, --rerank)")
        if req.n not in (None, 1):
            raise bad_request("n > 1 is not supported: a request gets one answer")
        if req.id_slot is not None and not -1 <= req.id_slot < len(self.engine.lanes):
            raise bad_request(f"id_slot {req.id_slot}: the server has {len(self.engine.lanes)} (-np)")
        if req.model and req.model != self.options.alias:  # (as llama-server: any)
            self.warn_once(("model", req.model), "requests for model %r: served by %r", req.model, self.options.alias)
        for name, value in (req.model_extra or {}).items():
            neutral = NEUTRAL.get(name, value)
            if name in NEUTRAL and value not in (neutral, None, [], {}, ""):
                self.warn_once((name, repr(value)), "%s = %r is not supported: ignored", name, value)

    def constraint(self, req: GenerationRequest, *, route: str = "chat", tools: bool = False
                   ) -> tuple[structured.Constraint, str] | None:
        """The request's structured output and the field it came from (``constraints.constraint_of``; none: the
        server's, ``--grammar``, ``--json-schema``, which a request with ``tools`` takes only as JSON), or None."""
        got = constraint_of(req.constraint_fields(), route=route)
        default = self.options.constraint
        if got is None and default is not None and tools and not default.json:
            self.warn_once(("constraint", "tools"), "--grammar: requests with tools are answered without it (a call "
                           "or free text; a JSON schema would go with them)")
        elif got is None and default is not None:
            got = (default, "--grammar / --json-schema")
        return got

    async def compile(self, got: tuple[structured.Constraint, str] | None, *, calls: Calls | None = None,
                      chat: bool = False) -> structured.Compiled | None:
        """A request's structured output (``constraint``) and its tool calls (``calls``: their grammar,
        ``call_grammar.answer``, with a JSON response format as the answer that is no call), compiled against the
        model's tokens (in a thread: the first request may wait for them); ``chat``: a chat answer's (a model of blocks:
        the answer its text block after its thoughts, ``structured.compiled``'s ``blocks``; a call grammar holds its
        whole turn). A field that does not compile: a 400 that names it."""
        if got is None and calls is None:
            return None
        v = await self.vocab()
        constraint, field = got if got is not None else (None, "tools")
        if calls is not None:
            field = "tools" if got is None else f"tools with {field}"
        blocks = self.engine.model.blocks if chat else None
        try:  # (in a thread: a custom tool's grammar is analysed too, ``lexemes``)
            return await run_in_threadpool(lambda: structured.compiled(
                call_grammar.answer(calls, v, constraint, blocks=blocks) if calls is not None else constraint, v,
                blocks=blocks))
        except structured.ConstraintError as e:
            raise bad_request(f"{field}: {e}") from None

    async def vocab(self) -> structured.Vocab:
        """The model's tokens for structured output (``Engine.vocab``: in a thread)."""
        try:
            return await run_in_threadpool(self.engine.vocab)
        except ValueError as e:
            raise not_supported(str(e)) from None

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
            answer_mark: bool = False, grammar: structured.Compiled | None = None, ban: tuple[int, ...] = (),
            opened: bool | None = None, budget: tuple[int, str, str, bool] = (-1, "", "", False), cid: str = "",
            continued: bool = False, chat: bool = False) -> tuple[Job, asyncio.Queue[tuple[str, dict[str, Any]]]]:
        """A job for the prompt and its queue of events (filled from the worker thread; ``answer_mark``:
        ``Job.answer_mark``; ``grammar``: the answer's structured output, ``compile``; ``ban``: tokens it never
        has; ``opened``: the prompt ends inside a thought, None: where it ends with the open marker; ``budget``: the
        thought's, ``thought_budget``; ``cid``: the chat completion's id; ``continued``, ``chat``: ``Job``'s)."""
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
        opened = self.prompts.inside(ids) if opened is None else opened
        j = Job(ids=ids, emit=emit, sampling=self.request_sampling(req, think), max_tokens=n, drafts=req.spec_n_max,
                n_probs=n_probs, post_sampling=bool(req.post_sampling_probs),
                seed=req.seed if req.seed is not None else options.seed, stop=self.stops(req), reasoning=reasoning,
                inside=reasoning and opened, opened=opened, think=think,
                cache_prompt=options.cache_prompt if req.cache_prompt is None else req.cache_prompt,
                progress=bool(req.return_progress), conv_id=conv_id, tools=tools, answer_mark=answer_mark,
                grammar=grammar, ban=ban, lane=req.id_slot if req.id_slot is not None and req.id_slot >= 0 else None,
                budget=budget[0], budget_message=budget[1], budget_end=budget[2], control=budget[3], cid=cid,
                continued=continued, chat=chat)
        try:
            self.engine.submit(j)
        except RuntimeError as e:
            raise unavailable(str(e)) from None
        return j, events

    def thought_budget(self, req: ChatRequest) -> tuple[int, str, str, bool]:
        """llama-server's reasoning budget of a chat request: a thought's tokens at most (its
        ``reasoning_budget_tokens``, else ``thinking_budget_tokens``, the web UI's and Anthropic's; -1 or none: its
        ``reasoning_effort``'s, ``EFFORT_BUDGETS``, where the template renders no levels of its own - Qwen3.8's and
        Spark-X's keep theirs -, else ``--reasoning-budget``; < 0: any), the message that ends it early (``reasoning_budget_message``, else ``--reasoning-budget-message``;
        ``Job.thought_end``: on a paragraph of its own), the template's own end of a thought after it (``thought_end``:
        Qwen's ``"\\n</think>\\n\\n"``, so that the next request, which renders the cut thought again, continues
        the cached tokens; llama-server forces the message and the close marker alone) and whether the control route
        may end it (``reasoning_control``)."""
        model = self.engine.model
        n = req.reasoning_budget_tokens if req.reasoning_budget_tokens is not None else req.thinking_budget_tokens
        if n is not None and n < -1:
            raise bad_request("reasoning_budget_tokens: -1 (no budget) or a number of tokens")
        if model.reasoning is None:
            return -1, "", "", False
        if n is None or n == -1:
            n = EFFORT_BUDGETS.get(req.reasoning_effort or "") if not set(self.efforts) - {"none"} else None
            n = self.options.reasoning_budget if n is None else n
        message = req.reasoning_budget_message if req.reasoning_budget_message is not None else \
            self.options.reasoning_budget_message
        return n, message.strip(), self.thought_end, bool(req.reasoning_control)

    async def start(self, req: ChatRequest, *, api: str = "chat", conv_id: str | None = None, cid: str = ""
                    ) -> tuple[Job, asyncio.Queue[tuple[str, dict[str, Any]]]]:
        """The job of a chat request (of ``api``: its overflow's words; ``cid``: the chat completion's id) and its
        queue of events. A request that continues its last message (``PromptBuilder.continuation``) gets the new text
        alone."""
        prompts, options = self.prompts, self.options
        self.check(req)
        cont = prompts.continuation(req)
        if "audio" in ((req.model_extra or {}).get("modalities") or []):
            raise not_supported("audio output is not supported by artlite-server")
        fns, calls = prompts.calls_of(req)
        got = self.constraint(req, tools=calls is not None)
        grammar = await self.compile(got, calls=calls, chat=True)
        ban: tuple[int, ...] = ()
        if fns and calls is None and grammar is None and self.syntax is not None:  # (tool_choice none: no call text)
            blocks = self.engine.model.blocks
            if blocks is not None:  # (a turn of thought and text blocks alone)
                v = await self.vocab()
                grammar = await run_in_threadpool(lambda: structured.compiled(call_grammar.no_calls(blocks, v), v))
            else:  # (an opener of one token: Qwen's, Gemma's; Llama's: none)
                opener = self.engine.model.tokenizer.encode(getattr(self.syntax.parser_class, "OPEN", ""))
                ban = tuple(opener) if len(opener) == 1 else ()
        ids, think = await prompts.render(req.messages, req.chat_template_kwargs, effort=req.reasoning_effort,
                                          tools=fns, cont=cont)
        budget = self.thought_budget(req)
        if budget[:2] == (0, "") and not budget[3] and cont is None and prompts.inside(ids):
            # a thought of no token and no message: the template's prompt without a thought where the next request
            # renders this answer's empty thought in its tokens (Qwen's "<think>\n\n</think>\n\n"; the end forced
            # after the opener's newline would be other tokens), else the end forced after the opener
            off, _ = await prompts.render(req.messages, {**(req.chat_template_kwargs or {}), "enable_thinking": False},
                                          effort=req.reasoning_effort, tools=fns)
            empty = Message(role="assistant", content="", reasoning_content="")
            echo, _ = await prompts.render([*req.messages, empty], req.chat_template_kwargs,
                                           effort=req.reasoning_effort, tools=fns, cont="content")
            if token_ids(off) == token_ids(echo):
                ids, think, budget = off, False, (-1, "", "", False)
        self.fits(ids, api)
        # a constraint of the whole answer (not a structural tag's free text; beside tools, the JSON that is no call):
        # the content is that answer alone, the thought split off where reasoning_format none leaves it in the content
        whole = got is not None and got[0].kind != "structural_tag" and (calls is None or calls.choice == "auto")
        reasoning = whole or (req.reasoning_format or options.reasoning_format) != "none"
        # (a template that drops the earlier answers' thoughts renders this answer's opener otherwise next time)
        keeps = prompts.template_options(req.chat_template_kwargs)[1] and "preserve_thinking" in self.info.chat_template
        return self.job(req, ids, think=think, reasoning=reasoning, max_tokens=req.max_completion_tokens,
                        conv_id=conv_id, tools=calls, answer_mark=not keeps, grammar=grammar, ban=ban,
                        opened=True if cont == "reasoning_content" else None, budget=budget, cid=cid,
                        continued=cont is not None, chat=True)
