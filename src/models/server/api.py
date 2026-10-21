"""The server's HTTP side (FastAPI): llama-server's routes that apply to one model on the NPU, their request and
response shapes, server-sent events, the API keys, CORS and OpenAI's error shape. The generations run on the engine's
worker (``engine.Engine``); a route renders the prompt (in a thread: a long prompt's tokenizer takes a while), checks
that it fits the KV cache, queues a job and turns its events into the response."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import secrets
import time
import urllib.parse
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import jinja2
from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Receive, Scope, Send

from ..common import MEDIA_MARKER, Sampling
from .engine import Engine, Job

log = logging.getLogger("artlite.server")

TOP_K_ALL = 1000  # the candidates of top_k <= 0 (llama.cpp: all tokens; past these the probabilities are negligible)
SAMPLERS = ["penalties", "top_k", "temperature", "top_p", "min_p"]  # in the order Sampling applies them
PENALTIES = ("repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty")  # (Sampling's, Options')
# sampler fields the NPU path does not have, with their neutral values (others are ignored with a log line)
NEUTRAL: dict[str, object] = {
    "typical_p": 1.0, "typ_p": 1.0, "dry_multiplier": 0.0, "xtc_probability": 0.0, "mirostat": 0,
    "dynatemp_range": 0.0, "top_n_sigma": -1.0, "n_probs": 0, "logprobs": False, "top_logprobs": 0,
    "ignore_eos": False, "logit_bias": None, "grammar": None, "json_schema": None, "response_format": None,
    "tools": None, "thinking_budget_tokens": None,
}
# llama-server's other sampler settings, as /props reports them (the parameters of samplers that are off)
OTHER_PARAMS = {"dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_n_sigma": -1.0, "xtc_probability": 0.0,
                "xtc_threshold": 0.1, "typical_p": 1.0, "dry_multiplier": 0.0, "dry_base": 1.75,
                "dry_allowed_length": 2, "dry_penalty_last_n": -1, "mirostat": 0, "mirostat_tau": 5.0,
                "mirostat_eta": 0.1, "n_keep": 0, "ignore_eos": False, "n_probs": 0, "min_keep": 0,
                "post_sampling_probs": False, "backend_sampling": False, "lora": []}
ROLES = ("system", "developer", "user", "assistant", "tool")


@dataclass
class Options:
    """The server's settings (the command line, ``main``)."""

    alias: str  # the model's id in the API
    n_ctx: int  # the KV cache's positions
    api_keys: frozenset[str] = frozenset()  # none: no key needed
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    n_predict: int = -1  # a request's default max_tokens (< 0: no limit)
    temp: float | None = None  # sampling over the model's defaults (None: the model's)
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int = -1  # a request's default seed (< 0: random)
    reasoning: str = "off"  # think: on, off, auto (when the model can)
    reasoning_format: str = "auto"  # auto: thoughts in reasoning_content; none: in content
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)  # under each request's
    cache_prompt: bool = True
    sse_ping_interval: float = 30.0  # seconds between pings while a stream waits (<= 0: none)
    slots: bool = True  # GET /slots
    ui: Callable[[FastAPI], None] | None = None  # serves the web UI's files (``ui.mount``); None: none
    no_ui: str = "The web UI is off: the API is at /v1."  # what / shows without one
    build_info: str = "Artlite"


@dataclass(frozen=True)
class ModelInfo:
    """What the API reports of the model file (its GGUF header)."""

    path: str
    chat_template: str = ""
    bos_token: str = ""
    eos_token: str = ""
    n_ctx_train: int = 0
    n_embd: int = 0
    n_vocab: int = 0
    n_params: int = 0
    size: int = 0
    ftype: str = ""

    @classmethod
    def from_gguf(cls, path: str) -> ModelInfo:
        from artlite.gguf_model import GGUFFile

        from ..bench import file_info, file_type_name

        g = GGUFFile(path)
        meta, arch = g.meta, str(g.arch)
        fields, n_vocab = file_info(path, gguf=g)
        ids = [int(meta[k]) for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id") if k in meta]
        names = json.loads(g.strings_json("tokenizer.ggml.tokens", ids)) if ids else []
        special = dict(zip([k for k in ("bos", "eos") if f"tokenizer.ggml.{k}_token_id" in meta], names, strict=True))
        return cls(path=path, chat_template=str(meta.get("tokenizer.chat_template", "")),
                   bos_token=special.get("bos", ""), eos_token=special.get("eos", ""),
                   n_ctx_train=int(meta.get(f"{arch}.context_length", 0)),
                   n_embd=int(meta.get(f"{arch}.embedding_length", 0)), n_vocab=n_vocab,
                   n_params=int(fields["model_n_params"]), size=int(fields["model_size"]),
                   ftype=file_type_name(meta.get("general.file_type")))


class ApiError(Exception):
    """An error response in OpenAI's shape: ``{"error": {"code", "message", "type", ...}}`` (``extra``: more fields,
    e.g. a context overflow's ``n_prompt_tokens`` and ``n_ctx``)."""

    def __init__(self, status: int, kind: str, message: str, **extra: object):
        super().__init__(message)
        self.status, self.kind, self.message, self.extra = status, kind, message, extra

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.status, "message": self.message, "type": self.kind, **self.extra}}


def image_bytes(url: object) -> bytes:
    """An ``image_url`` part's image: a data URI's bytes (base64 or percent-encoded)."""
    u = url.get("url") if isinstance(url, dict) else url
    if not isinstance(u, str) or not u.startswith("data:") or "," not in u:
        raise bad_request("an image_url must be a data URI (data:image/...;base64,...)")
    head, data = u.split(",", 1)
    try:
        return base64.b64decode(data, validate=True) if head.endswith(";base64") else urllib.parse.unquote_to_bytes(data)
    except ValueError:
        raise bad_request("an image_url's base64 data does not decode") from None


def bad_request(message: str) -> ApiError:
    return ApiError(400, "invalid_request_error", message)


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------


class Message(BaseModel):
    """A chat message: its other fields (tool_calls, name, ...) go to the template as they are."""

    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


class Generation(BaseModel):
    """The fields that every generation request shares (the others are accepted: clients send many)."""

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    stream: bool | None = False
    max_tokens: int | None = None
    n_predict: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int | None = 1
    cache_prompt: bool | None = None
    return_progress: bool | None = False
    timings_per_token: bool | None = False
    sse_ping_interval: float | None = None
    stream_options: dict[str, Any] | None = None


class ChatRequest(Generation):
    messages: list[Message]
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning_format: str | None = None
    continue_final_message: bool | None = False


class CompletionRequest(Generation):
    prompt: str | list[int] | list[str]


class TokenizeRequest(BaseModel):
    content: str
    add_special: bool = False
    with_pieces: bool = False


class DetokenizeRequest(BaseModel):
    tokens: list[int]


class TemplateRequest(BaseModel):
    messages: list[Message]
    chat_template_kwargs: dict[str, Any] | None = None


class EventStream(StreamingResponse):
    """A stream of server-sent events whose job is cancelled when the response ends early (its client went away)."""

    def __init__(self, events: AsyncIterator[str], job: Job):
        super().__init__(events, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        self.job = job

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job.cancel.set()


def sse(data: object) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------
# What the model's server and the router share
# ----------------------------------------------------------------------


def new_app(*, build_info: str, cors_origins: list[str], lifespan: Callable | None = None) -> FastAPI:
    """A FastAPI app whose errors all answer in OpenAI's shape, with CORS."""
    app = FastAPI(title="artlite-server", version=build_info, docs_url="/docs", redoc_url=None, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ApiError)
    async def api_error(request: Request, e: ApiError) -> JSONResponse:
        return JSONResponse(e.body(), status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, e: RequestValidationError) -> JSONResponse:
        errors = "; ".join(f"{'.'.join(str(x) for x in err['loc'][1:])}: {err['msg']}" for err in e.errors())
        return JSONResponse(bad_request(errors or "invalid request").body(), status_code=400)

    @app.exception_handler(Exception)
    async def failure(request: Request, e: Exception) -> JSONResponse:
        log.exception("%s %s failed", request.method, request.url.path)
        return JSONResponse(ApiError(500, "server_error", f"{type(e).__name__}: {e}").body(), status_code=500)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, e: StarletteHTTPException) -> JSONResponse:
        kind = {401: "authentication_error", 403: "permission_error", 404: "not_found_error"}.get(e.status_code,
                                                                                                 "invalid_request_error")
        return JSONResponse(ApiError(e.status_code, kind, str(e.detail)).body(), status_code=e.status_code)

    return app


def key_check(api_keys: frozenset[str]) -> Callable[..., Awaitable[None]]:
    """The dependency of the routes that need an API key (``Authorization: Bearer`` or ``X-Api-Key``), when the
    server has keys."""

    async def authorized(authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> None:
        if not api_keys:
            return
        key = x_api_key or (authorization[7:].strip() if authorization and authorization.startswith("Bearer ")
                            else "")
        if not any(secrets.compare_digest(key.encode(), k.encode()) for k in api_keys):
            raise ApiError(401, "authentication_error", "Invalid API Key")

    return authorized


NOT_SUPPORTED = ["/v1/chat/completions/control", "/v1/responses", "/responses", "/v1/messages",
                 "/v1/messages/count_tokens", "/v1/embeddings", "/embeddings", "/embedding", "/rerank", "/reranking",
                 "/v1/rerank", "/v1/reranking", "/infill", "/v1/audio/transcriptions", "/audio/transcriptions",
                 "/lora-adapters", "/metrics"]


def add_unsupported(router: APIRouter) -> None:
    """llama-server's routes that Artlite does not have (501), and ``/tools``: 403, which the web UI reads as its server
    tools being off."""

    @router.get("/tools")
    @router.post("/tools")
    async def tools() -> JSONResponse:
        return JSONResponse({"error": {"message": "this feature is disabled", "type": "feature_disabled"}},
                            status_code=403)

    async def unsupported(request: Request) -> None:
        raise ApiError(501, "not_supported_error", f"{request.url.path} is not supported by artlite-server")

    for path in NOT_SUPPORTED:
        router.add_api_route(path, unsupported, methods=["GET", "POST"], include_in_schema=False)


def add_ui(app: FastAPI, ui: Callable[[FastAPI], None] | None, no_ui: str) -> None:
    """The web UI's files at ``/`` (after the app's routes), or a page that says why there is none."""
    if ui is not None:
        ui(app)
        return

    @app.get("/", include_in_schema=False)
    async def page() -> HTMLResponse:
        return HTMLResponse(f"<!doctype html><title>artlite-server</title><p>{no_ui}</p>")


# ----------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------


def create_app(engine: Engine, options: Options, info: ModelInfo) -> FastAPI:
    """The FastAPI app of the server: the routes on ``engine`` with ``options``; ``info``: the model file's."""
    app = new_app(build_info=options.build_info, cors_origins=options.cors_origins)
    authorized = key_check(options.api_keys)
    warned: set[tuple[str, str]] = set()

    async def loaded() -> None:
        if not engine.ready:
            raise ApiError(503, "unavailable_error", "Loading model")

    public = APIRouter()  # no key, any state
    listed = APIRouter(dependencies=[Depends(authorized)])  # also while the model loads
    api = APIRouter(dependencies=[Depends(authorized), Depends(loaded)])

    # ------------------------------------------------------------------
    # Settings of a request
    # ------------------------------------------------------------------

    def default_think() -> bool:
        return options.reasoning != "off" and engine.model.thinks  # (on, auto: when the model can)

    def sampling(think: bool) -> Sampling:
        """The defaults of a request: the model's, then the command line's."""
        s = engine.model.sampling(think=think)
        given = {"temp": options.temp, "top_k": options.top_k, "top_p": options.top_p, "min_p": options.min_p,
                 **{k: getattr(options, k) for k in PENALTIES}}
        return dataclasses.replace(s, **{k: v for k, v in given.items() if v is not None})

    def params(s: Sampling) -> dict[str, Any]:
        """llama-server's generation settings (/props, /slots)."""
        return {"seed": options.seed, "temperature": 0.0 if s.greedy else s.temp, "top_k": s.top_k, "top_p": s.top_p,
                "min_p": s.min_p, **{k: getattr(s, k) for k in PENALTIES}, **OTHER_PARAMS,
                "max_tokens": options.n_predict, "n_predict": options.n_predict,
                "stream": False, "reasoning_format": options.reasoning_format, "samplers": SAMPLERS,
                "timings_per_token": False}

    def check(req: Generation) -> None:
        """What every generation request must satisfy; fields the NPU path ignores are logged."""
        if req.n not in (None, 1):
            raise bad_request("n > 1 is not supported: one sequence at a time")
        if req.model and req.model != options.alias and ("model", req.model) not in warned:  # (as llama-server: any)
            warned.add(("model", req.model))
            log.info("requests for model %r: served by %r", req.model, options.alias)
        for name, value in (req.model_extra or {}).items():
            neutral = NEUTRAL.get(name, value)
            if name in NEUTRAL and value not in (neutral, None, [], {}, "") and (name, repr(value)) not in warned:
                warned.add((name, repr(value)))
                log.info("%s = %r is not supported: ignored", name, value)

    def request_sampling(req: Generation, think: bool) -> Sampling:
        s = sampling(think)
        given = {"temp": req.temperature, "top_k": req.top_k, "top_p": req.top_p, "min_p": req.min_p,
                 **{k: getattr(req, k) for k in PENALTIES}}
        try:
            s = dataclasses.replace(s, **{k: v for k, v in given.items() if v is not None})
        except ValueError as e:  # (a penalty out of range)
            raise bad_request(str(e)) from None
        if req.temperature is not None and req.temperature > 0:
            s = dataclasses.replace(s, greedy=False)
        return dataclasses.replace(s, top_k=min(s.top_k, TOP_K_ALL) if s.top_k > 0 else TOP_K_ALL)

    def stops(req: Generation) -> list[str]:
        return [req.stop] if isinstance(req.stop, str) else list(req.stop or [])

    def fits(ids: list[int]) -> None:
        capacity = engine.npu.capacity
        if not ids:
            raise bad_request("the prompt is empty")
        if len(ids) + 1 > capacity:
            raise ApiError(400, "exceed_context_size_error", f"request ({len(ids)} tokens) exceeds the available "
                           f"context size ({capacity} tokens), try increasing it", n_prompt_tokens=len(ids),
                           n_ctx=capacity)

    def messages_of(messages: list[Message], *, media: bool = False) -> list[dict[str, object]]:
        """The messages for the chat template: text content (parts joined by newlines), with images (``image_url``
        parts of data URIs, a model with a vision encoder) as parts ``{"type": "image", "image": bytes}`` among the
        text parts; ``media``: images as the text ``MEDIA_MARKER`` (the template's text alone)."""
        out = []
        vision = engine.model.vision is not None
        for m in messages:
            if m.role not in ROLES:
                raise bad_request(f"message role {m.role!r} (one of {', '.join(ROLES)})")
            content = m.content
            if isinstance(content, list):
                parts: list[dict[str, object]] = []
                for part in content:
                    kind = part.get("type")
                    if kind == "text":
                        parts.append({"type": "text", "text": str(part.get("text", ""))})
                    elif kind == "image_url" and vision:
                        parts.append({"type": "text", "text": MEDIA_MARKER} if media else
                                     {"type": "image", "image": image_bytes(part.get("image_url"))})
                    else:
                        raise bad_request(f"content of type {kind!r} is not supported: this server's model takes "
                                          f"text{' and images (image_url)' if vision else ' only'}")
                content = parts if any(p["type"] == "image" for p in parts) else "\n".join(p["text"] for p in parts)
            d: dict[str, object] = {**(m.model_extra or {}), "role": m.role, "content": content or ""}
            if m.reasoning_content:
                d["reasoning_content"] = m.reasoning_content
            out.append(d)
        if not out:
            raise bad_request("no messages")
        return out

    def thinking(kwargs: dict[str, Any] | None) -> bool:
        kw = {**options.chat_template_kwargs, **(kwargs or {})}
        for k in kw.keys() - {"enable_thinking"}:
            if ("chat_template_kwargs", k) not in warned:
                warned.add(("chat_template_kwargs", k))
                log.info("chat_template_kwargs.%s is not supported: ignored", k)
        think = bool(kw["enable_thinking"]) if "enable_thinking" in kw else default_think()
        return think and engine.model.thinks

    async def render(messages: list[Message], kwargs: dict[str, Any] | None, *, media: bool = False
                     ) -> tuple[list[int], bool]:
        """The prompt's tokens through the chat template (with images: ids and embedding rows, the encoder run on the
        engine's worker between its jobs; ``media``: images as text, ``messages_of``), and whether the model thinks."""
        think = thinking(kwargs)
        msgs = messages_of(messages, media=media)
        try:
            if any(isinstance(m["content"], list) for m in msgs):
                return await asyncio.wrap_future(engine.run(lambda: engine.model.chat(msgs, think=think))), think
            return await run_in_threadpool(engine.model.chat, msgs, think=think), think
        except jinja2.TemplateError as e:
            raise bad_request(f"the chat template failed: {e}") from None
        except (OSError, ValueError) as e:  # (an image that does not open, or does not fit the prompt)
            raise bad_request(str(e)) from None

    def inside(ids: list[int]) -> bool:
        """Whether a prompt ends inside a thought (its last text is the model's open marker)."""
        markers = engine.model.reasoning
        if markers is None:
            return False
        last = [int(t) for t in getattr(ids, "ids", ids)[-8:] if t >= 0]  # (a sequence's ids; an image's rows: none)
        tail = engine.model.tokenizer.decode_bytes(last, special=True).decode("utf-8", errors="replace")
        return tail.rstrip().endswith(markers[0])

    def job(req: Generation, ids: list[int], *, think: bool, reasoning: bool = False, max_tokens: int | None = None,
            conv_id: str | None = None) -> tuple[Job, asyncio.Queue[tuple[str, dict[str, Any]]]]:
        """A job for the prompt and its queue of events (filled from the worker thread)."""
        loop, events = asyncio.get_running_loop(), asyncio.Queue()

        def emit(kind: str, data: dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(events.put_nowait, (kind, data))
            except RuntimeError:  # (the event loop closed: the server is gone)
                pass

        n = max_tokens if max_tokens is not None else req.max_tokens if req.max_tokens is not None else \
            req.n_predict if req.n_predict is not None else options.n_predict
        j = Job(ids=ids, emit=emit, sampling=request_sampling(req, think), max_tokens=n,
                seed=req.seed if req.seed is not None else options.seed, stop=stops(req), reasoning=reasoning,
                inside=reasoning and inside(ids),
                cache_prompt=options.cache_prompt if req.cache_prompt is None else req.cache_prompt,
                progress=bool(req.return_progress), conv_id=conv_id)
        try:
            engine.submit(j)
        except RuntimeError as e:
            raise ApiError(503, "unavailable_error", str(e)) from None
        return j, events

    async def events_of(events: asyncio.Queue, ping: float) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """A job's events up to its last, with ("ping", {}) after every ``ping`` seconds without one."""
        while True:
            try:
                kind, data = await asyncio.wait_for(events.get(), ping if ping > 0 else None)
            except TimeoutError:
                yield "ping", {}
                continue
            yield kind, data
            if kind in ("done", "error", "cancelled"):
                return

    async def collect(j: Job, events: asyncio.Queue, request: Request) -> tuple[dict[str, str], dict[str, Any]]:
        """A job's text by kind and its last event (a response without streaming); the client going away cancels
        it."""
        texts: dict[str, list[str]] = {"content": [], "reasoning_content": []}
        loop = asyncio.get_running_loop()
        check = loop.time()
        try:
            async for kind, data in events_of(events, 0.5):
                if loop.time() >= check:  # (every half second)
                    check = loop.time() + 0.5
                    if await request.is_disconnected():
                        j.cancel.set()
                if kind == "delta":
                    for k, t in data["parts"]:
                        texts[k].append(t)
                elif kind == "done":
                    return {k: "".join(v) for k, v in texts.items()}, data
                elif kind == "error":
                    raise ApiError(500, "server_error", data["message"])
                elif kind == "cancelled":
                    raise ApiError(503, "unavailable_error", "the request was cancelled")
        finally:
            j.cancel.set()  # (nothing when it ended; the handler cancelled: shutdown)
        raise AssertionError("unreachable")

    def stream_error(kind: str, data: dict[str, Any]) -> str:
        e = ApiError(500, "server_error", data["message"]) if kind == "error" else \
            ApiError(503, "unavailable_error", "the request was cancelled")
        return sse(e.body())

    def ping_of(req: Generation) -> float:
        return options.sse_ping_interval if req.sse_ping_interval is None else req.sse_ping_interval

    def usage(done: dict[str, Any]) -> dict[str, Any]:
        t = done["timings"]
        return {"completion_tokens": t["predicted_n"], "prompt_tokens": done["n_prompt"],
                "total_tokens": t["predicted_n"] + done["n_prompt"],
                "prompt_tokens_details": {"cached_tokens": t["cache_n"]}}

    # ------------------------------------------------------------------
    # Status and information
    # ------------------------------------------------------------------

    @public.get("/health")
    @public.get("/v1/health")
    async def health() -> JSONResponse:
        if engine.ready:
            return JSONResponse({"status": "ok"})
        return JSONResponse(ApiError(503, "unavailable_error", "Loading model").body(), status_code=503)

    @api.get("/props")
    async def props() -> dict[str, Any]:
        return {"default_generation_settings": {"n_ctx": engine.npu.capacity, "params": params(sampling(
                    default_think()))},
                "total_slots": 1, "model_alias": options.alias, "model_path": info.path,
                "modalities": {"vision": engine.model.vision is not None, "audio": False, "video": False},
                "endpoint_slots": options.slots, "endpoint_props": False, "endpoint_metrics": False,
                "ui": options.ui is not None, "ui_settings": {},
                "chat_template": info.chat_template, "bos_token": info.bos_token, "eos_token": info.eos_token,
                "build_info": options.build_info, "is_sleeping": False, "cors_proxy_enabled": False}

    @listed.get("/v1/models")
    @listed.get("/models")
    async def models() -> dict[str, Any]:
        name, vision = options.alias, engine.model is not None and engine.model.vision is not None
        return {"object": "list",
                "models": [{"name": name, "model": name, "modified_at": "", "size": "", "digest": "", "type": "model",
                            "description": "", "tags": [""], "capabilities": ["completion"] + ["multimodal"] * vision,
                            "parameters": "",
                            "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""],
                                        "parameter_size": "", "quantization_level": ""}}],
                "data": [{"id": name, "aliases": [], "tags": [], "object": "model",
                          "architecture": {"input_modalities": ["text"] + ["image"] * vision,
                                           "output_modalities": ["text"]},
                          "created": int(time.time()), "owned_by": "artlite",
                          "meta": {"n_vocab": info.n_vocab, "n_ctx": options.n_ctx, "n_ctx_train": info.n_ctx_train,
                                   "n_embd": info.n_embd, "n_params": info.n_params, "size": info.size,
                                   "ftype": info.ftype}}]}

    @api.get("/slots")
    async def slots() -> list[dict[str, Any]]:
        if not options.slots:
            raise ApiError(501, "not_supported_error", "This server does not support slots endpoint.")
        j = engine.current
        return [{"id": 0, "id_task": -1 if j is None else 0, "n_ctx": engine.npu.capacity,
                 "is_processing": j is not None, "n_cached": len(engine.cached), "n_waiting": engine.waiting(),
                 "params": params(sampling(default_think()) if j is None else j.sampling),
                 "next_token": [{"has_next_token": j is not None and j.state == "generating",
                                 "n_decoded": 0 if j is None else j.n_gen}]}]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @api.post("/v1/chat/completions")
    @api.post("/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request,
                               x_conversation_id: str | None = Header(None)) -> Any:
        check(req)
        if req.continue_final_message:
            raise bad_request("continue_final_message is not supported")
        ids, think = await render(req.messages, req.chat_template_kwargs)
        fits(ids)
        reasoning = (req.reasoning_format or options.reasoning_format) != "none"
        max_tokens = req.max_completion_tokens if req.max_completion_tokens is not None else None
        j, events = job(req, ids, think=think, reasoning=reasoning, max_tokens=max_tokens, conv_id=x_conversation_id)
        cid, created, name = f"chatcmpl-{secrets.token_hex(16)}", int(time.time()), options.alias
        head = {"created": created, "id": cid, "model": name, "system_fingerprint": options.build_info}
        if not req.stream:
            texts, done = await collect(j, events, request)
            message = {"role": "assistant", "content": texts["content"]}
            if texts["reasoning_content"]:
                message["reasoning_content"] = texts["reasoning_content"]
            return {"choices": [{"finish_reason": done["finish_reason"], "index": 0, "message": message}], **head,
                    "object": "chat.completion", "usage": usage(done), "timings": done["timings"]}
        include_usage = bool((req.stream_options or {}).get("include_usage"))

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: object) -> dict[str, Any]:
            return {"choices": [{"finish_reason": finish, "index": 0, "delta": delta}], **head,
                    "object": "chat.completion.chunk", **extra}

        async def stream() -> AsyncIterator[str]:
            yield sse(chunk({"role": "assistant", "content": None}))
            async for kind, data in events_of(events, ping_of(req)):
                if kind == "ping":
                    yield ":\n\n"
                elif kind == "progress":
                    yield sse(chunk({"role": "assistant", "content": None}, prompt_progress=data))
                elif kind == "delta":
                    parts = data["parts"]
                    for i, (k, text) in enumerate(parts):
                        last = req.timings_per_token and i == len(parts) - 1
                        yield sse(chunk({k: text}, **({"timings": data["timings"]} if last else {})))
                elif kind == "done":
                    end = [chunk({}, data["finish_reason"])]
                    if include_usage:
                        end.append({"choices": [], **head, "object": "chat.completion.chunk", "usage": usage(data)})
                    end[-1]["timings"] = data["timings"]
                    for c in end:
                        yield sse(c)
                    yield "data: [DONE]\n\n"
                else:
                    yield stream_error(kind, data)

        return EventStream(stream(), j)

    def valid(ids: list[int]) -> list[int]:
        """Token ids within the vocabulary."""
        bad = [t for t in ids if not 0 <= t < info.n_vocab]
        if bad:
            raise bad_request(f"token id {bad[0]} out of range (the vocabulary has {info.n_vocab})")
        return ids

    def prompt_ids(prompt: str | list[int] | list[str]) -> list[int]:
        """A raw prompt's tokens: a text (after the model's BOS), or token ids."""
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return valid(list(prompt))
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise bad_request("one prompt at a time")
            prompt = prompt[0]
        return engine.model.encode(prompt)

    @api.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Any:
        """OpenAI's completions: a raw prompt."""
        check(req)
        ids = await run_in_threadpool(prompt_ids, req.prompt)
        fits(ids)
        j, events = job(req, ids, think=False)
        cid, created, name = f"cmpl-{secrets.token_hex(16)}", int(time.time()), options.alias
        head = {"created": created, "model": name, "system_fingerprint": options.build_info,
                "object": "text_completion", "id": cid}

        def body(text: str, finish: str | None, **extra: object) -> dict[str, Any]:
            return {"choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}], **head,
                    **extra}

        if not req.stream:
            texts, done = await collect(j, events, request)
            return body(texts["content"], done["finish_reason"], usage=usage(done), timings=done["timings"])

        async def stream() -> AsyncIterator[str]:
            async for kind, data in events_of(events, ping_of(req)):
                if kind == "ping":
                    yield ":\n\n"
                elif kind == "progress":
                    yield sse(body("", None, prompt_progress=data))
                elif kind == "delta":
                    text = "".join(t for _, t in data["parts"])
                    yield sse(body(text, None, **({"timings": data["timings"]} if req.timings_per_token else {})))
                elif kind == "done":
                    yield sse(body("", data["finish_reason"], usage=usage(data), timings=data["timings"]))
                    yield "data: [DONE]\n\n"
                else:
                    yield stream_error(kind, data)

        return EventStream(stream(), j)

    @api.post("/completion")
    @api.post("/completions")
    async def completion(req: CompletionRequest, request: Request) -> Any:
        """llama-server's own completion: a raw prompt; its responses as llama-server's (``content``, ``stop``,
        ``timings``, ``tokens_cached``, ...)."""
        check(req)
        ids = await run_in_threadpool(prompt_ids, req.prompt)
        fits(ids)
        j, events = job(req, ids, think=False)
        n_prompt = len(ids)

        def final(content: str, done: dict[str, Any]) -> dict[str, Any]:
            t, finish = done["timings"], done["finish_reason"]
            stop_type = "word" if done["stop_word"] else "eos" if finish == "stop" else "limit"
            return {"index": 0, "content": content, "tokens": [], "id_slot": 0, "stop": True, "model": options.alias,
                    "tokens_predicted": t["predicted_n"], "tokens_evaluated": n_prompt,
                    "generation_settings": params(j.sampling), "prompt": req.prompt if isinstance(req.prompt, str)
                    else "", "has_new_line": "\n" in content, "truncated": False, "stop_type": stop_type,
                    "stopping_word": done["stop_word"] or "", "tokens_cached": done["n_cached"], "timings": t}

        if not req.stream:
            texts, done = await collect(j, events, request)
            return final(texts["content"], done)

        async def stream() -> AsyncIterator[str]:
            async for kind, data in events_of(events, ping_of(req)):
                if kind == "ping":
                    yield ":\n\n"
                elif kind == "progress":
                    yield sse({"index": 0, "content": "", "tokens": [], "stop": False, "id_slot": 0,
                               "tokens_predicted": 0, "tokens_evaluated": n_prompt, "prompt_progress": data})
                elif kind == "delta":
                    yield sse({"index": 0, "content": "".join(t for _, t in data["parts"]), "tokens": [],
                               "stop": False, "id_slot": 0, "tokens_predicted": data["timings"]["predicted_n"],
                               "tokens_evaluated": n_prompt,
                               **({"timings": data["timings"]} if req.timings_per_token else {})})
                elif kind == "done":
                    yield sse(final("", data))
                else:
                    yield stream_error(kind, data)

        return EventStream(stream(), j)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @api.post("/tokenize")
    async def tokenize(req: TokenizeRequest) -> dict[str, Any]:
        model = engine.model
        ids = await run_in_threadpool(model.encode, req.content)
        if not req.add_special:
            bos = model.encode("")
            ids = ids[len(bos):] if ids[:len(bos)] == bos else ids
        if not req.with_pieces:
            return {"tokens": ids}
        pieces = []
        for t in ids:
            b = model.tokenizer.decode_bytes([t], special=True)
            try:
                piece: str | list[int] = b.decode("utf-8")
            except UnicodeDecodeError:
                piece = list(b)
            pieces.append({"id": t, "piece": piece})
        return {"tokens": pieces}

    @api.post("/detokenize")
    async def detokenize(req: DetokenizeRequest) -> dict[str, str]:
        text = engine.model.tokenizer.decode_bytes(valid(req.tokens), special=True).decode("utf-8", errors="replace")
        return {"content": text}

    @api.post("/apply-template")
    async def apply_template(req: TemplateRequest) -> dict[str, str]:
        ids, _ = await render(req.messages, req.chat_template_kwargs, media=True)
        return {"prompt": engine.model.tokenizer.decode_bytes(ids, special=True).decode("utf-8", errors="replace")}

    # ------------------------------------------------------------------
    # The web UI's other calls
    # ------------------------------------------------------------------

    @api.delete("/v1/stream")
    async def stop_stream(conv_id: str = "") -> dict[str, Any]:
        """The UI's Stop: cancel its conversation's request."""
        return {"success": True, "cancelled": engine.cancel(conv_id) if conv_id else 0}

    @api.get("/v1/stream")
    async def resume_stream() -> None:
        raise ApiError(404, "not_found_error", "no resumable streams")

    @api.post("/v1/streams/lookup")
    async def lookup_streams() -> list[Any]:
        return []

    add_unsupported(listed)
    app.include_router(public)
    app.include_router(listed)
    app.include_router(api)
    add_ui(app, options.ui, options.no_ui)
    return app
