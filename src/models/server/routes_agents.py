"""The agents' APIs, on the chat route's jobs (``agents`` converts their requests to chat completions and the job's
parts to their shapes): Anthropic's Messages (``/v1/messages``, ``/v1/messages/count_tokens``: Claude Code) and
OpenAI's Responses (``/v1/responses``, ``/v1/responses/input_tokens``, the stored responses: Codex)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from . import agents
from .app_base import ANTHROPIC, NO_CACHE
from .context import ApiContext
from .errors import ApiError, bad_request, cancelled, server_error
from .schemas import ChatRequest, chat_request
from .streaming import Format, collect, stream_response


class AnthropicFormat(Format):
    """Anthropic's stream of ``out`` (``agents.Anthropic``)."""

    def __init__(self, out: agents.Anthropic):
        self.out = out

    def ping(self) -> list[str]:
        return [agents.event("ping", {"type": "ping"})]

    def progress(self, data: dict[str, Any]) -> list[str]:
        if self.out.started:
            return []
        return ["".join(self.out.start(n_prompt=data["total"], cached=data["cache"]))]

    def delta(self, data: dict[str, Any]) -> list[str]:
        return ["".join(self.out.parts(data["parts"]))]

    def done(self, data: dict[str, Any]) -> list[str]:
        return ["".join(self.out.end(data))]

    def error(self, kind: str, data: dict[str, Any]) -> list[str]:
        e = server_error(data["message"]) if kind == "error" else cancelled()
        return [agents.event("error", agents.anthropic_error(e.status, e.message))]


class ResponsesFormat(Format):
    """The Responses API's stream of ``out`` (``agents.Responses``); ``keep``: the finished response is stored
    (``store``, with ``inputs``: its own input items)."""

    def __init__(self, out: agents.Responses, *, store: agents.Store | None, inputs: list[dict[str, Any]]):
        self.out, self.store, self.inputs = out, store, inputs

    def start(self) -> list[str]:
        return ["".join(self.out.start())]

    def delta(self, data: dict[str, Any]) -> list[str]:
        return ["".join(self.out.parts(data["parts"]))]

    def done(self, data: dict[str, Any]) -> list[str]:
        end, r = self.out.end(data)
        if self.store is not None:
            self.store.put(r, self.inputs)
        return ["".join(end)]

    def error(self, kind: str, data: dict[str, Any]) -> list[str]:
        return ["".join(self.out.failed("server_error", data["message"] if kind == "error" else
                                        "the request was cancelled"))]


def add_routes(api: APIRouter, ctx: ApiContext) -> None:
    """The Anthropic and Responses routes on ``api``."""
    options, store = ctx.options, ctx.store

    def anthropic_request(body: dict[str, Any]) -> ChatRequest:
        try:
            return chat_request(agents.anthropic_chat(body, ctx.ignored(ANTHROPIC)))
        except ValueError as e:
            raise bad_request(str(e)) from None

    @api.post("/v1/messages")
    async def messages(body: dict[str, Any], request: Request) -> Any:
        """Anthropic's Messages (``agents.Anthropic``)."""
        req = anthropic_request(body)
        req.return_progress = bool(req.stream)  # (the job's start: its message_start, with the cached tokens)
        j, events = await ctx.start(req, api="anthropic")
        out = agents.Anthropic(options.alias)
        if not req.stream:
            out.blocks, done = await collect(j, events, request)
            return out.message(done)
        return await stream_response(j, events, ctx.ping_of(req), AnthropicFormat(out))

    @api.post("/v1/messages/count_tokens")
    async def count_tokens(body: dict[str, Any]) -> dict[str, int]:
        return {"input_tokens": await ctx.prompts.counted(anthropic_request(body))}

    def responses_request(body: dict[str, Any]) -> tuple[ChatRequest, list[dict[str, Any]]]:
        """A Responses request as a chat request, after the response it continues (``previous_response_id``), and its
        own input items."""
        prev = body.get("previous_response_id")
        try:
            history = store.items(prev) if prev else []
        except KeyError:
            raise ApiError(400, "invalid_request_error", f"Previous response with id '{prev}' not found.",
                           code="previous_response_not_found", param="previous_response_id") from None
        try:
            return chat_request(agents.responses_chat(body, history, ctx.ignored("/v1/responses"))), \
                agents.input_items(body.get("input"))
        except ValueError as e:
            raise bad_request(str(e)) from None

    @api.post("/v1/responses")
    @api.post("/responses")
    async def responses(body: dict[str, Any], request: Request) -> Any:
        """OpenAI's Responses (``agents.Responses``); a response is stored unless ``store`` is false."""
        req, inputs = responses_request(body)
        out = agents.Responses(options.alias, body)
        keep = body.get("store") is not False
        try:
            j, events = await ctx.start(req, api="responses")
        except ApiError as e:  # (an overflow while streaming: response.failed, as clients expect it)
            if not req.stream or e.extra.get("code") != "context_length_exceeded":
                raise
            lines = out.start() + out.failed("context_length_exceeded", e.message)

            async def failed() -> AsyncIterator[str]:
                yield "".join(lines)

            return StreamingResponse(failed(), media_type="text/event-stream", headers=NO_CACHE)
        if not req.stream:
            out.blocks, done = await collect(j, events, request)
            r = out.response(done)
            if keep:
                store.put(r, inputs)
            return r
        return await stream_response(j, events, ctx.ping_of(req),
                                     ResponsesFormat(out, store=store if keep else None, inputs=inputs))

    @api.post("/v1/responses/input_tokens")
    @api.post("/responses/input_tokens")
    async def input_tokens(body: dict[str, Any]) -> dict[str, Any]:
        return {"object": "response.input_tokens",
                "input_tokens": await ctx.prompts.counted(responses_request(body)[0])}

    @api.get("/v1/responses/{rid}")
    @api.get("/responses/{rid}")
    async def stored_response(rid: str) -> dict[str, Any]:
        r = store.get(rid)
        if r is None:
            raise ApiError(404, "not_found_error", f"No response with id '{rid}' found.")
        return r

    @api.delete("/v1/responses/{rid}")
    @api.delete("/responses/{rid}")
    async def delete_response(rid: str) -> dict[str, Any]:
        if not store.delete(rid):
            raise ApiError(404, "not_found_error", f"No response with id '{rid}' found.")
        return {"id": rid, "object": "response.deleted", "deleted": True}
