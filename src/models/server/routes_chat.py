"""OpenAI's chat completions (llama-server's shapes): ``/v1/chat/completions``, streamed (``ChatFormat``) or not, with
``reasoning_content``, tool calls, ``logprobs``, ``timings``, ``prompt_progress`` and ``usage``; a request with audio
output goes to ``speech.chat_audio``."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Header, Request
from starlette.concurrency import run_in_threadpool

from .app_base import sse
from .context import ApiContext
from .ids import hex_id
from .requests import ChatRequest, Message
from .speech import chat_audio
from .streaming import Format, collect, stream_response, usage
from .tools import ToolCall


def call_delta(c: ToolCall) -> dict[str, Any]:
    """A tool call's part as a streamed ``delta.tool_calls`` entry (its first part: id, type and name)."""
    if c.id is None:
        return {"index": c.index, "function": {"arguments": c.arguments}}
    return {"index": c.index, "id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}


class ChatFormat(Format):
    """A chat completion's chunks (``head``: the fields every chunk has; ``timings``: ``timings_per_token``;
    ``include_usage``: ``stream_options``'s)."""

    def __init__(self, head: dict[str, Any], *, timings: bool, include_usage: bool):
        self.head, self.timings, self.include_usage = head, timings, include_usage

    def chunk(self, delta: dict[str, Any], finish: str | None = None, probs: list | None = None, **extra: object
              ) -> dict[str, Any]:
        choice = {"finish_reason": finish, "index": 0, "delta": delta}
        if probs:
            choice["logprobs"] = {"content": probs}
        return {"choices": [choice], **self.head, "object": "chat.completion.chunk", **extra}

    def start(self) -> list[str]:
        return [sse(self.chunk({"role": "assistant", "content": None}))]

    def progress(self, data: dict[str, Any]) -> list[str]:
        return [sse(self.chunk({"role": "assistant", "content": None}, prompt_progress=data))]

    def delta(self, data: dict[str, Any]) -> list[str]:
        parts, out = data["parts"], []
        for i, (k, v) in enumerate(parts):
            last = i == len(parts) - 1
            out.append(sse(self.chunk({"tool_calls": [call_delta(v)]} if k == "tool_call" else {k: v},
                                      probs=data.get("probs") if last else None,
                                      **({"timings": data["timings"]} if last and self.timings else {}))))
        return out

    def done(self, data: dict[str, Any]) -> list[str]:
        end = [self.chunk({}, data["finish_reason"], probs=data["probs"])]
        if self.include_usage:
            end.append({"choices": [], **self.head, "object": "chat.completion.chunk", "usage": usage(data)})
        end[-1]["timings"] = data["timings"]
        return [sse(c) for c in end] + ["data: [DONE]\n\n"]


def add_routes(api: APIRouter, ctx: ApiContext) -> None:
    """``/v1/chat/completions`` on ``api``."""
    engine, options = ctx.engine, ctx.options

    @api.post("/v1/chat/completions")
    @api.post("/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request,
                               x_conversation_id: str | None = Header(None)) -> Any:
        if "audio" in ((req.model_extra or {}).get("modalities") or []):  # (a spoken reply)
            body = req.model_dump(exclude_none=True)
            msgs = await run_in_threadpool(lambda: ctx.prompts.messages_of(
                [Message.model_validate(m) for m in ctx.replies.messages(body["messages"])]))
            return await chat_audio(engine, body, msgs, alias=options.alias, replies=ctx.replies)
        j, events = await ctx.start(req, conv_id=x_conversation_id)
        cid, created, name = hex_id("chatcmpl-", 16), int(time.time()), options.alias
        head = {"created": created, "id": cid, "model": name, "system_fingerprint": options.build_info}
        if not req.stream:
            blocks, done = await collect(j, events, request)
            calls = [{"id": b.call.id, "type": "function", "function": {"name": b.call.name, "arguments": b.text}}
                     for b in blocks.complete(done["n_calls"]) if b.kind == "call"]
            message = {"role": "assistant", "content": blocks.text("text") or (None if calls else "")}
            if blocks.text("thinking"):
                message["reasoning_content"] = blocks.text("thinking")
            if calls:
                message["tool_calls"] = calls
            choice = {"finish_reason": done["finish_reason"], "index": 0, "message": message}
            if done["probs"]:
                choice["logprobs"] = {"content": done["probs"]}
            return {"choices": [choice], **head, "object": "chat.completion", "usage": usage(done),
                    "timings": done["timings"]}
        fmt = ChatFormat(head, timings=bool(req.timings_per_token),
                         include_usage=bool((req.stream_options or {}).get("include_usage")))
        return await stream_response(j, events, ctx.ping_of(req), fmt)
