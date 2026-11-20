"""OpenAI's chat completions (llama-server's shapes): ``/v1/chat/completions``, streamed (``ChatFormat``) or not, with
``reasoning_content``, tool calls (a custom tool's: its input), ``logprobs``, ``timings``, ``prompt_progress`` and ``usage``; a request with audio
output goes to ``speech.chat_audio``, one whose answer is one of a few options with ``logprobs`` to ``answers`` (their
probabilities: ``decide``)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Collection
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import answers
from .app_base import sse
from .context import ApiContext
from .errors import bad_request, unavailable
from .ids import hex_id
from .jobs import timings
from .schemas import ChatRequest, Message
from .speech import chat_audio
from .streaming import Format, collect, stream_response, usage
from .tools import InputStream, ToolCall, custom_input, customs


def call_message(c: ToolCall, arguments: str, custom: Collection[str]) -> dict[str, Any]:
    """A complete tool call as a message's ``tool_calls`` entry (a custom tool's: its input)."""
    if c.name in custom:
        return {"id": c.id, "type": "custom", "custom": {"name": c.name, "input": custom_input(arguments)}}
    return {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": arguments}}


class ChatFormat(Format):
    """A chat completion's chunks (``head``: the fields every chunk has; ``timings``: ``timings_per_token``;
    ``include_usage``: ``stream_options``'s; ``custom``: the custom tools' names, whose calls stream their input)."""

    def __init__(self, head: dict[str, Any], *, timings: bool, include_usage: bool, custom: Collection[str] = ()):
        self.head, self.timings, self.include_usage, self.custom = head, timings, include_usage, custom
        self.inputs: dict[int, InputStream] = {}  # (the custom calls' inputs by index)

    def call_delta(self, c: ToolCall) -> dict[str, Any]:
        """A tool call's part as a streamed ``delta.tool_calls`` entry (its first part: id, type and name)."""
        if c.id is not None and c.name in self.custom:
            self.inputs[c.index] = InputStream()
        if c.index in self.inputs:
            out = {"index": c.index, "custom": {"input": self.inputs[c.index].feed(c.arguments)}}
            return out if c.id is None else {**out, "id": c.id, "type": "custom",
                                              "custom": {"name": c.name, **out["custom"]}}
        if c.id is None:
            return {"index": c.index, "function": {"arguments": c.arguments}}
        return {"index": c.index, "id": c.id, "type": "function", "function": {"name": c.name,
                                                                                "arguments": c.arguments}}

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
            out.append(sse(self.chunk({"tool_calls": [self.call_delta(v)]} if k == "tool_call" else {k: v},
                                      probs=data.get("probs") if last else None,
                                      **({"timings": data["timings"]} if last and self.timings else {}))))
        return out

    def done(self, data: dict[str, Any]) -> list[str]:
        end = [self.chunk({}, data["finish_reason"], probs=data["probs"])]
        if self.include_usage:
            end.append({"choices": [], **self.head, "object": "chat.completion.chunk", "usage": usage(data)})
        end[-1]["timings"] = data["timings"]
        return [sse(c) for c in end] + ["data: [DONE]\n\n"]


async def decide(ctx: ApiContext, req: ChatRequest, head: dict[str, Any]) -> dict[str, Any] | None:
    """A decision's chat completion (``answers``: a response format of options, ``logprobs``; no tools, no thought, no
    continued message), or None: a generation."""
    opts = answers.request_options({"logprobs": req.logprobs, "response_format": req.response_format,
                                    "json_schema": req.json_schema})
    if opts is None or req.tools or ctx.prompts.continuation(req) is not None:
        return None
    ctx.check(req)
    ids, think = await ctx.prompts.render(req.messages, req.chat_template_kwargs, effort=req.reasoning_effort)
    if think and ctx.engine.model.reasoning is not None and ctx.prompts.inside(ids):
        return None  # (the template opened a thought: the answer comes after it)
    ctx.fits(ids)
    model, t0 = ctx.engine.model, time.perf_counter()
    texts = [opts.text(v).encode() for v in opts.values]
    await ctx.vocab()  # (the model's tokens: built aside, in a thread)
    try:
        logprobs = await asyncio.wrap_future(ctx.engine.answers(ids, texts, cache_prompt=req.cache_prompt is not False))
    except RuntimeError as e:
        raise unavailable(str(e)) from None
    best = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))
    n_out = len(model.tokenizer.encode_plain(texts[best].decode()))
    t = timings(cache_n=0, prompt_n=len(ids), prompt_s=time.perf_counter() - t0, predicted_n=n_out, predicted_s=0.0)
    return answers.answer_body(opts, logprobs, head=head, timings=t,
                               usage={"prompt_tokens": len(ids), "completion_tokens": n_out,
                                      "total_tokens": len(ids) + n_out})


async def decision_events(body: dict[str, Any]) -> AsyncIterator[str]:
    """A decision as a stream: its message in one chunk, its end."""
    choice = body["choices"][0]
    chunk = {k: v for k, v in body.items() if k not in ("choices", "usage", "timings")}
    yield sse({**chunk, "object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": None, "delta": {
        "role": "assistant", "content": choice["message"]["content"]}, "logprobs": choice["logprobs"]}]})
    yield sse({**chunk, "object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": "stop",
                                                                         "delta": {}}], "usage": body["usage"]})
    yield "data: [DONE]\n\n"


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
        cid, created, name = hex_id("chatcmpl-", 16), int(time.time()), options.alias
        head = {"created": created, "id": cid, "model": name, "system_fingerprint": options.build_info}
        decision = await decide(ctx, req, head)  # (one of a few options, with logprobs: their probabilities)
        if decision is not None:
            return decision if not req.stream else StreamingResponse(decision_events(decision),
                                                                     media_type="text/event-stream")
        j, events = await ctx.start(req, conv_id=x_conversation_id, cid=cid)
        custom = customs(req.tools).keys()  # (checked: ``start``)
        if not req.stream:
            blocks, done = await collect(j, events, request)
            calls = [call_message(b.call, b.text, custom) for b in blocks.complete(done["n_calls"]) if b.kind == "call"]
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
                         include_usage=bool((req.stream_options or {}).get("include_usage")), custom=custom)
        return await stream_response(j, events, ctx.ping_of(req), fmt)

    @api.post("/v1/chat/completions/control")
    async def control(body: dict[str, Any]) -> dict[str, Any]:
        """llama-server's control of a running chat completion (its ``id``): ``reasoning_end`` ends its thought where
        the request armed it (``reasoning_control``)."""
        cid, action = body.get("id"), body.get("action")
        if not isinstance(cid, str) or not cid:
            raise bad_request("missing completion id")
        if action != "reasoning_end":
            raise bad_request("unknown control action")
        j = engine.job_of(cid)
        if j is None:
            return {"success": False, "message": "no active completion for this id"}
        if not j.control:
            return {"success": False, "message": "reasoning control not enabled for this completion"}
        j.end_thought.set()
        return {"success": True}
