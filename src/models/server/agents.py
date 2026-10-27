"""The APIs that agent clients speak besides chat completions: Anthropic's Messages (``/v1/messages``: Claude Code,
the Anthropic SDKs) and OpenAI's Responses (``/v1/responses``: Codex CLI, the Agents SDK). Each request becomes a
chat-completions body (``anthropic_chat``, ``responses_chat``: llama-server's conversions, ``server-chat.cpp``), which
the chat path renders and runs; the job's parts become the API's response and its stream of events (``Anthropic``,
``Responses``), through ``Blocks``: a generation's thought, text and calls as the ordered blocks of one message, which
Anthropic's content blocks and the Responses API's output items both are.

Where the conversions differ from llama-server's:

- an Anthropic message's blocks keep their order: tool results come before the text that follows them (llama-server
  puts a message's text first, a user turn between the calls and their results);
- Anthropic's ``tool_choice`` ``tool`` names its function, ``none`` and ``disable_parallel_tool_use`` apply;
  ``thinking`` ``enabled`` and ``adaptive`` turn the model's thought on, ``disabled`` off (absent: the server's
  ``-rea``), as a Responses request's ``reasoning.effort`` (``none``: off); thinking budgets are not enforced;
- Claude Code's billing header has a stamp per request (``cch=``, which llama.cpp fixes), its version's build hash,
  which differs between sessions, and on the requests its scheduled wakeups send a ``cc_workload``: the stamp and the
  hash are fixed, the workload dropped, so the system prompt's tokens stay a cached prefix;
- a Responses request may continue a stored response (``previous_response_id``, ``Store``; llama-server refuses
  it); a reasoning item without content gives its summary, an assistant message needs no ``type``;
- tools other than functions (Anthropic's server tools, Codex's ``namespace`` and ``web_search``) are left out.

The streams follow the APIs' own: Anthropic's closes each block before the next opens (``signature_delta`` before a
thought's stop) and stops for a stop string with ``stop_sequence``; the Responses stream numbers its events and
sends each item's ``.done`` events when the item ends, ``response.incomplete`` at ``max_output_tokens``. Both clients
act on a context overflow by its wording: Claude Code reads "prompt is too long: N tokens > M maximum", Codex the code
``context_length_exceeded`` (``overflow``)."""

from __future__ import annotations

import collections
import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .tools import ALNUM, Part, ToolCall

Note = Callable[[str], None]  # a field the conversion ignores: logged once

ANTHROPIC_MAX_TOKENS = 4096  # an Anthropic request's max_tokens when it has none (llama-server's)
BILLING = "x-anthropic-billing-header:"
STORED = 1024  # the responses kept for previous_response_id


def token(n: int = 24) -> str:
    return "".join(secrets.choice(ALNUM) for _ in range(n))


def event(name: str, data: dict[str, Any]) -> str:
    """A server-sent event with its name (both APIs name their events)."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def overflow(n_prompt: int, n_ctx: int, api: str) -> str:
    """A context overflow's message as the API's clients recognize it."""
    if api == "anthropic":
        return f"prompt is too long: {n_prompt} tokens > {n_ctx} maximum"
    return f"Your input exceeds the context window of this model: {n_prompt} tokens > {n_ctx}."


# ----------------------------------------------------------------------
# The blocks of a message
# ----------------------------------------------------------------------


@dataclass(eq=False)
class Block:
    """A block of the generated message: ``kind`` thinking, text or call; ``text``: its text so far (a call's
    arguments' JSON text); ``call``: a call's first part (its id and name); ``index``: its place in the message."""

    kind: str
    index: int
    text: str = ""
    call: ToolCall | None = None


KINDS = {"reasoning_content": "thinking", "content": "text", "tool_call": "call"}


class Blocks:
    """A generation's parts as the blocks of one message, in order: a part of another kind than the open block, or
    another call, closes the open block and opens the next. ``feed`` and ``end`` return what changed: ("start",
    block, ""), ("delta", block, text), ("stop", block, "")."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.open: Block | None = None

    def feed(self, parts: list[Part]) -> list[tuple[str, Block, str]]:
        out: list[tuple[str, Block, str]] = []
        for kind, value in parts:
            k = KINDS[kind]
            new_call = k == "call" and value.id is not None
            if self.open is None or self.open.kind != k or new_call:
                out += self.end()
                self.open = Block(k, len(self.blocks), call=value if k == "call" else None)
                self.blocks.append(self.open)
                out.append(("start", self.open, ""))
            text = value.arguments if k == "call" else value
            if text:
                self.open.text += text
                out.append(("delta", self.open, text))
        return out

    def end(self) -> list[tuple[str, Block, str]]:
        """The open block closed (the generation ended, or another block begins)."""
        if self.open is None:
            return []
        b, self.open = self.open, None
        return [("stop", b, "")]

    def complete(self, n_calls: int) -> list[Block]:
        """The blocks of a whole response: the first ``n_calls`` calls (a call cut off is left out)."""
        calls = [b for b in self.blocks if b.kind == "call"][:n_calls]
        return [b for b in self.blocks if b.kind != "call" or b in calls]

    def text(self, kind: str) -> str:
        return "".join(b.text for b in self.blocks if b.kind == kind)


def arguments(block: Block) -> dict[str, Any]:
    """A complete call's arguments (its JSON text, an object)."""
    try:
        args = json.loads(block.text) if block.text.strip() else {}
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


# ----------------------------------------------------------------------
# Anthropic's Messages
# ----------------------------------------------------------------------


def billing(text: str) -> str:
    """Claude Code's billing header with its per-request stamp (``cch=``) and its version's build hash fixed, its
    ``cc_workload`` dropped."""
    if not text.startswith(BILLING):
        return text
    text = re.sub(r"(cc_version=\d+\.\d+\.\d+)\.[0-9a-f]+;", r"\1;", text, count=1)
    text = re.sub(r" ?\bcc_workload=[^;]*;", "", text, count=1)
    return re.sub(r"\bcch=[^;]{5};", "cch=fffff;", text, count=1)


def image_part(block: dict[str, Any]) -> dict[str, Any]:
    """An Anthropic image block as an ``image_url`` part (a base64 source as a data URI)."""
    source = block.get("source") or {}
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type') or 'image/jpeg'};base64,{source.get('data', '')}"
    else:
        url = str(source.get("url", ""))
    return {"type": "image_url", "image_url": {"url": url}}


def tool_result(content: object) -> str | list[dict[str, Any]]:
    """A tool result's content: its text, or text and image parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [{"type": "text", "text": str(b.get("text", ""))} if b.get("type") == "text" else image_part(b)
             for b in content if isinstance(b, dict) and b.get("type") in ("text", "image")]
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def anthropic_messages(messages: object) -> list[dict[str, Any]]:
    """Anthropic's messages as chat messages: a message's blocks in order, its text, images, thoughts and calls one
    message, each tool result a ``tool`` message where it stands. ValueError: a block the server does not take."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages: a list of messages")
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError(f"a message is an object with a role: {m!r}")
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, list):
            if content is not None or role != "assistant":
                out.append({"role": role, "content": content or ""})
            continue
        msg: dict[str, Any] | None = None  # (the message the next blocks go to)
        for b in content:
            kind = b.get("type") if isinstance(b, dict) else None
            if kind == "tool_result":
                out.append({"role": "tool", "tool_call_id": str(b.get("tool_use_id", "")),
                            "content": tool_result(b.get("content"))})
                msg = None
                continue
            if kind not in ("text", "image", "thinking", "redacted_thinking", "tool_use"):
                raise ValueError(f"content blocks of type {kind!r} are not supported")
            if msg is None:
                msg = {"role": role, "content": []}
                out.append(msg)
            if kind == "text":
                msg["content"].append({"type": "text", "text": str(b.get("text", ""))})
            elif kind == "image":
                msg["content"].append(image_part(b))
            elif kind == "thinking":
                msg["reasoning_content"] = msg.get("reasoning_content", "") + str(b.get("thinking", ""))
            elif kind == "tool_use":
                msg.setdefault("tool_calls", []).append(
                    {"id": str(b.get("id", "")), "type": "function",
                     "function": {"name": str(b.get("name", "")), "arguments": b.get("input") or {}}})
    return out


def anthropic_chat(body: dict[str, Any], note: Note) -> dict[str, Any]:
    """An Anthropic Messages request as a chat-completions body. ValueError: one the server does not take."""
    messages = []
    system = body.get("system")
    if isinstance(system, str):
        messages.append({"role": "system", "content": billing(system)})
    elif isinstance(system, list):
        messages.append({"role": "system", "content": "".join(billing(str(b.get("text", ""))) for b in system
                                                              if isinstance(b, dict) and b.get("type") == "text")})
    messages += anthropic_messages(body.get("messages"))
    out: dict[str, Any] = {"messages": messages, "max_tokens": body.get("max_tokens", ANTHROPIC_MAX_TOKENS)}
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict) or t.get("type") not in (None, "custom"):
            note(f"tools of type {t.get('type') if isinstance(t, dict) else t!r}")
            continue
        tools.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description"),
                                                       "parameters": t.get("input_schema") or {}}})
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif kind in ("auto", "any", "none"):
            out["tool_choice"] = {"any": "required"}.get(kind, kind)
        else:
            raise ValueError(f"tool_choice of type {kind!r} (auto, any, tool or none)")
        if choice.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    for k in ("temperature", "top_p", "top_k", "stream"):
        if body.get(k) is not None:
            out[k] = body[k]
    kwargs = dict(body.get("chat_template_kwargs") or {})
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive", "disabled"):
        kwargs.setdefault("enable_thinking", thinking["type"] != "disabled")
        if thinking.get("budget_tokens") is not None:
            note("thinking.budget_tokens")
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    if (body.get("output_config") or {}).get("effort") is not None:
        note("output_config.effort")
    return out


def anthropic_usage(*, n_prompt: int, cached: int, output: int) -> dict[str, int]:
    return {"input_tokens": n_prompt - cached, "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
            "output_tokens": output}


def stop_reason(done: dict[str, Any]) -> str:
    return {"length": "max_tokens", "tool_calls": "tool_use"}.get(done["finish_reason"],
                                                                 "stop_sequence" if done["stop_word"] else "end_turn")


def anthropic_block(b: Block) -> dict[str, Any]:
    """A whole block as Anthropic's content block."""
    if b.kind == "thinking":
        return {"type": "thinking", "thinking": b.text, "signature": ""}
    if b.kind == "text":
        return {"type": "text", "text": b.text}
    return {"type": "tool_use", "id": b.call.id, "name": b.call.name, "input": arguments(b)}


class Anthropic:
    """A response of the Messages API: ``message`` the whole one; streamed, ``start`` (once the job began: its
    prompt's cached tokens are known), ``parts`` and ``end`` the events."""

    def __init__(self, model: str):
        self.id, self.model = f"msg_{token()}", model
        self.blocks = Blocks()
        self.started = False

    def message(self, done: dict[str, Any]) -> dict[str, Any]:
        t = done["timings"]
        return {"id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [anthropic_block(b) for b in self.blocks.complete(done["n_calls"])],
                "stop_reason": stop_reason(done), "stop_sequence": done["stop_word"],
                "usage": anthropic_usage(n_prompt=done["n_prompt"], cached=t["cache_n"], output=t["predicted_n"])}

    def start(self, *, n_prompt: int, cached: int) -> list[str]:
        self.started = True
        return [event("message_start", {"type": "message_start", "message": {
            "id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": anthropic_usage(n_prompt=n_prompt, cached=cached, output=0)}})]

    def _events(self, changes: list[tuple[str, Block, str]]) -> list[str]:
        out = []
        for what, b, text in changes:
            if what == "start":  # (empty: the block's text so far may have come after its start)
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""}}.get(b.kind) or \
                    {"type": "tool_use", "id": b.call.id, "name": b.call.name, "input": {}}
                out.append(event("content_block_start", {"type": "content_block_start", "index": b.index,
                                                         "content_block": block}))
            elif what == "delta":
                delta = {"thinking": {"type": "thinking_delta", "thinking": text},
                         "text": {"type": "text_delta", "text": text},
                         "call": {"type": "input_json_delta", "partial_json": text}}[b.kind]
                out.append(event("content_block_delta", {"type": "content_block_delta", "index": b.index,
                                                         "delta": delta}))
            else:
                if b.kind == "thinking":
                    out.append(event("content_block_delta", {"type": "content_block_delta", "index": b.index,
                                                             "delta": {"type": "signature_delta", "signature": ""}}))
                out.append(event("content_block_stop", {"type": "content_block_stop", "index": b.index}))
        return out

    def parts(self, parts: list[Part]) -> list[str]:
        return self._events(self.blocks.feed(parts))

    def end(self, done: dict[str, Any]) -> list[str]:
        t = done["timings"]
        return self._events(self.blocks.end()) + [
            event("message_delta", {"type": "message_delta",
                                    "delta": {"stop_reason": stop_reason(done), "stop_sequence": done["stop_word"]},
                                    "usage": anthropic_usage(n_prompt=done["n_prompt"], cached=t["cache_n"],
                                                             output=t["predicted_n"])}),
            event("message_stop", {"type": "message_stop"})]


ANTHROPIC_ERRORS = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                    404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error", 503: "overloaded_error"}


def anthropic_error(status: int, message: str) -> dict[str, Any]:
    """An error in Anthropic's shape."""
    return {"type": "error", "error": {"type": ANTHROPIC_ERRORS.get(status, "api_error"), "message": message}}


# ----------------------------------------------------------------------
# OpenAI's Responses
# ----------------------------------------------------------------------


def input_items(value: object) -> list[dict[str, Any]]:
    """A Responses request's ``input`` as items (a text: one user message). ValueError: neither."""
    if isinstance(value, str):
        return [{"type": "message", "role": "user", "content": value}]
    if not isinstance(value, list) or not all(isinstance(i, dict) for i in value):
        raise ValueError("input: a text or a list of items")
    return value


def content_parts(content: object) -> list[dict[str, Any]]:
    """A Responses message's or tool output's content as chat parts. ValueError: a part the server does not take."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for p in content if isinstance(content, list) else []:
        kind = p.get("type") if isinstance(p, dict) else None
        if kind in ("input_text", "output_text", "text"):
            out.append({"type": "text", "text": str(p.get("text", ""))})
        elif kind == "refusal":
            out.append({"type": "text", "text": str(p.get("refusal", ""))})
        elif kind == "input_image":
            url = p.get("image_url")
            if not isinstance(url, str):
                raise ValueError("an input_image needs an image_url (a data URI)")
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ValueError(f"content of type {kind!r} is not supported")
    return out


def responses_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses items as chat messages: an assistant's reasoning, messages and calls that follow each other are one
    assistant message. ValueError: an item the server does not take."""
    out: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type", "message")
        prev = out[-1] if out and out[-1]["role"] == "assistant" else None
        if kind == "message":
            role, parts = item.get("role"), content_parts(item.get("content"))
            if role == "assistant" and prev is not None:
                prev["content"] += parts
            else:
                out.append({"role": role, "content": parts})
        elif kind == "function_call":
            call = {"id": str(item.get("call_id", "")), "type": "function",
                    "function": {"name": str(item.get("name", "")), "arguments": item.get("arguments") or "{}"}}
            if prev is not None:
                prev.setdefault("tool_calls", []).append(call)
            else:
                out.append({"role": "assistant", "content": [], "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            out.append({"role": "tool", "tool_call_id": str(item.get("call_id", "")),
                        "content": output if isinstance(output, str) else content_parts(output)})
        elif kind == "reasoning":
            texts = [c.get("text", "") for c in item.get("content") or [] if isinstance(c, dict)] or \
                [s.get("text", "") for s in item.get("summary") or [] if isinstance(s, dict)]
            text = "\n\n".join(str(t) for t in texts)
            if prev is not None:
                prev["reasoning_content"] = prev.get("reasoning_content", "") + text
            else:
                out.append({"role": "assistant", "content": [], "reasoning_content": text})
        else:
            raise ValueError(f"input items of type {kind!r} are not supported")
    return out


def responses_chat(body: dict[str, Any], history: list[dict[str, Any]], note: Note) -> dict[str, Any]:
    """A Responses request as a chat-completions body, after ``history`` (the items of the response it continues).
    ValueError: one the server does not take."""
    for k in ("conversation", "background"):
        if body.get(k):
            raise ValueError(f"{k} is not supported")
    instructions = body.get("instructions")
    messages = [{"role": "system", "content": instructions}] if isinstance(instructions, str) and instructions else []
    messages += responses_messages(history + input_items(body.get("input")))
    out: dict[str, Any] = {"messages": messages}
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            note(f"tools of type {t.get('type') if isinstance(t, dict) else t!r}")
            continue
        tools.append({"type": "function", "function": {k: v for k, v in t.items() if k not in ("type", "strict")}})
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") != "function":
            raise ValueError(f"tool_choice of type {choice.get('type')!r} (auto, none, required or a function)")
        out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
    elif choice is not None:
        out["tool_choice"] = choice
    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
    for k in ("temperature", "top_p", "stream", "parallel_tool_calls", "chat_template_kwargs"):
        if body.get(k) is not None:
            out[k] = body[k]
    effort = (body.get("reasoning") or {}).get("effort")
    if effort is not None:
        out["chat_template_kwargs"] = {"enable_thinking": effort != "none", **(body.get("chat_template_kwargs") or {})}
    if ((body.get("text") or {}).get("format") or {}).get("type", "text") != "text":
        note("text.format")
    return out


class Store:
    """The last ``size`` responses, for ``previous_response_id``: each its request's input items, its output items,
    the response it continues and its response object; a conversation's responses stay while it is used."""

    def __init__(self, size: int = STORED):
        self.size = size
        self._entries: collections.OrderedDict[str, tuple[str | None, list, dict]] = collections.OrderedDict()

    def put(self, response: dict[str, Any], inputs: list[dict[str, Any]]) -> None:
        self._entries[response["id"]] = (response.get("previous_response_id"), inputs, response)
        while len(self._entries) > self.size:
            self._entries.popitem(last=False)

    def get(self, rid: str) -> dict[str, Any] | None:
        entry = self._entries.get(rid)
        return entry[2] if entry is not None else None

    def delete(self, rid: str) -> bool:
        return self._entries.pop(rid, None) is not None

    def items(self, rid: str) -> list[dict[str, Any]]:
        """The items of the conversation up to response ``rid`` (its inputs and outputs, after its predecessors').
        KeyError: a response of the chain is not stored."""
        chain = []
        while rid is not None:
            prev, inputs, response = self._entries[rid]
            chain.append((rid, inputs, response["output"]))
            rid = prev
        out = []
        for rid, inputs, output in reversed(chain):
            self._entries.move_to_end(rid)
            out += inputs + output
        return out


ECHOED = ("instructions", "max_output_tokens", "parallel_tool_calls", "previous_response_id", "reasoning", "store",
          "temperature", "text", "tool_choice", "tools", "top_p", "truncation", "metadata")


class Responses:
    """A response of the Responses API: ``response`` the whole one; streamed, ``start``, ``parts``, ``end`` and
    ``failed`` the events. ``body``: the request (its fields the response echoes; ``include``
    ``reasoning.encrypted_content``: reasoning items carry an empty one)."""

    def __init__(self, model: str, body: dict[str, Any]):
        self.id, self.model, self.created = f"resp_{secrets.token_hex(24)}", model, int(time.time())
        self.echo = {"tool_choice": "auto", "tools": [], "parallel_tool_calls": True, "store": True,
                     "truncation": "disabled", "metadata": {}, "text": {"format": {"type": "text"}}} | \
            {k: body[k] for k in ECHOED if k in body}
        self.encrypted = "reasoning.encrypted_content" in (body.get("include") or [])
        self.blocks = Blocks()
        self.ids: dict[int, str] = {}  # (each block's item id)
        self.seq = 0

    def item_id(self, b: Block) -> str:
        return self.ids.setdefault(b.index, {"thinking": "rs_", "text": "msg_", "call": "fc_"}[b.kind] + (
            b.call.id.removeprefix("call_") if b.kind == "call" else token()))

    def item(self, b: Block, *, status: str = "completed") -> dict[str, Any]:
        """A block as an output item (``status`` in_progress: as just added, without its text)."""
        done, rid = status == "completed", self.item_id(b)
        if b.kind == "thinking":
            out = {"id": rid, "type": "reasoning", "summary": [],
                   "content": [{"type": "reasoning_text", "text": b.text}] if done else [], "status": status}
            return out | ({"encrypted_content": ""} if self.encrypted else {})
        if b.kind == "text":
            return {"id": rid, "type": "message", "role": "assistant", "status": status,
                    "content": [self._text(b.text)] if done else []}
        return {"id": rid, "type": "function_call", "status": status, "call_id": b.call.id, "name": b.call.name,
                "arguments": b.text if done else ""}

    @staticmethod
    def _text(text: str) -> dict[str, Any]:
        return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}

    def response(self, done: dict[str, Any] | None, *, error: dict[str, Any] | None = None) -> dict[str, Any]:
        """The response object: in progress (``done`` None), failed (``error``) or ended (``done``: its output, its
        usage, incomplete at ``max_output_tokens``)."""
        status, usage, incomplete, output = "in_progress", None, None, []
        if error is not None:
            status = "failed"
        elif done is not None:
            output = [self.item(b) for b in self.blocks.complete(done["n_calls"])]
            t = done["timings"]
            usage = {"input_tokens": done["n_prompt"], "input_tokens_details": {"cached_tokens": t["cache_n"]},
                     "output_tokens": t["predicted_n"], "output_tokens_details": {"reasoning_tokens": 0},
                     "total_tokens": done["n_prompt"] + t["predicted_n"]}
            status = "incomplete" if done["finish_reason"] == "length" else "completed"
            incomplete = {"reason": "max_output_tokens"} if status == "incomplete" else None
        return {"id": self.id, "object": "response", "created_at": self.created, "status": status,
                "model": self.model, "output": output, "usage": usage, "error": error,
                "incomplete_details": incomplete, **self.echo}

    def _event(self, kind: str, **data: object) -> str:
        self.seq += 1
        return event(kind, {"type": kind, "sequence_number": self.seq - 1, **data})

    def start(self) -> list[str]:
        r = self.response(None)
        return [self._event("response.created", response=r), self._event("response.in_progress", response=r)]

    def _events(self, changes: list[tuple[str, Block, str]]) -> list[str]:
        out = []
        for what, b, text in changes:
            at = {"item_id": self.item_id(b), "output_index": b.index}
            if what == "start":
                out.append(self._event("response.output_item.added", output_index=b.index,
                                       item=self.item(b, status="in_progress")))
                if b.kind == "text":
                    out.append(self._event("response.content_part.added", **at, content_index=0, part=self._text("")))
            elif what == "delta":
                if b.kind == "call":
                    out.append(self._event("response.function_call_arguments.delta", **at, delta=text))
                else:
                    kind = "response.reasoning_text.delta" if b.kind == "thinking" else "response.output_text.delta"
                    out.append(self._event(kind, **at, content_index=0, delta=text,
                                           **({"logprobs": []} if b.kind == "text" else {})))
            else:
                if b.kind == "thinking":
                    out.append(self._event("response.reasoning_text.done", **at, content_index=0, text=b.text))
                elif b.kind == "text":
                    out.append(self._event("response.output_text.done", **at, content_index=0, text=b.text,
                                           logprobs=[]))
                    out.append(self._event("response.content_part.done", **at, content_index=0,
                                           part=self._text(b.text)))
                else:
                    out.append(self._event("response.function_call_arguments.done", **at, name=b.call.name,
                                           arguments=b.text))
                out.append(self._event("response.output_item.done", output_index=b.index, item=self.item(b)))
        return out

    def parts(self, parts: list[Part]) -> list[str]:
        return self._events(self.blocks.feed(parts))

    def end(self, done: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """The last events and the whole response."""
        out = self._events(self.blocks.end())
        r = self.response(done)
        return out + [self._event(f"response.{r['status']}", response=r)], r

    def failed(self, code: str, message: str) -> list[str]:
        return [self._event("response.failed", response=self.response(None, error={"code": code,
                                                                                   "message": message}))]
