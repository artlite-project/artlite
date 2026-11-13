"""A chat request's prompt: its messages as the chat template takes them (``PromptBuilder.messages_of``: media parts
as bytes for the model's encoders, the system messages a template that takes one first, a spoken reply as the model said
it), the template's variables from the request and the server's options, the tokens through the template, and the
request's tools as the template and the parser need them.

A chat's messages go to the template as llama-server hands them over, and more: ``developer`` is ``system``
(llama.cpp's rule), and a template that takes a system message only first (Qwen's raise otherwise) gets the leading
ones joined and the later ones as user messages - Claude Code sends system messages between the turns, Codex its
``instructions`` and a developer message."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Any

import jinja2
from starlette.concurrency import run_in_threadpool

from ..model import MEDIA_MARKER
from ..sequence import data_bytes
from ..templating import TEMPLATE_RESERVED, chat_template
from .errors import bad_request
from .listing import Modalities
from .schemas import ROLES, ChatRequest, Message
from .tools import Calls, Vocabulary, call_arguments, functions

if TYPE_CHECKING:
    from .context import ApiContext

log = logging.getLogger("artlite.server")

MEDIA_TYPES = ("image", "audio", "video")  # (the kinds of a message's media parts, as the template gets them)


def image_bytes(url: object) -> bytes:
    """An ``image_url`` part's image: a data URI's bytes (base64 or percent-encoded)."""
    u = url.get("url") if isinstance(url, dict) else url
    if not isinstance(u, str) or not u.startswith("data:") or "," not in u:
        raise bad_request("an image_url must be a data URI (data:image/...;base64,...)")
    try:
        return data_bytes(u)
    except ValueError:
        raise bad_request("an image_url's base64 data does not decode") from None


def video_bytes(part: dict[str, Any]) -> bytes:
    """An ``input_video`` / ``video_url`` part's video (llama-server's: ``data`` or ``url``, a data URL or base64)."""
    v = part.get(part.get("type")) or {}
    u = v.get("data") or v.get("url") if isinstance(v, dict) else v
    if not isinstance(u, str) or not u:
        raise bad_request(f"a {part.get('type')} part without data (a data URL or base64)")
    try:
        return data_bytes(u) if u.startswith("data:") else base64.b64decode(u, validate=True)
    except ValueError:
        raise bad_request(f"a {part.get('type')} part's data does not decode") from None


def audio_clip(part: dict[str, Any]) -> bytes:
    """An ``input_audio`` part's clip (base64 data) or an ``audio_url`` part's (a data URI)."""
    from ..lfm3a.host import part_audio  # (numpy only: the model's host side)
    try:
        return part_audio(part)
    except ValueError as e:
        raise bad_request(str(e)) from None


def takes_parts(template: str) -> bool:
    """Whether a chat template renders a message's content given as a list of text parts (Qwen's and Gemma's join
    them their own way; others take text, which the server joins with newlines: llama.cpp's rule)."""
    parts = [{"type": "text", "text": "\x01a"}, {"type": "text", "text": "\x01b"}]
    try:
        text = chat_template(template, [{"role": "user", "content": parts}])
    except Exception:  # noqa: BLE001 (any failure: it does not)
        return False
    return "\x01a" in text and "\x01b" in text and "'type'" not in text


def system_first(template: str) -> bool:
    """Whether a chat template takes a system message only as the first message (Qwen's raise for a later one)."""
    try:
        chat_template(template, [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}])
    except jinja2.TemplateError:
        return True
    except Exception:  # noqa: BLE001 (another failure: not this one)
        return False
    return False


def text_of(content: object) -> str:
    """A message's text: its content, or its text parts joined by blank lines."""
    if isinstance(content, list):
        return "\n\n".join(str(p.get("text", "")) for p in content if p.get("type") == "text")
    return str(content or "")


class PromptBuilder:
    """The prompts of ``ctx``'s model: its chat template's properties and the server's options."""

    def __init__(self, ctx: ApiContext):
        self.ctx = ctx

    def messages_of(self, messages: list[Message], *, media: bool = False) -> list[dict[str, object]]:
        """The messages for the chat template: content as text parts if the template takes them (else joined by
        newlines), with images (``image_url`` parts of data URIs, a model with a vision encoder) as parts
        ``{"type": "image", "image": bytes}``, audio clips (``input_audio`` parts, ``audio_url`` ones of data URIs,
        a model with a speech encoder) as parts ``{"type": "audio", "audio": bytes}`` and videos (``input_video`` /
        ``video_url`` parts, a model that takes them: ``Capabilities.video``) as ``{"type": "video", "video":
        bytes}`` among the text parts; ``media``: images, clips and videos as the text ``MEDIA_MARKER`` (the
        template's text alone)."""
        ctx = self.ctx
        out = []
        mods = Modalities.of(ctx.engine)
        vision, audio, video = mods.vision, mods.audio, mods.video
        for m in messages:
            if m.role not in ROLES:
                raise bad_request(f"message role {m.role!r} (one of {', '.join(ROLES)})")
            role = "system" if m.role == "developer" else m.role
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
                    elif kind in ("input_audio", "audio_url") and audio:
                        parts.append({"type": "text", "text": MEDIA_MARKER} if media else
                                     {"type": "audio", "audio": audio_clip(part)})
                    elif kind in ("input_video", "video_url") and video:
                        parts.append({"type": "text", "text": MEDIA_MARKER} if media else
                                     {"type": "video", "video": video_bytes(part)})
                    else:
                        takes = [n for n, ok in (("images (image_url)", vision), ("audio (input_audio)", audio),
                                                 ("videos (input_video)", video)) if ok]
                        raise bad_request(f"content of type {kind!r} is not supported: this server's model takes "
                                          f"text{' and ' + ' and '.join(takes) if takes else ' only'}")
                content = parts if ctx.typed or any(p["type"] in MEDIA_TYPES for p in parts) else \
                    "\n".join(p["text"] for p in parts)
            d: dict[str, object] = ctx.replies.said_as({**(m.model_extra or {}), "role": role,
                                                        "content": content or ""})
            if m.reasoning_content:
                d["reasoning_content"] = m.reasoning_content
            try:
                out.append(call_arguments(d))
            except ValueError as e:
                raise bad_request(str(e)) from None
        if not out:
            raise bad_request("no messages")
        if ctx.first:  # (the template takes a system message only first)
            lead = next((i for i, d in enumerate(out) if d["role"] != "system"), len(out))
            if lead > 1:
                out[:lead] = [{"role": "system",
                               "content": "\n\n".join(text_of(d["content"]) for d in out[:lead]).strip()}]
            for d in out[1:]:
                if d["role"] == "system":
                    d["role"] = "user"
        return out

    def template_options(self, kwargs: dict[str, Any] | None, effort: str | None = None
                         ) -> tuple[bool, bool, dict[str, Any]]:
        """Whether the model thinks (``enable_thinking``), whether the earlier answers keep their thoughts
        (``preserve_reasoning``, llama-server's, or the templates' own ``preserve_thinking``) and the template's other
        variables (``reasoning_effort``, ...), of a request's ``chat_template_kwargs`` over its ``reasoning_effort``
        (``effort``: ``none``, no thought; another level: a thought and the template's ``reasoning_effort``) over the
        server's (``--reasoning-effort`` over ``--chat-template-kwargs``)."""
        ctx = self.ctx
        options = ctx.options
        server = {**options.chat_template_kwargs,
                  **({"reasoning_effort": options.reasoning_effort} if options.reasoning_effort else {})}
        asked = {} if effort is None else {"enable_thinking": effort != "none"} | \
            ({"reasoning_effort": effort} if effort != "none" else {})
        kw = {**server, **asked, **(kwargs or {})}
        if effort == "none" and "reasoning_effort" not in (kwargs or {}):
            kw.pop("reasoning_effort", None)
        for k in kw.keys() & TEMPLATE_RESERVED:
            ctx.warn_once(("chat_template_kwargs", k), "chat_template_kwargs.%s is the request's, not a variable: "
                          "ignored", k)
        think = bool(kw["enable_thinking"]) if "enable_thinking" in kw else ctx.default_think()
        given = next((d[k] for d in (kwargs or {}, options.chat_template_kwargs)
                      for k in ("preserve_thinking", "preserve_reasoning") if k in d), None)
        preserve = options.preserve_reasoning if given is None else bool(given)
        variables = {k: v for k, v in kw.items()
                     if k not in TEMPLATE_RESERVED | {"enable_thinking", "preserve_reasoning", "preserve_thinking"}}
        return think and ctx.engine.model.thinks, preserve, variables

    async def render(self, messages: list[Message], kwargs: dict[str, Any] | None, *, effort: str | None = None,
                     media: bool = False, tools: list[dict[str, Any]] | None = None) -> tuple[list[int], bool]:
        """The prompt's tokens through the chat template (with images: ids and embedding rows, the encoder run on the
        engine's worker between its jobs; ``media``: images as text, ``messages_of``; ``tools``: the functions the
        model may call; ``kwargs``, ``effort``: a request's ``chat_template_kwargs`` and ``reasoning_effort``,
        ``template_options``), and whether the model thinks. A request's ``reasoning_effort`` level that the template
        refuses (Qwen3.8's takes xhigh, medium and low; clients send high or minimal) gives way to the server's, else
        the template's default, with a log line; one in ``chat_template_kwargs`` fails the request."""
        ctx = self.ctx
        engine = ctx.engine
        think, preserve, variables = self.template_options(kwargs, effort)
        msgs = self.messages_of(messages, media=media)
        tools = tools or None

        async def run(variables: dict[str, Any]) -> list[int]:
            def chat() -> list[int]:
                return engine.model.chat(msgs, think=think, tools=tools, preserve=preserve, variables=variables)

            if any(p.get("type") in MEDIA_TYPES for m in msgs if isinstance(m["content"], list)
                   for p in m["content"]):  # (an encoder on the NPU: between the worker's jobs)
                return await asyncio.wrap_future(engine.run(chat))
            return await run_in_threadpool(chat)

        try:
            try:
                return await run(variables), think
            except jinja2.TemplateError as e:
                if not effort or variables.get("reasoning_effort") != effort or "reasoning_effort" in (kwargs or {}):
                    raise
                # (the server's, else the template's)
                default = self.template_options(kwargs)[2].get("reasoning_effort")
                ids = await run({**variables, "reasoning_effort": default} if default else
                                {k: v for k, v in variables.items() if k != "reasoning_effort"})
                # (the template took the level away: it refused it)
                ctx.warn_once(("reasoning_effort", effort), "reasoning_effort %r: the chat template refuses it (%s): "
                              "%s instead", effort, e, f"the server's {default!r}" if default else "the template's "
                              "default")
                return ids, think
        except jinja2.TemplateError as e:
            raise bad_request(f"the chat template failed: {e}") from None
        except (OSError, ValueError) as e:  # (an image or video that does not open, or does not fit the prompt)
            raise bad_request(str(e)) from None

    def inside(self, ids: list[int]) -> bool:
        """Whether a prompt ends inside a thought (its last text is the model's open marker)."""
        model = self.ctx.engine.model
        markers = model.reasoning
        if markers is None:
            return False
        last = [int(t) for t in getattr(ids, "ids", ids)[-8:] if t >= 0]  # (a sequence's ids; an image's rows: none)
        tail = model.tokenizer.decode_bytes(last, special=True).decode("utf-8", errors="replace")
        return tail.rstrip().endswith(markers[0])

    def calls_of(self, req: ChatRequest) -> tuple[list[dict[str, Any]], Calls | None]:
        """A request's functions for the template and how to read its answer's calls (None: as text: no tools, or
        ``tool_choice`` none). ``tool_choice`` required or a named function: the answer begins with a call (its opener
        forced; a named function's call alone). A template that takes no tools: they are ignored (the web UI sends its
        own)."""
        syntax = self.ctx.syntax
        try:
            fns = functions(req.tools)
        except ValueError as e:
            raise bad_request(str(e)) from None
        choice = "auto" if req.tool_choice is None else req.tool_choice
        name = None
        if isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") if choice.get("type") == "function" else None
            if not isinstance(name, str):
                raise bad_request('tool_choice: "auto", "none", "required" or {"type": "function", "function": '
                                  '{"name": ...}}')
            if name not in {f["function"]["name"] for f in fns}:
                raise bad_request(f"tool_choice names {name!r}, not one of the tools")
        elif choice not in ("auto", "none", "required"):
            raise bad_request(f'tool_choice {choice!r} (one of "auto", "none", "required" or a function)')
        if fns and syntax is None:
            self.ctx.warn_once(("tools", ""), "tools are ignored: this model's chat template takes none")
            fns = []
        if not fns:
            if choice not in ("auto", "none"):
                raise bad_request("tool_choice without tools" if syntax is not None else
                                  f"tool_choice {choice!r}: this model's chat template takes no tools")
            return [], None
        if choice == "none":
            return fns, None
        force = syntax.opener(name) if choice == "required" or name else None
        parallel = req.parallel_tool_calls is not False and name is None  # (a named function: that call alone)
        return fns, Calls(syntax, fns, parallel=parallel, force=force)

    async def vocabulary(self) -> Vocabulary | None:
        """The model's tokens by their text (``tools.Vocabulary``: a few tenths of a second, in a thread, once), or
        None without the vocabulary's size."""
        ctx = self.ctx
        async with ctx.vocab_lock:
            if not ctx.vocab and ctx.info.n_vocab:
                ctx.vocab.append(await run_in_threadpool(Vocabulary, ctx.engine.model.tokenizer, ctx.info.n_vocab))
        return ctx.vocab[0] if ctx.vocab else None

    async def counted(self, req: ChatRequest) -> int:
        """The prompt tokens of a chat request."""
        ids, _ = await self.render(req.messages, req.chat_template_kwargs, effort=req.reasoning_effort,
                                   tools=self.calls_of(req)[0])
        return len(ids)
