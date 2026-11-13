"""OpenAI's realtime API as llama-server's: live transcription sessions (the websocket
``/v1/realtime?intent=transcription`` of a model with LFM3-Audio's live ASR: ``models.lfm3a.live``, its speech encoder
and the ``asr`` chat template: ``LiveAsrSession``; a model with a speech encoder and no live ASR, LFM3-Omni: each
committed buffer transcribed alone, ``TranscriptionSession``) and conversations (``ConversationSession``: other
sessions of a model that speaks, continued turn by turn on the prompt cache: ``conversation.Conversation``).

A session (one stream): ``session.created`` (PCM16 mono at 16 kHz in), then the client's events -
``input_audio_buffer.append`` (base64 PCM16), ``input_audio_buffer.commit`` (the audio so far padded with silence to a
chunk's end: its text, then the next item; ``"final": true`` ends the stream), ``input_audio_buffer.clear``,
``session.update`` (answered, nothing to set) - and the server's ``conversation.item.input_audio_transcription.delta``
as each chunk's text comes out, ``...completed`` per item (the transcript). The stream's chunks run on the engine's
worker between other requests' jobs (the stream's text sequence back through the prompt cache, its encoder state
restored when a clip ran meanwhile).

Every session is a ``Session``: it receives the client's JSON events and dispatches each by its ``type`` to a handler
(``handlers``; a handler that returns True ends the session), and answers what no handler takes with an error event."""

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..detokenizer import StreamDecoder
from ..sequence import data_bytes
from ..templating import file_template
from .conversation import Conversation
from .engine import Engine
from .ids import hex_id
from .speech import speaks, worker
from .tools import functions, syntax_of

log = logging.getLogger("artlite.server")

MODES = {None: "auto", ("text",): "text", ("audio",): "audio", ("audio", "text"): "mixed"}  # (output_modalities)
Handler = Callable[[dict[str, Any]], Awaitable[bool | None]]
TRANSCRIPTION_EVENT = "conversation.item.input_audio_transcription"


def _responses_tools(tools: object) -> list[dict[str, Any]]:
    """A realtime session's ``tools`` (the Responses API's form, ``{"type": "function", "name", ...}``) as the chat
    templates take them (``tools.functions``: ValueError, not a list of functions)."""
    if isinstance(tools, list):
        tools = [t if not isinstance(t, dict) or "function" in t else
                 {"type": t.get("type"), "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
                 for t in tools]
    return functions(tools)


class Session:
    """A realtime websocket session: the server's events out (``event``, ``error``), the client's in (``loop``).
    A subclass sets ``handlers`` (the client's event types to coroutines of the event; True ends the session)."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.handlers: dict[str, Handler] = {}

    async def event(self, ev: dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps({"event_id": hex_id("event_", 8), **ev}))

    async def error(self, message: str, kind: str = "invalid_request_error") -> None:
        await self.event({"type": "error", "error": {"type": kind, "code": None, "message": message, "param": None}})

    async def pcm(self, ev: dict[str, Any]) -> np.ndarray | None:
        """An ``input_audio_buffer.append``'s samples (base64 PCM16 as float32); None (the client told): not base64."""
        try:
            pcm = base64.b64decode(str(ev.get("audio", "")), validate=True)
        except binascii.Error:
            await self.error("invalid base64 audio")
            return None
        return np.frombuffer(pcm[:len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768

    async def before(self) -> None:
        """Before each event's handler."""

    async def loop(self) -> None:
        """The client's events, each to its handler, until a handler returns True."""
        while True:
            try:
                ev = json.loads(await self.ws.receive_text())
                kind = ev.get("type") if isinstance(ev, dict) else None
            except ValueError:
                await self.error("invalid JSON")
                continue
            await self.before()
            handler = self.handlers.get(kind) if isinstance(kind, str) else None
            if handler is None:
                await self.error(f"unknown event type: {kind}")
            elif await handler(ev):
                return


class LiveAsrSession(Session):
    """A live transcription session over ``models.lfm3a.live.LiveAsr`` (``live``), whose steps run on the engine's
    worker: ``input_audio_buffer.append`` pushes samples, ``.commit`` ends a chunk's item (``final``: the stream)."""

    def __init__(self, ws: WebSocket, engine: Engine, alias: str, live: Any):
        super().__init__(ws)
        self.engine, self.live = engine, live
        self.session = {"id": hex_id("sess_", 12), "object": "realtime.transcription_session",
                        "type": "transcription",
                        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000},
                                            "transcription": {"model": alias}, "turn_detection": None,
                                            "noise_reduction": None}}}
        self.item, self.prev, self.text = hex_id("item_", 12), None, []
        self.handlers = {"session.update": self.update, "transcription_session.update": self.update,
                         "input_audio_buffer.append": self.append, "input_audio_buffer.commit": self.commit,
                         "input_audio_buffer.clear": self.clear}

    @classmethod
    async def open(cls, ws: WebSocket, engine: Engine, alias: str) -> "LiveAsrSession | None":
        """The session of the engine's model, or None (the client told: the model refuses)."""
        from ..lfm3a import live as lv

        model = engine.model
        try:
            live = await asyncio.wrap_future(engine.live(lambda: lv.LiveAsr(model, engine.npu, model.audio.encoder,
                                                                            sync=(engine.hold, engine.ran))))
        except Exception as e:  # (the model refuses)
            log.exception("realtime transcription failed")
            await Session(ws).error(f"{type(e).__name__}: {e}", "server_error")
            return None
        return cls(ws, engine, alias, live)

    async def step(self, fn: Callable[[], str]) -> None:
        out = await asyncio.wrap_future(self.engine.live(fn))
        if out:
            self.text.append(out)
            await self.event({"type": f"{TRANSCRIPTION_EVENT}.delta", "item_id": self.item, "content_index": 0,
                              "delta": out})

    async def completed(self) -> None:
        await self.event({"type": f"{TRANSCRIPTION_EVENT}.completed", "item_id": self.item, "content_index": 0,
                          "transcript": "".join(self.text)})
        self.text.clear()

    async def update(self, ev: dict[str, Any]) -> None:
        await self.event({"type": "session.updated", "session": self.session})

    async def append(self, ev: dict[str, Any]) -> None:
        x = await self.pcm(ev)
        if x is not None and len(x):
            await self.step(lambda: self.live.push(x))

    async def commit(self, ev: dict[str, Any]) -> bool | None:
        if ev.get("final"):
            await self.step(self.live.end)
            await self.completed()
            return True
        await self.event({"type": "input_audio_buffer.committed", "previous_item_id": self.prev,
                          "item_id": self.item})
        await self.step(self.live.commit)
        await self.completed()
        self.prev, self.item = self.item, hex_id("item_", 12)
        return None

    async def clear(self, ev: dict[str, Any]) -> None:
        await self.event({"type": "input_audio_buffer.cleared"})

    async def run(self) -> None:
        await self.event({"type": "session.created", "session": self.session})
        try:
            await self.loop()
        except WebSocketDisconnect:
            return
        except Exception as e:  # (a failure in the model: the session's error)
            log.exception("realtime transcription failed")
            await self.error(f"{type(e).__name__}: {e}", "server_error")
        live = self.live
        log.info("realtime transcription: %d chunks; the encoder %.2f s, the text %.2f s", live.chunks,
                 live.stream.t_encode, live.t_text)
        await self.ws.close()


class TranscriptionSession(Session):
    """A transcription session of a model not trained for live ASR (LFM3-Omni; llama-server's): each committed buffer
    transcribed alone with the offline ASR prompt (``transcribe(samples, language)``: its text's pieces as they come),
    as ``conversation.item.input_audio_transcription.delta`` events and the item's ``...completed``."""

    def __init__(self, ws: WebSocket, alias: str, transcribe: Callable[[np.ndarray, str], AsyncIterator[str]]):
        super().__init__(ws)
        self.transcribe = transcribe
        self.session = {"id": hex_id("sess_", 12), "object": "realtime.transcription_session",
                        "type": "transcription",
                        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000},
                                            "transcription": {"model": alias, "language": None},
                                            "turn_detection": None, "noise_reduction": None}}}
        self.buf: list[np.ndarray] = []
        self.prev: str | None = None
        self.handlers = {"session.update": self.update, "transcription_session.update": self.update,
                         "input_audio_buffer.append": self.append, "input_audio_buffer.commit": self.commit,
                         "input_audio_buffer.clear": self.clear}

    async def update(self, ev: dict[str, Any]) -> None:
        tr = (((ev.get("session") or {}).get("audio") or {}).get("input") or {}).get("transcription") or {}
        if "language" in tr:
            self.session["audio"]["input"]["transcription"]["language"] = tr["language"]
        await self.event({"type": "session.updated", "session": self.session})

    async def append(self, ev: dict[str, Any]) -> None:
        x = await self.pcm(ev)
        if x is not None:
            self.buf.append(x)

    async def commit(self, ev: dict[str, Any]) -> bool | None:
        x = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        self.buf.clear()
        final = bool(ev.get("final"))
        if not len(x):
            if not final:
                await self.error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty")
            return final
        item = hex_id("item_", 12)
        await self.event({"type": "input_audio_buffer.committed", "previous_item_id": self.prev, "item_id": item})
        text = []
        async for piece in self.transcribe(x, self.session["audio"]["input"]["transcription"]["language"] or ""):
            text.append(piece)
            await self.event({"type": f"{TRANSCRIPTION_EVENT}.delta", "item_id": item, "content_index": 0,
                              "delta": piece})
        await self.event({"type": f"{TRANSCRIPTION_EVENT}.completed", "item_id": item, "content_index": 0,
                          "transcript": "".join(text)})
        self.prev = item
        return final

    async def clear(self, ev: dict[str, Any]) -> None:
        self.buf.clear()
        await self.event({"type": "input_audio_buffer.cleared"})

    async def run(self) -> None:
        await self.event({"type": "session.created", "session": self.session})
        try:
            await self.loop()
        except WebSocketDisconnect:
            return
        except Exception as e:  # (a failure in the model: the session's error)
            log.exception("realtime transcription failed")
            await self.error(f"{type(e).__name__}: {e}", "server_error")
        await self.ws.close()


class ConversationSession(Session):
    """A realtime conversation (llama-server's "Realtime conversation", a model that speaks: ``Conversation``): the
    session's ``instructions`` (the system prompt), ``tools`` (the Responses API's functions; the model's calls as
    ``function_call`` items, their outputs back as ``function_call_output`` items), ``output_modalities`` (left out: the
    model speaks or not by itself; ["text"], ["audio"]: speech, then text; ["text", "audio"]: interleaved),
    ``max_output_tokens``, the voice (``audio.output.voice``: female, male); the user turns' audio
    (``input_audio_buffer.append``: PCM16 at 16 kHz, run on the model as it arrives; ``.commit``), text and images
    (``conversation.item.create``: ``input_text``, ``input_image`` parts of data URLs: a camera frame);
    ``response.create`` generates the reply on the engine's worker after the turn's last rows, streamed as
    ``response.output_text.delta`` and ``response.output_audio.delta`` (base64 PCM16 at 24 kHz) between the
    ``response.*`` start and done events; ``response.cancel`` stops it. The instructions and tools are fixed at the
    session's first audio, message or response."""

    def __init__(self, ws: WebSocket, engine: Engine, alias: str):
        super().__init__(ws)
        self.engine, self.model = engine, engine.model
        self.session: dict[str, Any] = {
            "id": hex_id("sess_", 12), "object": "realtime.session", "type": "realtime", "model": alias,
            "instructions": "", "output_modalities": None, "max_output_tokens": "inf", "tools": [],
            "tool_choice": "auto",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}},
                      "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "female"}}}
        self.conv = Conversation(engine, self.model)
        self.fns: list[dict[str, Any]] = []
        self.task: asyncio.Task | None = None  # (the response being generated)
        self.stop: threading.Event | None = None  # (its cancel)
        self.last: str | None = None  # (the last item's id)
        self.has_audio = False
        self.handlers = {"session.update": self.update, "input_audio_buffer.append": self.append,
                         "input_audio_buffer.commit": self.commit_audio, "input_audio_buffer.clear": self.clear,
                         "conversation.item.create": self.create_item, "response.create": self.create_response,
                         "response.cancel": self.cancel_response}

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def template(self) -> str:
        return file_template(self.model.gguf, self.model.chat_template)

    async def on_worker(self, fn: Callable[[], Any]) -> Any:
        return await asyncio.wrap_future(self.engine.run(fn))

    async def started(self) -> None:
        if self.conv.turns is None:
            await self.on_worker(lambda: self.conv.start(self.session["instructions"], self.fns))

    async def item_done(self, item: dict[str, Any]) -> None:
        await self.event({"type": "conversation.item.added", "previous_item_id": self.last, "item": item})
        await self.event({"type": "conversation.item.done", "previous_item_id": self.last, "item": item})
        self.last = item["id"]

    async def commit(self) -> bool:
        """The user turn's audio ends (False: there is none)."""
        if not self.has_audio:
            return False
        await self.on_worker(self.conv.commit)
        self.has_audio = False
        item = hex_id("item_", 12)
        await self.event({"type": "input_audio_buffer.committed", "previous_item_id": self.last, "item_id": item})
        await self.item_done({"id": item, "object": "realtime.item", "type": "message", "role": "user",
                              "status": "completed", "content": [{"type": "input_audio"}]})
        return True

    async def before(self) -> None:
        if self.task is not None and self.task.done() and self.task.exception() is not None:
            log.error("realtime response failed: %s", self.task.exception())

    # ------------------------------------------------------------------
    # The client's events
    # ------------------------------------------------------------------

    async def update(self, ev: dict[str, Any]) -> None:
        session = self.session
        up = ev.get("session") or {}
        if self.conv.turns is not None and any(k in up for k in ("instructions", "tools")):
            await self.error("session.update: instructions and tools cannot change after the first audio or message")
            return
        if "tools" in up:
            try:
                self.fns = _responses_tools(up["tools"])
            except ValueError as e:
                await self.error(f"session.update: {e}")
                return
            if self.fns and syntax_of(self.template) is None:
                await self.error("session.update: the model's chat template has no tool calls")
                self.fns = []
                return
        for k in ("instructions", "output_modalities", "max_output_tokens", "type", "tools", "tool_choice"):
            if k in up:
                session[k] = up[k]
        v = ((up.get("audio") or {}).get("output") or {}).get("voice")
        if v in ("female", "male"):
            session["audio"]["output"]["voice"] = v
            session["instructions"] = session["instructions"] or ("" if v == "female" else "Respond in a male voice.")
        await self.event({"type": "session.updated", "session": session})

    async def append(self, ev: dict[str, Any]) -> None:
        x = await self.pcm(ev)
        if x is None or not len(x):
            return
        if self.busy:
            await self.error("input_audio_buffer.append: a response is in progress")
            return
        await self.started()
        await self.on_worker(lambda: self.conv.audio(x))
        self.has_audio = True

    async def commit_audio(self, ev: dict[str, Any]) -> None:
        if not await self.commit():
            await self.error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty")

    async def clear(self, ev: dict[str, Any]) -> None:
        await self.event({"type": "input_audio_buffer.cleared"})  # (what ran stays: the model heard it)

    async def create_item(self, ev: dict[str, Any]) -> None:
        conv = self.conv
        it = ev.get("item") or {}
        if self.busy:
            await self.error("conversation.item.create: a response is in progress")
            return
        if it.get("type") == "function_call_output":
            if not self.fns or conv.n == 0 or conv.open:
                await self.error("conversation.item.create: a function call output must follow the response that "
                                 "called it")
                return
            out = str(it.get("output", ""))
            await self.on_worker(lambda: conv.output(out))
            await self.item_done({"id": it.get("id") or hex_id("item_", 12), "object": "realtime.item",
                                  "type": "function_call_output", "call_id": it.get("call_id", ""), "output": out,
                                  "status": "completed"})
            return
        if it.get("role", "user") != "user" or it.get("type", "message") != "message":
            await self.error("conversation.item.create: only user messages and function call outputs are supported")
            return
        content = [c for c in it.get("content") or [] if isinstance(c, dict)]
        bad = [c.get("type") for c in content if c.get("type") not in ("input_text", "text", "input_image")]
        if bad or not content:
            await self.error(f"conversation.item.create: unsupported content {bad or 'none'} (input_text, "
                             "input_image)")
            return
        if any(c.get("type") == "input_image" for c in content) and self.model.vision is None:
            await self.error("conversation.item.create: the model takes no images (an mmproj with a vision tower)")
            return
        await self.started()
        if self.has_audio:
            await self.commit()
        try:
            for c in content:
                if c.get("type") == "input_image":
                    url = c.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    data = data_bytes(str(url))
                    await self.on_worker(lambda data=data: conv.image(data))
                else:
                    t = str(c.get("text", ""))
                    await self.on_worker(lambda t=t: conv.text(t))
        except (OSError, ValueError) as e:
            await self.error(f"conversation.item.create: {e}")
            return
        await self.item_done({"id": it.get("id") or hex_id("item_", 12), "object": "realtime.item",
                              "type": "message", "role": "user", "status": "completed",
                              "content": [{"type": c.get("type")} | ({"text": c.get("text", "")}
                                          if c.get("type") != "input_image" else {}) for c in content]})

    async def create_response(self, ev: dict[str, Any]) -> None:
        if self.busy:
            await self.error("a response is in progress")
            return
        await self.started()
        await self.commit()
        if not self.conv.open and not self.conv.results:
            await self.error("response.create: commit audio or create a message first", "conversation_empty")
            return
        self.task = asyncio.create_task(self.respond(ev.get("response") or {}))

    async def cancel_response(self, ev: dict[str, Any]) -> None:
        if self.stop is not None:
            self.stop.set()

    # ------------------------------------------------------------------
    # A response
    # ------------------------------------------------------------------

    async def respond(self, over: dict[str, Any]) -> None:
        """A response (``over``: the request's own ``output_modalities`` and ``max_output_tokens``): the reply
        generated on the worker, streamed as the ``response.*`` events."""
        session, model, conv = self.session, self.model, self.conv
        mods = over.get("output_modalities", session["output_modalities"])
        mode = MODES.get(tuple(sorted(mods)) if mods else None)
        if mode is None:
            await self.error(f"output_modalities {mods!r}: text and / or audio")
            return
        mx = over.get("max_output_tokens", session["max_output_tokens"])
        mx = 4096 if mx in (None, "inf") else int(mx)
        syntax = syntax_of(self.template) if self.fns else None
        parser = syntax.parser(self.fns) if syntax is not None else None

        def produce(push: Callable[[tuple[str, Any]], None], stop: threading.Event) -> None:
            from ..lfm3a.speech import SpeechStats
            self.stop = stop
            dec, st = StreamDecoder(model.tokenizer), SpeechStats()
            for kind, x in conv.respond(mode, mx, st):
                if stop.is_set():
                    break
                if kind == "text":
                    if piece := dec.feed(x):
                        push(("text", piece))
                elif len(x):
                    push(("audio", x))
            push(("done", (stop.is_set(), st)))

        q, self.stop = worker(self.engine, produce, taken=False)
        out = ResponseEvents(self, hex_id("resp_", 12), hex_id("item_", 12))
        await out.created()
        status, n_out = "completed", 0
        while True:
            kind, x = await q.get()
            if kind == "text":
                await (out.parts(parser.parts([("content", x)])) if parser is not None else out.content(x))
            elif kind == "audio":
                await out.audio(x)
            elif kind == "error":
                await self.error(x, "server_error")
                status = "failed"
                break
            else:
                stopped, st = x
                status, n_out = "cancelled" if stopped else "completed", st.frames
                break
        if parser is not None:
            await out.parts(parser.flush())
        await out.finish(status, n_out)

    async def run(self) -> None:
        await self.event({"type": "session.created", "session": self.session})
        try:
            await self.loop()
        except WebSocketDisconnect:
            if self.stop is not None:
                self.stop.set()
            return
        finally:
            if self.task is not None and not self.task.done():
                if self.stop is not None:
                    self.stop.set()
                with contextlib.suppress(Exception):
                    await self.task


class ResponseEvents:
    """The ``response.*`` events of one response of a conversation session: its message item (the text and audio
    parts, by content index) and the function call items the parser finds in the text."""

    def __init__(self, session: ConversationSession, rid: str, item: str):
        self.session, self.rid, self.item = session, rid, item
        self.output: list[dict[str, Any]] = [{"id": item, "object": "realtime.item", "type": "message",
                                              "role": "assistant", "status": "in_progress", "content": []}]
        self.base = {"response_id": rid, "item_id": item, "output_index": 0}
        self.text: list[str] = []
        self.kinds: dict[str, int] = {}  # (the message's parts by kind: their content index)
        self.calls: dict[int, int] = {}  # (a call's index: its output item)

    async def created(self) -> None:
        event = self.session.event
        await event({"type": "response.created", "response": {"id": self.rid, "object": "realtime.response",
                                                               "status": "in_progress", "output": []}})
        await event({"type": "response.output_item.added", "response_id": self.rid, "output_index": 0,
                     "item": self.output[0]})

    async def part(self, kind: str) -> int:
        if kind not in self.kinds:
            self.kinds[kind] = len(self.kinds)
            p = {"type": kind, "text": ""} if kind == "output_text" else {"type": kind, "transcript": ""}
            await self.session.event({"type": "response.content_part.added", **self.base,
                                      "content_index": self.kinds[kind], "part": p})
        return self.kinds[kind]

    async def content(self, piece: str) -> None:
        self.text.append(piece)
        await self.session.event({"type": "response.output_text.delta", **self.base,
                                  "content_index": await self.part("output_text"), "delta": piece})

    async def audio(self, samples: np.ndarray) -> None:
        pcm = (np.clip(samples, -1, 1) * 32767).round().astype("<i2").tobytes()
        await self.session.event({"type": "response.output_audio.delta", **self.base,
                                  "content_index": await self.part("output_audio"),
                                  "delta": base64.b64encode(pcm).decode()})

    async def call(self, c: Any) -> None:
        event, output, calls = self.session.event, self.output, self.calls
        if c.index not in calls:
            calls[c.index] = len(output)
            output.append({"id": hex_id("item_", 12), "object": "realtime.item", "type": "function_call",
                           "status": "in_progress", "name": c.name, "call_id": c.id, "arguments": ""})
            await event({"type": "response.output_item.added", "response_id": self.rid,
                         "output_index": calls[c.index], "item": output[calls[c.index]]})
        it = output[calls[c.index]]
        if c.arguments:
            it["arguments"] += c.arguments
            await event({"type": "response.function_call_arguments.delta", "response_id": self.rid,
                         "item_id": it["id"], "output_index": calls[c.index], "call_id": it["call_id"],
                         "delta": c.arguments})

    async def parts(self, ps: list[tuple[str, Any]]) -> None:
        for k, v in ps:
            await (self.content(v) if k == "content" else self.call(v) if k == "tool_call" else asyncio.sleep(0))

    async def finish(self, status: str, n_out: int) -> None:
        """The parts', calls' and items' done events, then ``response.done``."""
        event, base, rid, output = self.session.event, self.base, self.rid, self.output
        transcript = "".join(self.text)
        ps = []
        for kind, i in sorted(self.kinds.items(), key=lambda kv: kv[1]):
            if kind == "output_text":
                await event({"type": "response.output_text.done", **base, "content_index": i, "text": transcript})
                ps.append({"type": kind, "text": transcript})
            else:
                await event({"type": "response.output_audio.done", **base, "content_index": i})
                ps.append({"type": kind, "transcript": transcript})
            await event({"type": "response.content_part.done", **base, "content_index": i, "part": ps[-1]})
        output[0]["content"] = ps
        for i in self.calls.values():
            it = output[i]
            await event({"type": "response.function_call_arguments.done", "response_id": rid, "item_id": it["id"],
                         "output_index": i, "call_id": it["call_id"], "name": it["name"],
                         "arguments": it["arguments"]})
        for i, it in enumerate(output):
            it["status"] = status
            await event({"type": "response.output_item.done", "response_id": rid, "output_index": i, "item": it})
            self.session.last = it["id"]
        await event({"type": "response.done", "response": {"id": rid, "object": "realtime.response", "status": status,
                                                           "output": output, "usage": {"output_tokens": n_out}}})


def add_route(app: FastAPI, *, engine: Engine, alias: str, authorized: Callable[[str | None, str | None], bool],
              transcribe: Callable[[np.ndarray, str], AsyncIterator[str]] | None = None) -> None:
    """``/v1/realtime``: ``authorized(authorization, x_api_key)``: the request's headers pass the server's API keys;
    ``transcribe``: a clip's offline transcript as its pieces come (the transcription sessions of a model without live
    ASR)."""

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        if not authorized(ws.headers.get("authorization"), ws.headers.get("x-api-key")):
            await ws.close(code=1008, reason="Invalid API Key")
            return
        await ws.accept()
        model = engine.model if engine.ready else None
        transcribing = ws.query_params.get("intent") == "transcription"
        if not transcribing and model is not None and speaks(engine):
            await ConversationSession(ws, engine, alias).run()
            return
        live_asr = model is not None and model.capabilities.live_asr
        if transcribing and model is not None and model.audio is not None and not live_asr and transcribe is not None:
            await TranscriptionSession(ws, alias, transcribe).run()
            return
        if not transcribing or model is None or model.audio is None or not live_asr:
            session = Session(ws)
            await session.error("The current model does not support realtime transcription (a model with a speech "
                                "encoder: LFM3-Audio, LFM3-Omni; ?intent=transcription).")
            await ws.close()
            return
        session = await LiveAsrSession.open(ws, engine, alias)
        if session is None:
            await ws.close()
            return
        await session.run()
