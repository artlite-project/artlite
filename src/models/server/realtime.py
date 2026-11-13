"""OpenAI's realtime API as llama-server's: live transcription sessions (the websocket
``/v1/realtime?intent=transcription`` of a model with LFM3-Audio's live ASR: ``models.lfm3a.live``, its speech encoder
and the ``asr`` chat template; a model with a speech encoder and no live ASR, LFM3-Omni: each committed buffer
transcribed alone, ``transcription``) and conversations (``conversation``: other sessions of a model that speaks,
continued turn by turn on the prompt cache: ``Conversation``).

A session (one stream): ``session.created`` (PCM16 mono at 16 kHz in), then the client's events -
``input_audio_buffer.append`` (base64 PCM16), ``input_audio_buffer.commit`` (the audio so far padded with silence to a
chunk's end: its text, then the next item; ``"final": true`` ends the stream), ``input_audio_buffer.clear``,
``session.update`` (answered, nothing to set) - and the server's ``conversation.item.input_audio_transcription.delta``
as each chunk's text comes out, ``...completed`` per item (the transcript). The stream's chunks run on the engine's
worker between other requests' jobs (the stream's text sequence back through the prompt cache, its encoder state
restored when a clip ran meanwhile)."""

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import secrets
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..common import StreamDecoder
from ..sequence import Sequence
from .cache import common_prefix
from .engine import Engine
from .speech import speaks, worker

log = logging.getLogger("artlite.server")


def add_route(app: FastAPI, *, engine: Engine, alias: str, authorized: Callable[[str | None, str | None], bool],
              transcribe: Callable[[np.ndarray, str], Any] | None = None) -> None:
    """``authorized(authorization, x_api_key)``: the request's headers pass the server's API keys; ``transcribe``: a
    clip's offline transcript as its pieces come (the transcription sessions of a model without live ASR)."""

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        if not authorized(ws.headers.get("authorization"), ws.headers.get("x-api-key")):
            await ws.close(code=1008, reason="Invalid API Key")
            return
        await ws.accept()

        async def event(ev: dict[str, Any]) -> None:
            await ws.send_text(json.dumps({"event_id": f"event_{secrets.token_hex(8)}", **ev}))

        async def error(message: str, kind: str = "invalid_request_error") -> None:
            await event({"type": "error", "error": {"type": kind, "code": None, "message": message, "param": None}})

        model = engine.model if engine.ready else None
        if ws.query_params.get("intent") != "transcription" and model is not None and speaks(engine):
            await conversation(ws, engine, alias, event, error)
            return
        live_asr = "tokenizer.chat_template.asr" in getattr(getattr(model, "gguf", None), "meta", {})
        if ws.query_params.get("intent") == "transcription" and model is not None and model.audio is not None and \
                not live_asr and transcribe is not None:
            await transcription(ws, transcribe, alias, event, error)
            return
        if ws.query_params.get("intent") != "transcription" or model is None or model.audio is None or not live_asr:
            await error("The current model does not support realtime transcription (a model with a speech encoder: "
                        "LFM3-Audio, LFM3-Omni; ?intent=transcription).")
            await ws.close()
            return
        from ..lfm3a import live as lv

        try:
            live = await asyncio.wrap_future(engine.live(lambda: lv.LiveAsr(model, engine.npu, model.audio.encoder,
                                                                            sync=(engine.hold, engine.ran))))
        except Exception as e:  # (the model refuses)
            log.exception("realtime transcription failed")
            await error(f"{type(e).__name__}: {e}", "server_error")
            await ws.close()
            return
        session = {"id": f"sess_{secrets.token_hex(12)}", "object": "realtime.transcription_session",
                   "type": "transcription",
                   "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}, "transcription": {"model": alias},
                                       "turn_detection": None, "noise_reduction": None}}}
        await event({"type": "session.created", "session": session})
        item, prev, text = f"item_{secrets.token_hex(12)}", None, []

        async def run(fn: Callable[[], str]) -> None:
            out = await asyncio.wrap_future(engine.live(fn))
            if out:
                text.append(out)
                await event({"type": "conversation.item.input_audio_transcription.delta", "item_id": item,
                             "content_index": 0, "delta": out})

        async def completed() -> None:
            await event({"type": "conversation.item.input_audio_transcription.completed", "item_id": item,
                         "content_index": 0, "transcript": "".join(text)})
            text.clear()

        try:
            while True:
                try:
                    ev = json.loads(await ws.receive_text())
                    kind = ev.get("type") if isinstance(ev, dict) else None
                except ValueError:
                    await error("invalid JSON")
                    continue
                if kind in ("session.update", "transcription_session.update"):
                    await event({"type": "session.updated", "session": session})
                elif kind == "input_audio_buffer.append":
                    try:
                        pcm = base64.b64decode(str(ev.get("audio", "")), validate=True)
                    except binascii.Error:
                        await error("invalid base64 audio")
                        continue
                    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768
                    if len(x):
                        await run(lambda x=x: live.push(x))
                elif kind == "input_audio_buffer.commit":
                    if ev.get("final"):
                        await run(live.end)
                        await completed()
                        break
                    await event({"type": "input_audio_buffer.committed", "previous_item_id": prev, "item_id": item})
                    await run(live.commit)
                    await completed()
                    prev, item = item, f"item_{secrets.token_hex(12)}"
                elif kind == "input_audio_buffer.clear":
                    await event({"type": "input_audio_buffer.cleared"})
                else:
                    await error(f"unknown event type: {kind}")
        except WebSocketDisconnect:
            return
        except Exception as e:  # (a failure in the model: the session's error)
            log.exception("realtime transcription failed")
            await error(f"{type(e).__name__}: {e}", "server_error")
        log.info("realtime transcription: %d chunks; the encoder %.2f s, the text %.2f s", live.chunks,
                 live.stream.t_encode, live.t_text)
        await ws.close()


MODES = {None: "auto", ("text",): "text", ("audio",): "audio", ("audio", "text"): "mixed"}  # (output_modalities)
ROWS_STEP = 8  # (a user turn's audio rows run on the language model as they come, 8 at a time: 0.64 s)


class Turns:
    """The text around a realtime conversation's turns, from the model's chat template rendered with placeholder
    messages (llama-server's ``derive_turn_text``): ``first`` (the system turn - instructions, tools - and the first user
    turn's header), ``user`` / ``user_end`` (a later user turn's header, a user turn's end), ``reply`` (the generation
    prompt: the assistant's header and its empty thought), ``reply_end`` (after an answer), ``tool`` / ``tool_end``
    (around a function's output), ``audio`` / ``image`` (around a clip's rows, an image's tokens in a user turn:
    LFM3-Omni's vision markers; LFM3-Audio's clip bare)."""

    def __init__(self, model: Any, instructions: str, tools: list[dict[str, Any]] | None):
        u, r, x = "@@USER@@", "@@REPLY@@", "@@RESULT@@"
        system = [{"role": "system", "content": instructions}] if instructions else []

        def render(msgs: list[dict[str, Any]], gen: bool = False) -> str:  # (``Lfm.chat_text``)
            return model.chat_text(system + msgs, tools=tools or None, generation=gen)
        user = {"role": "user", "content": u}
        a, g = render([user]), render([user], True)
        c = render([user, {"role": "assistant", "content": r}])
        d = render([user, {"role": "assistant", "content": r}, user])
        pa, pc, pd = a.find(u), c.find(r), d.find(u, len(c))
        if min(pa, pc, pd) < 0 or not (c.startswith(a) and d.startswith(c) and g.startswith(a)):
            raise ValueError("the chat template's turns do not compose: no realtime conversation")
        self.first, self.user_end, self.reply = a[:pa], a[pa + len(u):], g[len(a):]
        self.reply_end, self.user = c[pc + len(r):], d[len(c):pd]
        self.tool = self.tool_end = ""
        if tools:
            fn = {"name": tools[0]["function"]["name"], "arguments": {}}
            call = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": fn}]}
            e = render([user, call])
            f = render([user, call, {"role": "tool", "tool_call_id": "call_0", "content": x}])
            pf = f.find(x, len(e))
            if pf < 0 or not f.startswith(e):
                raise ValueError("the chat template's tool results do not compose: no realtime tools")
            self.tool, self.tool_end = f[len(e):pf], f[pf + len(x):]

        def around(kind: str, marker: str) -> tuple[str, str]:
            t = render([{"role": "user", "content": [{"type": "text", "text": "@@A@@"}, {"type": kind},
                                                     {"type": "text", "text": "@@B@@"}]}])
            mid = t[t.find("@@A@@") + 5:t.find("@@B@@")]
            i = mid.find(marker)
            return (mid[:i], mid[i + len(marker):]) if i >= 0 else ("", "")
        self.audio, self.image = around("audio", "<|audio|>"), around("image", "<|image|>")


class Conversation:
    """A realtime conversation's sequence on the model (llama-server's realtime slot): every token and embedding row
    it ran, in order (``seq``): the turns' text (``Turns``), a user turn's audio rows as the speech encoder's stream
    gives them (``models.lfm3a.live.Stream``), an image's tokens, each reply's tokens and its frames' feedback rows
    (``models.lfm3a.speech.Reply``: the next turns see a spoken reply as the model said it). Its methods run on the
    engine's worker; the device holds the sequence through the prompt cache (``Engine.hold``: requests that ran
    between its turns displace it, the cache brings it back; ``Engine.ran``), and a turn's audio runs on the language
    model while it arrives (``ROWS_STEP`` rows at a time), so that a response starts with its last rows and the
    assistant's header."""

    def __init__(self, engine: Engine, model: Any):
        self.engine, self.model = engine, model
        self.seq = Sequence.text([], model.cfg.hidden_size)
        self.turns: Turns | None = None
        self.open = False  # (a user turn is being filled)
        self.after = ""  # (the text before the next turn: the last answer's end)
        self.n = 0  # (responses so far)
        self.results = False  # (a function's output since the last response: the next answers it)
        self.stream: Any = None  # (the open turn's audio: a speech encoder stream)

    def start(self, instructions: str, tools: list[dict[str, Any]] | None) -> None:
        """The session's system turn (fixed at its first audio, message or response)."""
        if self.turns is None:
            self.turns = Turns(self.model, instructions, tools)

    def _text(self, text: str) -> None:
        if text:
            self.seq = self.seq + self.model.tokenizer.encode(text)

    def _turn(self) -> None:
        if not self.open:
            self._text(self.turns.first if self.n == 0 else self.after + self.turns.user)
            self.after, self.open = "", True

    def sync(self) -> None:
        """The device holds the sequence: what the prompt cache keeps of it, the rest run (nothing when the device's
        sequence is it already: a prompt cache keeps a whole prompt only with its last token's logits)."""
        cached = self.engine.cached
        if len(cached) == len(self.seq) and common_prefix(cached, self.seq) == len(self.seq):
            return
        kept = self.engine.hold(self.seq)
        if kept < len(self.seq):
            self.engine.npu.prompt(self.seq[kept:])
        self.engine.ran(self.seq)

    def audio(self, samples: np.ndarray) -> None:
        """More of the user turn's audio (16 kHz): its rows run as they come."""
        self._turn()
        if self.stream is None:
            self._text(self.turns.audio[0])
            self.stream = self.model.audio_stream()
        self.stream.push(samples)
        if self.stream.available() >= ROWS_STEP:
            self._rows(self.stream.available())
            self.sync()

    def _rows(self, n: int) -> None:
        from ..lfm3a.host import clip_sequence
        self.seq = self.seq + clip_sequence(self.stream.take(n))

    def commit(self) -> bool:
        """The turn's audio ends (its last rows, the text after a clip); False: no audio."""
        if self.stream is None:
            return False
        self.stream.end()
        if self.stream.available():
            self._rows(self.stream.available())
        self.stream = None
        self._text(self.turns.audio[1])
        self.sync()
        return True

    def text(self, text: str) -> None:
        self._turn()
        self._text(text)

    def image(self, data: bytes) -> None:
        """An image in the user turn (a camera frame): its tokens between the template's text around one."""
        self._turn()
        self._text(self.turns.image[0])
        try:
            self.seq = self.seq + self.model.vision.sequence(data)
        finally:
            self.model._vision_ran()
        self._text(self.turns.image[1])

    def output(self, text: str) -> None:
        """A function's output (the last response called it)."""
        self._text(self.after + self.turns.tool + text + self.turns.tool_end)
        self.after, self.results = "", True

    def respond(self, mode: str, max_tokens: int, stats: Any, seed: int | None = None) -> Any:
        """The reply's ("text", token) and ("audio", samples) as they come (``Lfm.reply``; ``seed``: its speech codes'
        sampling); its tokens and frames stay in the sequence."""
        from ..lfm3a.speech import Reply
        if self.open:
            self._text(self.turns.user_end)
        self._text(self.turns.reply)
        self.seq = self.seq + self.model.opener(mode)
        stats.prompt = len(self.seq)
        kept = self.engine.hold(self.seq)
        reply = Reply()
        try:
            yield from self.model.reply(self.seq[kept:], mode=mode, seed=seed, max_tokens=max_tokens, reply=reply,
                                        stats=stats)
        finally:
            self.seq = self.seq + reply.sequence(self.model.cfg.hidden_size)
            self.engine.ran(self.seq)
            self.after, self.open, self.results = self.turns.reply_end, False, False
            self.n += 1


def _responses_tools(tools: object) -> list[dict[str, Any]]:
    """A realtime session's ``tools`` (the Responses API's form, ``{"type": "function", "name", ...}``) as the chat
    templates take them (``tools.functions``: ValueError, not a list of functions)."""
    from .tools import functions
    if isinstance(tools, list):
        tools = [t if not isinstance(t, dict) or "function" in t else
                 {"type": t.get("type"), "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
                 for t in tools]
    return functions(tools)


async def conversation(ws: WebSocket, engine: Engine, alias: str, event: Callable[[dict[str, Any]], Any],
                       error: Callable[..., Any]) -> None:
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
    from ..sequence import data_bytes
    from .tools import syntax_of

    model = engine.model
    session: dict[str, Any] = {"id": f"sess_{secrets.token_hex(12)}", "object": "realtime.session", "type": "realtime",
                               "model": alias, "instructions": "", "output_modalities": None, "max_output_tokens": "inf",
                               "tools": [], "tool_choice": "auto",
                               "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}},
                                         "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "female"}}}
    await event({"type": "session.created", "session": session})
    conv = Conversation(engine, model)
    fns: list[dict[str, Any]] = []
    task: asyncio.Task | None = None
    cancel: list[Any] = [None]
    last: list[str | None] = [None]  # (the last item's id)
    has_audio = [False]

    async def on_worker(fn: Callable[[], Any]) -> Any:
        return await asyncio.wrap_future(engine.run(fn))

    async def started() -> None:
        if conv.turns is None:
            await on_worker(lambda: conv.start(session["instructions"], fns))

    async def item_done(item: dict[str, Any]) -> None:
        await event({"type": "conversation.item.added", "previous_item_id": last[0], "item": item})
        await event({"type": "conversation.item.done", "previous_item_id": last[0], "item": item})
        last[0] = item["id"]

    async def commit() -> bool:
        if not has_audio[0]:
            return False
        await on_worker(conv.commit)
        has_audio[0] = False
        item = f"item_{secrets.token_hex(12)}"
        await event({"type": "input_audio_buffer.committed", "previous_item_id": last[0], "item_id": item})
        await item_done({"id": item, "object": "realtime.item", "type": "message", "role": "user", "status": "completed",
                         "content": [{"type": "input_audio"}]})
        return True

    async def respond(over: dict[str, Any]) -> None:
        mods = over.get("output_modalities", session["output_modalities"])
        mode = MODES.get(tuple(sorted(mods)) if mods else None)
        if mode is None:
            await error(f"output_modalities {mods!r}: text and / or audio")
            return
        mx = over.get("max_output_tokens", session["max_output_tokens"])
        mx = 4096 if mx in (None, "inf") else int(mx)
        rid, item = f"resp_{secrets.token_hex(12)}", f"item_{secrets.token_hex(12)}"
        syntax = syntax_of(model.chat_template or str(model.gguf.meta["tokenizer.chat_template"])) if fns else None
        parser = syntax.parser(fns) if syntax is not None else None

        def produce(push: Callable[[tuple[str, Any]], None], stop: Any) -> None:
            from ..lfm3a.speech import SpeechStats
            cancel[0] = stop
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

        q, stop = worker(engine, produce, taken=False)
        cancel[0] = stop
        await event({"type": "response.created", "response": {"id": rid, "object": "realtime.response",
                                                              "status": "in_progress", "output": []}})
        output: list[dict[str, Any]] = [{"id": item, "object": "realtime.item", "type": "message", "role": "assistant",
                                         "status": "in_progress", "content": []}]
        await event({"type": "response.output_item.added", "response_id": rid, "output_index": 0, "item": output[0]})
        base = {"response_id": rid, "item_id": item, "output_index": 0}
        text, kinds, calls = [], {}, {}  # (the message's parts by kind: their content index; calls: index -> item)

        async def part(kind: str) -> int:
            if kind not in kinds:
                kinds[kind] = len(kinds)
                p = {"type": kind, "text": ""} if kind == "output_text" else {"type": kind, "transcript": ""}
                await event({"type": "response.content_part.added", **base, "content_index": kinds[kind], "part": p})
            return kinds[kind]

        async def content(piece: str) -> None:
            text.append(piece)
            await event({"type": "response.output_text.delta", **base, "content_index": await part("output_text"),
                         "delta": piece})

        async def call(c: Any) -> None:
            if c.index not in calls:
                calls[c.index] = len(output)
                output.append({"id": f"item_{secrets.token_hex(12)}", "object": "realtime.item", "type": "function_call",
                               "status": "in_progress", "name": c.name, "call_id": c.id, "arguments": ""})
                await event({"type": "response.output_item.added", "response_id": rid, "output_index": calls[c.index],
                             "item": output[calls[c.index]]})
            it = output[calls[c.index]]
            if c.arguments:
                it["arguments"] += c.arguments
                await event({"type": "response.function_call_arguments.delta", "response_id": rid, "item_id": it["id"],
                             "output_index": calls[c.index], "call_id": it["call_id"], "delta": c.arguments})

        async def parts(ps: list[tuple[str, Any]]) -> None:
            for k, v in ps:
                await (content(v) if k == "content" else call(v) if k == "tool_call" else asyncio.sleep(0))

        status, n_out = "completed", 0
        while True:
            kind, x = await q.get()
            if kind == "text":
                await (parts(parser.parts([("content", x)])) if parser is not None else content(x))
            elif kind == "audio":
                pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes()
                await event({"type": "response.output_audio.delta", **base, "content_index": await part("output_audio"),
                             "delta": base64.b64encode(pcm).decode()})
            elif kind == "error":
                await error(x, "server_error")
                status = "failed"
                break
            else:
                stopped, st = x
                status, n_out = "cancelled" if stopped else "completed", st.frames
                break
        if parser is not None:
            await parts(parser.flush())
        transcript = "".join(text)
        ps = []
        for kind, i in sorted(kinds.items(), key=lambda kv: kv[1]):
            if kind == "output_text":
                await event({"type": "response.output_text.done", **base, "content_index": i, "text": transcript})
                ps.append({"type": kind, "text": transcript})
            else:
                await event({"type": "response.output_audio.done", **base, "content_index": i})
                ps.append({"type": kind, "transcript": transcript})
            await event({"type": "response.content_part.done", **base, "content_index": i, "part": ps[-1]})
        output[0]["content"] = ps
        for i in calls.values():
            it = output[i]
            await event({"type": "response.function_call_arguments.done", "response_id": rid, "item_id": it["id"],
                         "output_index": i, "call_id": it["call_id"], "name": it["name"], "arguments": it["arguments"]})
        for i, it in enumerate(output):
            it["status"] = status
            await event({"type": "response.output_item.done", "response_id": rid, "output_index": i, "item": it})
            last[0] = it["id"]
        await event({"type": "response.done", "response": {"id": rid, "object": "realtime.response", "status": status,
                                                           "output": output, "usage": {"output_tokens": n_out}}})

    try:
        while True:
            try:
                ev = json.loads(await ws.receive_text())
                kind = ev.get("type") if isinstance(ev, dict) else None
            except ValueError:
                await error("invalid JSON")
                continue
            if task is not None and task.done() and task.exception() is not None:
                log.error("realtime response failed: %s", task.exception())
            busy = task is not None and not task.done()
            if kind == "session.update":
                up = ev.get("session") or {}
                fixed = conv.turns is not None
                if fixed and any(k in up for k in ("instructions", "tools")):
                    await error("session.update: instructions and tools cannot change after the first audio or message")
                    continue
                if "tools" in up:
                    try:
                        fns = _responses_tools(up["tools"])
                    except ValueError as e:
                        await error(f"session.update: {e}")
                        continue
                    if fns and syntax_of(model.chat_template or str(model.gguf.meta["tokenizer.chat_template"])) is None:
                        await error("session.update: the model's chat template has no tool calls")
                        fns = []
                        continue
                for k in ("instructions", "output_modalities", "max_output_tokens", "type", "tools", "tool_choice"):
                    if k in up:
                        session[k] = up[k]
                v = ((up.get("audio") or {}).get("output") or {}).get("voice")
                if v in ("female", "male"):
                    session["audio"]["output"]["voice"] = v
                    session["instructions"] = session["instructions"] or (
                        "" if v == "female" else "Respond in a male voice.")
                await event({"type": "session.updated", "session": session})
            elif kind == "input_audio_buffer.append":
                try:
                    pcm = base64.b64decode(str(ev.get("audio", "")), validate=True)
                except binascii.Error:
                    await error("invalid base64 audio")
                    continue
                x = np.frombuffer(pcm[:len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768
                if len(x):
                    if busy:
                        await error("input_audio_buffer.append: a response is in progress")
                        continue
                    await started()
                    await on_worker(lambda x=x: conv.audio(x))
                    has_audio[0] = True
            elif kind == "input_audio_buffer.commit":
                if not await commit():
                    await error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty")
            elif kind == "input_audio_buffer.clear":
                await event({"type": "input_audio_buffer.cleared"})  # (what ran stays: the model heard it)
            elif kind == "conversation.item.create":
                it = ev.get("item") or {}
                if busy:
                    await error("conversation.item.create: a response is in progress")
                    continue
                if it.get("type") == "function_call_output":
                    if not fns or conv.n == 0 or conv.open:
                        await error("conversation.item.create: a function call output must follow the response that "
                                    "called it")
                        continue
                    out = str(it.get("output", ""))
                    await on_worker(lambda out=out: conv.output(out))
                    await item_done({"id": it.get("id") or f"item_{secrets.token_hex(12)}", "object": "realtime.item",
                                     "type": "function_call_output", "call_id": it.get("call_id", ""), "output": out,
                                     "status": "completed"})
                    continue
                if it.get("role", "user") != "user" or it.get("type", "message") != "message":
                    await error("conversation.item.create: only user messages and function call outputs are supported")
                    continue
                content = [c for c in it.get("content") or [] if isinstance(c, dict)]
                bad = [c.get("type") for c in content if c.get("type") not in ("input_text", "text", "input_image")]
                if bad or not content:
                    await error(f"conversation.item.create: unsupported content {bad or 'none'} (input_text, "
                                "input_image)")
                    continue
                if any(c.get("type") == "input_image" for c in content) and model.vision is None:
                    await error("conversation.item.create: the model takes no images (an mmproj with a vision tower)")
                    continue
                await started()
                if has_audio[0]:
                    await commit()
                try:
                    for c in content:
                        if c.get("type") == "input_image":
                            url = c.get("image_url")
                            url = url.get("url") if isinstance(url, dict) else url
                            data = data_bytes(str(url))
                            await on_worker(lambda data=data: conv.image(data))
                        else:
                            t = str(c.get("text", ""))
                            await on_worker(lambda t=t: conv.text(t))
                except (OSError, ValueError) as e:
                    await error(f"conversation.item.create: {e}")
                    continue
                await item_done({"id": it.get("id") or f"item_{secrets.token_hex(12)}", "object": "realtime.item",
                                 "type": "message", "role": "user", "status": "completed",
                                 "content": [{"type": c.get("type")} | ({"text": c.get("text", "")}
                                             if c.get("type") != "input_image" else {}) for c in content]})
            elif kind == "response.create":
                if busy:
                    await error("a response is in progress")
                    continue
                await started()
                await commit()
                if not conv.open and not conv.results:
                    await error("response.create: commit audio or create a message first", "conversation_empty")
                    continue
                task = asyncio.create_task(respond(ev.get("response") or {}))
            elif kind == "response.cancel":
                if cancel[0] is not None:
                    cancel[0].set()
            else:
                await error(f"unknown event type: {kind}")
    except WebSocketDisconnect:
        if cancel[0] is not None:
            cancel[0].set()
        return
    finally:
        if task is not None and not task.done():
            if cancel[0] is not None:
                cancel[0].set()
            with contextlib.suppress(Exception):
                await task


async def transcription(ws: WebSocket, transcribe: Callable[[np.ndarray, str], Any], alias: str,
                        event: Callable[[dict[str, Any]], Any], error: Callable[..., Any]) -> None:
    """A transcription session of a model not trained for live ASR (LFM3-Omni; llama-server's): each committed buffer
    transcribed alone with the offline ASR prompt (``transcribe(samples, language)``: its text's pieces as they come),
    as ``conversation.item.input_audio_transcription.delta`` events and the item's ``...completed``."""
    session = {"id": f"sess_{secrets.token_hex(12)}", "object": "realtime.transcription_session",
               "type": "transcription",
               "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000},
                                   "transcription": {"model": alias, "language": None}, "turn_detection": None,
                                   "noise_reduction": None}}}
    await event({"type": "session.created", "session": session})
    buf: list[np.ndarray] = []
    prev = None
    try:
        while True:
            try:
                ev = json.loads(await ws.receive_text())
                kind = ev.get("type") if isinstance(ev, dict) else None
            except ValueError:
                await error("invalid JSON")
                continue
            if kind in ("session.update", "transcription_session.update"):
                tr = (((ev.get("session") or {}).get("audio") or {}).get("input") or {}).get("transcription") or {}
                if "language" in tr:
                    session["audio"]["input"]["transcription"]["language"] = tr["language"]
                await event({"type": "session.updated", "session": session})
            elif kind == "input_audio_buffer.append":
                try:
                    pcm = base64.b64decode(str(ev.get("audio", "")), validate=True)
                except binascii.Error:
                    await error("invalid base64 audio")
                    continue
                buf.append(np.frombuffer(pcm[:len(pcm) // 2 * 2], np.int16).astype(np.float32) / 32768)
            elif kind == "input_audio_buffer.commit":
                x = np.concatenate(buf) if buf else np.zeros(0, np.float32)
                buf.clear()
                if not len(x):
                    if not ev.get("final"):
                        await error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty")
                    if ev.get("final"):
                        break
                    continue
                item = f"item_{secrets.token_hex(12)}"
                await event({"type": "input_audio_buffer.committed", "previous_item_id": prev, "item_id": item})
                text = []
                async for piece in transcribe(x, session["audio"]["input"]["transcription"]["language"] or ""):
                    text.append(piece)
                    await event({"type": "conversation.item.input_audio_transcription.delta", "item_id": item,
                                 "content_index": 0, "delta": piece})
                await event({"type": "conversation.item.input_audio_transcription.completed", "item_id": item,
                             "content_index": 0, "transcript": "".join(text)})
                prev = item
                if ev.get("final"):
                    break
            elif kind == "input_audio_buffer.clear":
                buf.clear()
                await event({"type": "input_audio_buffer.cleared"})
            else:
                await error(f"unknown event type: {kind}")
    except WebSocketDisconnect:
        return
    except Exception as e:  # (a failure in the model: the session's error)
        log.exception("realtime transcription failed")
        await error(f"{type(e).__name__}: {e}", "server_error")
    await ws.close()
