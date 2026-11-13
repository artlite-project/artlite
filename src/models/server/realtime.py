"""OpenAI's realtime API as llama-server's: live transcription sessions (the websocket
``/v1/realtime?intent=transcription`` of a model with LFM3-Audio's live ASR: ``models.lfm3a.live``, its speech encoder
and the ``asr`` chat template) and conversations (``conversation``: other sessions of a model that speaks).

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
from .engine import Engine
from .speech import speaks, worker

log = logging.getLogger("artlite.server")


def add_route(app: FastAPI, *, engine: Engine, alias: str, authorized: Callable[[str | None, str | None], bool]) -> None:
    """``authorized(authorization, x_api_key)``: the request's headers pass the server's API keys."""

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
        if ws.query_params.get("intent") != "transcription" or model is None or model.audio is None or \
                "tokenizer.chat_template.asr" not in getattr(getattr(model, "gguf", None), "meta", {}):
            await error("The current model does not support realtime transcription (a live ASR model: LFM3-Audio, "
                        "?intent=transcription).")
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


async def conversation(ws: WebSocket, engine: Engine, alias: str, event: Callable[[dict[str, Any]], Any],
                       error: Callable[..., Any]) -> None:
    """A realtime conversation (llama-server's "Realtime conversation", a model that speaks: ``Lfm.voice_reply``): the
    session's ``instructions`` (the system prompt), ``output_modalities`` (left out: the model speaks or not by itself;
    ["text"], ["audio"]: speech, then text; ["text", "audio"]: interleaved), ``max_output_tokens``, the voice
    (``audio.output.voice``: female, male); the user turns' audio (``input_audio_buffer.append``: PCM16 at 16 kHz,
    ``.commit``) and text (``conversation.item.create``: ``input_text``); ``response.create`` generates the reply on the
    engine's worker (the conversation so far as one prompt: a clip's rows kept by the speech encoder), streamed as
    ``response.output_text.delta`` and ``response.output_audio.delta`` (base64 PCM16 at 24 kHz) between the
    ``response.*`` start and done events; ``response.cancel`` stops it. Each reply stays in the conversation as its
    text."""
    model = engine.model
    session: dict[str, Any] = {"id": f"sess_{secrets.token_hex(12)}", "object": "realtime.session", "type": "realtime",
                               "model": alias, "instructions": "", "output_modalities": None, "max_output_tokens": "inf",
                               "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}},
                                         "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "female"}}}
    await event({"type": "session.created", "session": session})
    messages: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []  # (the user turn so far)
    buf: list[np.ndarray] = []  # (its audio not committed yet)
    task: asyncio.Task | None = None
    cancel: list[Any] = [None]

    async def commit() -> str | None:
        if not buf:
            return None
        item = f"item_{secrets.token_hex(12)}"
        parts.append({"type": "audio", "audio": np.concatenate(buf)})
        buf.clear()
        await event({"type": "input_audio_buffer.committed", "previous_item_id": None, "item_id": item})
        it = {"id": item, "type": "message", "role": "user", "content": [{"type": "input_audio"}]}
        await event({"type": "conversation.item.added", "item": it})
        await event({"type": "conversation.item.done", "item": it})
        return item

    async def respond(over: dict[str, Any]) -> None:
        mods = over.get("output_modalities", session["output_modalities"])
        mode = MODES.get(tuple(sorted(mods)) if mods else None)
        if mode is None:
            await error(f"output_modalities {mods!r}: text and / or audio")
            return
        if parts:
            messages.append({"role": "user", "content": list(parts)})
            parts.clear()
        msgs = ([{"role": "system", "content": session["instructions"]}] if session["instructions"] else []) + messages
        mx = over.get("max_output_tokens", session["max_output_tokens"])
        mx = 4096 if mx in (None, "inf") else int(mx)
        rid, item = f"resp_{secrets.token_hex(12)}", f"item_{secrets.token_hex(12)}"

        def produce(push: Callable[[tuple[str, Any]], None], stop: Any) -> None:
            cancel[0] = stop
            dec = StreamDecoder(model.tokenizer)
            for kind, x in model.voice_reply(msgs, mode=mode, max_tokens=mx):
                if stop.is_set():
                    break
                if kind == "text":
                    if piece := dec.feed(x):
                        push(("text", piece))
                elif len(x):
                    push(("audio", x))
            push(("done", stop.is_set()))

        q, stop = worker(engine, produce)
        cancel[0] = stop
        base = {"response_id": rid, "item_id": item, "output_index": 0, "content_index": 0}
        await event({"type": "response.created", "response": {"id": rid, "object": "realtime.response",
                                                              "status": "in_progress", "output": []}})
        await event({"type": "response.output_item.added", "response_id": rid, "output_index": 0,
                     "item": {"id": item, "type": "message", "role": "assistant", "content": []}})
        await event({"type": "response.content_part.added", **base, "part": {"type": "audio", "transcript": ""}})
        text, status = [], "completed"
        while True:
            kind, x = await q.get()
            if kind == "text":
                text.append(x)
                await event({"type": "response.output_text.delta", **base, "delta": x})
            elif kind == "audio":
                pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes()
                await event({"type": "response.output_audio.delta", **base, "delta": base64.b64encode(pcm).decode()})
            elif kind == "error":
                await error(x, "server_error")
                status = "failed"
                break
            else:
                status = "cancelled" if x else "completed"
                break
        transcript = "".join(text)
        messages.append({"role": "assistant", "content": transcript})
        await event({"type": "response.output_text.done", **base, "text": transcript})
        await event({"type": "response.output_audio.done", **base})
        await event({"type": "response.content_part.done", **base, "part": {"type": "audio", "transcript": transcript}})
        done = {"id": item, "type": "message", "role": "assistant", "status": status,
                "content": [{"type": "output_audio", "transcript": transcript}]}
        await event({"type": "response.output_item.done", "response_id": rid, "output_index": 0, "item": done})
        await event({"type": "response.done", "response": {"id": rid, "object": "realtime.response", "status": status,
                                                           "output": [done]}})

    try:
        while True:
            try:
                ev = json.loads(await ws.receive_text())
                kind = ev.get("type") if isinstance(ev, dict) else None
            except ValueError:
                await error("invalid JSON")
                continue
            if kind == "session.update":
                up = ev.get("session") or {}
                for k in ("instructions", "output_modalities", "max_output_tokens", "type"):
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
                    buf.append(x)
            elif kind == "input_audio_buffer.commit":
                await commit()
            elif kind == "input_audio_buffer.clear":
                buf.clear()
                await event({"type": "input_audio_buffer.cleared"})
            elif kind == "conversation.item.create":
                it = ev.get("item") or {}
                txt = "".join(c.get("text", "") for c in it.get("content") or [] if isinstance(c, dict)
                              and c.get("type") in ("input_text", "text"))
                if it.get("role", "user") != "user" or not txt:
                    await error("conversation.item.create: a user message with input_text content")
                    continue
                parts.append({"type": "text", "text": txt})
                done = {"id": it.get("id") or f"item_{secrets.token_hex(12)}", "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": txt}]}
                await event({"type": "conversation.item.added", "item": done})
                await event({"type": "conversation.item.done", "item": done})
            elif kind == "response.create":
                if task is not None and not task.done():
                    await error("a response is in progress")
                    continue
                await commit()
                if not parts and (not messages or messages[-1]["role"] != "user"):
                    await error("response.create: no user turn")
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
