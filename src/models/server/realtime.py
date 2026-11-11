"""OpenAI's realtime API as llama-server's live transcription sessions: the websocket ``/v1/realtime?intent=transcription``
of a model with LFM3-Audio's live ASR (``models.lfm3a.live``: its speech encoder and the ``asr`` chat template).

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
import json
import logging
import secrets
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .engine import Engine

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
