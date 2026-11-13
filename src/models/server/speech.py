"""The server's speech output (a model with LFM3-Audio's speech generator: ``Lfm.tts``, ``models.lfm3a.speech``):
llama-server's ``/tts`` and OpenAI's ``/v1/audio/speech``. A generation runs on the engine's worker (``Engine.run``:
between the jobs, the prompt cache told the device's sequence is lost: ``Engine.device_taken``), its samples streamed as
the codec makes them (every 4 frames, 320 ms of speech).

- ``/tts`` (llama-server's): ``input`` (alias ``prompt``), ``lang``, ``voice`` (female, male), ``seed`` (-1 random),
  ``n_predict`` (frames at most, default 512), ``response_format`` (``wav``; ``pcm``: float32 samples), ``stream`` (a
  WAV of unknown length, its sizes 0xFFFFFFFF, or the samples as they come); a JSON body or a form. ``speaker_ref``:
  400 (the model has no voice encoder); ``top_k`` / ``top_p`` are not used (llama.cpp ignores them for LFM3-Audio too).
- chat completions with ``modalities`` ["text", "audio"] and ``audio`` {voice, format} (OpenAI's chat audio output,
  ``chat_audio``): a spoken reply (``Lfm.voice_reply``: interleaved text and speech) as ``message.audio`` {id, data
  (base64 WAV, or 16-bit PCM for ``format`` pcm16), transcript, expires_at}, streamed as ``delta.audio`` (the
  transcript's pieces, base64 16-bit chunks). An earlier reply in the messages (an assistant message with ``audio``
  {id} the server still holds, ``REPLIES`` of them) goes in as its transcript.
- ``/v1/audio/speech`` (OpenAI's TTS): ``input``, ``voice`` (female, male: 400 with the list otherwise),
  ``response_format`` (``wav``, the default here: no mp3 / opus / aac / flac encoders, 400; a whole file; ``pcm``:
  16-bit samples, streamed),
  ``speed`` (1 only), ``stream_format`` (``audio``; ``sse``: ``speech.audio.delta`` events of base64 chunks, then
  ``speech.audio.done`` with the usage), ``instructions`` (not used, logged).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from ..common import StreamDecoder
from ..lfm3a.speech import SR, VOICES, SpeechStats, pcm16, wav_header
from .engine import Engine

log = logging.getLogger("artlite.server")
MAX_FRAMES = 512  # (a request's frames by default: llama-server's n_predict, 41 s)
REPLIES = 256  # (spoken replies whose transcripts the server keeps for later turns)
EXPIRES = 3600  # (a reply's audio.expires_at: seconds after it)


def speaks(engine: Engine) -> bool:
    """Whether the engine's model speaks (LFM3-Audio's language model with its mmproj's generator)."""
    m = engine.model
    return m is not None and hasattr(m, "tts") and getattr(m, "audio", None) is not None and m.audio.gen is not None


class SpeechStream(StreamingResponse):
    """A streamed response whose generation stops when the response ends early (its client went away)."""

    def __init__(self, body: AsyncIterator[bytes | str], cancel: threading.Event, media_type: str):
        super().__init__(body, media_type=media_type, headers={"Cache-Control": "no-cache"})
        self.cancel = cancel

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cancel.set()


def add_speech(api: APIRouter, engine: Engine, error: Callable[[int, str, str], Exception]) -> None:
    """The speech routes on ``api``; ``error(status, kind, message)``: the server's error responses."""

    def bad(message: str) -> Exception:
        return error(400, "invalid_request_error", message)

    def check() -> None:
        if not speaks(engine):
            raise error(501, "not_supported_error", "The current model does not support speech output.")

    def generate(text: str, voice: str, lang: str, seed: int, frames: int
                 ) -> tuple[asyncio.Queue[tuple[str, Any]], threading.Event]:
        """The generation on the worker: its queue of ("audio", samples), then ("done", stats) or ("error", message)."""
        def produce(push: Callable[[tuple[str, Any]], None], cancel: threading.Event) -> None:
            st = SpeechStats()
            for y in engine.model.tts(text, voice=voice, language=lang, seed=None if seed < 0 else seed,
                                      max_frames=frames, stats=st):
                if cancel.is_set():
                    break
                if len(y):
                    push(("audio", y))
            push(("done", st))
        return worker(engine, produce)

    async def samples(q: asyncio.Queue[tuple[str, Any]]) -> AsyncIterator[tuple[str, Any]]:
        while True:
            kind, x = await q.get()
            yield kind, x
            if kind in ("done", "error"):
                return

    async def whole(q: asyncio.Queue[tuple[str, Any]]) -> tuple[np.ndarray, SpeechStats]:
        parts: list[np.ndarray] = []
        async for kind, x in samples(q):
            if kind == "audio":
                parts.append(x)
            elif kind == "error":
                raise error(500, "server_error", x)
            else:
                return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), x
        raise error(500, "server_error", "the generation ended early")

    def voice_of(v: object) -> str:
        v = "female" if v in (None, "") else str(v).lower()
        if v not in VOICES:
            raise bad(f"voice {v!r}: the model's voices are {', '.join(VOICES)}")
        return v

    async def fields(request: Request) -> dict[str, Any]:
        """A JSON body or a form's fields."""
        if request.headers.get("content-type", "").startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            try:
                form = await request.form()
            except (AssertionError, ValueError) as e:
                raise bad(f"a form: {e}") from None
            return {k: v for k, v in form.items()}
        try:
            body = await request.json()
        except ValueError:
            raise bad("a JSON body") from None
        if not isinstance(body, dict):
            raise bad("a JSON object")
        return body

    @api.post("/tts")
    async def tts(request: Request) -> Response:
        """llama-server's speech generation (module docstring)."""
        check()
        f = await fields(request)
        text = f.get("input", f.get("prompt"))
        if not isinstance(text, str) or not text.strip():
            raise bad("input: the text to speak")
        if f.get("speaker_ref") is not None or f.get("speaker_ref_b64"):
            raise bad("speaker_ref: the model has no voice encoder (its voices: female, male)")
        fmt = str(f.get("response_format") or "wav")
        if fmt not in ("wav", "pcm"):
            raise bad(f"response_format {fmt!r}: wav or pcm")
        try:
            seed, frames = int(f.get("seed", -1)), int(f.get("n_predict", MAX_FRAMES))
        except (TypeError, ValueError):
            raise bad("seed, n_predict: integers") from None
        stream = str(f.get("stream", False)).lower() in ("true", "1")
        q, cancel = generate(text, voice_of(f.get("voice")), str(f.get("lang") or "en"), seed,
                             frames if frames > 0 else MAX_FRAMES)
        mt = "audio/wav" if fmt == "wav" else f"audio/pcm;rate={SR};encoding=float;bits=32"
        if not stream:
            y, _ = await whole(q)
            data = wav_header(len(y)) + pcm16(y) if fmt == "wav" else np.asarray(y, "<f4").tobytes()
            return Response(data, media_type=mt)

        async def body() -> AsyncIterator[bytes]:
            if fmt == "wav":
                yield wav_header(None)
            async for kind, x in samples(q):
                if kind == "audio":
                    yield pcm16(x) if fmt == "wav" else np.asarray(x, "<f4").tobytes()
                elif kind == "error":
                    log.error("/tts: %s", x)
        return SpeechStream(body(), cancel, mt)

    @api.post("/v1/audio/speech")
    @api.post("/audio/speech")
    async def speech(request: Request) -> Response:
        """OpenAI's text to speech (module docstring)."""
        check()
        f = await fields(request)
        text = f.get("input")
        if not isinstance(text, str) or not text.strip():
            raise bad("input: the text to speak")
        fmt = str(f.get("response_format") or "wav")
        if fmt in ("mp3", "opus", "aac", "flac"):
            raise bad(f"response_format {fmt!r}: not supported (no encoder): wav or pcm")
        if fmt not in ("wav", "pcm"):
            raise bad(f"response_format {fmt!r}: wav or pcm")
        try:
            speed = float(f.get("speed", 1.0))
        except (TypeError, ValueError):
            raise bad("speed: a number") from None
        if speed != 1.0:
            raise bad("speed: 1 only")
        if f.get("instructions"):
            log.info("/v1/audio/speech: instructions are not used")
        sse = str(f.get("stream_format") or "audio")
        if sse not in ("audio", "sse"):
            raise bad(f"stream_format {sse!r}: audio or sse")
        q, cancel = generate(text, voice_of(f.get("voice")), "en", -1, MAX_FRAMES)
        if sse == "audio" and fmt == "wav":  # (a whole file: its header's sizes)
            y, _ = await whole(q)
            return Response(wav_header(len(y)) + pcm16(y), media_type="audio/wav")
        if sse == "audio":  # (16-bit samples as they come: chunked)
            async def body() -> AsyncIterator[bytes]:
                async for kind, x in samples(q):
                    if kind == "audio":
                        yield pcm16(x)
                    elif kind == "error":
                        log.error("/v1/audio/speech: %s", x)
            return SpeechStream(body(), cancel, "audio/pcm")

        async def events() -> AsyncIterator[str]:
            n = 0
            async for kind, x in samples(q):
                if kind == "audio":
                    n += len(x)
                    yield "data: " + json.dumps({"type": "speech.audio.delta",
                                                 "audio": base64.b64encode(pcm16(x)).decode()}) + "\n\n"
                elif kind == "done":
                    yield "data: " + json.dumps({"type": "speech.audio.done", "usage": {
                        "input_tokens": 0, "output_tokens": x.frames, "total_tokens": x.frames}}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"type": "error", "error": {"message": x}}) + "\n\n"
        return SpeechStream(events(), cancel, "text/event-stream")


def worker(engine: Engine, produce: Callable[[Callable[[tuple[str, Any]], None], threading.Event], None]
           ) -> tuple[asyncio.Queue[tuple[str, Any]], threading.Event]:
    """``produce(push, cancel)`` on the engine's worker (the device's sequence lost to the prompt cache), its items into
    an asyncio queue; an exception: ("error", message)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancel = threading.Event()

    def push(item: tuple[str, Any]) -> None:
        loop.call_soon_threadsafe(q.put_nowait, item)

    def run() -> None:
        try:
            engine.device_taken()
            produce(push, cancel)
        except Exception as e:  # (the request's error)
            log.exception("speech failed")
            push(("error", f"{type(e).__name__}: {e}"))

    engine.run(run)
    return q, cancel


class Replies:
    """The transcripts of the spoken replies the server sent, by audio id (the last ``REPLIES``)."""

    def __init__(self) -> None:
        self.text: dict[str, str] = {}

    def add(self, aid: str, text: str) -> None:
        self.text[aid] = text
        while len(self.text) > REPLIES:
            self.text.pop(next(iter(self.text)))

    def messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI's messages with earlier spoken replies (``audio`` {id}, no content) as their transcripts."""
        out = []
        for m in messages:
            a = m.get("audio")
            if m.get("role") == "assistant" and isinstance(a, dict) and not m.get("content"):
                m = {k: v for k, v in m.items() if k != "audio"} | {"content": self.text.get(str(a.get("id")), "")}
            out.append(m)
        return out


async def chat_audio(engine: Engine, body: dict[str, Any], msgs: list[dict[str, object]], *, alias: str,
                     error: Callable[[int, str, str], Exception], replies: Replies) -> Response:
    """A chat completion with audio output (module docstring): ``body`` the request, ``msgs`` its messages for the chat
    template."""
    if not speaks(engine):
        raise error(501, "not_supported_error", "The current model does not support speech output.")
    a = body.get("audio") or {}
    voice = str(a.get("voice") or "female").lower()
    if voice not in VOICES:
        raise error(400, "invalid_request_error", f"audio.voice {voice!r}: the model's voices are {', '.join(VOICES)}")
    stream = bool(body.get("stream"))
    fmt = str(a.get("format") or ("pcm16" if stream else "wav"))
    if fmt not in ("wav", "pcm16") or (stream and fmt != "pcm16"):
        raise error(400, "invalid_request_error", f"audio.format {fmt!r}: wav or pcm16 (streamed: pcm16)")
    if voice != "female":
        msgs = [{"role": "system", "content": f"Respond with interleaved text and audio, in a {voice} voice."}] + \
            [m for m in msgs if m.get("role") != "system"]
    seed = body.get("seed")
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024)
    model = engine.model
    cid, aid, created = f"chatcmpl-{secrets.token_hex(16)}", f"audio_{secrets.token_hex(12)}", int(time.time())
    head = {"created": created, "id": cid, "model": alias}

    def produce(push: Callable[[tuple[str, Any]], None], cancel: threading.Event) -> None:
        dec, st, n = StreamDecoder(model.tokenizer), SpeechStats(), 0
        for kind, x in model.voice_reply(msgs, seed=None if seed is None else int(seed), max_tokens=max_tokens,
                                         stats=st):
            if cancel.is_set():
                break
            if kind == "text":
                n += 1
                if piece := dec.feed(x):
                    push(("text", piece))
            elif len(x):
                push(("audio", x))
        push(("done", (st, n)))

    q, cancel = worker(engine, produce)

    async def items() -> AsyncIterator[tuple[str, Any]]:
        while True:
            kind, x = await q.get()
            yield kind, x
            if kind in ("done", "error"):
                return

    def usage(st: SpeechStats, n: int) -> dict[str, Any]:
        return {"prompt_tokens": 0, "completion_tokens": n + st.frames, "total_tokens": n + st.frames,
                "completion_tokens_details": {"text_tokens": n, "audio_tokens": st.frames}}

    if not stream:
        text, parts, st, n = [], [], SpeechStats(), 0
        async for kind, x in items():
            if kind == "text":
                text.append(x)
            elif kind == "audio":
                parts.append(x)
            elif kind == "error":
                raise error(500, "server_error", x)
            else:
                st, n = x
        y = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        data = wav_header(len(y)) + pcm16(y) if fmt == "wav" else pcm16(y)
        transcript = "".join(text)
        replies.add(aid, transcript)
        audio = {"id": aid, "data": base64.b64encode(data).decode(), "expires_at": created + EXPIRES,
                 "transcript": transcript}
        return Response(json.dumps({"choices": [{"finish_reason": "stop", "index": 0, "message": {
            "role": "assistant", "content": None, "audio": audio}}], **head, "object": "chat.completion",
            "usage": usage(st, n)}), media_type="application/json")

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: object) -> str:
        return "data: " + json.dumps({"choices": [{"finish_reason": finish, "index": 0, "delta": delta}], **head,
                                      "object": "chat.completion.chunk", **extra}) + "\n\n"

    async def events() -> AsyncIterator[str]:
        yield chunk({"role": "assistant", "content": None})
        text: list[str] = []
        async for kind, x in items():
            if kind == "text":
                text.append(x)
                yield chunk({"audio": {"id": aid, "transcript": x}})
            elif kind == "audio":
                yield chunk({"audio": {"id": aid, "data": base64.b64encode(pcm16(x)).decode()}})
            elif kind == "done":
                replies.add(aid, "".join(text))
                inc = (body.get("stream_options") or {}).get("include_usage")
                yield chunk({"audio": {"id": aid, "expires_at": created + EXPIRES}}, "stop",
                            **({"usage": usage(*x)} if inc else {}))
                yield "data: [DONE]\n\n"
            else:
                yield "data: " + json.dumps({"error": {"message": x, "type": "server_error"}}) + "\n\n"
    return SpeechStream(events(), cancel, "text/event-stream")
