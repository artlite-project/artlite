"""The server's speech output (a model with LFM3-Audio's speech generator: ``Lfm.tts``, ``models.lfm3a.speech``):
llama-server's ``/tts`` and OpenAI's ``/v1/audio/speech``. A generation runs on the engine's worker (``Engine.run``:
between the jobs, the prompt cache told the device's sequence is lost: ``Engine.recover``), its samples streamed as
the codec makes them (every 4 frames, 320 ms of speech).

- ``/tts`` (llama-server's): ``input`` (alias ``prompt``), ``lang``, ``voice`` (female, male), ``seed`` (-1 random),
  ``n_predict`` (frames at most, default 512), ``response_format`` (``wav``; ``pcm``: float32 samples), ``stream`` (a
  WAV of unknown length, its sizes 0xFFFFFFFF, or the samples as they come); a JSON body or a form. ``speaker_ref``:
  400 (the model has no voice encoder); ``top_k`` / ``top_p`` are not used (llama.cpp ignores them for LFM3-Audio too).
- chat completions with ``modalities`` ["text", "audio"] and ``audio`` {voice, format} (OpenAI's chat audio output,
  ``chat_audio``): a spoken reply (``Lfm.voice_reply``: interleaved text and speech; ["audio"]: speech, then text) as
  ``message.audio`` {id, data
  (base64 WAV, or 16-bit PCM for ``format`` pcm16), transcript, expires_at}, streamed as ``delta.audio`` (the
  transcript's pieces, base64 16-bit chunks). An earlier reply in the messages (an assistant message with ``audio``
  {id} the server still holds, ``REPLIES`` of them) goes in as the model said it (``Replies``: its text and frames).
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
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..detokenizer import StreamDecoder
from ..lfm3a.speech import SR, VOICES, Reply, SpeechStats, pcm16, wav_header
from .app_base import NO_CACHE, CancellingResponse, sse
from .engine import Engine
from .errors import bad_request, not_supported, server_error
from .ids import hex_id

log = logging.getLogger("artlite.server")
MAX_FRAMES = 512  # (a request's frames by default: llama-server's n_predict, 41 s)
REPLIES = 256  # (spoken replies the server keeps for later turns)
EXPIRES = 3600  # (a reply's audio.expires_at: seconds after it)


def speaks(engine: Engine) -> bool:
    """Whether the engine's model speaks (LFM3-Audio's language model with its mmproj's generator)."""
    m = engine.model
    return m is not None and m.capabilities.speech


def audio_stream(body: AsyncIterator[bytes | str], cancel: threading.Event, media_type: str) -> CancellingResponse:
    """A streamed response whose generation stops when the response ends early (its client went away)."""
    return CancellingResponse(body, on_end=cancel.set, media_type=media_type, headers=NO_CACHE)


def add_speech(api: APIRouter, engine: Engine) -> None:
    """The speech routes on ``api``."""

    def check() -> None:
        if not speaks(engine):
            raise not_supported("The current model does not support speech output.")

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
                raise server_error(x)
            else:
                return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), x
        raise server_error("the generation ended early")

    def voice_of(v: object) -> str:
        v = "female" if v in (None, "") else str(v).lower()
        if v not in VOICES:
            raise bad_request(f"voice {v!r}: the model's voices are {', '.join(VOICES)}")
        return v

    async def fields(request: Request) -> dict[str, Any]:
        """A JSON body or a form's fields."""
        if request.headers.get("content-type", "").startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            try:
                form = await request.form()
            except (AssertionError, ValueError) as e:
                raise bad_request(f"a form: {e}") from None
            return {k: v for k, v in form.items()}
        try:
            body = await request.json()
        except ValueError:
            raise bad_request("a JSON body") from None
        if not isinstance(body, dict):
            raise bad_request("a JSON object")
        return body

    @api.post("/tts")
    async def tts(request: Request) -> Response:
        """llama-server's speech generation (module docstring)."""
        check()
        f = await fields(request)
        text = f.get("input", f.get("prompt"))
        if not isinstance(text, str) or not text.strip():
            raise bad_request("input: the text to speak")
        if f.get("speaker_ref") is not None or f.get("speaker_ref_b64"):
            raise bad_request("speaker_ref: the model has no voice encoder (its voices: female, male)")
        fmt = str(f.get("response_format") or "wav")
        if fmt not in ("wav", "pcm"):
            raise bad_request(f"response_format {fmt!r}: wav or pcm")
        try:
            seed, frames = int(f.get("seed", -1)), int(f.get("n_predict", MAX_FRAMES))
        except (TypeError, ValueError):
            raise bad_request("seed, n_predict: integers") from None
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
        return audio_stream(body(), cancel, mt)

    @api.post("/v1/audio/speech")
    @api.post("/audio/speech")
    async def speech(request: Request) -> Response:
        """OpenAI's text to speech (module docstring)."""
        check()
        f = await fields(request)
        text = f.get("input")
        if not isinstance(text, str) or not text.strip():
            raise bad_request("input: the text to speak")
        fmt = str(f.get("response_format") or "wav")
        if fmt in ("mp3", "opus", "aac", "flac"):
            raise bad_request(f"response_format {fmt!r}: not supported (no encoder): wav or pcm")
        if fmt not in ("wav", "pcm"):
            raise bad_request(f"response_format {fmt!r}: wav or pcm")
        try:
            speed = float(f.get("speed", 1.0))
        except (TypeError, ValueError):
            raise bad_request("speed: a number") from None
        if speed != 1.0:
            raise bad_request("speed: 1 only")
        if f.get("instructions"):
            log.info("/v1/audio/speech: instructions are not used")
        stream_format = str(f.get("stream_format") or "audio")
        if stream_format not in ("audio", "sse"):
            raise bad_request(f"stream_format {stream_format!r}: audio or sse")
        q, cancel = generate(text, voice_of(f.get("voice")), "en", -1, MAX_FRAMES)
        if stream_format == "audio" and fmt == "wav":  # (a whole file: its header's sizes)
            y, _ = await whole(q)
            return Response(wav_header(len(y)) + pcm16(y), media_type="audio/wav")
        if stream_format == "audio":  # (16-bit samples as they come: chunked)
            async def body() -> AsyncIterator[bytes]:
                async for kind, x in samples(q):
                    if kind == "audio":
                        yield pcm16(x)
                    elif kind == "error":
                        log.error("/v1/audio/speech: %s", x)
            return audio_stream(body(), cancel, "audio/pcm")

        async def events() -> AsyncIterator[str]:
            n = 0
            async for kind, x in samples(q):
                if kind == "audio":
                    n += len(x)
                    yield sse({"type": "speech.audio.delta", "audio": base64.b64encode(pcm16(x)).decode()},
                              ensure_ascii=True)
                elif kind == "done":
                    use = {"input_tokens": x.prompt, "output_tokens": x.frames, "total_tokens": x.prompt + x.frames}
                    yield sse({"type": "speech.audio.done", "usage": use}, ensure_ascii=True)
                else:
                    yield sse({"type": "error", "error": {"message": x}}, ensure_ascii=True)
        return audio_stream(events(), cancel, "text/event-stream")


def worker(engine: Engine, produce: Callable[[Callable[[tuple[str, Any]], None], threading.Event], None], *,
           taken: bool = True) -> tuple[asyncio.Queue[tuple[str, Any]], threading.Event]:
    """``produce(push, cancel)`` on the engine's worker (``taken``: the device's sequence lost to the prompt cache;
    else produce tells the cache what the device holds), its items into an asyncio queue; an exception: ("error",
    message)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancel = threading.Event()

    def push(item: tuple[str, Any]) -> None:
        loop.call_soon_threadsafe(q.put_nowait, item)

    def run() -> None:
        try:
            if taken:
                engine.recover()
            produce(push, cancel)
        except Exception as e:  # (the request's error)
            log.exception("speech failed")
            push(("error", f"{type(e).__name__}: {e}"))

    engine.run(run)
    return q, cancel


class Replies:
    """The spoken replies the server sent, by audio id (the last ``REPLIES``): their transcripts and the replies as
    the model said them (``parts``: the content parts a chat template renders them from, ``reply_parts``)."""

    def __init__(self) -> None:
        self.text: dict[str, str] = {}
        self.said: dict[str, list[dict[str, Any]]] = {}

    def add(self, aid: str, text: str, parts: list[dict[str, Any]] | None = None) -> None:
        self.text[aid] = text
        if parts is not None:
            self.said[aid] = parts
        while len(self.text) > REPLIES:
            self.said.pop(old := next(iter(self.text)), None)
            self.text.pop(old)

    def messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI's messages with earlier spoken replies (``audio`` {id}, no content) as their transcripts, the id kept
        (``REPLY_KEY``: ``said`` puts the reply as said in its place)."""
        out = []
        for m in messages:
            a = m.get("audio")
            if m.get("role") == "assistant" and isinstance(a, dict) and not m.get("content"):
                aid = str(a.get("id"))
                m = {k: v for k, v in m.items() if k != "audio"} | {"content": self.text.get(aid, ""), REPLY_KEY: aid}
            out.append(m)
        return out

    def said_as(self, message: dict[str, Any]) -> dict[str, Any]:
        """A message of ``messages`` (after the server's own conversion): an earlier reply as the model said it (its
        text and frames: ``reply_parts``), where the server keeps it; else its transcript."""
        aid = message.pop(REPLY_KEY, None)
        if aid in self.said:
            message["content"] = self.said[aid]
        return message


REPLY_KEY = "_artlite_reply"  # (an earlier spoken reply's audio id in a converted message: ``Replies``)


def reply_parts(model: Any, reply: Reply, opener: list[int]) -> list[dict[str, Any]]:
    """A spoken reply as an assistant message's content parts: its text runs (control tokens included: the opener,
    ``<|text_end|>``) and per frame an ``audio_out`` part with its feedback row (``Lfm.chat`` puts the row where the
    template writes ``<|audio_out|>``)."""
    text = lambda ids: model.tokenizer.decode_bytes(ids, special=True).decode("utf-8", errors="replace")
    parts: list[dict[str, Any]] = []
    run = list(opener)
    for i, t in enumerate(reply.ids):
        if i in reply.rows:
            if run:
                parts.append({"type": "text", "text": text(run)})
                run = []
            parts.append({"type": "audio_out", "row": reply.rows[i]})
        else:
            run.append(t)
    if run:
        parts.append({"type": "text", "text": text(run)})
    return parts


async def chat_audio(engine: Engine, body: dict[str, Any], msgs: list[dict[str, object]], *, alias: str,
                     replies: Replies) -> Response:
    """A chat completion with audio output (module docstring): ``body`` the request, ``msgs`` its messages for the chat
    template."""
    if not speaks(engine):
        raise not_supported("The current model does not support speech output.")
    a = body.get("audio") or {}
    voice = str(a.get("voice") or "female").lower()
    if voice not in VOICES:
        raise bad_request(f"audio.voice {voice!r}: the model's voices are {', '.join(VOICES)}")
    stream = bool(body.get("stream"))
    fmt = str(a.get("format") or ("pcm16" if stream else "wav"))
    if fmt not in ("wav", "pcm16") or (stream and fmt != "pcm16"):
        raise bad_request(f"audio.format {fmt!r}: wav or pcm16 (streamed: pcm16)")
    if voice != "female":
        msgs = [{"role": "system", "content": f"Respond with interleaved text and audio, in a {voice} voice."}] + \
            [m for m in msgs if m.get("role") != "system"]
    seed = body.get("seed")
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024)
    mode = "audio" if "text" not in (body.get("modalities") or []) else "mixed"  # (["audio"]: speech, then text)
    model = engine.model
    cid, aid, created = hex_id("chatcmpl-", 16), hex_id("audio_", 12), int(time.time())
    head = {"created": created, "id": cid, "model": alias}

    said = Reply()

    def produce(push: Callable[[tuple[str, Any]], None], cancel: threading.Event) -> None:
        dec, st, n = StreamDecoder(model.tokenizer), SpeechStats(), 0
        for kind, x in model.voice_reply(msgs, mode=mode, seed=None if seed is None else int(seed),
                                         max_tokens=max_tokens, stats=st, reply=said):
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
        out = n + st.frames
        return {"prompt_tokens": st.prompt, "completion_tokens": out, "total_tokens": st.prompt + out,
                "completion_tokens_details": {"text_tokens": n, "audio_tokens": st.frames}}

    if not stream:
        text, parts, st, n = [], [], SpeechStats(), 0
        async for kind, x in items():
            if kind == "text":
                text.append(x)
            elif kind == "audio":
                parts.append(x)
            elif kind == "error":
                raise server_error(x)
            else:
                st, n = x
        y = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        data = wav_header(len(y)) + pcm16(y) if fmt == "wav" else pcm16(y)
        transcript = "".join(text)
        replies.add(aid, transcript, reply_parts(model, said, model.opener(mode)))
        audio = {"id": aid, "data": base64.b64encode(data).decode(), "expires_at": created + EXPIRES,
                 "transcript": transcript}
        return Response(json.dumps({"choices": [{"finish_reason": "stop", "index": 0, "message": {
            "role": "assistant", "content": None, "audio": audio}}], **head, "object": "chat.completion",
            "usage": usage(st, n)}), media_type="application/json")

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: object) -> str:
        return sse({"choices": [{"finish_reason": finish, "index": 0, "delta": delta}], **head,
                    "object": "chat.completion.chunk", **extra}, ensure_ascii=True)

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
                replies.add(aid, "".join(text), reply_parts(model, said, model.opener(mode)))
                inc = (body.get("stream_options") or {}).get("include_usage")
                yield chunk({"audio": {"id": aid, "expires_at": created + EXPIRES}}, "stop",
                            **({"usage": usage(*x)} if inc else {}))
                yield "data: [DONE]\n\n"
            else:
                yield sse({"error": {"message": x, "type": "server_error"}}, ensure_ascii=True)
    return audio_stream(events(), cancel, "text/event-stream")
