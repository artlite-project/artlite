"""Audio input: OpenAI's transcriptions (``/v1/audio/transcriptions``, llama-server's) and the offline transcript of a
clip as its pieces come (``transcribe``: the realtime transcription sessions of a model without live ASR). Both are chat
requests with the model's ASR prompt (the system message of LFM3-Audio's card). The speech output: ``speech``."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from .app_base import sse
from .context import ApiContext
from .errors import bad_request, not_supported
from .requests import chat_request
from .streaming import Format, collect, events_of, stream_response, usage

TRANSCRIPT_TOKENS = (64, 8)  # a transcription's tokens at most by default: 64 + 8 per second of audio


def asr_usage(done: dict[str, Any]) -> dict[str, Any]:
    """The ``usage`` of a transcription (OpenAI's token form)."""
    u = usage(done)
    return {"type": "tokens", "input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
            "total_tokens": u["total_tokens"], "input_tokens_details": u["prompt_tokens_details"]}


class TranscriptFormat(Format):
    """A transcription's stream: ``transcript.text.delta`` events, then ``transcript.text.done`` with the text."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def delta(self, data: dict[str, Any]) -> list[str]:
        out = []
        for k, v in data["parts"]:
            if k == "content":
                self.parts.append(v)
                out.append(sse({"type": "transcript.text.delta", "delta": v}))
        return out

    def done(self, data: dict[str, Any]) -> list[str]:
        return [sse({"type": "transcript.text.done", "text": "".join(self.parts), "usage": asr_usage(data)}),
                "data: [DONE]\n\n"]


async def transcribe(ctx: ApiContext, x: np.ndarray, language: str) -> AsyncIterator[str]:
    """A clip's transcript (16 kHz samples; the transcriptions' chat request: the ASR prompt) as its pieces come."""
    from ..lfm3a.host import SR, asr_system, wav_bytes  # (numpy only)
    clip = {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes(x)).decode(), "format": "wav"}}
    _, events = await ctx.start(chat_request({
        "messages": [{"role": "system", "content": asr_system(language)}, {"role": "user", "content": [clip]}],
        "stream": True, "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": TRANSCRIPT_TOKENS[0] + TRANSCRIPT_TOKENS[1] * len(x) // SR}))
    async for kind, data in events_of(events, 0):
        if kind == "delta":
            for k, v in data["parts"]:
                if k == "content":
                    yield v
        elif kind in ("error", "cancelled"):
            raise RuntimeError(f"the transcription failed: {data}")


def add_routes(api: APIRouter, ctx: ApiContext) -> None:
    """``/v1/audio/transcriptions`` on ``api``."""
    engine = ctx.engine

    @api.post("/v1/audio/transcriptions")
    @api.post("/audio/transcriptions")
    async def transcriptions(request: Request) -> Any:
        """OpenAI's transcriptions (llama-server's): a multipart form's ``file`` (an audio file), ``language``,
        ``prompt`` (text before the clip in the user message), ``response_format`` (json, text), ``temperature``,
        ``max_tokens`` (default ``TRANSCRIPT_TOKENS``: a clip of minutes can loop), ``stream``; the model's ASR prompt
        (the system message of LFM3-Audio's card) as a chat request.
        The answer: ``{"type": "transcript.text.done", "text", "usage"}`` (stream: ``transcript.text.delta`` events
        first), or the text."""
        if engine.model.audio is None:
            raise not_supported("The current model does not support audio input.")
        try:
            form = await request.form()
        except (AssertionError, ValueError) as e:  # (no python-multipart, or not a multipart body)
            raise bad_request(f"a transcription takes a multipart form: {e}") from None
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise bad_request("No input file found for transcription")
        data = await f.read()
        get = lambda k, d="": str(form.get(k) or d)
        fmt = get("response_format", "json")
        if fmt not in ("json", "text"):
            raise bad_request(f"response_format {fmt!r}: json or text")
        from ..lfm3a.host import SR, asr_system, load_audio  # (numpy only)
        user = ([{"type": "text", "text": get("prompt")}] if get("prompt") else []) + \
            [{"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"}}]
        body: dict[str, Any] = {"messages": [{"role": "system", "content": asr_system(get("language"))},
                                             {"role": "user", "content": user}],
                                "stream": get("stream", "false").lower() == "true",
                                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            for k, cast in (("temperature", float), ("max_tokens", int), ("seed", int)):
                if form.get(k) not in (None, ""):
                    body[k] = cast(form.get(k))
            if "max_tokens" not in body:  # (a transcript that loops ends: 8 tokens a second of audio, ~2x speech's)
                body["max_tokens"] = TRANSCRIPT_TOKENS[0] + TRANSCRIPT_TOKENS[1] * len(
                    await run_in_threadpool(load_audio, data)) // SR
        except ValueError as e:
            raise bad_request(str(e)) from None
        req = chat_request(body)
        j, events = await ctx.start(req)
        if not req.stream:
            blocks, done = await collect(j, events, request)
            text = blocks.text("text") or ""
            if fmt == "text":
                return PlainTextResponse(text)
            return {"type": "transcript.text.done", "text": text, "usage": asr_usage(done)}
        return await stream_response(j, events, ctx.ping_of(req), TranscriptFormat())
