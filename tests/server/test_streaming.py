"""The server-sent events of every API (``models.server.streaming.Format`` and its subclasses) driven by one scripted
job on the host: the chunks of its first events, its end, and a failed or cancelled job in each API's error shape."""

import json
from typing import Any

import pytest

from models.server import agents
from models.server.errors import ApiError, bad_request, cancelled, not_supported, server_error, unavailable
from models.server.routes_agents import AnthropicFormat, ResponsesFormat
from models.server.routes_audio import TranscriptFormat
from models.server.routes_chat import ChatFormat
from models.server.routes_completion import CompletionFormat, LlamaCompletionFormat
from models.server.streaming import Format

TIMINGS = {"cache_n": 0, "prompt_n": 2, "prompt_ms": 1.0, "prompt_per_token_ms": 0.5, "prompt_per_second": 2000.0,
           "predicted_n": 2, "predicted_ms": 2.0, "predicted_per_token_ms": 1.0, "predicted_per_second": 1000.0}
DELTA = {"parts": [("content", "Hi"), ("content", " there")], "timings": TIMINGS}
DONE = {"finish_reason": "stop", "stop_word": None, "timings": TIMINGS, "n_prompt": 2, "n_cached": 4, "n_calls": 0,
        "probs": []}
PROGRESS = {"total": 2, "cache": 0, "processed": 2, "time_ms": 1.0}


def final(content: str, done: dict[str, Any]) -> dict[str, Any]:
    return {"content": content, "stop": True, "timings": done["timings"]}


def formats() -> dict[str, Format]:
    head = {"id": "x", "created": 0, "model": "m"}
    return {
        "chat": ChatFormat(head, timings=False, include_usage=True),
        "completion": CompletionFormat(head, timings=False),
        "llama": LlamaCompletionFormat(final, n_prompt=2, timings=False),
        "transcript": TranscriptFormat(),
        "anthropic": AnthropicFormat(agents.Anthropic("m")),
        "responses": ResponsesFormat(agents.Responses("m", {}), store=None, inputs=[]),
    }


def text(chunks: list[str]) -> str:
    assert all(isinstance(c, str) for c in chunks)
    return "".join(chunks)


@pytest.mark.parametrize("name", list(formats()))
def test_stream(name: str) -> None:
    """Every API's stream: its text comes with the deltas and the job's end closes it (the chat, completion and
    transcription streams with ``[DONE]``)."""
    fmt = formats()[name]
    out = text(fmt.start())
    if name == "anthropic":
        out += text(fmt.progress(PROGRESS))
    out += text(fmt.delta(DELTA)) + text(fmt.done(DONE))
    assert "Hi" in out and " there" in out
    assert out.endswith("[DONE]\n\n") == (name in ("chat", "completion", "transcript"))


@pytest.mark.parametrize("name", list(formats()))
def test_progress_and_ping(name: str) -> None:
    fmt = formats()[name]
    assert text(fmt.ping()) in (":\n\n", agents.event("ping", {"type": "ping"}))
    progress = text(fmt.progress(PROGRESS))
    assert ("prompt_progress" in progress) == (name in ("chat", "completion", "llama"))


@pytest.mark.parametrize("name", list(formats()))
def test_errors(name: str) -> None:
    """A failed job: its message; a cancelled one: the cancellation; in the API's shape."""
    fmt = formats()[name]
    failed, ended = text(fmt.error("error", {"message": "boom"})), text(fmt.error("cancelled", {}))
    assert "boom" in failed and "cancelled" in ended
    if name != "responses":
        body = json.loads(next(line for line in failed.split("\n") if line.startswith("data: "))[6:])
        assert "boom" in json.dumps(body)


def test_error_bodies() -> None:
    assert bad_request("x").body() == {"error": {"code": 400, "message": "x", "type": "invalid_request_error"}}
    assert not_supported("x").status == 501 and server_error("x").status == 500 and unavailable("x").status == 503
    e = ApiError(400, "exceed_context_size_error", "m", n_ctx=8)
    assert e.body()["error"]["n_ctx"] == 8 and cancelled().message == "the request was cancelled"
