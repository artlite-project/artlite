"""A job's events as a response: the events of its queue (``events_of``: with pings while it waits), the response of
a request without streaming (``collect``: the job's output as the blocks of its message) and the streamed one
(``stream_response``: server-sent events, each API's own through a ``Format``). A job that fails before its first
event or ping answers with the HTTP error a request without streaming gets (``opened``: llama-server's first error,
which clients show; a stream's error event they may not)."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from .agents import Blocks
from .app_base import NO_CACHE, CancellingResponse, sse
from .errors import cancelled, server_error
from .jobs import Job

Events = asyncio.Queue[tuple[str, dict[str, Any]]]


async def events_of(events: Events, ping: float) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """A job's events up to its last, with ("ping", {}) after every ``ping`` seconds without one."""
    while True:
        try:
            kind, data = await asyncio.wait_for(events.get(), ping if ping > 0 else None)
        except TimeoutError:
            yield "ping", {}
            continue
        yield kind, data
        if kind in ("done", "error", "cancelled"):
            return


async def opened(j: Job, events: Events, ping: float) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """A stream's events (``events_of``) once its job's first one or first ping has come: a job that fails before
    either answers with the HTTP error a request without streaming gets."""
    it = events_of(events, ping)
    try:
        first = await anext(it)
    except BaseException:  # (the handler cancelled: shutdown)
        j.cancel.set()
        raise
    if first[0] == "error":
        raise server_error(first[1]["message"])
    if first[0] == "cancelled":
        raise cancelled()

    async def rest() -> AsyncIterator[tuple[str, dict[str, Any]]]:
        yield first
        async for e in it:
            yield e

    return rest()


async def collect(j: Job, events: Events, request: Request) -> tuple[Blocks, dict[str, Any]]:
    """A job's output (a response without streaming) as the blocks of its message, and its last event (its
    ``probs``: every delta's and its own); the client going away cancels it."""
    blocks, probs = Blocks(), []
    loop = asyncio.get_running_loop()
    check = loop.time()
    try:
        async for kind, data in events_of(events, 0.5):
            if loop.time() >= check:  # (every half second)
                check = loop.time() + 0.5
                if await request.is_disconnected():
                    j.cancel.set()
            if kind == "delta":
                blocks.feed(data["parts"])
                probs += data.get("probs", [])
            elif kind == "done":
                blocks.end()
                return blocks, {**data, "probs": probs + data["probs"]}
            elif kind == "error":
                raise server_error(data["message"])
            elif kind == "cancelled":
                raise cancelled()
    finally:
        j.cancel.set()  # (nothing when it ended; the handler cancelled: shutdown)
    raise AssertionError("unreachable")


def usage(done: dict[str, Any]) -> dict[str, Any]:
    """OpenAI's ``usage`` of a finished job."""
    t = done["timings"]
    return {"completion_tokens": t["predicted_n"], "prompt_tokens": done["n_prompt"],
            "total_tokens": t["predicted_n"] + done["n_prompt"],
            "prompt_tokens_details": {"cached_tokens": t["cache_n"]}}


class Format:
    """How a stream of events becomes server-sent events: per event kind, the chunks to send (each is one piece of the
    response body). The defaults: a ping is an SSE comment, progress is dropped, an error is OpenAI's error body as
    an event; ``delta`` and ``done`` are each API's own."""

    def start(self) -> list[str]:
        """The chunks before the first event."""
        return []

    def ping(self) -> list[str]:
        return [":\n\n"]

    def progress(self, data: dict[str, Any]) -> list[str]:
        return []

    def delta(self, data: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def done(self, data: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def error(self, kind: str, data: dict[str, Any]) -> list[str]:
        """The job's ``error`` or ``cancelled`` event."""
        e = server_error(data["message"]) if kind == "error" else cancelled()
        return [sse(e.body())]


async def stream_response(j: Job, events: Events, ping: float, fmt: Format) -> CancellingResponse:
    """The job's events as server-sent events of ``fmt``; the response ending early (its client went away) cancels the
    job."""
    evs = await opened(j, events, ping)

    async def body() -> AsyncIterator[str]:
        for chunk in fmt.start():
            yield chunk
        async for kind, data in evs:
            if kind == "ping":
                chunks = fmt.ping()
            elif kind == "progress":
                chunks = fmt.progress(data)
            elif kind == "delta":
                chunks = fmt.delta(data)
            elif kind == "done":
                chunks = fmt.done(data)
            else:
                chunks = fmt.error(kind, data)
            for chunk in chunks:
                yield chunk

    return CancellingResponse(body(), media_type="text/event-stream", headers=NO_CACHE, on_end=j.cancel.set)
