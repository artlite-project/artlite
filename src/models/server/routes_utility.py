"""Utilities: ``/tokenize``, ``/detokenize``, ``/apply-template``, and the web UI's other calls (its Stop of a
conversation's request, the resumable streams it asks about: none)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from .context import ApiContext
from .errors import ApiError, bad_request
from .requests import DetokenizeRequest, TemplateRequest, TokenizeRequest
from .tools import functions


def add_routes(api: APIRouter, ctx: ApiContext) -> None:
    """The routes on ``api``."""
    engine = ctx.engine

    @api.post("/tokenize")
    async def tokenize(req: TokenizeRequest) -> dict[str, Any]:
        model = engine.model
        ids = await run_in_threadpool(model.encode, req.content)
        if not req.add_special:
            bos = model.encode("")
            ids = ids[len(bos):] if ids[:len(bos)] == bos else ids
        if not req.with_pieces:
            return {"tokens": ids}
        pieces = []
        for t in ids:
            b = model.tokenizer.decode_bytes([t], special=True)
            try:
                piece: str | list[int] = b.decode("utf-8")
            except UnicodeDecodeError:
                piece = list(b)
            pieces.append({"id": t, "piece": piece})
        return {"tokens": pieces}

    @api.post("/detokenize")
    async def detokenize(req: DetokenizeRequest) -> dict[str, str]:
        text = engine.model.tokenizer.decode_bytes(ctx.valid(req.tokens), special=True).decode("utf-8",
                                                                                              errors="replace")
        return {"content": text}

    @api.post("/apply-template")
    async def apply_template(req: TemplateRequest) -> dict[str, str]:
        try:
            fns = functions(req.tools) if ctx.syntax is not None else []
        except ValueError as e:
            raise bad_request(str(e)) from None
        ids, _ = await ctx.prompts.render(req.messages, req.chat_template_kwargs, effort=req.reasoning_effort,
                                          media=True, tools=fns)
        return {"prompt": engine.model.tokenizer.decode_bytes(ids, special=True).decode("utf-8", errors="replace")}

    @api.delete("/v1/stream")
    async def stop_stream(conv_id: str = "") -> dict[str, Any]:
        """The UI's Stop: cancel its conversation's request."""
        return {"success": True, "cancelled": engine.cancel(conv_id) if conv_id else 0}

    @api.get("/v1/stream")
    async def resume_stream() -> None:
        raise ApiError(404, "not_found_error", "no resumable streams")

    @api.post("/v1/streams/lookup")
    async def lookup_streams() -> list[Any]:
        return []
