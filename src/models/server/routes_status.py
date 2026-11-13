"""Status and information: ``/health``, ``/props``, ``/v1/models``, ``/slots``."""

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..lfm3a.speech import VOICES
from .context import ApiContext
from .errors import not_supported, unavailable
from .listing import Modalities, architecture, model_entries, model_meta


def add_routes(public: APIRouter, listed: APIRouter, api: APIRouter, ctx: ApiContext) -> None:
    """The routes: ``/health`` on ``public``, ``/v1/models`` on ``listed`` (both also while the model loads), the
    others on ``api``."""
    engine, options, info = ctx.engine, ctx.options, ctx.info

    @public.get("/health")
    @public.get("/v1/health")
    async def health() -> JSONResponse:
        if engine.ready:
            return JSONResponse({"status": "ok"})
        return JSONResponse(unavailable("Loading model").body(), status_code=503)

    @api.get("/props")
    async def props() -> dict[str, Any]:
        mods = Modalities.of(engine)
        return {"default_generation_settings": {"n_ctx": engine.npu.capacity, "params": ctx.params(ctx.sampling(
                    ctx.default_think()))},
                "total_slots": 1, "model_alias": options.alias, "model_path": info.path,
                "modalities": {"vision": mods.vision, "audio": mods.audio, "video": mods.video},
                **({"voices": list(VOICES)} if mods.speech else {}),
                "endpoint_slots": options.slots, "endpoint_props": False, "endpoint_metrics": False,
                "ui": options.ui is not None, "ui_settings": {},
                "chat_template": info.chat_template, "bos_token": info.bos_token, "eos_token": info.eos_token,
                "chat_template_caps": {"supports_tools": ctx.syntax is not None,
                                       "supports_tool_calls": ctx.syntax is not None,
                                       "supports_parallel_tool_calls": ctx.syntax is not None,
                                       "supports_preserve_reasoning": "preserve_thinking" in info.chat_template},
                "build_info": options.build_info, "is_sleeping": False, "cors_proxy_enabled": False}

    @listed.get("/v1/models")
    @listed.get("/models")
    async def models() -> dict[str, Any]:
        mods = Modalities.of(engine)
        data, details = model_entries(
            options.alias, aliases=[], created=int(time.time()), multimodal=mods.vision or mods.audio,
            arch=architecture(mods.media, decision=bool(info.decision)),
            meta=model_meta(n_vocab=info.n_vocab, n_ctx=options.n_ctx, n_ctx_train=info.n_ctx_train,
                            n_embd=info.n_embd, n_params=info.n_params, size=info.size, ftype=info.ftype))
        return {"object": "list", "models": [details], "data": [data]}

    @api.get("/slots")
    async def slots() -> list[dict[str, Any]]:
        if not options.slots:
            raise not_supported("This server does not support slots endpoint.")
        j = engine.current
        return [{"id": 0, "id_task": -1 if j is None else 0, "n_ctx": engine.npu.capacity,
                 "is_processing": j is not None, "n_cached": len(engine.cached), "n_waiting": engine.waiting(),
                 "params": ctx.params(ctx.sampling(ctx.default_think())) if j is None
                 else ctx.params(j.sampling, j.drafts),
                 "next_token": [{"has_next_token": j is not None and j.state == "generating",
                                 "n_decoded": 0 if j is None else j.n_gen}]}]
