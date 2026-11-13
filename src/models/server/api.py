"""The model server's HTTP side (FastAPI): llama-server's routes that apply to one model on the NPU, their request and
response shapes, server-sent events, the API keys, CORS and OpenAI's error shape (Anthropic's on its routes). The
generations run on the engine's worker (``engine.Engine``); a route renders the prompt (in a thread: a long prompt's
tokenizer takes a while), checks that it fits the KV cache, queues a job and turns its events into the response
(``streaming``). ``create_app`` makes the context the routes share (``context.ApiContext``) and registers the route
groups: ``routes_status``, ``routes_chat``, ``routes_audio``, ``routes_agents`` (Anthropic's Messages and OpenAI's
Responses: chat requests converted by ``agents``), ``routes_completion``, ``routes_utility``, ``routes_readouts``,
``routes_decisions``, ``speech``, ``realtime``."""

from fastapi import APIRouter, Depends, FastAPI

from . import (
    realtime,
    routes_agents,
    routes_audio,
    routes_chat,
    routes_completion,
    routes_decisions,
    routes_readouts,
    routes_status,
    routes_utility,
)
from .app_base import add_ui, add_unsupported, key_check, key_valid, new_app
from .context import ApiContext
from .engine import Engine
from .errors import unavailable
from .settings import ModelInfo, Options
from .speech import add_speech


def create_app(engine: Engine, options: Options, info: ModelInfo) -> FastAPI:
    """The FastAPI app of the server: the routes on ``engine`` with ``options``; ``info``: the model file's."""
    app = new_app(build_info=options.build_info, cors_origins=options.cors_origins)
    ctx = ApiContext(engine, options, info)

    async def loaded() -> None:
        if not engine.ready:
            raise unavailable("Loading model")

    authorized = key_check(options.api_keys)
    public = APIRouter()  # no key, any state
    listed = APIRouter(dependencies=[Depends(authorized)])  # also while the model loads
    api = APIRouter(dependencies=[Depends(authorized), Depends(loaded)])

    routes_status.add_routes(public, listed, api, ctx)
    routes_chat.add_routes(api, ctx)
    routes_audio.add_routes(api, ctx)
    routes_agents.add_routes(api, ctx)
    routes_completion.add_routes(api, ctx)
    routes_utility.add_routes(api, ctx)
    routes_readouts.add_routes(api, ctx)
    routes_decisions.add_routes(api, ctx)
    realtime.add_route(app, engine=engine, alias=options.alias,
                       authorized=lambda a, x: key_valid(options.api_keys, a, x),
                       transcribe=lambda x, language: routes_audio.transcribe(ctx, x, language))
    add_speech(api, engine)
    add_unsupported(listed)
    app.include_router(public)
    app.include_router(listed)
    app.include_router(api)
    add_ui(app, options.ui, options.no_ui)
    return app
