"""What the model's server and the router's app share: the FastAPI app whose errors all answer in the API's shape
(OpenAI's, Anthropic's on its routes), the API keys, the routes of llama-server's features that Artlite does not have,
the web UI's page, server-sent events and the streaming response that stops its work when its client goes away."""

import inspect
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Receive, Scope, Send

from . import agents
from .errors import ApiError, bad_request, not_supported

log = logging.getLogger("artlite.server")

ANTHROPIC = "/v1/messages"  # (the routes under it answer errors in Anthropic's shape)
NOT_SUPPORTED = ["/v1/chat/completions/control", "/infill", "/lora-adapters", "/metrics"]
NO_CACHE = {"Cache-Control": "no-cache"}


def sse(data: object, *, ensure_ascii: bool = False) -> str:
    """A server-sent event of a JSON value."""
    return f"data: {json.dumps(data, ensure_ascii=ensure_ascii)}\n\n"


class CancellingResponse(StreamingResponse):
    """A streamed response whose work stops when the response ends, early or not (its client went away): ``on_end``
    (a function, or a coroutine function) runs then."""

    def __init__(self, body: AsyncIterator[Any], *, on_end: Callable[[], Awaitable[None] | None],
                 media_type: str | None = None, headers: dict[str, str] | None = None, status_code: int = 200):
        super().__init__(body, status_code=status_code, headers=headers, media_type=media_type)
        self.on_end = on_end

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if inspect.isawaitable(done := self.on_end()):
                await done


def validation_message(errors: list[dict[str, Any]], *, skip: int = 1) -> str:
    """A request's validation errors as one line (``skip``: the leading parts of their locations, FastAPI's
    ``body``)."""
    return "; ".join(f"{'.'.join(str(x) for x in err['loc'][skip:])}: {err['msg']}" for err in errors) or \
        "invalid request"


def error_response(request: Request, e: ApiError) -> JSONResponse:
    """An error in the shape of the request's API: Anthropic's on its routes, else OpenAI's."""
    if request.url.path.startswith(ANTHROPIC):
        return JSONResponse(agents.anthropic_error(e.status, e.message), status_code=e.status)
    return JSONResponse(e.body(), status_code=e.status)


def new_app(*, build_info: str, cors_origins: list[str], lifespan: Callable | None = None) -> FastAPI:
    """A FastAPI app whose errors all answer in the API's shape (``error_response``), with CORS."""
    app = FastAPI(title="artlite-server", version=build_info, docs_url="/docs", redoc_url=None, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ApiError)
    async def api_error(request: Request, e: ApiError) -> JSONResponse:
        return error_response(request, e)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, e: RequestValidationError) -> JSONResponse:
        return error_response(request, bad_request(validation_message(e.errors())))

    @app.exception_handler(Exception)
    async def failure(request: Request, e: Exception) -> JSONResponse:
        log.exception("%s %s failed", request.method, request.url.path)
        return error_response(request, ApiError(500, "server_error", f"{type(e).__name__}: {e}"))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, e: StarletteHTTPException) -> JSONResponse:
        kind = {401: "authentication_error", 403: "permission_error", 404: "not_found_error"}.get(e.status_code,
                                                                                                 "invalid_request_error")
        return error_response(request, ApiError(e.status_code, kind, str(e.detail)))

    return app


def key_valid(api_keys: frozenset[str], authorization: str | None, x_api_key: str | None) -> bool:
    """Whether a request's ``Authorization: Bearer`` or ``X-Api-Key`` is one of the server's keys (any, without keys)."""
    if not api_keys:
        return True
    key = x_api_key or (authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else "")
    return any(secrets.compare_digest(key.encode(), k.encode()) for k in api_keys)


def key_check(api_keys: frozenset[str]) -> Callable[..., Awaitable[None]]:
    """The dependency of the routes that need an API key (``key_valid``), when the server has keys."""

    async def authorized(authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> None:
        if not key_valid(api_keys, authorization, x_api_key):
            raise ApiError(401, "authentication_error", "Invalid API Key")

    return authorized


def add_unsupported(router: APIRouter) -> None:
    """llama-server's routes that Artlite does not have (501), and ``/tools``: 403, which the web UI reads as its server
    tools being off."""

    @router.get("/tools")
    @router.post("/tools")
    async def tools() -> JSONResponse:
        return JSONResponse({"error": {"message": "this feature is disabled", "type": "feature_disabled"}},
                            status_code=403)

    async def unsupported(request: Request) -> None:
        raise not_supported(f"{request.url.path} is not supported by artlite-server")

    for path in NOT_SUPPORTED:
        router.add_api_route(path, unsupported, methods=["GET", "POST"], include_in_schema=False)


def add_ui(app: FastAPI, ui: Callable[[FastAPI], None] | None, no_ui: str) -> None:
    """The web UI's files at ``/`` (after the app's routes), or a page that says why there is none."""
    if ui is not None:
        ui(app)
        return

    @app.get("/", include_in_schema=False)
    async def page() -> HTMLResponse:
        return HTMLResponse(f"<!doctype html><title>artlite-server</title><p>{no_ui}</p>")
