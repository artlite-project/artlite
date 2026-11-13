"""The server's error responses: ``ApiError`` (OpenAI's error shape, which ``app_base.new_app``'s handlers answer
with, Anthropic's on its routes) and the constructors of the errors the routes share. A leaf module: every route
module imports it, none is imported."""

from typing import Any


class ApiError(Exception):
    """An error response in OpenAI's shape: ``{"error": {"code", "message", "type", ...}}`` (``extra``: more fields,
    e.g. a context overflow's ``n_prompt_tokens`` and ``n_ctx``)."""

    def __init__(self, status: int, kind: str, message: str, **extra: object):
        super().__init__(message)
        self.status, self.kind, self.message, self.extra = status, kind, message, extra

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.status, "message": self.message, "type": self.kind, **self.extra}}


def bad_request(message: str) -> ApiError:
    """400: the request is wrong."""
    return ApiError(400, "invalid_request_error", message)


def not_supported(message: str) -> ApiError:
    """501: llama-server's feature that Artlite does not have."""
    return ApiError(501, "not_supported_error", message)


def server_error(message: str) -> ApiError:
    """500: a failure in the model or the server."""
    return ApiError(500, "server_error", message)


def unavailable(message: str) -> ApiError:
    """503: the server cannot answer now (loading, shutting down)."""
    return ApiError(503, "unavailable_error", message)


def cancelled() -> ApiError:
    """503: a job that ended early (its client went away, the UI's Stop, the shutdown)."""
    return unavailable("the request was cancelled")
