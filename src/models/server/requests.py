"""The request bodies of the routes that take chat or completion requests (the generation fields every one shares,
the others accepted: clients send many)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .app_base import validation_message
from .errors import bad_request

ROLES = ("system", "developer", "user", "assistant", "tool")


class Message(BaseModel):
    """A chat message: its other fields (tool_calls, name, ...) go to the template as they are."""

    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


class Generation(BaseModel):
    """The fields that every generation request shares (the others are accepted: clients send many)."""

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    stream: bool | None = False
    max_tokens: int | None = None
    n_predict: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int | None = 1
    cache_prompt: bool | None = None
    return_progress: bool | None = False
    timings_per_token: bool | None = False
    sse_ping_interval: float | None = None
    stream_options: dict[str, Any] | None = None
    n_probs: int | None = None
    post_sampling_probs: bool | None = None
    spec_n_max: int | None = Field(None, alias="speculative.n_max")  # (llama-server's: draft tokens per step at most)


class ChatRequest(Generation):
    messages: list[Message]
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    reasoning_format: str | None = None
    continue_final_message: bool | None = False
    tools: list[Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


class CompletionRequest(Generation):
    prompt: str | list[int] | list[str]
    logprobs: int | None = None  # (n_probs: llama-server's alias)


class TokenizeRequest(BaseModel):
    content: str
    add_special: bool = False
    with_pieces: bool = False


class DetokenizeRequest(BaseModel):
    tokens: list[int]


class TemplateRequest(BaseModel):
    messages: list[Message]
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    tools: list[Any] | None = None


def chat_request(body: dict[str, Any]) -> ChatRequest:
    """A converted request (the agents' APIs', the transcriptions') as the chat route's."""
    try:
        return ChatRequest.model_validate(body)
    except ValidationError as e:
        raise bad_request(validation_message(e.errors(), skip=0)) from None
