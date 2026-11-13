"""Raw-prompt completions: OpenAI's ``/v1/completions`` and llama-server's own ``/completion`` (its responses:
``content``, ``stop``, ``timings``, ``tokens_cached``, ...), streamed or not."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from .app_base import sse
from .context import ApiContext
from .errors import bad_request
from .ids import hex_id
from .schemas import CompletionRequest
from .streaming import Format, collect, stream_response, usage


class CompletionFormat(Format):
    """OpenAI's completion chunks (``body``: a chunk, the response's shape; ``timings``: ``timings_per_token``)."""

    def __init__(self, head: dict[str, Any], *, timings: bool):
        self.head, self.timings = head, timings

    def body(self, text: str, finish: str | None, probs: list | None = None, **extra: object) -> dict[str, Any]:
        return {"choices": [{"text": text, "index": 0, "logprobs": {"content": probs} if probs else None,
                             "finish_reason": finish}], **self.head, **extra}

    def progress(self, data: dict[str, Any]) -> list[str]:
        return [sse(self.body("", None, prompt_progress=data))]

    def delta(self, data: dict[str, Any]) -> list[str]:
        text = "".join(t for _, t in data["parts"])
        return [sse(self.body(text, None, data.get("probs"),
                              **({"timings": data["timings"]} if self.timings else {})))]

    def done(self, data: dict[str, Any]) -> list[str]:
        return [sse(self.body("", data["finish_reason"], data["probs"], usage=usage(data), timings=data["timings"])),
                "data: [DONE]\n\n"]


class LlamaCompletionFormat(Format):
    """llama-server's ``/completion`` chunks (``final``: the last one's shape; ``n_prompt``: the prompt's tokens)."""

    def __init__(self, final: Any, *, n_prompt: int, timings: bool):
        self.final, self.n_prompt, self.timings = final, n_prompt, timings

    def progress(self, data: dict[str, Any]) -> list[str]:
        return [sse({"index": 0, "content": "", "tokens": [], "stop": False, "id_slot": 0,
                     "tokens_predicted": 0, "tokens_evaluated": self.n_prompt, "prompt_progress": data})]

    def delta(self, data: dict[str, Any]) -> list[str]:
        return [sse({"index": 0, "content": "".join(t for _, t in data["parts"]), "tokens": [],
                     "stop": False, "id_slot": 0, "tokens_predicted": data["timings"]["predicted_n"],
                     "tokens_evaluated": self.n_prompt,
                     **({"completion_probabilities": data["probs"]} if data.get("probs") else {}),
                     **({"timings": data["timings"]} if self.timings else {})})]

    def done(self, data: dict[str, Any]) -> list[str]:
        return [sse(self.final("", data))]


def add_routes(api: APIRouter, ctx: ApiContext) -> None:
    """``/v1/completions``, ``/completion`` on ``api``."""
    engine, options = ctx.engine, ctx.options

    def prompt_ids(prompt: str | list[int] | list[str]) -> list[int]:
        """A raw prompt's tokens: a text (after the model's BOS), or token ids."""
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return ctx.valid(list(prompt))
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise bad_request("one prompt at a time")
            prompt = prompt[0]
        return engine.model.encode(prompt)

    @api.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Any:
        """OpenAI's completions: a raw prompt."""
        ctx.check(req)
        ids = await run_in_threadpool(prompt_ids, req.prompt)
        ctx.fits(ids)
        j, events = ctx.job(req, ids, think=False)
        cid, created, name = hex_id("cmpl-", 16), int(time.time()), options.alias
        head = {"created": created, "model": name, "system_fingerprint": options.build_info,
                "object": "text_completion", "id": cid}
        fmt = CompletionFormat(head, timings=bool(req.timings_per_token))
        if not req.stream:
            blocks, done = await collect(j, events, request)
            return fmt.body(blocks.text("text"), done["finish_reason"], done["probs"], usage=usage(done),
                            timings=done["timings"])
        return await stream_response(j, events, ctx.ping_of(req), fmt)

    @api.post("/completion")
    @api.post("/completions")
    async def completion(req: CompletionRequest, request: Request) -> Any:
        """llama-server's own completion: a raw prompt; its responses as llama-server's (``content``, ``stop``,
        ``timings``, ``tokens_cached``, ...)."""
        ctx.check(req)
        ids = await run_in_threadpool(prompt_ids, req.prompt)
        ctx.fits(ids)
        j, events = ctx.job(req, ids, think=False)
        n_prompt = len(ids)

        def final(content: str, done: dict[str, Any]) -> dict[str, Any]:
            t, finish = done["timings"], done["finish_reason"]
            stop_type = "word" if done["stop_word"] else "eos" if finish == "stop" else "limit"
            return {"index": 0, "content": content, "tokens": [], "id_slot": 0, "stop": True, "model": options.alias,
                    "tokens_predicted": t["predicted_n"], "tokens_evaluated": n_prompt,
                    "generation_settings": ctx.params(j.sampling), "prompt": req.prompt if isinstance(req.prompt, str)
                    else "", "has_new_line": "\n" in content, "truncated": False, "stop_type": stop_type,
                    "stopping_word": done["stop_word"] or "", "tokens_cached": done["n_cached"], "timings": t,
                    **({"completion_probabilities": done["probs"]} if done["probs"] else {})}

        if not req.stream:
            blocks, done = await collect(j, events, request)
            return final(blocks.text("text"), done)
        fmt = LlamaCompletionFormat(final, n_prompt=n_prompt, timings=bool(req.timings_per_token))
        return await stream_response(j, events, ctx.ping_of(req), fmt)
