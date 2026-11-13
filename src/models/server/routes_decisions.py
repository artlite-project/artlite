"""``/v1/systemone``'s route (``decisions``: the questions' prompts and scores): the request's media encoded on the
engine's worker, the prompts' output rows read there (``Engine.readouts``), the answers."""

import asyncio
import base64
from typing import Any

import numpy as np
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from ..sequence import Sequence, data_bytes
from .context import ApiContext
from .decisions import D1, IMAGE_TYPES, TYPES, BadRequest, Decisions, Unsupported, spoken
from .errors import bad_request, not_supported


def add_routes(api: APIRouter, ctx: ApiContext) -> None:
    """``/v1/systemone`` on ``api`` (the model file's decision type: None, not a decision model; its ``Decisions``,
    ``ModelInfo.decisions`` or built on the first request)."""
    engine, alias, kind, info = ctx.engine, ctx.options.alias, ctx.info.decision, ctx.info
    made: list[Decisions] = []
    lock = asyncio.Lock()

    async def decisions() -> Decisions:
        async with lock:
            if not made:
                if kind not in TYPES:
                    raise not_supported(f"decision model type {kind!r} is not supported by artlite-server (it answers "
                                      f"{', '.join(TYPES)})")
                made.append(await run_in_threadpool(
                    lambda: info.decisions or Decisions.of(info.path, engine.model.tokenizer.encode)))
        return made[0]

    @api.post("/v1/systemone")
    async def systemone(body: dict[str, Any]) -> Any:
        """Answers to questions about a state (module docstring)."""
        if kind is None:
            raise not_supported("This model is not a decision model")
        d = await decisions()
        try:
            questions = d.questions(body)
            state, images = d.state(body)
            set_size = d.set_size(body) if d.type in D1 else 1
        except BadRequest as e:
            raise bad_request(str(e)) from None
        except Unsupported as e:
            raise not_supported(str(e)) from None
        model = engine.model
        kinds = [m.kind for m in images]
        if d.type not in D1 and any(k != "image" for k in kinds):
            raise not_supported(f"artlite-server reads images in decision requests of {d.type}, not audio or video")
        clips = {i: q.instructions["input_audio"]["data"] for i, q in enumerate(questions) if spoken(q.instructions)}
        for k, enc in (("image", model.vision), ("video", model.vision), ("audio", model.audio)):
            if (k in kinds or k == "audio" and clips) and enc is None:
                raise not_supported(f"This server does not support {k} input for decisions. For a model that supports it, "
                                  "start it with `--mmproj`")
        if images and d.type not in IMAGE_TYPES:
            raise not_supported(f"artlite-server reads images in decision requests of {', '.join(IMAGE_TYPES)}, not "
                              f"{d.type}")
        media: list[Sequence] = []
        heard: dict[int, Sequence] = {}
        if images or clips:  # (the encoders on the worker, between its jobs)
            def encode_media() -> tuple[list[Sequence], dict[int, Sequence]]:
                if d.type not in D1:
                    return [model.image_tokens(b) for b in images], {}
                said = {i: model.media_tokens(data_bytes(c) if c.startswith("data:") else base64.b64decode(c), "audio")
                        for i, c in clips.items()}
                return [model.image_tokens(b) if k == "image" else model.media_tokens(b, k)
                        for b, k in zip(images, kinds, strict=True)], said
            try:
                media, heard = await asyncio.wrap_future(engine.run(encode_media))
            except (OSError, ValueError) as e:  # (media that do not open or are too large)
                raise bad_request(f"media: {e}") from None
        npu = engine.npu
        whole = npu.prefill.sizes[-1] if not npu.causal else 0  # (a non-causal model reads a prompt in one chunk)
        try:
            prompts = await run_in_threadpool(d.prompts, state, questions, model.tokenizer.encode, media=media,
                                              set_size=set_size, max_tokens=whole, spoken_media=heard)
        except BadRequest as e:
            raise bad_request(str(e)) from None
        for p in prompts:
            ctx.fits(p.ids)
            if whole and len(p.ids) > whole:
                raise bad_request(f"the prompt has {len(p.ids)} tokens, this model reads at most {whole} in one pass")
        sets = [p.sets for p in prompts] if any(p.sets for p in prompts) else None
        out = await asyncio.wrap_future(engine.readouts([p.ids for p in prompts], [p.rows for p in prompts],
                                                        cache_prompt=d.shares_prompt, name="decisions", sets=sets))
        scores: list[list[np.ndarray]] = [[] for _ in questions]
        for p, rows in zip(prompts, out, strict=True):
            if len(p.questions) == 1:
                scores[p.question].append(d.scores(questions[p.question], rows))
                continue
            for j, i in enumerate(p.questions):  # (a sequence's questions: a row each)
                scores[i].append(d.scores(questions[i], rows[j:j + 1]))
        n = sum(len(p.ids) for p in prompts)
        return {"model": body.get("model") or alias,
                "answers": {q.id: d.answer(q, s, media=bool(media)) for q, s in zip(questions, scores, strict=True)},
                "usage": {"input_tokens": n, "output_tokens": 0}}
