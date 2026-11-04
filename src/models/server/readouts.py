"""llama-server's readout routes (llama.cpp 7182ef71e's shapes and messages): embeddings (``/v1/embeddings``,
OpenAI's; ``/embeddings``, ``/embedding``, llama-server's own) and rerank (``/rerank``, ``/reranking``,
``/v1/rerank``, ``/v1/reranking``: Jina's and TEI's shapes), with OpenAI's ``dimensions`` and Cohere's
``return_documents``; vLLM's ``/classify``: a classifier head's label probabilities per input.
Each request's inputs run on the worker one after another (``Engine.readouts``: prefill chunks only, after what the
prompt cache keeps), their rows pooled and scored on the host (``models.readouts``). Like llama-server's, the routes
need ``--embedding`` (rerank and classify: ``--rerank``, a model file with a classifier head) and answer 501
otherwise."""

import asyncio
import base64
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from ..readouts import Readouts, normalize, pool, rows_of
from .engine import Engine

ApiErrors = tuple[Callable[[str], Exception], Callable[[str], Exception]]  # (400, 501)


def add_routes(api: APIRouter, *, engine: Engine, alias: str, embedding: bool, pooling: str | None,
               embd_normalize: int, readouts: Readouts, cache_prompt: bool, errors: ApiErrors,
               fits: Callable[[list[int]], None], valid: Callable[[list[int]], list[int]]) -> None:
    """The routes on ``api``: ``embedding`` (``--embedding``), ``pooling`` (``--pooling``, else the file's: None or
    "none" give per-token rows), ``embd_normalize`` (the server's default), ``readouts`` (the file's), ``errors``: a
    400 and a 501 from a message, ``fits``: refuses a prompt the KV cache does not hold, ``valid``: checks token ids."""
    bad, unsupported = errors
    pooling = pooling or "none"

    def text_of(item: object, what: str) -> str:
        """A rerank query's or document's text: a string or {"text": a string or strings}."""
        if isinstance(item, str):
            return item
        if not isinstance(item, dict):
            raise bad(f"{what} must be a string or an object of the form {{\"text\": ...}}")
        if "image" in item or "video" in item:
            raise bad(f"images in a {what} are not supported")
        text = item.get("text", "")
        if isinstance(text, list) and all(isinstance(t, str) for t in text):
            return "".join(text)
        if not isinstance(text, str):
            raise bad("\"text\" must be a string or an array of strings")
        return text

    async def rows(inputs: list[list[int]], at: list[np.ndarray], name: str) -> list[np.ndarray]:
        for ids in inputs:
            fits(ids)
        try:
            return await asyncio.wrap_future(engine.readouts(inputs, at, cache_prompt=cache_prompt, name=name))
        except NotImplementedError as e:  # (a model without readouts: Gemma 4)
            raise unsupported(str(e)) from None

    def embedding_ids(item: object) -> list[int]:
        if isinstance(item, list) and all(isinstance(t, int) for t in item):
            return valid(list(item))
        if not isinstance(item, str):
            raise bad("an input is a string or an array of token ids")
        return readouts.embedding_ids(engine.model.encode, item)

    async def embeddings(body: dict[str, Any], oai: bool) -> Any:
        if not embedding:
            raise unsupported("This server does not support embeddings. Start it with `--embeddings`")
        if oai and pooling == "none":
            raise bad("Pooling type 'none' is not OAI compatible. Please use a different pooling type")
        if "input" in body:
            prompt = body["input"]
        elif "content" in body:
            prompt, oai = body["content"], False  # ("content" is not OpenAI's)
        else:
            raise bad("\"input\" or \"content\" must be provided")
        fmt = body.get("encoding_format", "float")
        if fmt not in ("float", "base64"):
            raise bad("The format to return the embeddings in. Can be either float or base64")
        norm = body.get("embd_normalize", embd_normalize)
        dims = body.get("dimensions")
        if not isinstance(norm, int) or dims is not None and (not isinstance(dims, int) or dims < 1):
            raise bad("\"embd_normalize\" and \"dimensions\" are integers (dimensions at least 1)")
        items = prompt if isinstance(prompt, list) and not (prompt and all(isinstance(t, int) for t in prompt)) \
            else [prompt]
        if not items:
            raise bad("\"input\" must not be empty")
        inputs = await run_in_threadpool(lambda: [embedding_ids(p) for p in items])
        if not all(inputs):
            raise bad("Input content cannot be empty")
        out = await rows(inputs, [rows_of(pooling, len(ids)) for ids in inputs], "embeddings")
        vecs = []
        for r in out:
            v = pool(r, pooling)
            if pooling != "none":  # (llama-server normalizes pooled embeddings only)
                v = normalize(v[:dims], norm) if dims is not None else normalize(v, norm)
            vecs.append(v)
        if not oai:
            return [{"index": i, "embedding": (v if v.ndim == 2 else v[None]).tolist()} for i, v in enumerate(vecs)]
        n = sum(map(len, inputs))
        data = [{"embedding": base64.b64encode(v.astype("<f4").tobytes()).decode(), "index": i, "object": "embedding",
                 "encoding_format": "base64"} if fmt == "base64" else
                {"embedding": v.tolist(), "index": i, "object": "embedding"} for i, v in enumerate(vecs)]
        return {"model": body.get("model") or alias, "object": "list",
                "usage": {"prompt_tokens": n, "total_tokens": n}, "data": data}

    @api.post("/v1/embeddings")
    async def embeddings_oai(body: dict[str, Any]) -> Any:
        """OpenAI's embeddings (pooled, normalized: ``embd_normalize``, Euclidean by default)."""
        return await embeddings(body, oai=True)

    @api.post("/embeddings")
    @api.post("/embedding")
    async def embeddings_own(body: dict[str, Any]) -> Any:
        """llama-server's own: per input its pooled embedding, or every token's (pooling none)."""
        return await embeddings(body, oai=False)

    async def rerank(body: dict[str, Any]) -> Any:
        """Jina's rerank (TEI's with ``texts``): the documents' relevance to the query, most relevant first."""
        if not embedding or pooling != "rank" or readouts.head is None:
            raise unsupported("This server does not support reranking. Start it with `--reranking`")
        tei = "texts" in body
        if "query" not in body:
            raise bad("\"query\" must be provided")
        query = text_of(body["query"], "query")
        docs = body.get("documents", body.get("texts", []))
        if not isinstance(docs, list) or not docs:
            raise bad("\"documents\" must be a non-empty array of strings or objects")
        texts = [text_of(d, "document") for d in docs]
        top_n = body.get("top_n", len(texts))
        if not isinstance(top_n, int) or top_n < 0:
            raise bad("\"top_n\" must be >= 0")
        model = engine.model
        bos = await run_in_threadpool(model.encode, "")
        inputs = await run_in_threadpool(lambda: [readouts.rerank_ids(model.tokenizer.encode, bos, query, t)
                                                  for t in texts])
        out = await rows(inputs, [rows_of("rank", len(ids)) for ids in inputs], "rerank")
        label = "score" if tei else "relevance_score"
        results = [{"index": i, label: readouts.head.score(r[0])} for i, r in enumerate(out)]
        results.sort(key=lambda e: -e[label])
        results = results[:top_n]
        for e in results:
            if tei and body.get("return_text"):
                e["text"] = texts[e["index"]]
            elif not tei and body.get("return_documents"):  # (Cohere's)
                e["document"] = {"text": texts[e["index"]]}
        if tei:
            return results
        n = sum(map(len, inputs))
        return {"model": body.get("model") or alias, "object": "list",
                "usage": {"prompt_tokens": n, "total_tokens": n}, "results": results}

    for path in ("/rerank", "/reranking", "/v1/rerank", "/v1/reranking"):
        api.add_api_route(path, rerank, methods=["POST"])

    async def classify(body: dict[str, Any]) -> Any:
        """vLLM's classification: per input (a text or token ids) its label probabilities, the most likely label."""
        if not embedding or pooling != "rank" or readouts.head is None:
            raise unsupported("This server does not support classification. Start it with `--reranking` on a model "
                              "file with a classifier head (cls.output)")
        prompt = body.get("input")
        items = prompt if isinstance(prompt, list) and not (prompt and all(isinstance(t, int) for t in prompt)) \
            else [prompt]
        if prompt is None or not items:
            raise bad("\"input\" must be a non-empty string, array of strings or of token ids")
        inputs = await run_in_threadpool(lambda: [embedding_ids(p) for p in items])
        if not all(inputs):
            raise bad("Input content cannot be empty")
        out = await rows(inputs, [rows_of("rank", len(ids)) for ids in inputs], "classify")
        data = []
        for i, r in enumerate(out):
            p = readouts.head.probs(r[0])
            data.append({"index": i, "label": readouts.head.label(int(p.argmax())), "probs": p.tolist(),
                         "num_classes": len(p)})
        n = sum(map(len, inputs))
        return {"id": f"classify-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()),
                "model": body.get("model") or alias, "data": data,
                "usage": {"prompt_tokens": n, "total_tokens": n, "completion_tokens": 0, "prompt_tokens_details": None}}

    api.add_api_route("/classify", classify, methods=["POST"])
