"""The model server's settings: the command line's (``Options``) and what the API reports of the model file
(``ModelInfo``)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI

from ..readouts import Readouts
from . import cli_options, decisions

PENALTIES = ("repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty")  # (Sampling's, Options')


@dataclass
class Options:
    """The server's settings (the command line, ``main``)."""

    alias: str  # the model's id in the API
    n_ctx: int  # the KV cache's positions
    api_keys: frozenset[str] = frozenset()  # none: no key needed
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    n_predict: int = -1  # a request's default max_tokens (< 0: no limit)
    temp: float | None = None  # sampling over the model's defaults (None: the model's)
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int = -1  # a request's default seed (< 0: random)
    reasoning: str = "off"  # think: on, off, auto (when the model can)
    reasoning_format: str = "auto"  # auto: thoughts in reasoning_content; none: in content
    preserve_reasoning: bool = True  # earlier answers keep their thoughts (the templates' preserve_thinking)
    reasoning_effort: str | None = None  # the templates' reasoning_effort (None: the template's own)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)  # under each request's
    cache_prompt: bool = True
    embedding: bool = False  # the embedding routes (llama-server's --embedding; with pooling rank: rerank too)
    pooling: str | None = None  # the embeddings' pooling (--pooling, else the model file's; None: none)
    embd_normalize: int = 2  # the embeddings' normalization by default (models.readouts.normalize)
    sse_ping_interval: float = 30.0  # seconds between pings while a stream waits (<= 0: none)
    slots: bool = True  # GET /slots
    ui: Callable[[FastAPI], None] | None = None  # serves the web UI's files (``ui.mount``); None: none
    no_ui: str = "The web UI is off: the API is at /v1."  # what / shows without one
    build_info: str = "Artlite"

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, alias: str, n_ctx: int, api_keys: frozenset[str],
                  reasoning_effort: str | None, penalties: dict[str, Any], pooling: str | None) -> Options:
        """The options of a model server's command line (``alias``, ``n_ctx``, ``api_keys``, ``reasoning_effort``,
        ``penalties``, ``pooling``: what ``main`` derived from it first; the web UI's fields come after: ``web_ui``)."""
        return cls(alias=alias, n_ctx=n_ctx, api_keys=api_keys, n_predict=args.n_predict, temp=args.temp,
                   top_k=args.top_k, top_p=args.top_p, min_p=args.min_p, **penalties, seed=args.seed,
                   reasoning=args.reasoning, reasoning_format="none" if args.reasoning_format == "none" else "auto",
                   preserve_reasoning=args.reasoning_preserve, reasoning_effort=reasoning_effort,
                   chat_template_kwargs=args.chat_template_kwargs, cache_prompt=args.cache_prompt,
                   embedding=args.embedding or args.rerank, pooling=pooling, embd_normalize=args.embd_normalize,
                   sse_ping_interval=args.sse_ping_interval, slots=args.slots,
                   cors_origins=cli_options.parse_cors(args.cors_origins), build_info=cli_options.build_info())

@dataclass(frozen=True)
class ModelInfo:
    """What the API reports of the model file (its GGUF header; ``chat_template``: the server's in place of the
    file's, ``--chat-template``)."""

    path: str
    chat_template: str = ""
    bos_token: str = ""
    eos_token: str = ""
    n_ctx_train: int = 0
    n_embd: int = 0
    n_vocab: int = 0
    n_params: int = 0
    size: int = 0
    ftype: str = ""
    readouts: Readouts = field(default_factory=Readouts)  # (its pooling, classifier head, rerank prompt)
    decision: str | None = None  # its decision type (``{arch}.decision.type``; None: not a decision model)
    decisions: decisions.Decisions | None = None  # (its decision context made ahead: else on the first request)

    @classmethod
    def from_gguf(cls, path: str) -> ModelInfo:
        from artlite.gguf_model import GGUFFile

        from ..bench import file_info, file_type_name

        g = GGUFFile(path)
        meta, arch = g.meta, str(g.arch)
        fields, n_vocab = file_info(path, gguf=g)
        ids = {k: int(meta[f"tokenizer.ggml.{k}_token_id"]) for k in ("bos", "eos") if f"tokenizer.ggml.{k}_token_id" in meta}
        names = {i: t for t, i in json.loads(g.strings_json("tokenizer.ggml.tokens", list(ids.values()))).items()} \
            if ids else {}  # (by id: BOS and EOS may be one token, Qwen3-Embedding's)
        special = {k: names[i] for k, i in ids.items()}
        return cls(path=path, chat_template=str(meta.get("tokenizer.chat_template", "")),
                   bos_token=special.get("bos", ""), eos_token=special.get("eos", ""),
                   n_ctx_train=int(meta.get(f"{arch}.context_length", 0)),
                   n_embd=int(meta.get(f"{arch}.embedding_length", 0)), n_vocab=n_vocab,
                   n_params=int(fields["model_n_params"]), size=int(fields["model_size"]),
                   ftype=file_type_name(meta.get("general.file_type")), readouts=Readouts.of(path),
                   decision=decisions.decision_type(meta, arch))
