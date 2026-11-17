"""A chat request whose answer is one of a few options, with ``logprobs`` (Halogen's decisions, on any chat model):
``response_format`` (or ``json_schema``) a JSON schema of options - an ``enum``, a ``oneOf`` / ``anyOf`` of ``const``,
``{"type": "boolean"}``, or an object of one required property that is one of those -: the options' exact
probabilities instead of a generation. ``message.content`` is the most likely option as JSON in the schema's shape,
``logprobs.content[0].top_logprobs`` one entry per option (``token``: the option, a string as it is, another value as
JSON; ``logprob``), as Halogen returns them.

An option's probability is its JSON text's (then the end of the turn) as the model's next tokens, along its own
tokens, as a generation held to the options would write it (``score``): the canonical tokens of the whole text are not
the model's (Qwen's pre-tokenizer merges an opening quote with the word: ``"very``, while a model writes ``"`` and
then ``very``). The prompt runs as one prompt; where its logits part the options (a token only one of them continues),
they are all; a token several continue (the opening quote) runs as a decode step from a checkpoint, a step or two in
all. A model that thinks first (its template opened a thought) generates as any request: no decision there."""

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..npu_model import NpuModel
from ..sampling import LogitsLike
from ..structured import Vocab

MAX_OPTIONS = 64


@dataclass(frozen=True)
class Options:
    """A schema's options (``values``) and, for an object of one property, its ``key``."""

    values: list[Any]
    key: str | None = None

    def text(self, v: Any) -> str:
        """An option's answer as JSON in the schema's shape."""
        value = json.dumps(v, ensure_ascii=False)
        return value if self.key is None else json.dumps({self.key: v}, ensure_ascii=False)

    @staticmethod
    def token(v: Any) -> str:
        """An option as a ``top_logprobs`` entry's token (a string as it is, another value as JSON)."""
        return v if isinstance(v, str) else json.dumps(v)


def options_of(schema: object) -> Options | None:
    """The options of a JSON schema (the module docstring's shapes, 2 .. ``MAX_OPTIONS``), or None."""
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict) and \
            len(schema["properties"]) == 1 and list(schema.get("required") or []) == list(schema["properties"]):
        (key, sub), = schema["properties"].items()
        inner = options_of(sub)
        return Options(inner.values, key) if inner is not None and inner.key is None else None
    if isinstance(schema.get("enum"), list):
        values = schema["enum"]
    elif any(isinstance(schema.get(k), list) for k in ("oneOf", "anyOf")):
        alts = schema.get("oneOf") or schema.get("anyOf")
        if not all(isinstance(a, dict) and "const" in a for a in alts):
            return None
        values = [a["const"] for a in alts]
    elif schema.get("type") == "boolean" and len(schema) <= 3 and not set(schema) - {"type", "description", "title"}:
        values = [True, False]
    else:
        return None
    unique = [v for i, v in enumerate(values) if v not in values[:i]]
    if not 2 <= len(unique) <= MAX_OPTIONS or not all(isinstance(v, str | int | float | bool) or v is None
                                                       for v in unique):
        return None
    return Options(unique)


def request_options(fields: dict[str, Any]) -> Options | None:
    """A chat request's options: its ``response_format`` json_schema's (or ``json_schema``'s), with ``logprobs``."""
    if fields.get("logprobs") is not True:
        return None
    rf = fields.get("response_format")
    schema = None
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        inner = rf.get("json_schema") if isinstance(rf.get("json_schema"), dict) else rf
        schema = inner.get("schema")
    elif fields.get("json_schema") is not None:
        schema = fields["json_schema"]
    return options_of(schema)


def _values(row: LogitsLike, ids: list[int], cap: Any) -> np.ndarray:
    """The logits (float64) of tokens ``ids`` in a row (an array, or a screened head's row: exact), under the model's
    soft cap ``cap`` (None: capped already)."""
    r = row.at(np.asarray(ids, np.int32)) if hasattr(row, "at") else np.asarray(row, np.float32)[np.asarray(ids)]
    r = np.asarray(r, np.float32)
    return np.asarray(cap(r) if cap is not None else r, np.float64)


def _after(npu: NpuModel, token: int) -> tuple[LogitsLike, bool]:
    """The next token's logits after running ``token`` (an overlay's: its raw row, a screened head's exact where read;
    another model's: ``step``'s, capped) and whether they are capped."""
    if getattr(npu, "overlay", None) is None:
        return npu.step(token), True
    row = npu.steps([token])[0]
    return (row if hasattr(row, "at") else np.array(row, copy=True)), False


def _pieces(v: Vocab) -> dict[bytes, list[int]]:
    """The tokens of a vocabulary by their bytes (the special ones left out: no option's text holds them)."""
    out = _by_bytes.get(id(v))
    if out is None:
        out = {}
        for t, b in enumerate(v.pieces):
            if b and t not in v.special:
                out.setdefault(b, []).append(t)
        _by_bytes[id(v)] = out
    return out


_by_bytes: dict[int, dict[bytes, list[int]]] = {}
MAX_TOKEN = 64  # a token's bytes at most (the prefixes of an option's text looked up)


def score(npu: NpuModel, logits: LogitsLike, texts: list[bytes], v: Vocab, *, capped: bool = True,
          max_steps: int = 8) -> list[float]:
    """The log-probabilities of answers ``texts`` (their bytes, then the end of the turn) after the device's position,
    whose next token's logits are ``logits`` (``capped``: under the model's soft cap already, ``NpuModel.forward``'s),
    along the model's own tokens: at each point the tokens whose bytes continue some answer's text, their
    probabilities among them (the end of the turn one choice of all its tokens, for the answers complete); a token one
    answer alone continues gives it its whole share, the most likely one several continue runs next (a decode step
    from where the device is, a checkpoint taken first), the others' share dropped; the answers' probabilities
    renormalized. The device ends where it began."""
    pieces, eos = _pieces(v), sorted(v.eos)
    cap = getattr(npu, "cap", None)
    p0, cp = npu.position, None
    out = [-math.inf] * len(texts)
    END = -1

    def cands(rest: dict[int, bytes]) -> dict[int, list[int]]:
        """Token -> the answers it continues (END: the complete ones)."""
        c: dict[int, list[int]] = {}
        for i, r in rest.items():
            if not r:
                c.setdefault(END, []).append(i)
                continue
            for n in range(1, min(len(r), MAX_TOKEN) + 1):
                for t in pieces.get(r[:n], ()):
                    c.setdefault(t, []).append(i)
        return c

    def walk(row: LogitsLike, rest: dict[int, bytes], logp: float, is_capped: bool, steps: int) -> None:
        nonlocal cp
        c = cands(rest)
        toks = [t for t in c if t != END]
        vals = _values(row, toks + eos, None if is_capped else cap)
        score = dict(zip(toks, vals[:len(toks)], strict=True))
        if END in c:
            score[END] = float(np.logaddexp.reduce(vals[len(toks):]))
        norm = float(np.logaddexp.reduce(list(score.values())))
        shared = []
        for t, ids in c.items():
            lp = logp + score[t] - norm
            if len(set(ids)) == 1:  # (one answer's alone: its text's rest follows)
                out[ids[0]] = float(np.logaddexp(out[ids[0]], lp))
            else:
                shared.append((lp, t, ids))
        if not shared:
            return
        lp, t, ids = max(shared)
        if steps == 0:  # (no more steps: the share among those answers as the token's)
            for i in set(ids):
                out[i] = float(np.logaddexp(out[i], lp - math.log(len(set(ids)))))
            return
        if cp is None:  # (the device back at p0 at the end)
            cp = npu.checkpoint()
        nxt, nxt_capped = _after(npu, t)  # (the walk follows one token at a time: it goes on from there)
        tb = v.pieces[t]
        walk(nxt, {i: rest[i][len(tb):] for i in set(ids)}, lp, nxt_capped, steps - 1)

    walk(logits, dict(enumerate(texts)), 0.0, capped, max_steps)
    if cp is not None:
        npu.restore(p0, cp)
    norm = float(np.logaddexp.reduce(out))
    return [x - norm for x in out]


def answer_body(opts: Options, logprobs: list[float], *, head: dict[str, Any], usage: dict[str, Any],
                timings: dict[str, Any]) -> dict[str, Any]:
    """The chat completion of a decision: the most likely option as JSON, one ``top_logprobs`` entry per option."""
    best = int(np.argmax(logprobs))
    text = opts.text(opts.values[best])

    def entry(v: Any, lp: float) -> dict[str, Any]:
        tok = Options.token(v)
        return {"token": tok, "logprob": lp, "bytes": list(tok.encode())}

    order = sorted(range(len(logprobs)), key=lambda i: -logprobs[i])
    top = [entry(opts.values[i], logprobs[i]) for i in order]
    content = {**entry(opts.values[best], logprobs[best]), "top_logprobs": top}
    return {"choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": {"content": [content]}}], **head, "object": "chat.completion", "usage": usage,
            "timings": timings}

