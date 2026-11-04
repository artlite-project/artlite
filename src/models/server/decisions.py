"""llama-server's ``/v1/systemone`` (TypeSafe's System One API; llama.cpp 7182ef71e's ``server-decision.cpp``): a
decision model answers questions about a state, each in one forward pass, no token generated. A GGUF file is one when
its metadata has ``{arch}.decision.type``; its ``tokenizer.chat_template.systemone`` renders a question's prompt,
``{arch}.decision.temperature.<question type>[.<options bucket>]`` and ``{arch}.decision.score_bias.<levels>.<level>``
calibrate the answers.

A request: ``state`` (text, JSON, or chat messages), ``questions`` {id: {type choice / score / noul, instructions,
criteria}}; the answers: a choice's ``choice``, ``probabilities`` and ``confidence``, a score's expected level
``score``, ``legend``, ``probabilities``, ``confidence``, a noul's probability of true ``noul``; ``usage`` counts every
prompt's tokens. A question's prompt shows its options, in two orders for lev's choices (``n_variants``: the two
softmaxes averaged); its scores (``Decisions.scores``) come from the prompt's output rows (``NpuModel.outputs``,
output-normed) on the host:

- openjev, lev, nimble, pplx-decider, vllm-d3: the logits of the options' label tokens at the last token (the label
  tokens' rows of the LM head, dequantized once: ``Decisions.head``): openjev's letters A .. Z a .. z, the others' codes
  A .. Z, AA .. ZZ that are one token (at most 255); lev reads a noul from a rating scale (9 labels: the expected
  rating / 8), nimble's prompt lists every question of the request, vllm-d3's model is non-causal (each prompt whole,
  from position 0);
- kev: ``cls.output`` (and its bias) of the rows at each option's end token (``<|box_end|>``) and at the last token;
  an option's score is the dot product of the last token's first half with its second half over sqrt(half);
- vllm-decision2 (a qwen3 trunk): a joint head (``decision.*`` tensors, fp32: ``JointHead``) over the rows at the end of
  each option's span and of the question's; the prompt is pieces the template separates (``CLEF_SEP``), tokenized one
  by one as the model was trained, a piece that starts with ``CLEF_QUESTION`` / ``CLEF_OPTION`` a span.

Other types (laya, clef, lfm2-d1, lfm2-d1-omni: other architectures or heads) and images (``files``, ``images``, a
chat message's image parts: d3's and pplx's encoders are Qwen3-VL's) are refused (501). A causal model's questions
share the prompt's prefix through the prompt cache (``Engine.readouts``: the state once)."""

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from artlite.gguf_model import GGUFFile

from ..common import render_template
from .engine import Engine

TYPES = ("openjev", "lev", "nimble", "pplx-decider", "vllm-d3", "kev", "vllm-decision2")  # (the types Artlite answers)
LABEL_TYPES = ("openjev", "lev", "nimble", "pplx-decider", "vllm-d3")  # (their scores: label logits)
# (the questions' prompts share a prefix the model continues)
SHARED = ("openjev", "lev", "kev", "nimble", "pplx-decider", "vllm-decision2")
CLEF_MARKER, CLEF_SEP = "<<clef:", "<<clef:sep>>"  # (a joint prompt's pieces: ``Decisions.joint_prompt``)
CLEF_QUESTION, CLEF_OPTION = "<<clef:question>>", "<<clef:option>>"
LEV_RATINGS = 9  # lev's noul rating scale: 0 = certainly no .. 8 = certainly yes
MAX_IMAGES = 8
KINDS = ("choice", "score", "noul")

ApiErrors = tuple[Callable[[str], Exception], Callable[[str], Exception]]  # (400, 501)


class BadRequest(ValueError):
    """A request llama-server answers with 400 (its ``std::invalid_argument``)."""


def decision_type(meta: dict[str, Any], arch: str) -> str | None:
    """A file's ``{arch}.decision.type``, or None: not a decision model."""
    t = meta.get(f"{arch}.decision.type")
    return None if t is None else str(t)


def escape_special(text: str) -> str:
    """``<|name|>`` written with U+00A6 for "|" (``server_decision_escape_special_tokens``: a text's special-token
    lookalikes are not parsed as them)."""
    out, done, i = [], 0, text.find("<|")
    while i >= 0:
        j = i + 2
        while j < len(text) and (text[j].isascii() and (text[j].isalnum() or text[j] == "_")):
            j += 1
        if j > i + 2 and text.startswith("|>", j):
            out += [text[done:i], "<¦", text[i + 2:j], "¦>"]
            done = j + 2
        i = text.find("<|", max(i + 1, done))
    return text if not done else "".join(out) + text[done:]


def _sort_keys(v: Any) -> Any:
    """Every object's keys sorted (lev was trained so; the legends' canonical text)."""
    if isinstance(v, list):
        return [_sort_keys(x) for x in v]
    if isinstance(v, dict):
        return {k: _sort_keys(v[k]) for k in sorted(v)}
    return v


def _kev_render(v: Any, indent: int = 0) -> str:
    """kev's text of a JSON value (kev/api.py's ``render``): an object's keys as labels, a list as "- " items."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, list):
        return "\n".join(pad + "- " + _kev_render(x, indent + 1).lstrip(" \t\n\r") for x in v)
    if isinstance(v, dict):
        out = []
        for k, x in v.items():
            nested = isinstance(x, dict | list)
            out.append(pad + k + (":\n" if nested else ": ") + _kev_render(x, indent + 1 if nested else 0))
        return "\n".join(out)
    return json.dumps(v)  # (a number as nlohmann's dump writes it)


def _kev_text(v: Any) -> str:
    return escape_special(_kev_render(v))


def _replace_text(v: Any, old: str, new: str) -> Any:
    """Every string value (not the keys) with ``old`` replaced (``decision_replace_text``)."""
    if isinstance(v, str):
        return v.replace(old, new)
    if isinstance(v, list):
        return [_replace_text(x, old, new) for x in v]
    if isinstance(v, dict):
        return {k: _replace_text(x, old, new) for k, x in v.items()}
    return v


def _payload(v: Any) -> bool:
    """Text, an object or an array (``decision_is_payload``: what vllm-decision2 takes)."""
    return isinstance(v, str | dict | list)


@dataclass(frozen=True)
class JointHead:
    """vllm-decision2's head (llama.cpp's ``llama_model_vllm_decision2::graph::build_head``, fp32 tensors): the option
    rows o and the question's row q, each LayerNorm'd (``eps``); an option's score is (K o) . (Q q) / sqrt(h) + s .
    gelu(Mo o + bo + Mq q), h the head's width."""

    t: dict[str, np.ndarray]
    eps: float

    @classmethod
    def of(cls, g: GGUFFile, eps: float) -> "JointHead":
        names = ("option_norm.weight", "option_norm.bias", "query_norm.weight", "query_norm.bias", "key.weight",
                 "query.weight", "option_mlp.weight", "option_mlp.bias", "query_mlp.weight", "scorer_out.weight")
        t = {n: np.asarray(g.tensor("decision." + n).dequantize(), np.float64).reshape(g.tensor("decision." + n).rows, -1)
             for n in names}
        return cls({n: v[0] if v.shape[0] == 1 else v for n, v in t.items()}, eps)

    def _norm(self, x: np.ndarray, which: str) -> np.ndarray:
        x = x - x.mean(-1, keepdims=True)
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps)
        return x * self.t[f"{which}_norm.weight"] + self.t[f"{which}_norm.bias"]

    def scores(self, options: np.ndarray, query: np.ndarray) -> np.ndarray:
        t = self.t
        o, q = self._norm(options, "option"), self._norm(query, "query")
        k, qq = o @ t["key.weight"].T, t["query.weight"] @ q
        bilinear = k @ qq / math.sqrt(k.shape[1])
        m = o @ t["option_mlp.weight"].T + t["option_mlp.bias"] + t["query_mlp.weight"] @ q
        gelu = 0.5 * m * (1 + np.vectorize(math.erf)(m / math.sqrt(2)))
        return bilinear + gelu @ t["scorer_out.weight"]


@dataclass(frozen=True)
class Question:
    id: str
    kind: str  # choice, score, noul
    instructions: Any
    options: list[tuple[str, Any]]  # (key, description: None if not given), in the order of the model's outputs


@dataclass
class Prompt:
    """One variant of a question: its tokens and the rows its scores read (indices into them)."""

    question: int
    variant: int
    ids: list[int]
    rows: np.ndarray


@dataclass
class Decisions:
    """A decision model's prompts and answers (``of``: from its file and tokenizer)."""

    type: str
    template: str
    temperatures: dict[str, float] = field(default_factory=dict)  # "<kind>" or "<kind>.<options bucket>"
    score_bias: dict[int, np.ndarray] = field(default_factory=dict)  # added to a score's logits, by its levels
    labels: list[int] = field(default_factory=list)  # (label types) the label tokens
    label_texts: list[str] = field(default_factory=list)  # (codes given to the template as an option's label)
    head: np.ndarray | None = None  # label types: the labels' LM rows (labels, hidden); kev: cls.output (out, hidden)
    bias: np.ndarray | None = None  # kev: cls.output's bias
    joint: JointHead | None = None  # vllm-decision2
    marker: int = -1  # kev: the token that ends an option
    n_options_max: int = 0
    noul_true_first: bool = False
    noul_criteria_order: bool = False  # (a noul's options in its criteria's order when they name both)
    strict_payload: bool = False  # (states, instructions and descriptions: text, objects or arrays)
    canonical_legend: bool = False

    @classmethod
    def of(cls, path: str, encode: Callable[[str], list[int]]) -> "Decisions | None":
        """The decision context of a model file (None: not a decision model); ``encode``: the model's tokenizer
        (special tokens parsed, no BOS). Raises ValueError for a type Artlite does not answer."""
        g = GGUFFile(path)
        meta, arch = g.meta, g.arch
        t = decision_type(meta, arch)
        if t is None:
            return None
        if t not in TYPES:
            raise ValueError(f"decision model type {t!r} is not supported by artlite-server (it answers "
                             f"{', '.join(TYPES)})")
        template = meta.get("tokenizer.chat_template.systemone")
        if template is None:
            raise ValueError("decision model has no \"systemone\" template")
        pre = f"{arch}.decision."
        d = cls(type=t, template=str(template))
        for k, v in meta.items():
            if k.startswith(pre + "temperature."):
                if float(v) <= 0:
                    raise ValueError(f"invalid decision temperature: {k} = {v}")
                d.temperatures[k[len(pre + "temperature."):]] = float(v)
            elif k.startswith(pre + "score_bias."):
                parts = k[len(pre + "score_bias."):].split(".")
                n, level = (int(x) if x.isdigit() else -1 for x in parts) if len(parts) == 2 else (0, 0)
                if not (2 <= n <= 255 and 0 <= level < n):
                    raise ValueError(f"invalid decision score bias: {k}")
                d.score_bias.setdefault(n, np.zeros(n))[level] = float(v)
        d.with_labels(encode)
        if t in LABEL_TYPES:
            lm = "output.weight" if "output.weight" in g else "token_embd.weight"
            d.head = np.asarray(g.rows(lm, d.labels), np.float64)
        elif t == "vllm-decision2":
            d.joint = JointHead.of(g, float(meta.get(f"{arch}.attention.layer_norm_epsilon", 1e-5)))
        else:
            w = g.tensor("cls.output.weight")
            d.head = np.asarray(g.rows("cls.output.weight", np.arange(w.rows)), np.float64)
            d.bias = np.asarray(g.rows("cls.output.bias", [0]), np.float64).reshape(-1) \
                if "cls.output.bias" in g else np.zeros(w.rows)
        return d

    def with_labels(self, encode: Callable[[str], list[int]]) -> "Decisions":
        """The type's label tokens (or kev's marker) and option limits from the model's tokenizer."""
        if self.type == "openjev":  # (one letter per option, each one token)
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                ids = encode(c)
                if len(ids) != 1:
                    raise ValueError(f"decision label {c!r} is not a single token")
                self.labels.append(ids[0])
            self.n_options_max, self.noul_true_first = len(self.labels), True
        elif self.type in LABEL_TYPES:  # (codes A .. Z, AA .. ZZ: those of one token)
            A = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            for code in A + [a + b for a in A for b in A]:
                ids = encode(code)
                if len(ids) == 1 and len(self.labels) < 255:
                    self.labels.append(ids[0])
                    self.label_texts.append(code)
            self.n_options_max, self.canonical_legend = len(self.labels), self.type == "vllm-d3"
        elif self.type == "vllm-decision2":
            self.n_options_max, self.noul_criteria_order, self.strict_payload, self.canonical_legend = 255, True, True, True
        else:  # kev
            ids = encode("<|box_end|>")
            if len(ids) != 1:
                raise ValueError("decision model has no <|box_end|> token")
            self.marker, self.n_options_max = ids[0], 255
        return self

    @property
    def shares_prompt(self) -> bool:
        """The questions of a request start with the same tokens, which the model continues from (the state once)."""
        return self.type in SHARED

    # ------------------------------------------------------------------
    # The request
    # ------------------------------------------------------------------

    def questions(self, body: dict[str, Any]) -> list[Question]:
        """A request's questions (BadRequest: a bad one)."""
        if "state" not in body or body["state"] is None and self.type != "vllm-d3":
            raise BadRequest("\"state\" must be provided")
        if self.strict_payload and not _payload(body["state"]):
            raise BadRequest("\"state\" must be text, an object or an array")
        qs = body.get("questions")
        if not isinstance(qs, dict) or not qs:
            raise BadRequest("\"questions\" must be a non-empty object")
        out = []
        for qid, q in qs.items():
            def err(msg: str, qid: str = qid) -> BadRequest:
                return BadRequest(f"questions.{qid}: {msg}")

            if not isinstance(q, dict):
                raise err("must be an object")
            instructions = q.get("instructions")  # (vllm-d3's prompt has a default question)
            if instructions is None and self.type != "vllm-d3":
                raise err("\"instructions\" must be provided")
            if self.strict_payload and (not _payload(instructions) or instructions == ""):
                raise err("\"instructions\" must be non-empty text, an object or an array")
            kind = q.get("type") if isinstance(q.get("type"), str) else ""
            crit = q.get("criteria")
            if kind == "choice":
                if not isinstance(crit, dict) or not crit:
                    raise err("\"criteria\" must be a non-empty object")
                options = list(crit.items())
                if self.strict_payload and any(not k or not (d is None or _payload(d)) for k, d in options):
                    raise err("option keys must be non-empty, descriptions must be null, text, an object or an array")
                if self.strict_payload and len(options) < 2:
                    raise err("\"criteria\" must have at least 2 options")
            elif kind == "score":
                if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                    raise err("\"criteria\" must be an array of 2 to 10 levels")
                if self.strict_payload and not all(_payload(c) for c in crit):
                    raise err("level descriptions must be text, an object or an array")
                options = [(str(i), c) for i, c in enumerate(crit)]
            elif kind == "noul":
                if crit is not None and not isinstance(crit, dict):
                    raise err("\"criteria\" must be an object")
                if self.strict_payload and crit and any(k not in ("false", "true") or not _payload(d)
                                                        for k, d in crit.items()):
                    raise err("\"criteria\" may only describe \"false\" and \"true\" with text, an object or an array")
                options = [(k, crit.get(k) if crit else None) for k in ("false", "true")]
                if self.noul_true_first or self.noul_criteria_order and crit and len(crit) == 2 and \
                        next(iter(crit)) == "true":
                    options.reverse()
            else:
                raise err("\"type\" must be one of: choice, score, noul")
            if len(options) > self.n_options_max:
                raise err(f"too many options ({len(options)}), this model supports at most {self.n_options_max}")
            out.append(Question(qid, kind, instructions, options))
        return out

    @staticmethod
    def state(body: dict[str, Any]) -> tuple[Any, int]:
        """A request's state without its images, and how many it has (``files`` / ``images``, a chat message's
        ``image_url`` and ``input_audio`` parts; BadRequest: a bad one)."""
        n = 0

        def image(url: object) -> None:
            nonlocal n
            if not isinstance(url, str) or not url.startswith("data:"):
                raise BadRequest("images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)")
            if n >= MAX_IMAGES:
                raise BadRequest(f"too many images, the maximum is {MAX_IMAGES}")
            n += 1

        def audio(data: object) -> None:
            nonlocal n
            if not isinstance(data, str) or data.startswith(("http", "file://")):
                raise BadRequest("audio must be base64 data")
            if n >= MAX_IMAGES:
                raise BadRequest(f"too many media files, the maximum is {MAX_IMAGES}")
            n += 1

        if body.get("videos"):
            raise BadRequest("\"videos\" is not supported")
        for key in ("files", "images"):
            if body.get(key) is None:
                continue
            if not isinstance(body[key], list):
                raise BadRequest(f"\"{key}\" must be an array")
            for url in body[key]:
                image(url)
        state = body["state"]
        wrapped = isinstance(state, dict) and "messages" in state
        messages = state["messages"] if wrapped else state
        if not isinstance(messages, list):
            return state, n
        kept = []
        for msg in messages:
            if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
                kept.append(msg)
                continue
            content = []
            for part in msg["content"]:
                kind = part.get("type") if isinstance(part, dict) and isinstance(part.get("type"), str) else ""
                if kind == "image_url" and "image_url" in part:
                    u = part["image_url"]
                    image(u["url"] if isinstance(u, dict) and "url" in u else u)
                elif kind == "input_audio" and "input_audio" in part:
                    a = part["input_audio"] if isinstance(part["input_audio"], dict) else {}
                    audio(a["data"] if "data" in a else a.get("url"))
                else:
                    content.append(part)
            kept.append({**msg, "content": content})
        return ({**state, "messages": kept} if wrapped else kept), n

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def n_variants(self, q: Question) -> int:
        """Prompts of a question (lev shows a choice's options in two orders: its preference for the first label
        cancels)."""
        return 2 if self.type == "lev" and q.kind == "choice" and len(q.options) > 1 else 1

    def n_outputs(self, q: Question) -> int:
        return LEV_RATINGS if self.type == "lev" and q.kind == "noul" else len(q.options)

    def _options(self, q: Question, variant: int) -> list[dict[str, Any]]:
        n = len(q.options)
        out = []
        for i in range(n):
            key, desc = q.options[i if variant == 0 else n - 1 - i]
            o = {"key": key, "description": desc}
            if self.type == "kev":
                o = {"key": _kev_text(key), "description": None if desc is None else _kev_text(desc)}
            if self.label_texts:
                o["label"] = self.label_texts[i]
            out.append(o)
        return out

    def render(self, state: Any, questions: list[Question], q: Question, variant: int) -> str:
        """The prompt text of a question's variant (the template takes the raw JSON values: it serializes those that
        are not strings)."""
        inp: dict[str, Any] = {"id": q.id, "type": q.kind, "instructions": q.instructions, "state": state,
                               "options": self._options(q, variant)}
        if self.type == "nimble":  # (its prompt lists all the questions)
            inp["questions"] = [{"id": x.id, "type": x.kind, "instructions": x.instructions,
                                 "options": self._options(x, 0)} for x in questions]
        if self.type == "lev":
            inp = _sort_keys(inp)
        if self.type == "kev":  # (text only)
            inp["state"], inp["instructions"] = _kev_text(state), _kev_text(q.instructions)
        inp["images"] = []
        return render_template(self.template, **inp)

    def joint_prompt(self, state: Any, q: Question, encode: Callable[[str], list[int]]) -> tuple[list[int], np.ndarray]:
        """vllm-decision2's prompt of a question (``fill_task_joint``): the template's text (the values with sorted keys,
        none of them a piece marker) split into its pieces, each tokenized alone; the rows: the last token of each
        option's span, then of the question's."""
        inp = {"state": state, "questions": [{"id": q.id, "type": q.kind, "instructions": q.instructions,
                                              "options": [{"key": k, "description": d} for k, d in q.options]}]}
        inp = _replace_text(_sort_keys(inp), CLEF_MARKER, "<<clef ")
        text = render_template(self.template, **inp, images=[], sep=CLEF_SEP, mark_question=CLEF_QUESTION,
                               mark_option=CLEF_OPTION)
        ids: list[int] = []
        options: list[int] = []
        question = -1
        for piece in text.split(CLEF_SEP):
            mark = next((m for m in (CLEF_QUESTION, CLEF_OPTION) if piece.startswith(m)), None)
            toks = encode(piece[len(mark):] if mark else piece)
            if mark and not toks:
                raise BadRequest("the instructions and the options of a question must not be empty")
            ids += toks
            if mark == CLEF_OPTION:
                options.append(len(ids) - 1)
            elif mark == CLEF_QUESTION:
                if question >= 0:
                    raise RuntimeError("unexpected layout of the decision prompt")
                question = len(ids) - 1
        if question < 0 or len(options) != len(q.options):
            raise RuntimeError("unexpected layout of the decision prompt")
        return ids, np.array([*options, question])

    def prompts(self, state: Any, questions: list[Question], encode: Callable[[str], list[int]]) -> list[Prompt]:
        """Every question's variants as prompts (``encode``: the model's tokenizer, special tokens parsed)."""
        out = []
        for i, q in enumerate(questions):
            if self.joint is not None:
                out.append(Prompt(i, 0, *self.joint_prompt(state, q, encode)))
                continue
            for v in range(self.n_variants(q)):
                ids = encode(self.render(state, questions, q, v))
                if self.type == "kev":  # (an option read at its end token, the question at the last)
                    marks = [j for j, t in enumerate(ids) if t == self.marker]
                    if len(marks) != len(q.options):
                        raise RuntimeError("unexpected layout of the decision prompt")
                    rows = np.array(marks + [len(ids) - 1])
                else:
                    rows = np.array([len(ids) - 1])
                out.append(Prompt(i, v, ids, rows))
        return out

    # ------------------------------------------------------------------
    # Answers
    # ------------------------------------------------------------------

    def scores(self, q: Question, rows: np.ndarray) -> np.ndarray:
        """A variant's raw outputs from its prompt's rows (``Prompt.rows``, output-normed)."""
        x = np.asarray(rows, np.float64)
        if self.joint is not None:
            return self.joint.scores(x[:-1], x[-1])
        if self.type == "kev":
            e = x @ self.head.T + self.bias  # (rows, out)
            half = e.shape[1] // 2
            return e[:-1, half:] @ e[-1, :half] / math.sqrt(half)
        return self.head[:self.n_outputs(q)] @ x[-1]

    def temperature(self, q: Question) -> float:
        """A question's temperature: by kind and number of options (the buckets its fit used), else by kind, else 1."""
        n = len(q.options)
        if self.type == "lev":
            bucket = "small" if n <= 8 else "mid" if n <= 26 else "large"
        else:
            bucket = "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"
        return self.temperatures.get(f"{q.kind}.{bucket}", self.temperatures.get(q.kind, 1.0))

    def confidence(self, kind: str, p: np.ndarray) -> float:
        """TypeSafe's confidence formulas (vllm-d3, vllm-decision2: the normalized entropy)."""
        n = len(p)
        if n < 2:
            return 1.0
        if self.type in ("vllm-d3", "vllm-decision2"):
            h = -sum(float(x) * math.log(x) for x in p if x > 0)
            return max(0.0, min(1.0, 1.0 - h / math.log(n)))
        if kind == "choice":
            return max(0.0, (float(p.max()) - 1 / n) / (1 - 1 / n))
        mode = int(np.argmax(p))
        dist = sum(float(p[i]) * abs(i - mode) for i in range(n))
        uniform = sum(abs(i - (n - 1) / 2) / n for i in range(n))
        return max(0.0, 1.0 - dist / uniform)

    def answer(self, q: Question, scores: list[np.ndarray]) -> dict[str, Any]:
        """A question's answer from its variants' scores (each softmaxed at the question's temperature, the second
        variant's in reverse order, then averaged)."""
        n = self.n_outputs(q)
        if len(scores) != self.n_variants(q) or any(len(s) != n for s in scores):
            raise RuntimeError("decision result does not match the number of options")
        T = self.temperature(q)
        probs = np.zeros(n)
        for v, s in enumerate(scores):
            z = np.asarray(s, np.float64)
            if q.kind == "score" and n in self.score_bias:
                z = z + self.score_bias[n]
            p = np.exp((z - z.max()) / T)
            p /= p.sum()
            probs += (p if v == 0 else p[::-1]) / len(scores)
        out: dict[str, Any] = {"type": q.kind}
        if q.kind == "noul":
            if self.type == "lev":
                out["noul"] = float(sum(probs[i] * i / (n - 1) for i in range(n)))
            else:
                out["noul"] = float(probs[[k for k, _ in q.options].index("true")])
            return out
        pr = {k: float(p) for (k, _), p in zip(q.options, probs, strict=True)}
        if q.kind == "choice":
            out |= {"choice": q.options[int(np.argmax(probs))][0], "probabilities": pr,
                    "confidence": self.confidence(q.kind, probs)}
        else:
            legend = {k: json.dumps(_sort_keys(d), ensure_ascii=False, separators=(",", ":"))
                      if self.canonical_legend and not isinstance(d, str) else d for k, d in q.options}
            out |= {"score": float(sum(i * p for i, p in enumerate(probs))), "legend": legend, "probabilities": pr,
                    "confidence": self.confidence(q.kind, probs)}
        return out


def add_routes(api: APIRouter, *, engine: Engine, alias: str, kind: str | None, make: Callable[[], Decisions],
               errors: ApiErrors, fits: Callable[[list[int]], None]) -> None:
    """``/v1/systemone`` on ``api`` (``kind``: the model file's decision type, None: not a decision model; ``make``: its
    ``Decisions``, built on the first request); ``errors``: a 400 and a 501 from a message; ``fits``: refuses a prompt
    the KV cache does not hold."""
    bad, unsupported = errors
    ctx: list[Decisions] = []
    lock = asyncio.Lock()

    async def decisions() -> Decisions:
        async with lock:
            if not ctx:
                if kind not in TYPES:
                    raise unsupported(f"decision model type {kind!r} is not supported by artlite-server (it answers "
                                      f"{', '.join(TYPES)})")
                ctx.append(await run_in_threadpool(make))
        return ctx[0]

    @api.post("/v1/systemone")
    async def systemone(body: dict[str, Any]) -> Any:
        """Answers to questions about a state (module docstring)."""
        if kind is None:
            raise unsupported("This model is not a decision model")
        d = await decisions()
        try:
            questions = d.questions(body)
            state, images = d.state(body)
        except BadRequest as e:
            raise bad(str(e)) from None
        if images:
            raise unsupported("This server does not support image input for decisions")
        prompts = await run_in_threadpool(d.prompts, state, questions, engine.model.tokenizer.encode)
        npu = engine.npu
        whole = npu.prefill.sizes[-1] if not npu.causal else 0  # (a non-causal model reads a prompt in one chunk)
        for p in prompts:
            fits(p.ids)
            if whole and len(p.ids) > whole:
                raise bad(f"the prompt has {len(p.ids)} tokens, this model reads at most {whole} in one pass")
        out = await asyncio.wrap_future(engine.readouts([p.ids for p in prompts], [p.rows for p in prompts],
                                                        cache_prompt=d.shares_prompt, name="decisions"))
        scores: list[list[np.ndarray]] = [[] for _ in questions]
        for p, rows in zip(prompts, out, strict=True):
            scores[p.question].append(d.scores(questions[p.question], rows))
        n = sum(len(p.ids) for p in prompts)
        return {"model": body.get("model") or alias,
                "answers": {q.id: d.answer(q, s) for q, s in zip(questions, scores, strict=True)},
                "usage": {"input_tokens": n, "output_tokens": 0}}


__all__ = ["TYPES", "BadRequest", "Decisions", "JointHead", "Prompt", "Question", "add_routes", "decision_type",
           "escape_special"]
