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
  by one as the model was trained, a piece that starts with ``CLEF_QUESTION`` / ``CLEF_OPTION`` a span;
- lfm3-d1 (LiquidAI's d1 on an LFM3 trunk, non-causal on its full-attention layers): per option a group of single
  tokens (its code, the code after a space; yes / Yes / YES, no / ...; a level's digit: ``d1_labels``), its score the
  group's largest logit at the question's last token; a choice's confidence is its probability, a score the median
  level; the texts of the control tokens are broken in the input (``d1_break``). Questions about a state share one
  sequence as the model's runner reads them (its ``set`` attention, up to ``D1_SET_SIZE`` a sequence; the request's
  ``set_size``, 1: each question alone as llama-server reads them): the state, then each question after "QUESTION:\n"
  (``Decisions.prompts``); on the full-attention layers the state's tokens see every token, a question's the state
  and its own (``Prompt.sets``, ``NpuModel.outputs``' ``sets``).

Images (``files``, ``images``: data URLs; a chat state's ``image_url`` parts; at most ``MAX_IMAGES``) for lfm3-d1 and
vllm-d3 (``IMAGE_TYPES``) on a server with the model's vision encoder: each runs the encoder once
(``Model.image_tokens``: its tokens between the encoder's markup), the template gets a ``MEDIA_MARKER`` per image (at
the head of the user turn in both: the state's first tokens) and the prompts are sequences of ids and embedding rows
(``sequence.splice``, as llama.cpp's mtmd); a file's ``decision.temperature.image`` is the temperature of a request with
images. lfm3-d1 also takes videos (their frames, d1's runner's: ``Lfm.media_tokens``) and lfm3-d1-omni audio clips and
spoken questions (a question's instructions an ``input_audio`` part: its clip where its text would be, after the
state's media: ``spoken``); a file's kind is its data URL's or part's type, else its content's (``Media``). Other types
(laya, clef, lfm2-d1, lfm2-d1-omni: other architectures or heads), images for the others, audio and video for
non-d1 types are refused (501). A causal model's questions share the prompt's prefix through the prompt
cache (``Engine.readouts``: the state once)."""

import base64
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from artlite.gguf_model import GGUFFile

from ..model import MEDIA_MARKER
from ..sequence import Sequence, data_bytes, splice
from ..templating import render_template

TYPES = ("openjev", "lev", "nimble", "pplx-decider", "vllm-d3", "kev", "vllm-decision2", "lfm3-d1",
         "lfm3-d1-omni")  # (Artlite's)
D1 = ("lfm3-d1", "lfm3-d1-omni")  # (d1 on an LFM3 trunk; -omni: with LFM3-Omni's speech encoder, audio and spoken questions)
LABEL_TYPES = ("openjev", "lev", "nimble", "pplx-decider", "vllm-d3")  # (their scores: label logits)
NULL_STATE = ("vllm-d3", *D1)  # (a null state: the question alone)
IMAGE_TYPES = ("vllm-d3", *D1)  # (images in the state; llama-server's openjev and pplx-decider too: no files here)
# (the questions' prompts share a prefix the model continues)
SHARED = ("openjev", "lev", "kev", "nimble", "pplx-decider", "vllm-decision2")
CLEF_MARKER, CLEF_SEP = "<<clef:", "<<clef:sep>>"  # (a joint prompt's pieces: ``Decisions.joint_prompt``)
CLEF_QUESTION, CLEF_OPTION = "<<clef:question>>", "<<clef:option>>"
LEV_RATINGS = 9  # lev's noul rating scale: 0 = certainly no .. 8 = certainly yes
MAX_IMAGES = 8
D1_SET_SIZE = 16  # lfm3-d1: the questions a sequence holds after their state (d1-2B's ``d1_set_size``, its runner's default)
D1_HEADER = "QUESTION:\n"  # lfm3-d1: what opens each question after the first in a sequence (the runner's ``question_header``)
D1_POOL = ([chr(c) for c in range(65, 91)] + [f"{i:02d}" for i in range(100)] + [chr(c) for c in range(97, 123)]
           + [f"#{i}" for i in range(200)] + [a + b for a in map(chr, range(65, 91)) for b in map(chr, range(65, 91))])
ZWSP = "\u200b"
KINDS = ("choice", "score", "noul")


class BadRequest(ValueError):
    """A request llama-server answers with 400 (its ``std::invalid_argument``)."""


class Unsupported(ValueError):
    """A request Artlite does not answer (501)."""


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


def d1_break(v: Any, spellings: re.Pattern | None) -> Any:
    """lfm3-d1: every string of a JSON value, keys too, with a zero-width space after the first character of each
    control token's text in it (``server_decision_context::d1_break``, the runner's ``plain``: it reads the same and is
    no longer the token), again where a break made a new one; ``spellings``: the texts, longest first."""
    if spellings is None:
        return v
    if isinstance(v, str):
        n = 1
        while n:
            v, n = spellings.subn(lambda m: m[0][0] + ZWSP + m[0][1:], v)
        return v
    if isinstance(v, list):
        return [d1_break(x, spellings) for x in v]
    if isinstance(v, dict):
        return {d1_break(k, spellings): d1_break(x, spellings) for k, x in v.items()}
    return v


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


def spoken(instructions: Any) -> bool:
    """A spoken question's instructions (lfm3-d1-omni, llama-server's): an ``input_audio`` part."""
    return isinstance(instructions, dict) and instructions.get("type") == "input_audio"


class Media(bytes):
    """A request's media file: its bytes and ``kind`` ("image", "audio", "video": its data URL's or part's type, else
    its content's: ``media_kind``)."""

    kind: str

    def __new__(cls, data: bytes, kind: str | None = None) -> Self:
        m = super().__new__(cls, data)
        m.kind = kind or media_kind(data)
        return m


def url_kind(url: str) -> str | None:
    """The media kind a data URL's type names (``data:image/...``, ``data:audio/...``, ``data:video/...``), or None."""
    t = url[5:url.find("/")] if url.startswith("data:") and "/" in url[:30] else ""
    return t if t in ("image", "audio", "video") else None


def media_kind(data: bytes) -> str:
    """A media file's kind by its content, as llama.cpp's mtmd helper tells them: "audio" (WAV, MP3, FLAC, Ogg), "image"
    (what Pillow opens), else "video"."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE" or data[:3] == b"ID3" or data[:4] in (b"fLaC", b"OggS") or \
            len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        return "audio"
    import io

    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)):
            return "image"
    except (OSError, ValueError):
        return "video"


def _d1_key(q: "Question") -> tuple:
    return q.kind, tuple(k for k, _ in q.options)


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
    """One variant of a question: its tokens (with images: a sequence of ids and embedding rows) and the rows its
    scores read (indices into them). lfm3-d1: a sequence of the questions ``questions`` after their state (each read at
    one row, in their order), ``sets`` its non-causal attention's bounds (the state's end, then each question's:
    ``NpuModel.outputs``)."""

    question: int
    variant: int
    ids: list[int] | Sequence
    rows: np.ndarray
    questions: list[int] = field(default_factory=list)  # (lfm3-d1; else [question])
    sets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.questions:
            self.questions = [self.question]


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
    spellings: re.Pattern | None = None  # lfm3-d1: the control tokens' texts the input's are broken at (``d1_break``)
    lm_rows: Callable[[list[int]], np.ndarray] | None = None  # lfm3-d1: LM rows of tokens (fp64, (n, hidden))
    _rows: dict[int, np.ndarray] = field(default_factory=dict)  # (lfm3-d1: the LM rows read so far)
    _d1: dict[tuple, tuple[list[str], list[list[int]]]] = field(default_factory=dict)  # (lfm3-d1: ``d1_labels``)

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
        lm = "output.weight" if "output.weight" in g else "token_embd.weight"
        if t in LABEL_TYPES:
            d.head = np.asarray(g.rows(lm, d.labels), np.float64)
        elif t in D1:
            types = np.asarray(meta["tokenizer.ggml.token_type"])
            texts = g.strings("tokenizer.ggml.tokens")
            ctrl = sorted((texts[i] for i in np.flatnonzero(types == 3) if len(texts[i].encode()) >= 2), key=len,
                          reverse=True)  # (stable: llama.cpp's order among equal lengths)
            d.spellings = re.compile("|".join(map(re.escape, ctrl))) if ctrl else None
            d.lm_rows = lambda ids: np.asarray(g.rows(lm, ids), np.float64)
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
        elif self.type in D1:  # (labels per question: ``d1_labels``)
            self.n_options_max, self.noul_true_first = 255, True
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
        if "state" not in body or body["state"] is None and self.type not in NULL_STATE:
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
            if spoken(instructions) and self.type != "lfm3-d1-omni":
                raise err("spoken instructions need a model with audio input")
            if spoken(instructions) and not isinstance((instructions.get("input_audio") or {}).get("data"), str):
                raise err("spoken instructions: an input_audio part's \"data\" (base64 audio)")
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

    def state(self, body: dict[str, Any]) -> tuple[Any, list[bytes]]:
        """A request's state without its media, and its media (``Media``: bytes and kind), in order (``files`` /
        ``images``, then a chat message's ``image_url``, ``input_audio`` and (lfm3-d1) ``video_url`` parts;
        BadRequest: a bad one or too many)."""
        media: list[bytes] = []

        def take(data: bytes, kind: str | None) -> None:
            if len(media) >= MAX_IMAGES:
                raise BadRequest(f"too many media files, the maximum is {MAX_IMAGES}")
            media.append(Media(data, kind))

        def image(url: object, kind: str | None = None) -> None:
            if not isinstance(url, str) or not url.startswith("data:"):
                raise BadRequest("media must be data URLs (data:image/..., data:audio/... or data:video/...;base64,...)")
            try:
                data = data_bytes(url)
            except ValueError as e:
                raise BadRequest(str(e)) from None
            take(data, url_kind(url) or kind)

        def audio(data: object) -> None:
            if not isinstance(data, str) or data.startswith(("http", "file://")):
                raise BadRequest("audio must be base64 data")
            try:
                clip = data_bytes(data) if data.startswith("data:") else base64.b64decode(data, validate=True)
            except ValueError as e:
                raise BadRequest(f"audio: {e}") from None
            take(clip, "audio")

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
            return state, media
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
                elif kind == "video_url" and "video_url" in part and self.type in D1:  # (llama-server's media)
                    u = part["video_url"]
                    image(u["url"] if isinstance(u, dict) and "url" in u else u, "video")
                else:
                    content.append(part)
            kept.append({**msg, "content": content})
        return ({**state, "messages": kept} if wrapped else kept), media

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
            if self.type in D1:
                o["label"] = self._d1[_d1_key(q)][0][i]
            out.append(o)
        return out

    def d1_labels(self, q: Question, encode: Callable[[str], list[int]]) -> tuple[list[str], list[list[int]]]:
        """lfm3-d1: each option's code (the template's label) and its group of single tokens, whose largest logit is
        the option's score (llama-server's ``d1_labels``, the runner's ``readout_ids``): a level's digit, a noul's yes /
        Yes / YES (no / ...), a choice's code and the code after a space; the codes are the options' keys when each is
        a letter, else A .. Z, then AA, AB ..; a code of several tokens or one taken: the first free of ``D1_POOL``.
        Kept for the question's answer (``scores``)."""
        key = _d1_key(q)
        if key in self._d1:
            return self._d1[key]

        def singles(forms: list[str]) -> list[int]:
            out: list[int] = []
            for f in forms:
                ids = encode(f)
                if len(ids) == 1 and ids[0] not in out:
                    out.append(ids[0])
            return out

        texts: list[str] = []
        groups: list[list[int]] = []
        if q.kind != "choice":
            for k, _ in q.options:
                g = singles([k] if q.kind == "score" else ["yes", "Yes", "YES"] if k == "true" else ["no", "No", "NO"])
                if not g:
                    raise RuntimeError(f"decision label is not a single token: {k}")
                texts.append(k)
                groups.append(g)
        else:
            keys = [k for k, _ in q.options]
            n = len(keys)
            if all(len(k.encode()) == 1 and k.isalpha() for k in keys):
                codes = keys
            else:
                codes = [chr(65 + i) if i < 26 else chr(65 + (i - 26) // 26) + chr(65 + (i - 26) % 26) for i in range(n)]
            used: set[int] = set()

            def take(code: str) -> bool:
                ids = encode(code)
                if len(ids) != 1 or ids[0] in used:
                    return False
                used.add(ids[0])
                texts.append(code)
                groups.append([ids[0]] + [t for t in singles([" " + code]) if t != ids[0]])
                return True

            for code in codes:
                if not take(code) and not any(take(c) for c in D1_POOL):
                    raise BadRequest(f"no single-token label left for {n} options")
        self._d1[key] = (texts, groups)
        return texts, groups

    def render(self, state: Any, questions: list[Question], q: Question, variant: int, n_images: int = 0) -> str:
        """The prompt text of a question's variant (the template takes the raw JSON values: it serializes those that
        are not strings), with ``n_images`` images: a ``MEDIA_MARKER`` each where the template puts them (none in the
        input's values: llama-server's)."""
        inp: dict[str, Any] = {"id": q.id, "type": q.kind, "instructions": q.instructions, "state": state,
                               "options": self._options(q, variant)}
        if self.type == "nimble":  # (its prompt lists all the questions)
            inp["questions"] = [{"id": x.id, "type": x.kind, "instructions": x.instructions,
                                 "options": self._options(x, 0)} for x in questions]
        if self.type == "lev":
            inp = _sort_keys(inp)
        if self.type == "kev":  # (text only)
            inp["state"], inp["instructions"] = _kev_text(state), _kev_text(q.instructions)
        if self.type in D1:
            inp = d1_break(inp, self.spellings)
        if n_images or spoken(q.instructions):
            inp = _replace_text(inp, MEDIA_MARKER, " ")
        inp["images"] = [MEDIA_MARKER] * n_images
        if spoken(q.instructions):  # (heard where its text would be, after the state's media: llama-server's)
            inp["instructions"] = MEDIA_MARKER
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

    def prompts(self, state: Any, questions: list[Question], encode: Callable[[str], list[int]], *,
                media: list[Sequence] = (), set_size: int = D1_SET_SIZE, max_tokens: int = 0,
                spoken_media: dict[int, Sequence] | None = None) -> list[Prompt]:
        """Every question's variants as prompts (``encode``: the model's tokenizer, special tokens parsed; ``media``:
        the request's media as the model reads them, ``Model.image_tokens`` / ``media_tokens``: the prompts are then
        sequences; ``spoken_media``: a spoken question's clip by its index, lfm3-d1-omni); lfm3-d1:
        sequences of up to ``set_size`` questions after their state (``_d1_prompts``), of at most ``max_tokens`` (0:
        any) where a question alone fits."""
        if self.type in D1:
            return self._d1_prompts(state, questions, encode, media=list(media), set_size=set_size,
                                    max_tokens=max_tokens, spoken=spoken_media or {})
        out = []
        for i, q in enumerate(questions):
            if self.joint is not None:
                out.append(Prompt(i, 0, *self.joint_prompt(state, q, encode)))
                continue
            for v in range(self.n_variants(q)):
                ids = splice(self.render(state, questions, q, v, len(media)), MEDIA_MARKER, list(media), encode)
                if self.type == "kev":  # (an option read at its end token, the question at the last)
                    marks = [j for j, t in enumerate(ids) if t == self.marker]
                    if len(marks) != len(q.options):
                        raise RuntimeError("unexpected layout of the decision prompt")
                    rows = np.array(marks + [len(ids) - 1])
                else:
                    rows = np.array([len(ids) - 1])
                out.append(Prompt(i, v, ids, rows))
        return out

    def _d1_prompts(self, state: Any, questions: list[Question], encode: Callable[[str], list[int]], *,
                    media: list[Sequence], set_size: int, max_tokens: int,
                    spoken: dict[int, Sequence]) -> list[Prompt]:
        """lfm3-d1's sequences (the runner's ``pack_set``): the prompt of a sequence's first question as the template
        writes it (the images, the state, "QUESTION:\\n", the question up to its answer slot), then each further
        question's part after ``D1_HEADER``, tokenized alone; the state is the tokens before the first question's own
        (its part's tokens alone: the images are the state's, as the runner's media). A question's part is its prompt
        after the template's head (its text with no state); a sequence closes at ``set_size`` questions, or before one
        that would take it past ``max_tokens``. A spoken question's clip (``spoken``) is its part's media marker."""
        sentinel = "\x00d1\x00"
        for q in questions:
            self.d1_labels(q, encode)
        probe = Question("", "noul", sentinel, [("true", None), ("false", None)])
        self.d1_labels(probe, encode)
        n = len(media)
        head = self.render(None, [], probe, 0, n)
        head = head[:head.find(sentinel)]
        parts, prefix = [], None
        for i, q in enumerate(questions):
            whole, alone = self.render(state, questions, q, 0, n), self.render(None, questions, q, 0, n)
            part = alone[len(head):]
            if not alone.startswith(head) or not whole.endswith(part) or prefix not in (None, whole[:-len(part)]) \
                    or part.count(MEDIA_MARKER) != (i in spoken):
                raise RuntimeError("unexpected layout of the decision prompt")
            prefix = whole[:-len(part)]
            parts.append(part)
        header = "" if state is None else D1_HEADER
        if not prefix.endswith(header):
            raise RuntimeError("unexpected layout of the decision prompt")
        heard = lambda i: [spoken[i]] if i in spoken else []
        more = [splice(header + p, MEDIA_MARKER, heard(i), encode) for i, p in enumerate(parts)]
        out: list[Prompt] = []
        i = 0
        while i < len(questions):
            ids = splice(prefix + parts[i], MEDIA_MARKER, media + heard(i), encode)
            S = len(ids) - len(splice(parts[i], MEDIA_MARKER, heard(i), encode))
            ends, qs = [len(ids)], [i]
            i += 1
            while i < len(questions) and len(qs) < set_size and (not max_tokens or len(ids) + len(more[i]) <= max_tokens):
                ids = ids + more[i]
                ends.append(len(ids))
                qs.append(i)
                i += 1
            out.append(Prompt(qs[0], 0, ids, np.array(ends) - 1, questions=qs, sets=(S, *ends)))
        return out

    @staticmethod
    def set_size(body: dict[str, Any]) -> int:
        """lfm3-d1: the request's ``set_size`` (questions a sequence holds after their state; 1: each alone, as
        llama-server reads them), default ``D1_SET_SIZE``."""
        n = body.get("set_size", D1_SET_SIZE)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise BadRequest("\"set_size\" must be a whole number from 1")
        return n

    # ------------------------------------------------------------------
    # Answers
    # ------------------------------------------------------------------

    def scores(self, q: Question, rows: np.ndarray) -> np.ndarray:
        """A variant's raw outputs from its prompt's rows (``Prompt.rows``, output-normed; lfm3-d1: the question's
        row)."""
        x = np.asarray(rows, np.float64)
        if self.type in D1:
            groups = self._d1[_d1_key(q)][1]
            missing = sorted({t for g in groups for t in g} - self._rows.keys())
            if missing:
                self._rows.update(zip(missing, self.lm_rows(missing), strict=True))
            return np.array([max(float(self._rows[t] @ x[-1]) for t in g) for g in groups])
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
        if self.type in D1:  # (the answer's probability: d1's api.py)
            return float(p.max())
        if kind == "choice":
            return max(0.0, (float(p.max()) - 1 / n) / (1 - 1 / n))
        mode = int(np.argmax(p))
        dist = sum(float(p[i]) * abs(i - mode) for i in range(n))
        uniform = sum(abs(i - (n - 1) / 2) / n for i in range(n))
        return max(0.0, 1.0 - dist / uniform)

    def answer(self, q: Question, scores: list[np.ndarray], *, media: bool = False) -> dict[str, Any]:
        """A question's answer from its variants' scores (each softmaxed at the question's temperature, or with
        ``media`` the file's ``image`` one where it has it, the second variant's in reverse order, then averaged)."""
        n = self.n_outputs(q)
        if len(scores) != self.n_variants(q) or any(len(s) != n for s in scores):
            raise RuntimeError("decision result does not match the number of options")
        T = self.temperatures["image"] if media and "image" in self.temperatures else self.temperature(q)
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
            score = float(sum(i * p for i, p in enumerate(probs)))
            if self.type in D1:  # (the median level: the first whose cumulative probability reaches one half)
                score = float(min(int(np.searchsorted(np.cumsum(probs), 0.5)), n - 1))
            out |= {"score": score, "legend": legend, "probabilities": pr, "confidence": self.confidence(q.kind, probs)}
        return out


__all__ = ["D1", "IMAGE_TYPES", "TYPES", "BadRequest", "Decisions", "JointHead", "Prompt", "Question", "Unsupported",
           "d1_break", "decision_type", "escape_special", "spoken"]
