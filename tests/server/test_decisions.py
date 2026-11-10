"""artlite-server's ``/v1/systemone`` (``models.server.decisions``) on the host, with the scripted fake model (its rows:
``fake_row``): llama-server's request errors (400) and 501s, the answers' shapes as its ``test_systemone``'s, the
arithmetic against numpy (temperatures, score biases, lev's two orders and rating scale, kev's dot products,
vllm-decision2's joint head, the confidences), the prompts (nimble's question list, lev's sorted keys, kev's text and
escapes, vllm-decision2's pieces and spans), a non-causal model's prompts read whole and a recurrent model's questions
sharing the state through a checkpoint; lfm3-d1's label groups, escapes and sequences of a state's questions."""

import base64
import math
import os
import re
import types
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from models.common import MEDIA_MARKER
from models.sequence import Sequence
from models.server import decisions as D
from models.server.cache import common_prefix
from models.server.decisions import Decisions

from .test_server import _stop_engines, app_of, fake_row, make  # noqa: F401 (_stop_engines: autouse)

STATE = "I was charged twice for my order last week and nobody has replied."
QUESTIONS = {  # (llama.cpp's test_systemone)
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
}
TEMPLATE = ("<|system|>{{ state if state is string else state | tojson }} question {{ id }} {{ instructions }}"
            "{% for o in options %} {{ o.label }} {{ o.key }}{% endfor %} <|assistant|>")
NIMBLE = TEMPLATE + "{% for q in questions %} also {{ q.id }}{% endfor %}"
KEV = ("<|system|>{{ state }} question {{ instructions }}{% for o in options %} {{ o.key }}{% if o.description %} "
       "{{ o.description }}{% endif %} <|user|>{% endfor %} <|assistant|>")


def context(kind: str, model: object, template: str = TEMPLATE, **kw: object) -> Decisions:
    """A decision context on the fake tokenizer: label rows (or kev's head) of random values."""
    d = Decisions(type=kind, template=template, **kw).with_labels(model.tokenizer.encode)
    rng = np.random.default_rng(1)
    if kind == "kev":
        d.marker = model.tokenizer.encode("<|user|>")[0]
        d.head, d.bias = rng.standard_normal((6, 8)), rng.standard_normal(6)
    else:
        d.head = rng.standard_normal((len(d.labels), 8)) * 3
    return d


def client_of(kind: str, template: str = TEMPLATE, *, npu_kw: dict | None = None, **kw: object
              ) -> Iterator[tuple[object, object, Decisions, TestClient]]:
    model, npu, engine = make(**(npu_kw or {}))
    d = context(kind, model, template, **kw)
    with TestClient(app_of(engine, decisions=d)) as client:
        assert engine.wait_ready(5)
        yield model, npu, d, client


@pytest.fixture
def pplx() -> Iterator[tuple[object, object, Decisions, TestClient]]:
    yield from client_of("pplx-decider", temperatures={"choice": 0.5, "score.3_5": 2.0},
                         score_bias={4: np.array([0.0, 0.5, 0.0, -1.0])})


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    p = np.exp((z - z.max()) / T)
    return p / p.sum()


def scores(d: Decisions, model: object, body: dict) -> dict[str, list[np.ndarray]]:
    """Each question's variants' raw outputs as the fake model's rows give them."""
    qs = d.questions(body)
    out: dict[str, list[np.ndarray]] = {q.id: [] for q in qs}
    for p in d.prompts(d.state(body)[0], qs, model.tokenizer.encode):
        rows = np.stack([fake_row(p.ids[:i + 1]) for i in p.rows])
        out[qs[p.question].id].append(d.scores(qs[p.question], rows))
    return out


def test_answers(pplx: tuple) -> None:
    model, _, d, client = pplx
    body = {"state": STATE, "questions": QUESTIONS}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert list(out["answers"]) == ["route", "urgency", "angry"] and out["usage"]["output_tokens"] == 0
    qs = d.questions(body)
    prompts = d.prompts(STATE, qs, model.tokenizer.encode)
    assert out["usage"]["input_tokens"] == sum(len(p.ids) for p in prompts)
    s = scores(d, model, body)
    route, urgency, angry = (out["answers"][k] for k in ("route", "urgency", "angry"))
    p = softmax(np.array(s["route"][0]), 0.5)  # (the choice temperature)
    assert route["type"] == "choice" and list(route["probabilities"]) == ["billing", "shipping", "technical"]
    assert np.allclose(list(route["probabilities"].values()), p)
    assert route["choice"] == ["billing", "shipping", "technical"][int(p.argmax())]
    assert math.isclose(route["confidence"], (p.max() - 1 / 3) / (2 / 3))
    p = softmax(np.array(s["urgency"][0]) + [0.0, 0.5, 0.0, -1.0], 2.0)  # (the bias, the 3_5 bucket's temperature)
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert np.allclose(list(urgency["probabilities"].values()), p)
    assert math.isclose(urgency["score"], float(np.arange(4) @ p))
    mode = int(p.argmax())
    dist = sum(p[i] * abs(i - mode) for i in range(4)) / sum(abs(i - 1.5) / 4 for i in range(4))
    assert math.isclose(urgency["confidence"], max(0.0, 1 - dist))
    p = softmax(np.array(s["angry"][0]))  # ([false, true])
    assert angry == {"type": "noul", "noul": pytest.approx(p[1])}
    # an object is given to the template as JSON
    j = client.post("/v1/systemone", json={"state": {"ticket": STATE, "plan": "pro"}, "questions": QUESTIONS})
    t = client.post("/v1/systemone", json={"state": '{"ticket": "' + STATE + '", "plan": "pro"}',
                                           "questions": QUESTIONS})
    assert j.json()["answers"] == t.json()["answers"] and j.json()["usage"] == t.json()["usage"]


@pytest.mark.parametrize("body", [
    {"questions": QUESTIONS},
    {"state": STATE},
    {"state": STATE, "questions": {}},
    {"state": STATE, "questions": {"q": {"type": "unknown", "instructions": "x"}}},
    {"state": STATE, "questions": {"q": {"type": "noul"}}},
    {"state": STATE, "questions": {"q": {"type": "choice", "instructions": "x"}}},
    {"state": STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
    {"state": STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}},
    {"state": STATE, "questions": QUESTIONS, "images": ["https://example.com/image.png"]},
    {"state": STATE, "questions": QUESTIONS, "images": ["data:image/png;base64,AAAA"] * 9},
    {"state": STATE, "questions": QUESTIONS, "videos": ["data:video/mp4;base64,AAAA"]},
])
def test_invalid_request(pplx: tuple, body: dict) -> None:
    r = pplx[3].post("/v1/systemone", json=body)
    assert r.status_code == 400 and "error" in r.json(), r.text


def test_not_supported(pplx: tuple) -> None:
    client = pplx[3]
    r = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS,
                                           "images": ["data:image/png;base64,AAAA"]})
    assert r.status_code == 501  # (no vision encoder for decisions)
    r = client.post("/v1/systemone", json={"state": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": STATE}]}],
        "questions": QUESTIONS})
    assert r.status_code == 501
    _, _, engine = make()
    with TestClient(app_of(engine)) as plain:  # (not a decision model)
        assert engine.wait_ready(5)
        r = plain.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS})
        assert r.status_code == 501 and "not a decision model" in r.json()["error"]["message"]
        assert plain.get("/v1/models").json()["data"][0]["architecture"]["output_modalities"] == ["text"]
    assert client.get("/v1/models").json()["data"][0]["architecture"]["output_modalities"] == ["decisions"]
    _, _, engine = make()
    with TestClient(app_of(engine, decisions=Decisions(type="clef", template=TEMPLATE))) as other:
        assert engine.wait_ready(5)
        r = other.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS})
        assert r.status_code == 501 and "clef" in r.json()["error"]["message"]


def test_lev() -> None:
    """Two orders of a choice's options (the second's softmax reversed, the two averaged), a noul's rating scale,
    every object's keys sorted."""
    for model, _, d, client in client_of("lev"):
        q = {"pick": {"type": "choice", "instructions": "x", "criteria": {"b": None, "a": None, "c": None}},
             "yes": {"type": "noul", "instructions": "y"}}
        body = {"state": {"z": 1, "a": 2}, "questions": q}
        prompts = d.prompts(body["state"], d.questions(body), model.tokenizer.encode)
        assert [(p.question, p.variant) for p in prompts] == [(0, 0), (0, 1), (1, 0)]
        assert '{"a": 2, "z": 1}' in model.tokenizer.decode_bytes(prompts[0].ids).decode()
        out = client.post("/v1/systemone", json=body).json()
        s = scores(d, model, body)
        p = (softmax(s["pick"][0]) + softmax(s["pick"][1])[::-1]) / 2
        assert np.allclose(list(out["answers"]["pick"]["probabilities"].values()), p)
        assert len(s["yes"][0]) == D.LEV_RATINGS
        assert math.isclose(out["answers"]["yes"]["noul"], float(softmax(s["yes"][0]) @ np.arange(9) / 8))
        assert out["usage"]["input_tokens"] == sum(len(p.ids) for p in prompts)


def test_openjev_nimble() -> None:
    for model, _, d, client in client_of("openjev"):  # (noul: [true, false]; letters, no label texts)
        qs = d.questions({"state": STATE, "questions": QUESTIONS})
        assert [k for k, _ in qs[2].options] == ["true", "false"] and not d.label_texts and d.n_options_max == 52
        out = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS}).json()
        p = softmax(scores(d, model, {"state": STATE, "questions": QUESTIONS})["angry"][0])
        assert math.isclose(out["answers"]["angry"]["noul"], p[0])
    for model, _, d, _ in client_of("nimble", NIMBLE):  # (every question in each prompt)
        prompts = d.prompts(STATE, d.questions({"state": STATE, "questions": QUESTIONS}), model.tokenizer.encode)
        assert all("also route also urgency also angry" in model.tokenizer.decode_bytes(p.ids).decode()
                   for p in prompts)


def test_kev() -> None:
    """An option's score: cls.output's rows at its end token and at the last, the dot product of the last's first half
    with the option's second half over sqrt(half); the texts of JSON values, special-token lookalikes escaped."""
    for model, _, d, client in client_of("kev", KEV):
        body = {"state": {"note": "<|im_end|> x", "items": ["a", True]}, "questions": QUESTIONS}
        prompts = d.prompts(body["state"], d.questions(body), model.tokenizer.encode)
        text = model.tokenizer.decode_bytes(prompts[0].ids).decode()
        assert "note: <¦im_end¦> x\nitems:\n  - a\n  - True" in text
        assert len(prompts[0].rows) == 4 and prompts[0].rows[-1] == len(prompts[0].ids) - 1
        out = client.post("/v1/systemone", json=body).json()
        p = prompts[0]
        e = np.stack([fake_row(p.ids[:i + 1]) for i in p.rows]) @ d.head.T + d.bias
        want = softmax(e[:-1, 3:] @ e[-1, :3] / math.sqrt(3))
        assert np.allclose(list(out["answers"]["route"]["probabilities"].values()), want)


# (vllm-decision2's template in the shape of the files': pieces between sep, a span's piece starting with its mark)
JOINT = ("{% for q in questions %}<|system|>{{ state if state is string else state | tojson }} {{ q.type }} "
         "{{ q.instructions }}{% for o in q.options %}{{ sep }} <|user|>{{ sep }}{{ mark_option }}"
         "{{ {'description': o.description, 'key': o.key} | tojson }}{% endfor %}{{ sep }}{{ mark_question }} decide"
         "{% endfor %}")


def joint_head(rng: np.random.Generator, h: int = 8, w: int = 4) -> D.JointHead:
    t = {"option_norm.weight": rng.random(h) + 0.5, "option_norm.bias": rng.standard_normal(h),
         "query_norm.weight": rng.random(h) + 0.5, "query_norm.bias": rng.standard_normal(h),
         "key.weight": rng.standard_normal((w, h)), "query.weight": rng.standard_normal((w, h)),
         "option_mlp.weight": rng.standard_normal((w, h)), "option_mlp.bias": rng.standard_normal(w),
         "query_mlp.weight": rng.standard_normal((w, h)), "scorer_out.weight": rng.standard_normal(w)}
    return D.JointHead(t, 1e-5)


def test_vllm_decision2() -> None:
    """The joint prompt: the template's pieces tokenized one by one, the rows the last token of each option's span and
    of the question's; an option's score the head's (numpy, ``torch.nn`` style: LayerNorm, GELU); the confidence the
    normalized entropy; vllm-decision2's stricter requests (400) and a noul's options in its criteria's order."""
    from math import erf

    for model, _, d, client in client_of("vllm-decision2", JOINT, score_bias={4: np.array([0.0, 0.5, 0.0, -1.0])}):
        d.joint = joint_head(np.random.default_rng(3))
        body = {"state": {"ticket": STATE, "plan": "pro"}, "questions": QUESTIONS}
        qs = d.questions(body)
        prompts = d.prompts(body["state"], qs, model.tokenizer.encode)
        assert [p.question for p in prompts] == [0, 1, 2]
        enc = model.tokenizer.encode
        for p, q in zip(prompts, qs, strict=True):
            assert len(p.rows) == len(q.options) + 1 and p.rows[-1] == len(p.ids) - 1
            assert all(p.ids[r + 1] == enc(" <|user|>")[0] for r in p.rows[:-2])  # (an option's span ends before the next)
        text = model.tokenizer.decode_bytes(prompts[0].ids).decode()
        assert text.index('"plan"') < text.index('"ticket"')  # (the values' keys sorted)
        out = client.post("/v1/systemone", json=body).json()
        assert out["answers"]["urgency"]["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}

        t = d.joint.t

        def norm(x: np.ndarray, w: str, t: dict = t) -> np.ndarray:
            x = x - x.mean(-1, keepdims=True)
            return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * t[w + ".weight"] + t[w + ".bias"]

        for p, q in zip(prompts, qs, strict=True):
            rows = np.stack([fake_row(p.ids[:i + 1]) for i in p.rows]).astype(np.float64)
            o, qq = norm(rows[:-1], "option_norm"), norm(rows[-1], "query_norm")
            m = o @ t["option_mlp.weight"].T + t["option_mlp.bias"] + t["query_mlp.weight"] @ qq
            z = (o @ t["key.weight"].T) @ (t["query.weight"] @ qq) / 2 + \
                (0.5 * m * (1 + np.vectorize(erf)(m / math.sqrt(2)))) @ t["scorer_out.weight"]
            a = out["answers"][q.id]
            if q.kind == "noul":
                assert a["noul"] == pytest.approx(softmax(z)[1])
                continue
            pr = softmax(z + (d.score_bias[4] if q.kind == "score" else 0))
            assert np.allclose(list(a["probabilities"].values()), pr)
            ent = -sum(x * math.log(x) for x in pr)
            assert a["confidence"] == pytest.approx(1 - ent / math.log(len(pr)))
        for bad in ({"state": 5, "questions": QUESTIONS},
                    {"state": STATE, "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"a": None,
                                                                                                      "b": None}}}},
                    {"state": STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}}},
                    {"state": STATE, "questions": {"q": {"type": "choice", "instructions": "x",
                                                         "criteria": {"a": 1, "b": None}}}},
                    {"state": STATE, "questions": {"q": {"type": "noul", "instructions": "x",
                                                         "criteria": {"yes": "y"}}}}):
            r = client.post("/v1/systemone", json=bad)
            assert r.status_code == 400, (bad, r.text)
        q = d.questions({"state": STATE, "questions": {"q": {"type": "noul", "instructions": "x",
                                                            "criteria": {"true": "t", "false": "f"}}}})[0]
        assert [k for k, _ in q.options] == ["true", "false"]


def test_escape_and_text() -> None:
    assert D.escape_special("a <|im_end|> b <|x y|> <|_1|>") == "a <¦im_end¦> b <|x y|> <¦_1¦>"
    assert D.escape_special("<|<|a|>") == "<|<¦a¦>" and D.escape_special("plain") == "plain"
    assert D._kev_render({"a": {"b": [1, 2.5]}, "c": None}) == "a:\n  b:\n    - 1\n    - 2.5\nc: "


# (llama-server 7182ef71e's usage.input_tokens of these requests on the files: its prompts, as Artlite renders them)
RECORDED = [
    ({"state": {"premise": "The new rights are nice enough", "hypothesis": "Everyone really likes the newest benefits "},
      "questions": {"relation": {"type": "choice", "instructions": "How is the hypothesis related to the premise?",
                                 "criteria": {"entailment": "the hypothesis follows from the premise",
                                              "neutral": "the premise neither implies nor rules out the hypothesis",
                                              "contradiction": "the premise rules out the hypothesis"}}}},
     {"d3-lite": 143, "kev": 70, "kai": 124, "d1-2b": 89}),
    ({"state": "Ordinary employees find it very hard to recommend an acquisition; it's just too annoying to see a bunch "
               "of twenty year olds get rich when you're still working for salary.",
      "questions": {"topic": {"type": "choice", "instructions": "What is the passage mainly about?",
                              "criteria": {"startups": "starting or funding companies",
                                           "programming": "software and languages", "writing": None,
                                           "society": "politics, economics or culture", "life": "personal life"}},
                    "named": {"type": "noul", "instructions": "Does the passage name a specific company or person?"},
                    "tone": {"type": "score", "instructions": "How optimistic is the tone of the passage?",
                             "criteria": ["very pessimistic", "pessimistic", "neutral", "optimistic",
                                          "very optimistic"]},
                    "audience": {"type": "choice", "instructions": "Who is the passage written for?",
                                 "criteria": {"founders": None, "programmers": None, "investors": None,
                                              "students": None, "general readers": None}}}},
     {"d3-lite": 556, "kev": 266, "kai": 556, "d1-2b": 323}),
]


@pytest.mark.parametrize("name", ["d3-lite", "kev", "kai", "d1-2b"])
def test_llama_server_prompts(name: str) -> None:
    """The files' prompts as llama-server tokenizes them (their token counts: its usage), from the files' templates
    (lfm3-d1: each question alone, as llama-server reads them)."""
    from artlite.gguf_model import GGUFFile
    from artlite.model_config import decision_gguf
    from models.tokenizer import BpeTokenizer

    path = decision_gguf(name)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    encode = BpeTokenizer(GGUFFile(path)).encode
    d = Decisions.of(path, encode)
    for body, tokens in RECORDED:
        prompts = d.prompts(d.state(body)[0], d.questions(body), encode, set_size=1)
        assert sum(len(p.ids) for p in prompts) == tokens[name]


def test_noncausal() -> None:
    """A non-causal model reads every prompt whole from position 0 (no prompt cache), at most its largest chunk."""
    for model, npu, d, client in client_of("vllm-d3"):
        npu.causal, npu.prefill = False, types.SimpleNamespace(sizes=(64, 128))
        body = {"state": STATE, "questions": QUESTIONS}
        npu.calls.clear()
        assert client.post("/v1/systemone", json=body).status_code == 200
        lens = [len(p.ids) for p in d.prompts(STATE, d.questions(body), model.tokenizer.encode)]
        assert [c for c in npu.calls if c[0] == "outputs"] == [("outputs", n) for n in lens]
        assert all(c[1] == 0 for c in npu.calls if c[0] == "restore")
        r = client.post("/v1/systemone", json={"state": " ".join(["word"] * 200), "questions": QUESTIONS})
        assert r.status_code == 400 and "128" in r.json()["error"]["message"]


def test_shared_state() -> None:
    """A recurrent model's questions: the state's prefix once (a checkpoint at the cache group below where the
    prompts part), each question's rest after it; the same answers as without."""
    long = " ".join(f"w{i}" for i in range(60))
    body = {"state": long, "questions": QUESTIONS}
    answers = []
    for share in (False, True):
        for model, npu, d, client in client_of("pplx-decider", npu_kw={"exact": True}):
            if share:  # (a chunk costs 10 ms and 0.1 ms a token: the prefix's runs pay)
                npu.prompt_seconds = lambda n, position=0: 0.01 * math.ceil(n / 8) + 1e-4 * n
            npu.calls.clear()
            out = client.post("/v1/systemone", json=body).json()
            answers.append(out["answers"])
            prompts = d.prompts(long, d.questions(body), model.tokenizer.encode)
            runs = [c[1] for c in npu.calls if c[0] == "outputs"]
            if share:
                p = min(common_prefix(prompts[0].ids, x.ids) for x in prompts[1:])
                p -= p % npu.group
                assert ("checkpoint", p) in npu.calls and runs == [p] + [len(x.ids) - p for x in prompts]
            else:
                assert runs == [len(x.ids) for x in prompts]
    assert answers[0] == answers[1]


# (lfm3-d1's template in the shape of d1-2B's: the images, the state, "QUESTION:", the question and its options up to
# the answer)
D1 = ("<s><|user|>{% for image in images %}{{ image }}{% endfor %}{% if state is not none %}"
      "{{ state if state is string else state | tojson(indent=2) }}"
      "{{ '\\n\\n\\nQUESTION:\\n' }}{% endif %}{{ instructions }}{% for o in options %} {{ o.label }} {{ o.key }}"
      "{% endfor %}<|assistant|><think></think>")


def d1_context(model: object) -> Decisions:
    """lfm3-d1 on the fake tokenizer: LM rows of random values per token, "<|user|>" a control token."""
    rng = np.random.default_rng(5)
    head = rng.standard_normal((2048, 8)) * 3
    d = Decisions(type="lfm3-d1", template=D1, temperatures={"choice": 1.3, "score": 1.3, "noul": 1.3},
                  spellings=re.compile(re.escape("<|user|>")), lm_rows=lambda ids: head[ids])
    return d.with_labels(model.tokenizer.encode)


def test_d1() -> None:
    """lfm3-d1: each option's score the largest logit of its label group (a choice's code and the code after a space;
    yes / Yes / YES; a level's digit), at the temperature, a choice's confidence its probability, a score the median
    level; the control tokens' texts broken in the input, keys too; a state's questions in sequences of ``set_size``
    (the state, then each question after "QUESTION:"; bounds per sequence, a row per question), the same answers'
    shapes; a null state; ``set_size`` and spoken questions refused (400)."""
    model, npu, engine = make()
    d = d1_context(model)
    enc = model.tokenizer.encode
    with TestClient(app_of(engine, decisions=d)) as client:
        assert engine.wait_ready(5)
        npu.causal, npu.prefill = False, types.SimpleNamespace(sizes=(64, 128, 256))
        body = {"state": {"note <|user|> x": "a <|user|> b"}, "questions": QUESTIONS}
        qs = d.questions(body)
        assert [k for k, _ in qs[2].options] == ["true", "false"]
        texts, groups = d.d1_labels(qs[0], enc)
        assert texts == ["A", "B", "C"] and groups == [[enc("A")[0], enc(" A")[0]], [enc("B")[0], enc(" B")[0]],
                                                       [enc("C")[0], enc(" C")[0]]]
        assert d.d1_labels(qs[1], enc)[0] == ["0", "1", "2", "3"]
        assert d.d1_labels(qs[2], enc)[1] == [[enc(f)[0] for f in forms] for forms in (("yes", "Yes", "YES"),
                                                                                       ("no", "No", "NO"))]
        lone = d.prompts(body["state"], qs, enc, set_size=1)
        text = model.tokenizer.decode_bytes(lone[0].ids).decode()
        assert text.count("<\u200b|user|>") == 2 and text.count("<|user|>") == 1  # (the template's own: a token)
        assert [p.questions for p in lone] == [[0], [1], [2]] and all(p.sets == (p.sets[0], len(p.ids)) for p in lone)
        both = d.prompts(body["state"], qs, enc, set_size=2)
        assert [p.questions for p in both] == [[0, 1], [2]]
        p0 = both[0]
        assert p0.ids[:p0.sets[1]] == lone[0].ids and p0.sets[0] == lone[0].sets[0]
        assert p0.ids[p0.sets[1]:] == enc("QUESTION:\n" + d.render(None, qs, qs[1], 0)[len("<s><|user|>"):])
        assert list(p0.rows) == [p0.sets[1] - 1, p0.sets[2] - 1]
        npu.calls.clear()
        out = client.post("/v1/systemone", json={**body, "set_size": 2}).json()
        assert [c for c in npu.calls if c[0] == "outputs"] == [("outputs", len(p.ids), p.sets) for p in both]
        assert out["usage"]["input_tokens"] == sum(len(p.ids) for p in both)
        for p in both:
            for j, i in enumerate(p.questions):
                q = qs[i]
                x = fake_row(p.ids[:p.rows[j] + 1]).astype(np.float64)
                z = np.array([max(d._rows[t] @ x for t in g) for g in d.d1_labels(q, enc)[1]])
                pr = softmax(z, 1.3)
                a = out["answers"][q.id]
                if q.kind == "noul":
                    assert a["noul"] == pytest.approx(pr[0])
                    continue
                assert np.allclose(list(a["probabilities"].values()), pr)
                assert a["confidence"] == pytest.approx(pr.max())
                if q.kind == "score":
                    assert a["score"] == float(np.searchsorted(np.cumsum(pr), 0.5))
        default = client.post("/v1/systemone", json=body).json()  # (one sequence: D1_SET_SIZE)
        assert default["usage"]["input_tokens"] == len(d.prompts(body["state"], qs, enc)[0].ids)
        r = client.post("/v1/systemone", json={"state": None, "questions": {"q": QUESTIONS["angry"]}})
        assert r.status_code == 200
        for bad in ({**body, "set_size": 0}, {**body, "set_size": "2"},
                    {"state": STATE, "questions": {"q": {"type": "noul", "instructions": {"type": "input_audio"}}}}):
            r = client.post("/v1/systemone", json=bad)
            assert r.status_code == 400, (bad, r.text)


IMAGES = [b"first image's bytes", b"second image's bytes"]
URLS = ["data:image/png;base64," + base64.b64encode(b).decode() for b in IMAGES]


def test_images() -> None:
    """Images (``files`` / ``images``, a chat state's ``image_url`` parts) in lfm3-d1's and vllm-d3's requests: each
    image through the encoder once, on the worker; the prompts sequences with the image's tokens between the encoder's
    markup where the template has its markers (the input's markers text: " "); lfm3-d1's sequences hold them in the
    state; usage counts their tokens; a file's image temperature; a null state with images; audio, video, images for
    another type or without an encoder refused (501)."""
    model, npu, engine = make(vision=True)
    d = d1_context(model)
    d.temperatures["image"] = 0.5
    enc = model.tokenizer.encode
    with TestClient(app_of(engine, decisions=d)) as client:
        assert engine.wait_ready(5)
        npu.causal, npu.prefill = False, types.SimpleNamespace(sizes=(64, 128, 256))
        state = f"a note {MEDIA_MARKER} with the marker"
        body = {"state": state, "questions": QUESTIONS, "images": URLS[:1], "set_size": 2}
        npu.calls.clear()
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200, r.text
        assert model.threads == ["artlite-npu"]
        qs = d.questions(body)
        media = [model.image_tokens(IMAGES[0])]
        assert len(media[0]) == 4 and list(media[0].ids[[0, 3]]) == enc("<|system|><|assistant|>")
        got = d.prompts(state, qs, enc, media=media, set_size=2)
        text = d.prompts(state.replace(MEDIA_MARKER, " "), qs, enc, set_size=2)
        assert [p.questions for p in got] == [[0, 1], [2]]
        for a, b in zip(got, text, strict=True):  # (the image after "<s><|user|>": the state's)
            assert isinstance(a.ids, Sequence) and len(a.ids) == len(b.ids) + 4 and list(a.ids.given) == \
                [False] * 3 + [True] * 2 + [False] * (len(b.ids) - 1)
            assert [*a.ids.ids[:2], *a.ids.ids[6:]] == b.ids and a.sets == tuple(x + 4 for x in b.sets)
            assert list(a.rows) == list(b.rows + 4)
        assert [c for c in npu.calls if c[0] == "outputs"] == [("outputs", len(p.ids), p.sets) for p in got]
        out = r.json()
        assert out["usage"]["input_tokens"] == sum(len(p.ids) for p in got)
        p = got[0]
        x = fake_row([int(t) for t in p.ids.ids[:p.rows[0] + 1]]).astype(np.float64)
        z = np.array([max(d._rows[t] @ x for t in g) for g in d.d1_labels(qs[0], enc)[1]])
        assert np.allclose(list(out["answers"]["route"]["probabilities"].values()), softmax(z, 0.5))  # (the image's)
        chat = {"state": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": URLS[1]}},
                                                       {"type": "text", "text": "what is it?"}]}],
                "questions": QUESTIONS, "files": URLS[:1]}
        model.threads.clear()
        r = client.post("/v1/systemone", json=chat)
        assert r.status_code == 200 and len(model.threads) == 2  # (files first, then the chat's)
        r = client.post("/v1/systemone", json={"state": None, "images": URLS, "questions": {"q": QUESTIONS["angry"]}})
        assert r.status_code == 200, r.text
        (lone,) = d.prompts(None, d.questions({"state": None, "questions": {"q": QUESTIONS["angry"]}}), enc,
                            media=[model.image_tokens(b) for b in IMAGES])
        assert lone.sets[0] == 2 + 8 and r.json()["usage"]["input_tokens"] == len(lone.ids)
        for bad in ({**body, "images": ["data:audio/wav;base64,AAAA"]},
                    {"state": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA"}}]}],
                     "questions": QUESTIONS},
                    {"state": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": URLS[0]}}]}],
                     "questions": QUESTIONS}):
            r = client.post("/v1/systemone", json=bad)
            assert r.status_code == 501, (bad, r.text)
        assert client.post("/v1/systemone", json={**body, "images": ["data:image/png;base64,A"]}).status_code == 400
    model, npu, engine = make(vision=True)
    d = context("vllm-d3", model, "{% for image in images %}{{ image }}{% endfor %}" + TEMPLATE)
    with TestClient(app_of(engine, decisions=d)) as client:
        assert engine.wait_ready(5)
        npu.causal, npu.prefill = False, types.SimpleNamespace(sizes=(64, 128))
        npu.calls.clear()
        r = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS, "images": URLS})
        assert r.status_code == 200 and len(model.threads) == 2
        qs = d.questions({"state": STATE, "questions": QUESTIONS})
        got = d.prompts(STATE, qs, model.tokenizer.encode, media=[model.image_tokens(b) for b in IMAGES])
        text = d.prompts(STATE, qs, model.tokenizer.encode)
        assert [len(p.ids) for p in got] == [len(p.ids) + 8 for p in text]
        assert [c for c in npu.calls if c[0] == "outputs"] == [("outputs", len(p.ids)) for p in got]
    for model, _, _, client in client_of("pplx-decider"):
        model.vision = types.SimpleNamespace(markup=("<|system|>", "<|assistant|>"))
        r = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS, "images": URLS})
        assert r.status_code == 501 and "pplx-decider" in r.json()["error"]["message"]
