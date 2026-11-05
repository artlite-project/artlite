"""artlite-server's readout routes (embeddings, rerank, classify) and ``logprobs`` on the host, with the scripted fake model
(``test_server``: its output rows are a function of the tokens up to each, ``fake_row``): shapes and errors as
llama-server's, pooling and normalization against numpy, base64, OpenAI's ``dimensions``, rerank's order, ``top_n``
and both formats, the prompt cache's prefix; the readout math of ``models.readouts``."""

import base64
import math
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from models.readouts import RankHead, Readouts, normalize, pool, pooling_of, rows_of
from models.server.cache import common_prefix

from .test_server import _stop_engines, app_of, chat, events, fake_row, make  # noqa: F401 (_stop_engines: autouse)

HEAD = RankHead(np.stack([np.arange(8, dtype=np.float32) / 8, -np.arange(8, dtype=np.float32) / 8]), None, True,
                ("yes", "no"))


def client_of(**options: object) -> Iterator[tuple[object, object, TestClient]]:
    readouts = options.pop("readouts", None)
    model, npu, engine = make()
    with TestClient(app_of(engine, readouts=readouts, **options)) as client:
        assert engine.wait_ready(5)
        yield model, npu, client


@pytest.fixture
def mean() -> Iterator[tuple[object, object, TestClient]]:
    yield from client_of(embedding=True, pooling="mean")


@pytest.fixture
def rank() -> Iterator[tuple[object, object, TestClient]]:
    yield from client_of(embedding=True, pooling="rank",
                         readouts=Readouts(pooling="rank", head=HEAD, rerank_template="{query}<|system|>{document}"))


def expected(model: object, text: str, pooling: str, p: int = 2) -> np.ndarray:
    ids = model.encode(text)
    rows = np.stack([fake_row(ids[:i + 1]) for i in rows_of(pooling, len(ids))])
    v = pool(rows, pooling)
    return v if pooling == "none" else normalize(v, p)


def test_math() -> None:
    x = np.array([[3.0, -4.0], [0.0, 0.0]], np.float32)
    assert np.allclose(normalize(x, 2), [[0.6, -0.8], [0, 0]])
    assert np.allclose(normalize(x, -1), x)
    assert np.allclose(normalize(x, 0), [[3 * 32760 / 4, -32760], [0, 0]])
    assert np.allclose(normalize(x, 1), [[3 / 7, -4 / 7], [0, 0]])
    assert list(rows_of("mean", 3)) == [0, 1, 2] and list(rows_of("cls", 3)) == [0] and list(rows_of("last", 3)) == [2]
    assert pooling_of({"qwen3.pooling_type": 3}, "qwen3") == "last" and pooling_of({}, "llama") is None
    z = HEAD.w @ np.ones(8)
    assert math.isclose(HEAD.score(np.ones(8)), 1 / (1 + math.exp(z[1] - z[0])), rel_tol=1e-6)
    assert Readouts(rerank_template="a{query}b{document}c").rerank_ids(lambda s: [len(s)], [], "{document}", "q") \
        == [len("a{document}bqc")]  # (one pass: the query's own braces stay)
    assert Readouts(add_eos=True, eos=1, add_sep=True, sep=2).rerank_ids(lambda s: [ord(s)], [0], "a", "b") == \
        [0, 97, 1, 2, 98, 1]


def test_off() -> None:
    for model, _npu, client in client_of():
        r = client.post("/v1/embeddings", json={"input": "hi"})
        assert r.status_code == 501 and "--embeddings" in r.json()["error"]["message"]
        assert client.post("/v1/rerank", json={"query": "q", "documents": ["d"]}).status_code == 501


def test_embeddings(mean: tuple) -> None:
    model, _npu, client = mean
    r = client.post("/v1/embeddings", json={"input": ["hello world", "a b c d e"], "model": "x"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list" and body["model"] == "x"
    assert [d["index"] for d in body["data"]] == [0, 1] and body["data"][0]["object"] == "embedding"
    for d, text in zip(body["data"], ["hello world", "a b c d e"], strict=True):
        assert np.allclose(d["embedding"], expected(model, text, "mean"), atol=1e-6)
    n = len(model.encode("hello world")) + len(model.encode("a b c d e"))
    assert body["usage"] == {"prompt_tokens": n, "total_tokens": n}
    b = client.post("/v1/embeddings", json={"input": "hello world", "encoding_format": "base64"}).json()["data"][0]
    assert b["encoding_format"] == "base64"
    assert np.allclose(np.frombuffer(base64.b64decode(b["embedding"]), "<f4"), expected(model, "hello world", "mean"))
    d = client.post("/v1/embeddings", json={"input": "hello world", "dimensions": 3}).json()["data"][0]["embedding"]
    assert np.allclose(d, normalize(expected(model, "hello world", "mean", -1)[:3], 2))  # (the first 3, renormalized)
    raw = client.post("/v1/embeddings", json={"input": "hello world", "embd_normalize": -1}).json()
    assert np.allclose(raw["data"][0]["embedding"], expected(model, "hello world", "mean", -1))
    ids = model.encode("x y")
    by_ids = client.post("/v1/embeddings", json={"input": ids}).json()["data"]
    assert len(by_ids) == 1 and np.allclose(by_ids[0]["embedding"], expected(model, "x y", "mean"))
    own = client.post("/embeddings", json={"content": "hello world"}).json()
    assert own[0]["index"] == 0 and np.allclose(own[0]["embedding"], [expected(model, "hello world", "mean")])


def test_embedding_errors(mean: tuple) -> None:
    _, _, client = mean
    for body, word in (({}, "must be provided"), ({"input": []}, "must not be empty"),
                       ({"input": "a", "encoding_format": "hex"}, "float or base64"),
                       ({"input": [999999]}, "out of range"), ({"input": "a", "dimensions": 0}, "dimensions")):
        r = client.post("/v1/embeddings", json=body)
        assert r.status_code == 400 and word in r.json()["error"]["message"], (body, r.text)
    r = client.post("/v1/embeddings", json={"input": "word " * 600})
    assert r.status_code == 400 and r.json()["error"]["type"] == "exceed_context_size_error"


def test_pooling_none() -> None:
    for model, _npu, client in client_of(embedding=True):
        r = client.post("/v1/embeddings", json={"input": "a b"})
        assert r.status_code == 400 and "not OAI compatible" in r.json()["error"]["message"]
        rows = client.post("/embedding", json={"content": "a b"}).json()[0]["embedding"]
        assert np.allclose(rows, expected(model, "a b", "none"))  # (every token's row, not normalized)


def test_cache_prefix() -> None:
    """Last pooling continues from the cached prefix (rows after it: the same as from scratch)."""
    for cached in (True, False):
        for model, npu, client in client_of(embedding=True, pooling="last", cache_prompt=cached):
            a = "one two three four five six seven"
            for text in (a, a + " eight nine", a + " eight nine"):
                v = client.post("/v1/embeddings", json={"input": text}).json()["data"][0]["embedding"]
                assert np.allclose(v, expected(model, text, "last"), atol=1e-6)
            runs = [c[1] for c in npu.calls if c[0] == "outputs"]
            n = len(model.encode(a))
            # (cached: the second extends the device's sequence; the third rewinds to the group below its last token)
            assert runs == ([n, 2, n + 2 - (n - n % 4)] if cached else [n, n + 2, n + 2]), runs


def test_rerank(rank: tuple) -> None:
    model, _npu, client = rank
    docs = ["alpha", "beta gamma", "delta", "epsilon zeta eta"]

    def score(q: str, d: str) -> float:
        ids = model.tokenizer.encode(f"{q}<|system|>{d}")
        return HEAD.score(fake_row(ids))  # (a rerank template's prompt: no BOS, as llama-server's)

    want = sorted(range(4), key=lambda i: -score("query text", docs[i]))
    r = client.post("/v1/rerank", json={"query": "query text", "documents": docs, "top_n": 3, "return_documents": True})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [e["index"] for e in res] == want[:3]
    assert all(math.isclose(e["relevance_score"], score("query text", docs[e["index"]]), rel_tol=1e-6) for e in res)
    assert res[0]["document"] == {"text": docs[want[0]]}
    tei = client.post("/rerank", json={"query": {"text": "query text"}, "texts": docs, "return_text": True}).json()
    assert [e["index"] for e in tei] == want and tei[0]["text"] == docs[want[0]] and "score" in tei[0]
    for body in ({"documents": docs}, {"query": "q", "documents": []}, {"query": "q", "documents": docs, "top_n": -1},
                 {"query": {"image": "x"}, "documents": docs}):
        assert client.post("/v1/reranking", json=body).status_code == 400, body


def test_rerank_shared_query() -> None:
    """A rerank request's documents after a long query (a model that rewinds): the query's prompt once (to the cache
    group below where the prompts part), each document's rest after it, the same scores as each pair whole."""
    import math as m

    query = " ".join(f"q{i}" for i in range(40))
    docs = ["alpha", "beta gamma", "delta", "epsilon zeta eta"]
    for share in (False, True):
        for model, npu, client in client_of(embedding=True, pooling="rank", readouts=Readouts(
                pooling="rank", head=HEAD, rerank_template="{query}<|system|>{document}")):
            if share:  # (a chunk costs 10 ms and 0.1 ms a token: the query's run once pays)
                npu.prompt_seconds = lambda n, position=0: 0.01 * m.ceil(n / 8) + 1e-4 * n
            npu.calls.clear()
            res = client.post("/v1/rerank", json={"query": query, "documents": docs}).json()["results"]
            prompts = [model.tokenizer.encode(f"{query}<|system|>{d}") for d in docs]
            want = {i: HEAD.score(fake_row(ids)) for i, ids in enumerate(prompts)}
            assert all(m.isclose(e["relevance_score"], want[e["index"]], rel_tol=1e-6) for e in res)
            runs = [c[1] for c in npu.calls if c[0] == "outputs"]
            if share:
                p = min(common_prefix(prompts[0], ids) for ids in prompts[1:])
                p -= p % npu.group
                assert runs == [p] + [len(ids) - p for ids in prompts], runs
            else:
                assert len(runs) == len(docs), runs


def test_classify(rank: tuple) -> None:
    """vLLM's /classify: per input the labels' softmax of the head's logits at its last token, the likeliest label."""
    model, _npu, client = rank
    texts = ["good movie", "bad plot and worse acting"]
    r = client.post("/classify", json={"input": texts, "model": "m"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["object"] == "list" and out["model"] == "m" and [d["index"] for d in out["data"]] == [0, 1]
    for d, t in zip(out["data"], texts, strict=True):
        ids = model.encode(t)
        z = HEAD.w.astype(np.float64) @ fake_row(ids)
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        assert np.allclose(d["probs"], p) and d["num_classes"] == 2 and d["label"] == ("yes", "no")[int(p.argmax())]
    assert out["usage"]["prompt_tokens"] == sum(len(model.encode(t)) for t in texts)
    assert client.post("/classify", json={"input": "one"}).json()["data"][0]["num_classes"] == 2
    assert client.post("/classify", json={"input": []}).status_code == 400
    for _model, _npu, plain in client_of(embedding=True, pooling="mean"):
        assert plain.post("/classify", json={"input": "a"}).status_code == 501


def test_no_readouts() -> None:
    """A model without readouts (Gemma 4) answers 501."""
    for _model, npu, client in client_of(embedding=True, pooling="mean"):
        npu.fail = NotImplementedError("Fake has no readouts")
        r = client.post("/v1/embeddings", json={"input": "a"})
        assert r.status_code == 501 and "no readouts" in r.json()["error"]["message"]


# ----------------------------------------------------------------------
# logprobs
# ----------------------------------------------------------------------


def test_logprobs() -> None:
    """Every sampled token's entry (the fake's logits: 0 for the next piece, -100 for the others), in the chat's
    response and its stream, /v1/completions' and /completion's; post-sampling probabilities."""
    for model, npu, client in client_of():
        npu.replies = [["hello", " world"]] * 4
        body = chat(client, "hi", logprobs=True, top_logprobs=2).json()
        content = body["choices"][0]["logprobs"]["content"]
        assert [e["token"] for e in content] == ["hello", " world", ""]  # (the end-of-sequence token's text: none)
        assert content[0]["id"] == model.tokenizer.id("hello") and content[0]["bytes"] == list(b"hello")
        assert math.isclose(content[0]["logprob"], -math.log1p(1023 * math.exp(-100)), abs_tol=1e-6)
        top = content[0]["top_logprobs"]
        assert len(top) == 2 and top[0]["id"] == content[0]["id"] and top[1]["logprob"] < -99
        evs = events(chat(client, "hi", logprobs=True, top_logprobs=2, stream=True))
        streamed = [e for ev in evs if isinstance(ev, dict) and ev.get("choices")
                    for e in (ev["choices"][0].get("logprobs") or {}).get("content", [])]
        assert [e["id"] for e in streamed] == [e["id"] for e in content]
        cmpl = client.post("/v1/completions", json={"prompt": "hi", "logprobs": 1}).json()
        assert len(cmpl["choices"][0]["logprobs"]["content"][0]["top_logprobs"]) == 1
        own = client.post("/completion", json={"prompt": "hi", "n_probs": 3, "post_sampling_probs": True}).json()
        first = own["completion_probabilities"][0]
        assert first["prob"] == 1.0 and "top_probs" in first and "logprob" not in first
        assert chat(client, "hi", top_logprobs=2).status_code == 400
        assert "logprobs" not in chat(client, "hi").json()["choices"][0]
