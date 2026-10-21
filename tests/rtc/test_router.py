"""artlite-server's router mode on the host (no NPU): the catalog of a directory of small GGUF files (ids, skipped
files, vision encoders, the header cache, presets), and the router under uvicorn with model servers that run
``models.server.child`` with test_server's fake model (it replies with its file's name; a file named ``fail-*`` fails
to load, ``waitload-*`` loads slowly, ``slow-*`` replies slowly and long). Each model server is a process (~2 s to
start). Serving the fake models with the web UI, for a manual check of its model picker in a browser:

    python tests/rtc/test_router.py [--port 8080]
"""

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import uvicorn
from gguf import GGMLQuantizationType, GGUFWriter

from models.server import catalog as catalog_module
from models.server import ui
from models.server.catalog import Catalog, read_preset, render
from models.server.main import arg_parser, listen
from models.server.router import Router, RouterOptions, create_router_app

CHILD = [sys.executable, os.path.abspath(__file__), "--child"]


def write_gguf(path: str, *, arch: str = "llama", n_embd: int = 64,
               qtype: GGMLQuantizationType = GGMLQuantizationType.Q4_0, projector: str = "minicpmv4_7") -> str:
    """A small GGUF file: a model of ``arch`` with one matrix of ``qtype``, or a vision encoder (``arch="clip"``);
    written next to the path and moved there (a server may have the old one mapped)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = GGUFWriter(path + ".tmp", arch)
    if arch == "clip":
        w.add_string("clip.projector_type", projector)
        w.add_uint32("clip.vision.projection_dim", n_embd)
        w.add_tensor("v.patch_embd.weight", np.zeros((4, 4), np.float32))
    else:
        w.add_context_length(8192)
        w.add_embedding_length(n_embd)
        w.add_tokenizer_model("gpt2")
        w.add_token_list(["<s>", "</s>", "a"])
        w.add_bos_token_id(0)
        w.add_eos_token_id(1)
        w.add_file_type(2)
        block, nbytes = {GGMLQuantizationType.Q4_0: (32, 18), GGMLQuantizationType.IQ2_XXS: (256, 66)}[qtype]
        w.add_tensor("token_embd.weight", np.zeros((n_embd, 256 // block * nbytes), np.uint8), raw_dtype=qtype)
        w.add_tensor("output_norm.weight", np.ones(n_embd, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(path + ".tmp", path)
    return path


@pytest.fixture
def models_dir(tmp_path: object) -> str:
    """org/a-GGUF: two quantizations and a drafter; org/vl-GGUF: a qwen35 model and two vision encoders;
    org/bad-GGUF: a file of a type without a tile format, one of an unknown architecture; a file at the top; a split
    file."""
    d = str(tmp_path / "models")
    for name in ("a-Q4_0.gguf", "a-Q8_0.gguf", "mtp-a-Q4_0.gguf"):
        write_gguf(f"{d}/org/a-GGUF/{name}")
    write_gguf(f"{d}/org/vl-GGUF/vl-1B-Q4_K_M.gguf", arch="qwen35", n_embd=96)
    write_gguf(f"{d}/org/vl-GGUF/mmproj-model-f16.gguf", arch="clip", n_embd=96)
    write_gguf(f"{d}/org/vl-GGUF/mmproj-model-f16-4x.gguf", arch="clip", n_embd=96)
    write_gguf(f"{d}/org/bad-GGUF/bad-IQ2_XXS.gguf", qtype=GGMLQuantizationType.IQ2_XXS)
    write_gguf(f"{d}/org/bad-GGUF/other-Q4_0.gguf", arch="qwen3")
    write_gguf(f"{d}/top.gguf")
    write_gguf(f"{d}/org/split-GGUF/s-Q4_0-00001-of-00002.gguf")
    return d


def catalog_of(models_dir: str, tmp_path: object, **kw: object) -> Catalog:
    return Catalog(models_dir, parser=arg_parser(env=False), cache=str(tmp_path / "cache"), **kw)


# ----------------------------------------------------------------------
# The catalog
# ----------------------------------------------------------------------


def test_catalog(models_dir: str, tmp_path: object, monkeypatch: pytest.MonkeyPatch,
                 caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    c = catalog_of(models_dir, tmp_path)
    entries = c.scan()
    assert sorted(entries) == ["org/a-GGUF:Q4_0", "org/a-GGUF:Q8_0", "org/vl-GGUF:Q4_K_M", "top"]
    a = entries["org/a-GGUF:Q4_0"]
    assert a.names == ["org/a-GGUF/a-Q4_0.gguf", "a-Q4_0.gguf"] and a.aliases == [] and a.mmproj is None
    assert a.header["n_embd"] == 64
    assert c.find("org/a-GGUF/a-Q4_0.gguf") is a and c.find("a-Q4_0.gguf") is a and c.find(a.path) is a
    assert c.find("org/a-GGUF:Q2_K") is None
    vl = entries["org/vl-GGUF:Q4_K_M"]
    assert os.path.basename(vl.mmproj) == "mmproj-model-f16.gguf"  # (the shortest name)
    skipped = caplog.text
    assert "bad-IQ2_XXS.gguf (type16 tensors: no tile format)" in skipped and "architecture 'qwen3'" in skipped
    assert "s-Q4_0-00001-of-00002.gguf (a split file)" in skipped and "mtp-" not in skipped
    assert c.argv(vl)[:6] == ["-m", vl.path, "-a", vl.id, "--mmproj", vl.mmproj]
    # the header cache: a second catalog reads no header, a changed file again
    read: list[str] = []
    real = catalog_module.header
    monkeypatch.setattr(catalog_module, "header", lambda p: read.append(p) or real(p))
    assert sorted(catalog_of(models_dir, tmp_path).scan()) == sorted(entries) and read == []
    write_gguf(a.path, n_embd=128)
    assert catalog_of(models_dir, tmp_path).scan()["org/a-GGUF:Q4_0"].header["n_embd"] == 128 and read == [a.path]
    # two files of one quantization in a directory: ids by file name
    write_gguf(f"{models_dir}/org/a-GGUF/b-Q4_0.gguf")
    assert {"org/a-GGUF:a-Q4_0", "org/a-GGUF:b-Q4_0"} <= set(catalog_of(models_dir, tmp_path).scan())


def test_preset(models_dir: str, tmp_path: object) -> None:
    preset = tmp_path / "models.ini"
    extra = write_gguf(str(tmp_path / "elsewhere" / "extra-Q4_0.gguf"))
    preset.write_text(f"""version = 1
; every model
[*]
c = 8192
no-warmup = true
ARTLITE_ARG_TEMP = 0.5

[org/a-GGUF:Q4_0]
ctx-size = 4096
alias = small, tiny
load-on-startup = true
kv-k = hilo

[org/vl-GGUF/vl-1B-Q4_K_M.gguf]
mmproj =
reasoning = on
chat-template-kwargs = {{"enable_thinking": false}}

[extra]
model = {extra}
cache-prompt = false
""")
    c = catalog_of(models_dir, tmp_path, preset=str(preset), options={"temp": 0.25})
    entries = c.scan()
    a = entries["org/a-GGUF:Q4_0"]
    assert a.options == {"ctx_size": 4096, "warmup": False, "temp": 0.25, "kv_k": "hilo"}  # (the command line's temp)
    assert a.load_on_startup and c.find("tiny") is a and c.find("small") is a
    assert c.argv(a)[4:] == ["--ctx-size", "4096", "--no-warmup", "--temp", "0.25", "--kv-k", "hilo"]
    vl = entries["org/vl-GGUF:Q4_K_M"]
    assert vl.mmproj is None and vl.options["reasoning"] == "on"
    assert "--chat-template-kwargs" in c.argv(vl) and '{"enable_thinking": false}' in c.argv(vl)
    e = entries["extra"]
    assert e.source == "preset" and e.path == extra and e.options["cache_prompt"] is False
    assert entries["org/a-GGUF:Q8_0"].options == {"ctx_size": 8192, "warmup": False, "temp": 0.25}
    p = arg_parser(env=False)
    for text, error in [("[*]\nport = 9000\n", "the router's option"), ("[*]\nctx = 1\n", "no such option"),
                        ("[*]\nkv-k = big\n", "one of single, hilo"), ("[*]\nc = many\n", "invalid literal"),
                        ("x = 1\n[*]\n", "keys before the first section"), ("[*]\nwarmup = maybe\n", "true or false")]:
        preset.write_text(text)
        with pytest.raises(ValueError, match=error):
            read_preset(str(preset), p)
    assert render({"slots": False, "no_prefill": True, "verbose": False, "seed": 7}, p) == \
        ["--no-slots", "--no-prefill", "--seed", "7"]


# ----------------------------------------------------------------------
# The router
# ----------------------------------------------------------------------


@contextlib.contextmanager
def live_router(models_dir: str, tmp_path: object, *, api_keys: frozenset[str] = frozenset(), ui: bool = False,
                **kw: object) -> Iterator[tuple[str, Router]]:
    """The router of ``models_dir`` under uvicorn on a free port (fake model servers; ``ui``: a web UI without files):
    its URL and the router; at the end every model server has ended."""
    router = Router(catalog_of(models_dir, tmp_path), command=CHILD, **kw)
    app = create_router_app(router, RouterOptions(api_keys=api_keys, ui=(lambda app: None) if ui else None))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    sock = listen(config)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", router
    finally:
        router.close()
        server.should_exit = True
        thread.join(30)
        router.wait()
        assert all(not c.alive for c in router.children)


class Feed:
    """The events of GET /models/sse, collected by a thread."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.events: list[dict] = []
        self._client = httpx.Client(timeout=None)
        self._thread = threading.Thread(target=self._read, args=(url, headers or {}), daemon=True)
        self._thread.start()
        while not self.connected:
            time.sleep(0.01)

    connected = False

    def _read(self, url: str, headers: dict[str, str]) -> None:
        with contextlib.suppress(httpx.HTTPError), \
                self._client.stream("GET", f"{url}/models/sse", headers=headers) as r:
            self.connected = True
            for line in r.iter_lines():
                if line.startswith("data: "):
                    self.events.append(json.loads(line[6:]))

    def statuses(self, model: str) -> list[str]:
        return [e["data"]["status"] for e in self.events if e["model"] == model and "progress" not in e["data"]]

    def wait(self, model: str, status: str, timeout: float = 30) -> None:
        t0 = time.monotonic()
        while status not in self.statuses(model):
            assert time.monotonic() - t0 < timeout, (model, status, self.events)
            time.sleep(0.02)


def ask(url: str, model: str | None, text: str = "hi", **body: object) -> httpx.Response:
    req = {"messages": [{"role": "user", "content": text}], **body} | ({"model": model} if model else {})
    return httpx.post(f"{url}/v1/chat/completions", json=req, timeout=60)


def reply(r: httpx.Response) -> str:
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["message"]["content"]


def status(url: str, model: str) -> dict:
    return next(m["status"] for m in httpx.get(f"{url}/v1/models").json()["data"] if m["id"] == model)


def test_router(models_dir: str, tmp_path: object) -> None:
    """Listing, autoload, switching models (the idle one stops), the feed, explicit load and unload, the model's own
    routes, a vision encoder's model."""
    A, B, VL = "org/a-GGUF:Q4_0", "org/a-GGUF:Q8_0", "org/vl-GGUF:Q4_K_M"
    with live_router(models_dir, tmp_path, ui=True) as (url, router):
        props = httpx.get(f"{url}/props", params={"autoload": "false"}).json()
        assert props["role"] == "router" and props["max_instances"] == 1 and props["models_autoload"] is True
        listing = httpx.get(f"{url}/v1/models").json()
        assert [m["id"] for m in listing["data"]] == [A, B, VL, "top"] == [m["model"] for m in listing["models"]]
        vl = next(m for m in listing["data"] if m["id"] == VL)
        assert vl["architecture"]["input_modalities"] == ["text", "image"] and vl["status"]["value"] == "unloaded"
        assert "--mmproj" in vl["status"]["args"] and vl["meta"]["n_ctx"] == 8192 and vl["context_length"] == 8192
        feed = Feed(url)
        assert reply(ask(url, A)) == "a-Q4_0"  # (autoload)
        assert feed.statuses(A) == ["loading", "loaded"] and status(url, A)["value"] == "loaded"
        r = httpx.get(f"{url}/props", params={"model": A, "autoload": "false"}).json()
        assert r["model_alias"] == A and r["default_generation_settings"]["n_ctx"] == 8192
        assert r["ui"] is True  # (the router's web UI: a model's server has none, and the UI hides "ui": false)
        assert httpx.get(f"{url}/props", params={"model": B, "autoload": "false"}).status_code == 400
        assert httpx.post(f"{url}/tokenize", json={"model": A, "content": "a b"}).json()["tokens"]
        assert reply(ask(url, None)) == "a-Q4_0"  # (no model: the one running)
        # requests on a kept-alive connection, the router's and a model's (no wait for delayed ACKs: TCP_NODELAY)
        with httpx.Client() as client:
            for path, body in [("/v1/models", None), ("/tokenize", {"model": A, "content": "a"})]:
                times = []
                for _ in range(8):
                    t0 = time.perf_counter()
                    r = client.request("POST" if body else "GET", url + path, json=body)
                    assert r.status_code == 200
                    times.append(time.perf_counter() - t0)
                assert sorted(times)[4] < 0.02, (path, times)
        # another model: the idle one stops first (one at a time), through the spare process
        spare = router.spare
        assert spare is not None and spare.alive
        assert reply(ask(url, "org/a-GGUF/a-Q8_0.gguf")) == "a-Q8_0"  # (by its path)
        assert router.instances[B].child is spare
        feed.wait(A, "unloaded")
        assert feed.statuses(A) == ["loading", "loaded", "unloaded"] and list(router.instances) == [B]
        assert "exit_code" not in next(e for e in feed.events if e["model"] == A and e["data"]["status"] == "unloaded"
                                       )["data"]
        assert ask(url, None).status_code == 200  # (the one running: B)
        # explicit unload and load
        assert httpx.post(f"{url}/models/unload", json={"model": A}).status_code == 400  # (not running)
        assert httpx.post(f"{url}/models/unload", json={"model": B}).json() == {"success": True}
        feed.wait(B, "unloaded")
        assert ask(url, None).status_code == 400  # (several models, none running: name one)
        assert httpx.post(f"{url}/models/load", json={"model": "nope"}).status_code == 404
        assert httpx.post(f"{url}/models/load", json={"model": VL}).json() == {"success": True}
        assert httpx.post(f"{url}/models/load", json={"model": VL}).status_code == 400  # (already running)
        feed.wait(VL, "loaded")
        assert reply(ask(url, VL)) == "vl-1B-Q4_K_M mmproj"  # (its vision encoder: the server's --mmproj)
        r = ask(url, "org/a-GGUF:Q2_K")
        assert r.status_code == 400 and "not found" in r.json()["error"]["message"]
        # without autoload
        router.autoload = False
        assert ask(url, A).json()["error"]["message"] == "model is not loaded"
        r = httpx.post(f"{url}/v1/chat/completions?autoload=true", json={"model": A, "messages": [
            {"role": "user", "content": "x"}]}, timeout=60)
        assert reply(r) == "a-Q4_0"


def test_router_failure(models_dir: str, tmp_path: object) -> None:
    """A model that fails to load: the request's error, its status, its feed event; the next model loads. An unload
    during a load (the UI's cancel): the request waiting for it fails, the model is unloaded (not failed)."""
    write_gguf(f"{models_dir}/org/f-GGUF/fail-Q4_0.gguf")
    write_gguf(f"{models_dir}/org/w-GGUF/waitload-Q4_0.gguf")
    F, W = "org/f-GGUF:Q4_0", "org/w-GGUF:Q4_0"
    with live_router(models_dir, tmp_path, spare=False) as (url, router):
        out: dict[str, httpx.Response] = {}
        t = threading.Thread(target=lambda: out.setdefault("r", ask(url, W)))
        t.start()
        while W not in router.instances:
            time.sleep(0.01)
        assert httpx.post(f"{url}/models/unload", json={"model": W}).json() == {"success": True}
        t.join(30)
        r = out["r"]
        assert r.status_code == 500 and "its load was cancelled (unload)" in r.json()["error"]["message"], r.text
        while W in router.instances:
            time.sleep(0.05)
        assert status(url, W) == {"value": "unloaded", "args": status(url, W)["args"]}
        feed = Feed(url)
        r = ask(url, F)
        assert r.status_code == 500 and "no kernels for it" in r.json()["error"]["message"], r.text
        st = status(url, F)
        assert st["value"] == "unloaded" and st["failed"] is True and st["exit_code"] == 1
        assert "no kernels for it" in st["error"]
        feed.wait(F, "unloaded")
        assert next(e for e in feed.events if e["model"] == F and e["data"]["status"] == "unloaded")["data"][
            "exit_code"] == 1
        assert reply(ask(url, "org/a-GGUF:Q4_0")) == "a-Q4_0" and router.spare is None


def test_router_busy(models_dir: str, tmp_path: object) -> None:
    """A model is not stopped under a request: the next model's load waits for it; the UI's Stop and a client that
    goes away reach the model's server."""
    write_gguf(f"{models_dir}/org/s-GGUF/slow-Q4_0.gguf")
    S, A = "org/s-GGUF:Q4_0", "org/a-GGUF:Q4_0"
    with live_router(models_dir, tmp_path) as (url, router):
        assert reply(ask(url, S, max_tokens=2)).startswith("slow-Q4_0")
        out: dict[str, object] = {}
        long = threading.Thread(target=lambda: out.setdefault("slow", reply(ask(url, S, max_tokens=200))))
        long.start()
        while router.instances[S].requests == 0:
            time.sleep(0.01)
        t = threading.Thread(target=lambda: out.setdefault("a", (reply(ask(url, A)), time.monotonic())))
        t.start()
        long.join(60)
        done = time.monotonic()
        t.join(60)
        assert str(out["slow"]).startswith("slow-Q4_0 w0 w1") and out["a"][0] == "a-Q4_0" and out["a"][1] > done
        # the UI's Stop of a streamed request (its X-Conversation-Id), at the model's server
        assert reply(ask(url, S, max_tokens=1))
        headers = {"X-Conversation-Id": "conv-1::" + S}
        body = {"model": S, "messages": [{"role": "user", "content": "go"}], "stream": True}
        with httpx.Client(timeout=30) as client, client.stream("POST", f"{url}/v1/chat/completions", json=body,
                                                               headers=headers) as r:
            lines = r.iter_lines()
            next(line for line in lines if '" w' in line)
            stop = httpx.delete(f"{url}/v1/stream", params={"conv_id": headers["X-Conversation-Id"]}).json()
            assert stop["cancelled"] == 1
            rest = list(lines)
        assert any("the request was cancelled" in line for line in rest) and "data: [DONE]" not in rest
        # a client that goes away cancels its generation at the model's server
        with httpx.Client(timeout=30) as client, client.stream("POST", f"{url}/v1/chat/completions",
                                                               json=body) as r:
            next(line for line in r.iter_lines() if '" w' in line)
        port = router.instances[S].port
        key = {"Authorization": f"Bearer {router.instances[S].child.key}"}
        t0 = time.monotonic()
        while httpx.get(f"http://127.0.0.1:{port}/slots", headers=key).json()[0]["is_processing"]:
            assert time.monotonic() - t0 < 2  # (the reply: 10 s)
            time.sleep(0.02)
        assert router.instances[S].requests == 0


def test_router_several(models_dir: str, tmp_path: object) -> None:
    """--models-max 2: two models loaded; a third stops the least recently used; the router's API key, the model
    servers' own."""
    A, B, VL = "org/a-GGUF:Q4_0", "org/a-GGUF:Q8_0", "org/vl-GGUF:Q4_K_M"
    keys = frozenset({"secret"})
    with live_router(models_dir, tmp_path, models_max=2, api_keys=keys) as (url, router):
        auth = {"Authorization": "Bearer secret"}
        assert httpx.get(f"{url}/v1/models").status_code == 401 and httpx.get(f"{url}/health").status_code == 200

        def ask_(model: str) -> str:
            r = httpx.post(f"{url}/v1/chat/completions", json={"model": model, "messages": [
                {"role": "user", "content": "hi"}]}, headers=auth, timeout=60)
            return reply(r)

        assert ask_(A) == "a-Q4_0" and ask_(B) == "a-Q8_0" and sorted(router.instances) == [A, B]
        port = router.instances[A].port
        assert httpx.get(f"http://127.0.0.1:{port}/props", headers=auth).status_code == 401  # (its own key)
        assert ask_(A) == "a-Q4_0"  # (B is now the least recently used)
        assert ask_(VL).startswith("vl-1B") and sorted(router.instances) == [A, VL]


def test_router_reload(models_dir: str, tmp_path: object) -> None:
    """GET /v1/models?reload=1: a new file is listed; a running model whose file changed stops; the feed says so."""
    A = "org/a-GGUF:Q4_0"
    with live_router(models_dir, tmp_path, spare=False) as (url, router):
        feed = Feed(url)
        assert reply(ask(url, A)) == "a-Q4_0"
        write_gguf(f"{models_dir}/org/n-GGUF/new-Q4_0.gguf")
        write_gguf(router.catalog.entries[A].path, n_embd=128)
        ids = [m["id"] for m in httpx.get(f"{url}/v1/models", params={"reload": "1"}).json()["data"]]
        assert "org/n-GGUF:Q4_0" in ids
        feed.wait(A, "unloaded")
        assert any(e["event"] == "models_reload" for e in feed.events)


# ----------------------------------------------------------------------
# The fake model servers (this file run with --child)
# ----------------------------------------------------------------------


def fake_child() -> None:
    """``models.server.child`` with test_server's fake model: it replies with its file's name (and " mmproj" with a
    vision encoder); ``fail-*`` fails to load, ``waitload-*`` takes 3 s, ``slow-*`` replies with 2000 more words, 5 ms
    each."""
    from test_server import FakeModel, FakeNpu

    from models.server import child, main

    class FileModel(FakeModel):
        def __init__(self, path: str, options: dict):
            super().__init__()
            self.file, self.options = os.path.basename(path).removesuffix(".gguf"), options

        def load(self, *, capacity: int, prefill: bool = True) -> FakeNpu:
            slow = self.file.startswith("slow")
            npu = FakeNpu(self, capacity=capacity, delay=0.005 if slow else 0.0)
            words = [self.file] + [" mmproj"] * ("mmproj" in self.options)
            words += [f" w{i % 100}" for i in range(2000 * slow)]
            npu.replies = [words] * 1000
            return npu

    def open_model(path: str, **options: object) -> FileModel:
        if os.path.basename(path).startswith("fail"):
            raise NotImplementedError(f"{os.path.basename(path)}: no kernels for it")
        if os.path.basename(path).startswith("waitload"):
            time.sleep(3)
        return FileModel(path, options)

    main.open_model = open_model
    child.main()


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        fake_child()
        sys.exit()
    a = argparse.ArgumentParser(description="Serve fake models with the web UI in router mode (a manual check)")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--path", help="the UI's files (default: the pinned build)")
    args = a.parse_args()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("org/a-GGUF/a-1B-Q4_0.gguf", "org/a-GGUF/a-1B-Q8_0.gguf", "org/s-GGUF/slow-3B-Q4_K_M.gguf",
                     "org/f-GGUF/fail-1B-Q4_0.gguf"):
            write_gguf(f"{tmp}/{name}")
        d = args.path or ui.fetch()
        r = Router(Catalog(tmp, parser=arg_parser(env=False), cache=tmp), command=CHILD)
        uvicorn.run(create_router_app(r, RouterOptions(ui=lambda app: ui.mount(app, d))), port=args.port,
                    lifespan="on")
        r.close()
        r.wait()
