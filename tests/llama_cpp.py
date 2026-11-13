"""llama.cpp's CPU build (``model_config.LLAMA_CPP_DIR``, ``ARTLITE_LLAMA_CPP_DIR``) for the references: its files
(a test without the build skips), the identity of its library (a rebuild invalidates what it computed), its
tokenizer (``llama-tokenize``), its embeddings (``llama-embedding``) and a decision model's answers (``llama-server``'s
``/v1/systemone``)."""

import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request

import pytest

from artlite.model_config import LLAMA_CPP_DIR


def llama_cpp_bin(name: str) -> str:
    """A file of the llama.cpp CPU build (``build/bin/<name>``), or skip the test without one."""
    path = os.path.join(LLAMA_CPP_DIR, "build", "bin", name)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing: no llama.cpp CPU build (ARTLITE_LLAMA_CPP_DIR)")
    return path


def lib_identity(name: str = "libllama.so") -> str:
    """The identity of one of the llama.cpp build's libraries (``libmtmd.so``: its media library): what the goldens come
    from (a rebuild invalidates them)."""
    lib = os.path.realpath(llama_cpp_bin(name))
    return f"{lib}:{os.path.getsize(lib)}:{int(os.path.getmtime(lib))}"


def llama_tokenize(model: str, text: str) -> list[int]:
    """llama.cpp's token ids of a text (``llama-tokenize``: special tokens parsed, no BOS)."""
    exe = llama_cpp_bin("llama-tokenize")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        out = subprocess.run([exe, "-m", model, "-f", path, "--ids", "--log-disable", "--no-bos", "--no-escape"],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    return json.loads(out.strip().splitlines()[-1])


def llama_embeddings(model: str, texts: list[str], pooling: str) -> list[list[float]]:
    """llama.cpp's embeddings of texts on the CPU (``llama-embedding``: tokens with the vocabulary's BOS / EOS, the
    pooling's rows of the output-normed hidden states, not normalized), one per text."""
    exe = llama_cpp_bin("llama-embedding")
    sep = "<#artlite#>"
    out = subprocess.run([exe, "-m", model, "-ngl", "0", "--pooling", pooling, "--embd-normalize", "-1",
                          "--embd-output-format", "json", "-c", "4096", "-b", "4096", "-ub", "4096",
                          "--embd-separator", sep, "-p", sep.join(texts)],
                         capture_output=True, text=True, check=True).stdout
    return [d["embedding"] for d in json.loads(out[out.index("{"):])["data"]]


def llama_server_decisions(model: str, mmproj: str, bodies: list[dict], args: tuple[str, ...] = ()) -> list[dict]:
    """llama-server's ``/v1/systemone`` responses to request bodies on the CPU (``-ngl 0``, the vision encoder ``mmproj``
    on the CPU too; one slot: each question alone, as it reads them); ``args``: more options."""
    exe = llama_cpp_bin("llama-server")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    p = subprocess.Popen([exe, "-m", model, "--mmproj", mmproj, "--port", str(port), "-np", "1", "-c", "8192", "-b", "8192",
                          "-ub", "8192", "-ngl", "0", "--no-mmproj-offload", "--no-webui", *args],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(url + "/health") as r:
                    if json.load(r).get("status") == "ok":
                        break
            except OSError:
                pass
            if p.poll() is not None:
                raise RuntimeError(f"llama-server exited ({p.returncode})")
            time.sleep(0.5)
        out = []
        for body in bodies:
            req = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                out.append(json.load(r))
        return out
    finally:
        p.terminate()
        p.wait(60)
