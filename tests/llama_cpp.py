"""llama.cpp's CPU build (``model_config.LLAMA_CPP_DIR``, ``ARTLITE_LLAMA_CPP_DIR``) for the references: its files
(a test without the build skips), the identity of its library (a rebuild invalidates what it computed) and its
tokenizer (``llama-tokenize``)."""

import json
import os
import subprocess
import tempfile

import pytest

from artlite.model_config import LLAMA_CPP_DIR


def llama_cpp_bin(name: str) -> str:
    """A file of the llama.cpp CPU build (``build/bin/<name>``), or skip the test without one."""
    path = os.path.join(LLAMA_CPP_DIR, "build", "bin", name)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing: no llama.cpp CPU build (ARTLITE_LLAMA_CPP_DIR)")
    return path


def _llama_lib() -> str:
    return llama_cpp_bin("libllama.so")


def _lib_identity() -> str:
    """The llama.cpp build the goldens come from (a rebuild invalidates them)."""
    lib = os.path.realpath(_llama_lib())
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
