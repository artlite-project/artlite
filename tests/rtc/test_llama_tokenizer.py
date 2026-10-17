"""The llama tokenizer (models.llama3.tokenizer, gigatoken) from a GGUF file's
vocabulary against llama.cpp's llama-tokenize and against the model's tokenizer.json (no NPU)."""

import os

import pytest

from artlite.model_config import llama_gguf, llama_safetensors
from models.llama3.tokenizer import load_tokenizer

from .common import ESSAYS_DIR, essay
from .qwen35_ref import llama_tokenize

TEXTS = [
    "Hello world, this is a test.\nNew line  two spaces",
    "  leading spaces and trailing  ",
    "\n\n\nmany\n\n\n\n\nnewlines\n\r\n",
    "Unicode: naïve café — “quotes” … 日本語のテキスト 한국어 العربية עברית 😀🎉👍🏽 é ạ̈",
    "def f(x):\n    return x ** 2 + 3 * x - 6  # comment\n\nprint(f(10))\n",
    "Numbers 1234567890 3.14159 1e-10 0xDEADBEEF 100,000,000",
    "I'm sure they'll say it's DON'T and YOU'RE, we'd've",
    "<|start_header_id|>user<|end_header_id|>\n\nFactorize x^2 + x - 6 = 0<|eot_id|>",
    "x" * 300 + " " * 40 + "y" * 5,
]


def _texts() -> list[str]:
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    return list(TEXTS) + [essay(n) for n in names[:4]]


@pytest.fixture(scope="module")
def tok() -> tuple[str, object, object | None]:
    path = llama_gguf("Q4_K_M", "1B")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    hf = os.path.join(os.path.dirname(llama_safetensors("1B")), "tokenizer.json")
    return path, load_tokenizer(path), load_tokenizer(path, hf) if os.path.exists(hf) else None


@pytest.mark.parametrize("i", range(len(_texts())))
def test_encode(tok: tuple[str, object, object | None], i: int) -> None:
    path, t, hf = tok
    text = _texts()[i]
    ids = t.encode(text)
    assert ids == llama_tokenize(path, text)
    assert hf is None or hf.encode(text) == ids
    assert t.decode(ids) == text
