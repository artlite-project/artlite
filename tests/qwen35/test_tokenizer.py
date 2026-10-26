"""The qwen35 tokenizer (models.qwen35.tokenizer) against llama.cpp's llama-tokenize: essays,
code, Unicode, special tokens, the chat template's prompts (no NPU)."""

import os

import pytest

from artlite.gguf_model import GGUFFile
from models.qwen35.tokenizer import QwenTokenizer, chat_prompt

from ..common import ESSAYS_DIR, essay
from ..llama_cpp import llama_tokenize
from .ref import qwen_gguf

TEXTS = [
    "Hello world, this is a test.\nNew line  two spaces",
    "  leading spaces and trailing  ",
    "\n\n\nmany\n\n\n\n\nnewlines\n\r\n",
    "tabs\tand\ttabs\t\t\tdone",
    "Unicode: naïve café — “quotes” … 日本語のテキスト 한국어 العربية עברית 😀🎉👍🏽 é ạ̈",
    "def f(x):\n    return x ** 2 + 3 * x - 6  # comment\n\nprint(f(10))\n",
    "Numbers 1234567890 3.14159 1e-10 0xDEADBEEF 100,000,000",
    "I'm sure they'll say it's DON'T and YOU'RE, we'd've",
    "<|im_start|>user\nFactorize x^2 + x - 6 = 0<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "A<|endoftext|>B<tool_call>C</tool_call>D<|vision_start|><|image_pad|><|vision_end|>",
    "x" * 300 + " " * 40 + "y" * 5,
    " ",
    "\n",
]
MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Factorize x^2 + x - 6 = 0"}]


@pytest.fixture(scope="module")
def tok() -> tuple[str, object]:
    path = qwen_gguf("Q4_K_M", "9B")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    return path, QwenTokenizer(GGUFFile(path))


def _texts() -> list[str]:
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    return list(TEXTS) + [essay(n) for n in names[:6]]


@pytest.mark.parametrize("i", range(len(_texts())))
def test_encode(tok: tuple[str, object], i: int) -> None:
    path, t = tok
    text = _texts()[i]
    ids = t.encode(text)
    assert ids == llama_tokenize(path, text)
    assert t.decode(ids, special=True) == text


@pytest.mark.parametrize("thinking", [False, True])
def test_chat_prompt(tok: tuple[str, object], thinking: bool) -> None:
    path, t = tok
    p = chat_prompt(GGUFFile(path), MESSAGES, enable_thinking=thinking)
    tail = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    assert p == ("<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nFactorize x^2 + x - 6 = 0<|im_end|>\n"
                 "<|im_start|>assistant\n" + tail)
    assert t.encode(p) == llama_tokenize(path, p)
