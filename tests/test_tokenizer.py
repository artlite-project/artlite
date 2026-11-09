"""The GGUF files' BPE tokenizer (models.tokenizer) against llama.cpp's llama-tokenize, on a file of each pre-tokenizer
(qwen35: Qwen3.5-9B, qwen2: Qwen3 0.6B, lfm2: LFM3-1.7B): essays, code, Unicode, special tokens, the chat template's
prompts; a template in place of the file's and the template's variables; LFM3's template (its ``generation`` block,
thinking, tool calls) as llama-server renders it (no NPU)."""

import os
from pathlib import Path

import jinja2
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import lfm3_gguf, qwen3_gguf, qwen_gguf
from models.common import template_option
from models.tokenizer import BpeTokenizer, chat_prompt

from .common import ESSAYS_DIR, essay
from .llama_cpp import llama_tokenize

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


@pytest.fixture(scope="module", params=["qwen35", "qwen2", "lfm2"])
def tok(request: pytest.FixtureRequest) -> tuple[str, object]:
    path = {"qwen35": qwen_gguf("Q4_K_M", "9B"), "qwen2": qwen3_gguf("0.6B"), "lfm2": lfm3_gguf()}[request.param]
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    return path, BpeTokenizer(GGUFFile(path))


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
    if GGUFFile(path).meta["tokenizer.ggml.pre"] == "lfm2":
        pytest.skip("LFM3's template: test_lfm3_template")
    p = chat_prompt(GGUFFile(path), MESSAGES, enable_thinking=thinking)
    opens = GGUFFile(path).meta["tokenizer.ggml.pre"] == "qwen35"  # (Qwen3's model writes its <think> itself)
    tail = ("<think>\n" if opens else "") if thinking else "<think>\n\n</think>\n\n"
    assert p == ("<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nFactorize x^2 + x - 6 = 0<|im_end|>\n"
                 "<|im_start|>assistant\n" + tail)
    assert t.encode(p) == llama_tokenize(path, p)


def test_template_option(tmp_path: Path) -> None:
    """--chat-template (the text) or --chat-template-file (a file's), checked by rendering a user message."""
    template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>{{ enable_thinking }}"
    f = tmp_path / "t.jinja"
    f.write_text(template)
    assert template_option(None, str(f)) == template_option(template, None) == template
    assert template_option(None, None) is None and template_option("", "") is None
    for text, file, error in [(template, str(f), "give one"), ("{% if %}", None, "TemplateSyntaxError"),
                              ("{{ raise_exception('no') }}", None, "no"), (None, str(tmp_path / "e"), "empty")]:
        if file and not os.path.exists(file):
            Path(file).write_text(" ")
        with pytest.raises(ValueError, match=error):
            template_option(text, file)
    with pytest.raises(OSError):
        template_option(None, str(tmp_path / "missing.jinja"))


def test_chat_prompt_template() -> None:
    """A template in place of the file's gets the model's variables (enable_thinking, ...), and the template's other
    variables go over them: Qwen3.8's reasoning_effort (xhigh, medium, low; another level raises)."""
    path = qwen3_gguf("0.6B")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}|{{ enable_thinking }}|{{ x }}"
    assert chat_prompt(g, MESSAGES, template=template, enable_thinking=True) == \
        "<system>You are terse.<user>Factorize x^2 + x - 6 = 0|True|"
    assert chat_prompt(g, MESSAGES, template=template, x=3, enable_thinking=False).endswith("|False|3")
    assert chat_prompt(g, MESSAGES, add_vision_id=True) == chat_prompt(g, MESSAGES)  # (a variable it does not use)
    path = qwen_gguf("Q4_K_M", "3.8-27B")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    assert "Reasoning effort is set to xhigh" in chat_prompt(g, MESSAGES, enable_thinking=True)
    assert "Reasoning effort is set to low" in chat_prompt(g, MESSAGES, enable_thinking=True, reasoning_effort="low")
    assert "Reasoning effort" not in chat_prompt(g, MESSAGES, reasoning_effort="low")  # (no thought)
    with pytest.raises(jinja2.TemplateError, match="Unexpected reasoning effort high"):
        chat_prompt(g, MESSAGES, enable_thinking=True, reasoning_effort="high")


LFM3_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}},
              {"type": "function", "function": {"name": "search", "description": "Search", "defer_loading": True,
                                                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


def test_lfm3_template() -> None:
    """LFM3's chat template (a ``{% generation %}`` block, the BOS, ``<think>`` opened or empty, an earlier thought kept
    with ``preserve_thinking``, tool calls in Qwen3-Coder's XML, deferred tools left out of the system prompt): the
    prompts llama-server's ``/apply-template`` renders (llama.cpp 80d2c0c2c; its ``preserve_reasoning`` default on)."""
    path = lfm3_gguf()
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    from models.lfm.lfm import Lfm
    m = Lfm(path)
    head = "<|startoftext|><|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    turn = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "think a"},
            {"role": "user", "content": "What is 2+2?"}]
    tail = "Hello!<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    for think in (True, False):
        end = "<think>" if think else "<think></think>"
        ids = m.chat(turn, think=think, preserve=True)
        assert m.tokenizer.decode(ids, special=True) == head + "<think>think a</think>" + tail + end
        assert ids == llama_tokenize(path, head + "<think>think a</think>" + tail + end)
        assert m.tokenizer.decode(m.chat(turn, think=think), special=True) == head + "<think></think>" + tail + end
    calls = [{"role": "user", "content": "Weather in Paris?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
             {"role": "tool", "content": "sunny"}]
    want = ('<|startoftext|><|im_start|>system\n<tools>\n{"name": "get_weather", "description": "Weather", "parameters": '
            '{"type": "object", "properties": {"city": {"type": "string"}}}}\n</tools><|im_end|>\n<|im_start|>user\n'
            'Weather in Paris?<|im_end|>\n<|im_start|>assistant\n<think></think><tool_call>\n<function=get_weather>\n'
            '<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>tool\n'
            '<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>assistant\n')
    for think in (True, False):
        ids = m.chat(calls, think=think, tools=LFM3_TOOLS, preserve=True)
        assert m.tokenizer.decode(ids, special=True) == want + ("<think>" if think else "<think></think>")
    assert m.encode("Hello") == llama_tokenize(path, "Hello")  # (no BOS: the file's add_bos_token)
