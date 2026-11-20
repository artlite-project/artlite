"""The GGUF files' BPE tokenizer (models.tokenizer) against llama.cpp's llama-tokenize, on a file of each pre-tokenizer
(qwen35: Qwen3.5-9B, qwen2: Qwen3 0.6B, lfm2: LFM3-1.7B, spark2_5: Spark-X2.6-1.7B, split natively): essays, code,
Unicode (CJK runs between digits and Latin, full-width digits), special tokens, the chat template's
prompts; a template in place of the file's and the template's variables; LFM3's template (its ``generation`` block,
thinking, tool calls) as llama-server renders it (no NPU)."""

import os
from pathlib import Path

import jinja2
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import flint_gguf, lfm3_gguf, qwen3_gguf, qwen_gguf, spark_gguf
from models.templating import chat_template, effort_levels, template_option
from models.tokenizer import BpeTokenizer, chat_prompt

from .common import ESSAYS_DIR, essay
from .llama_cpp import llama_tokenize
from .model_files import require

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
    "中文123测试abc，日本語のテキストとカタカナ・ひらがな２０２６年 ①②③ ½ x²+y³ 4K-8K-32K v1.2.3-rc4",
    "混合text中文 ok？「引用」—— 価格は1,234円です。\u3000全角スペース\u00a0nbsp\u200bzwsp\u2028end",
    "'s 's .net .NET __init__ #include <stdio.h> $x+=1; a->b; ...abc !!!WOW i++ ---- ****",
    ("<｜start▁of▁sentence｜><|System|>\nsys<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>hi<｜end▁of▁sentence｜>"
     "<｜start▁of▁sentence｜><|Bot|><think>x</think><tool_call>f<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>"),
    "<|system|>sys<|user|>hi\n<|assistant|><think>t</think>a<|observation|><tool_response>{}</tool_response><|end_of_text|>",
    "camelCaseWord HTTPServer getHTTPResponse iPhone McDonald's ÀÉÎ ǅ x'S y'Ll z'VE \t\tq ;\n\n  \r\n x",
]
MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Factorize x^2 + x - 6 = 0"}]


ESSAYS = {"qwen35": 2, "qwen2": 1, "lfm2": 1, "spark2_5": 1, "flint": 1}  # (essays per pre-tokenizer: prose is plain
# English, the pre-tokenizers differ in what TEXTS holds; qwen35's regex is the long-text sensitive one)
_TOKENIZERS: dict[str, tuple[str, BpeTokenizer]] = {}


def _tok(pre: str) -> tuple[str, BpeTokenizer]:
    if pre not in _TOKENIZERS:
        path = {"qwen35": qwen_gguf("Q4_K_M", "9B"), "qwen2": qwen3_gguf("0.6B"), "lfm2": lfm3_gguf(),
                "spark2_5": spark_gguf("1.7B"), "flint": flint_gguf()}[pre]
        require(path)
        _TOKENIZERS[pre] = path, BpeTokenizer(GGUFFile(path))
    return _TOKENIZERS[pre]


def _texts(pre: str) -> list[str]:
    names = sorted(n for n in os.listdir(ESSAYS_DIR) if n.endswith(".txt")) if os.path.isdir(ESSAYS_DIR) else []
    return list(TEXTS) + [essay(n) for n in names[:ESSAYS[pre]]]


@pytest.mark.parametrize("pre,i", [(pre, i) for pre, n in ESSAYS.items() for i in range(len(TEXTS) + n)],
                         ids=lambda v: str(v))
def test_encode(pre: str, i: int) -> None:
    path, t = _tok(pre)
    texts = _texts(pre)
    if i >= len(texts):
        pytest.skip("no essay")
    text = texts[i]
    ids = t.encode(text)
    assert ids == llama_tokenize(path, text)
    assert t.decode(ids, special=True) == text


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("path", [qwen_gguf("Q4_K_M", "9B"), qwen3_gguf("0.6B")], ids=["qwen35", "qwen2"])
def test_chat_prompt(path: str, thinking: bool) -> None:
    require(path)
    t = BpeTokenizer(GGUFFile(path))
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
    require(path)
    g = GGUFFile(path)
    template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}|{{ enable_thinking }}|{{ x }}"
    assert chat_prompt(g, MESSAGES, template=template, enable_thinking=True) == \
        "<system>You are terse.<user>Factorize x^2 + x - 6 = 0|True|"
    assert chat_prompt(g, MESSAGES, template=template, x=3, enable_thinking=False).endswith("|False|3")
    assert chat_prompt(g, MESSAGES, add_vision_id=True) == chat_prompt(g, MESSAGES)  # (a variable it does not use)
    path = qwen_gguf("Q4_K_M", "3.8-27B")
    require(path)
    g = GGUFFile(path)
    assert "Reasoning effort is set to xhigh" in chat_prompt(g, MESSAGES, enable_thinking=True)
    assert "Reasoning effort is set to low" in chat_prompt(g, MESSAGES, enable_thinking=True, reasoning_effort="low")
    assert "Reasoning effort" not in chat_prompt(g, MESSAGES, reasoning_effort="low")  # (no thought)
    with pytest.raises(jinja2.TemplateError, match="Unexpected reasoning effort high"):
        chat_prompt(g, MESSAGES, enable_thinking=True, reasoning_effort="high")


def test_continue_and_efforts() -> None:
    """A continued last message: the conversation cut where its content (its text parts too) or its thought ends, in the
    template's own form (Qwen3.6's: the answer after the last user message keeps its thought); a template that does not
    render the text refuses. The reasoning_effort levels a template renders distinctly (llama.cpp's probe)."""
    path = qwen_gguf("Q4_K_M", "27B")
    require(path)
    t = str(GGUFFile(path).meta["tokenizer.chat_template"])
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "The ans ", "reasoning_content": "Hmm"}]
    head = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert chat_template(t, msgs, continue_final_message="content") == head + "Hmm\n</think>\n\nThe ans "
    assert chat_template(t, msgs, continue_final_message="reasoning_content") == head + "Hmm"
    parts = [msgs[0], {**msgs[1], "content": [{"type": "text", "text": "The ans"}]}]
    assert chat_template(t, parts, continue_final_message="content").endswith("</think>\n\nThe ans")
    with pytest.raises(ValueError, match="does not render the last message's reasoning_content"):
        chat_template("{{ messages[-1].content }}", msgs, continue_final_message="reasoning_content")
    assert effort_levels(t) == []
    path = qwen_gguf("Q4_K_M", "3.8-27B")
    require(path)
    assert effort_levels(str(GGUFFile(path).meta["tokenizer.chat_template"])) == ["low", "medium", "xhigh"]


LFM3_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}},
              {"type": "function", "function": {"name": "search", "description": "Search", "defer_loading": True,
                                                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


def test_lfm3_template() -> None:
    """LFM3's chat template (a ``{% generation %}`` block, the BOS, ``<think>`` opened or empty, an earlier thought kept
    with ``preserve_thinking``, tool calls in Qwen3-Coder's XML, deferred tools left out of the system prompt): the
    prompts llama-server's ``/apply-template`` renders (llama.cpp 80d2c0c2c; its ``preserve_reasoning`` default on)."""
    path = lfm3_gguf()
    require(path)
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
