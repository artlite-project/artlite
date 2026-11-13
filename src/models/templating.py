"""Chat templates: a model file's Jinja2 templates rendered in a sandbox as transformers and llama.cpp render them
(``chat_template``, ``render_template``; ``render_chat``: a conversation with the file's own), and the options that give
another (``template_option``)."""

import functools
import json
from datetime import datetime
from typing import ClassVar

import jinja2.ext
import jinja2.parser
import jinja2.sandbox

from artlite.gguf_model import GGUFFile

TEMPLATE_KEY = "tokenizer.chat_template"  # (the model file's chat template; its variants: ``.asr``, ``.tts``, ...)


def _raise(message: str) -> None:
    raise jinja2.TemplateError(message)


def _tojson(value: object, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    """The templates' ``tojson`` as transformers and llama.cpp render it: the keys in their order, no HTML escapes
    (Jinja2's own sorts the keys and escapes ``<>&'``)."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class _Generation(jinja2.ext.Extension):
    """``{% generation %} .. {% endgeneration %}``: transformers' marker of the assistant's tokens (its training masks;
    LFM3's template), rendered as its body."""

    tags: ClassVar[set[str]] = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> list[jinja2.nodes.Node]:
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


@functools.lru_cache(maxsize=16)
def _compiled(template: str) -> jinja2.Template:
    """A chat template compiled (~10 ms: once per template, not per request)."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols, _Generation])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().astimezone().strftime(fmt)
    return env.from_string(template)


def render_template(template: str, **variables: object) -> str:
    """A template of a model file's (``tokenizer.chat_template.<name>``) rendered as the chat templates are (Jinja2 in
    the sandbox, compiled once)."""
    return _compiled(template).render(**variables)


def chat_template(template: str, messages: list[dict[str, object]], *, tools: list[dict[str, object]] | None = None,
                  **variables: object) -> str:
    """The prompt text of a conversation rendered with a model's chat template (a GGUF file's
    ``tokenizer.chat_template``, Jinja2 in a sandbox, as transformers renders it) up to the assistant's turn;
    ``tools``: the functions the model may call (OpenAI's ``{"type": "function", "function": {...}}``; a tool call's
    ``arguments`` in the messages are objects); ``variables``: the template's own (``enable_thinking``, ``bos_token``,
    ``date_string``, ...)."""
    return _compiled(template).render(messages=messages, add_generation_prompt=True, tools=tools or None, **variables)


TEMPLATE_RESERVED = frozenset({"messages", "tools", "add_generation_prompt"})  # (no variables: the conversation's)


def check_chat_template(template: str, **variables: object) -> None:
    """ValueError unless a chat template renders a user message with ``variables`` (llama.cpp's check of a template
    it is given: a broken one fails at load, not at the first request)."""
    try:
        chat_template(template, [{"role": "user", "content": "test"}], **variables)
    except Exception as e:  # noqa: BLE001 (any failure of the template's)
        given = f" with {json.dumps(variables, ensure_ascii=False)}" if variables else ""
        raise ValueError(f"the chat template does not render a user message{given}: {type(e).__name__}: {e}") from None


def template_option(text: str | None, file: str | None) -> str | None:
    """The chat template of the options ``--chat-template`` (its text) or ``--chat-template-file`` (a file's text),
    as llama.cpp's, in place of the model file's; None: neither (empty: not given). OSError: the file does not open;
    ValueError: both given, or the template does not render (``check_chat_template``)."""
    if text and file:
        raise ValueError("--chat-template and --chat-template-file: give one of them")
    if file:
        with open(file, encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError(f"--chat-template-file {file}: the file is empty")
    if not text:
        return None
    try:
        check_chat_template(text)
    except ValueError as e:
        raise ValueError(f"--chat-template{'-file ' + file if file else ''}: {e}") from None
    return text


def file_template(gguf: GGUFFile, template: str | None = None) -> str:
    """The chat template a model renders with: ``template`` (another in the file's place), else the file's."""
    return template or str(gguf.meta[TEMPLATE_KEY])


def render_chat(gguf: GGUFFile, messages: list[dict[str, object]], *, tools: list[dict[str, object]] | None = None,
                template: str | None = None, **variables: object) -> str:
    """The prompt text of a conversation rendered with the file's chat template (``template``: another in its place) up
    to the assistant's turn (``chat_template``: ``tools``, ``variables``); the models' ``chat_prompt`` add their own
    variables and fix-ups."""
    return chat_template(file_template(gguf, template), messages, tools=tools, **variables)
