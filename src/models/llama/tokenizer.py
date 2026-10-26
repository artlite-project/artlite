"""The Llama 3 tokenizer (gigatoken) and chat template.

The tokenizer is the ``tokenizer.json`` (Hugging Face) next to the model file, else the GGUF file's vocabulary (BPE
with Llama 3's split regex): both give the same ids. The chat template is the GGUF file's ``tokenizer.chat_template``.
"""

import json
import os

import gigatoken
import numpy as np

from artlite.gguf_model import GGUFFile, bpe_tokenizer_json

from ..common import chat_template

LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                  r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                  r"|\p{N}{1,3}"
                  r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+"
                  r"|\s+(?!\S)"
                  r"|\s+")
BOS = "<|begin_of_text|>"


class Tokenizer:
    """A gigatoken tokenizer (its native API), token ids as lists. Special tokens written in a text become
    their ids (as llama.cpp's ``parse_special``); ``special``: their ids by name."""

    def __init__(self, tokenizer: gigatoken.Tokenizer, special: dict[str, int]):
        self.tokenizer = tokenizer
        self.special = special
        self._special_ids = set(special.values())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).tolist()

    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes:
        """The UTF-8 bytes of the tokens (special tokens only when ``special``)."""
        if not special:
            ids = [i for i in ids if i not in self._special_ids]
        return self.tokenizer.decode(np.asarray(ids, np.uint32))

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids, special=True).decode("utf-8", errors="replace")


def load_tokenizer(model_path: str, tokenizer_path: str | None = None) -> Tokenizer:
    """The model's BPE tokenizer.

    Sources, first found: ``tokenizer_path`` or the ``tokenizer.json`` (Hugging Face) next to the model;
    the vocabulary of a GGUF file (``model_path``). For Llama 3 both give the same ids."""
    if tokenizer_path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "tokenizer.json")
        tokenizer_path = path if os.path.exists(path) else None
    if tokenizer_path is not None:
        with open(tokenizer_path, "rb") as f:
            data = f.read()
        special = {t["content"]: t["id"] for t in json.loads(data).get("added_tokens", [])}
        return Tokenizer(gigatoken.Tokenizer.from_json(data), special)
    if not model_path.endswith(".gguf"):
        raise FileNotFoundError(f"no tokenizer.json next to {model_path}, and it is not a GGUF file")
    g = GGUFFile(model_path)
    types = np.asarray(g.meta["tokenizer.ggml.token_type"])
    special = json.loads(g.strings_json("tokenizer.ggml.tokens", np.flatnonzero(types != 1)))
    # (ignore_merges: as Llama 3's tokenizer.json and llama.cpp's llama3 vocabulary)
    return Tokenizer(gigatoken.Tokenizer.from_json(bpe_tokenizer_json(g, LLAMA3_PATTERN, ignore_merges=True)), special)


def chat_prompt(gguf: GGUFFile, messages: list[dict[str, object]], *, date: str | None = None,
                tools: list[dict[str, object]] | None = None) -> str:
    """The prompt text of a conversation, rendered with the file's chat template (it begins with the BOS: encode it
    as it is) up to the assistant's turn; ``date`` ("26 Jul 2024") replaces today's date in the system header
    (reproducible prompts); ``tools``: the functions the model may call. The template takes one tool call per
    message: a message of several is rendered as one message per call."""
    messages = [one for m in messages
                for one in ([{**m, "tool_calls": [c]} for c in m["tool_calls"]] if len(m.get("tool_calls") or []) > 1
                            else [m])]
    return chat_template(str(gguf.meta["tokenizer.chat_template"]), messages, tools=tools, bos_token=BOS,
                         **({"date_string": date} if date else {}))
