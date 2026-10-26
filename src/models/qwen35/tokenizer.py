"""The qwen35 tokenizer and chat template, from the GGUF file's own metadata.

``tokenizer.ggml.model == "gpt2"`` with pre-tokenizer ``qwen35``: GPT-2 byte-level BPE, run by gigatoken (its native
API) from a tokenizer.json built from the file's vocabulary and merges (``gguf_model.bpe_tokenizer_json``). The split
regex is the tokenizer.json form of llama.cpp's (``src/llama-vocab.cpp``, ``LLAMA_VOCAB_PRE_TYPE_QWEN35``); control and
user-defined tokens (``<|im_start|>``, ``<think>``, ...) written in a text become their ids, as llama.cpp's
``parse_special``. No BOS. Checked against ``llama-tokenize`` (tests/qwen35/test_tokenizer.py).
"""

import gigatoken
import numpy as np

from artlite.gguf_model import GGUFFile, bpe_tokenizer_json

from ..common import chat_template

# the split regex of Qwen3.5's tokenizer.json (llama.cpp spells the case-insensitive contractions out)
QWEN35_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                  r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
                  r"|\p{N}"
                  r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+"
                  r"|\s+(?!\S)"
                  r"|\s+")
T_NORMAL, T_CONTROL, T_USER = 1, 3, 4  # tokenizer.ggml.token_type (5: unused, [PAD...]: not in the tokenizer)


class QwenTokenizer:
    """``encode`` / ``decode`` as llama.cpp tokenizes for this vocabulary (special tokens parsed)."""

    def __init__(self, gguf: GGUFFile):
        assert gguf.meta.get("tokenizer.ggml.pre") == "qwen35", gguf.meta.get("tokenizer.ggml.pre")
        self.tokenizer = gigatoken.Tokenizer.from_json(bpe_tokenizer_json(gguf, QWEN35_PATTERN))
        types = np.asarray(gguf.meta["tokenizer.ggml.token_type"])
        # the tokens decode_bytes writes: text and user-defined tokens; with special the control tokens too
        self._decoded = {False: np.isin(types, (T_NORMAL, T_USER)), True: np.isin(types, (T_NORMAL, T_USER, T_CONTROL))}
        self.eos = int(gguf.meta["tokenizer.ggml.eos_token_id"])

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).tolist()

    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes:
        """The UTF-8 bytes of the tokens (control tokens only when ``special``)."""
        ids = np.asarray(ids, np.int64)
        return self.tokenizer.decode(ids[self._decoded[special][ids]].astype(np.uint32))

    def decode(self, ids: list[int], *, special: bool = False) -> str:
        return self.decode_bytes(ids, special=special).decode("utf-8", errors="replace")


def chat_prompt(gguf: GGUFFile, messages: list[dict[str, object]], *, enable_thinking: bool = False,
                tools: list[dict[str, object]] | None = None) -> str:
    """The prompt text of a conversation, rendered with the file's ``tokenizer.chat_template``, up to the
    assistant's turn: ``<|im_start|>assistant\\n<think>\\n`` (thinking), or with an empty thought
    ``<think>\\n\\n</think>\\n\\n``; ``tools``: the functions the model may call."""
    return chat_template(str(gguf.meta["tokenizer.chat_template"]), messages, tools=tools,
                         enable_thinking=enable_thinking)
