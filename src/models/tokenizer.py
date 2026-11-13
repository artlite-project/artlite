"""The GPT-2 BPE tokenizer and the chat template of a GGUF file, from its own metadata: the qwen35 architecture's files
(Qwen3.5 on), the llama package's qwen3 ones (Qwen3, its embedding, reranker and decision models) and LFM3's.

``tokenizer.ggml.model == "gpt2"``: GPT-2 byte-level BPE, run by gigatoken (its native API) from a tokenizer.json built
from the file's vocabulary and merges (``gguf_model.bpe_tokenizer_json``). The split regex is the tokenizer.json form of
llama.cpp's for the file's pre-tokenizer (``src/llama-vocab.cpp``, ``LLAMA_VOCAB_PRE_TYPE_QWEN2`` / ``_QWEN35``: Qwen3.5
also keeps marks with the letters; ``lfm2``: Llama 3's split, a word in the vocabulary one token whatever the merges
say); control and user-defined tokens (``<|im_start|>``, ``<think>``, ...) written in a text become their ids, as
llama.cpp's ``parse_special``. No BOS (LFM3's chat template writes its own). Checked against ``llama-tokenize``
(tests/test_tokenizer.py).
"""

import gigatoken
import numpy as np

from artlite.gguf_model import GGUFFile, bpe_tokenizer_json

from .templating import render_chat

# the split regexes of the tokenizer.json files by ``tokenizer.ggml.pre`` (llama.cpp spells the case-insensitive
# contractions out)
PATTERNS = {"qwen35": (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                       r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
                       r"|\p{N}"
                       r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
                       r"|\s*[\r\n]+"
                       r"|\s+(?!\S)"
                       r"|\s+"),
            "qwen2": (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                      r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                      r"|\p{N}"
                      r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                      r"|\s*[\r\n]+"
                      r"|\s+(?!\S)"
                      r"|\s+"),
            "lfm2": (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                     r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                     r"|\p{N}{1,3}"
                     r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                     r"|\s*[\r\n]+"
                     r"|\s+(?!\S)"
                     r"|\s+")}
IGNORE_MERGES = {"lfm2"}  # (llama.cpp's ``ignore_merges``: a word in the vocabulary is one token)
T_NORMAL, T_CONTROL, T_USER = 1, 3, 4  # tokenizer.ggml.token_type (5: unused, [PAD...]: not in the tokenizer)


class BpeTokenizer:
    """``encode`` / ``decode`` as llama.cpp tokenizes for this vocabulary (special tokens parsed)."""

    def __init__(self, gguf: GGUFFile):
        pre = gguf.meta.get("tokenizer.ggml.pre")
        if pre not in PATTERNS:
            raise NotImplementedError(f"{gguf.path}: pre-tokenizer {pre!r} (supported: {', '.join(PATTERNS)})")
        self.tokenizer = gigatoken.Tokenizer.from_json(bpe_tokenizer_json(gguf, PATTERNS[str(pre)],
                                                                          ignore_merges=pre in IGNORE_MERGES))
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
                tools: list[dict[str, object]] | None = None, preserve_thinking: bool = False,
                template: str | None = None, **variables: object) -> str:
    """The prompt text of a conversation, rendered with the file's ``tokenizer.chat_template`` (``template``: another
    in its place), up to the assistant's turn: ``<|im_start|>assistant\\n<think>\\n`` (thinking), or with an empty
    thought ``<think>\\n\\n</think>\\n\\n``; ``tools``: the functions the model may call; ``preserve_thinking``:
    the answers before the last user message keep their thoughts (Qwen3.6's template and later; Qwen3.5's and Qwen3's
    drop them); ``variables``: the template's others (``reasoning_effort``: Qwen3.8's), over these."""
    return render_chat(gguf, messages, tools=tools, template=template,
                       **{"enable_thinking": enable_thinking, "preserve_thinking": preserve_thinking, **variables})
