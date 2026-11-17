"""The GPT-2 BPE tokenizer and the chat template of a GGUF file, from its own metadata: the qwen35 architecture's files
(Qwen3.5 on), the llama package's qwen3 ones (Qwen3, its embedding, reranker and decision models), LFM3's and
Spark-X's.

``tokenizer.ggml.model == "gpt2"``: GPT-2 byte-level BPE, run by gigatoken (its native API) from a tokenizer.json built
from the file's vocabulary and merges (``gguf_model.bpe_tokenizer_json``). The split regex is the tokenizer.json form of
llama.cpp's for the file's pre-tokenizer (``src/llama-vocab.cpp``, ``LLAMA_VOCAB_PRE_TYPE_QWEN2`` / ``_QWEN35``: Qwen3.5
also keeps marks with the letters; ``lfm2``: Llama 3's split, a word in the vocabulary one token whatever the merges
say); control and user-defined tokens (``<|im_start|>``, ``<think>``, ...) written in a text become their ids, as
llama.cpp's ``parse_special``. A pre-tokenizer gigatoken has no scanner for (``SPLITS``: ``spark2_5``, four regexes
in sequence, a CJK run split before the main one) encodes natively: the special tokens cut out, the rest split as
llama.cpp's regexes split it (``_native.pretokenize_spark2_5`` over the text's codepoints and their classes,
``unicode_classes``: the categories the regexes collapse a codepoint to) and each piece merged
(``_native.BpeEncoder``: gigatoken always splits with a scheme of its own); gigatoken still decodes. No BOS (the chat
templates write their own). Checked against ``llama-tokenize`` (tests/test_tokenizer.py).
"""

import functools
import re
import unicodedata

import gigatoken
import numpy as np

from artlite import _native
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
SPLITS = {"spark2_5": _native.pretokenize_spark2_5}  # (the pre-tokenizers split natively: module docstring)
T_NORMAL, T_CONTROL, T_USER = 1, 3, 4  # tokenizer.ggml.token_type (5: unused, [PAD...]: not in the tokenizer)

# A codepoint's class for the native splits (``_native.pretokenize_spark2_5``'s PreClass): an ASCII one by the ranges
# llama.cpp's regexes give the categories (src/unicode.cpp's k_ucat_map; \s: tab .. CR and space), a non-ASCII one by
# its Unicode category, whitespace first (llama.cpp's unicode_set_whitespace)
C_OTHER, C_N, C_L, C_P, C_M, C_S, C_WS = range(7)
_ASCII = np.zeros(128, np.uint8)
for _c in range(128):
    _ch = chr(_c)
    _ASCII[_c] = (C_N if _ch.isdigit() else C_L if _ch.isalpha() else C_WS if _c in (9, 10, 11, 12, 13, 32)
                  else C_S if _ch in "$+<=>^`|~" else C_P if 0x21 <= _c <= 0x7E and not _ch.isalnum() else C_OTHER)
_WHITESPACE = frozenset((0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000))
_CATEGORY = {"N": C_N, "L": C_L, "P": C_P, "M": C_M, "S": C_S}


@functools.cache
def _class(c: int) -> int:
    return C_WS if c in _WHITESPACE else _CATEGORY.get(unicodedata.category(chr(c))[0], C_OTHER)


def unicode_classes(cps: np.ndarray) -> np.ndarray:
    """The classes (uint8) of codepoints (uint32)."""
    out = _ASCII[np.minimum(cps, 127)]
    hi = cps >= 128
    if hi.any():
        u, inv = np.unique(cps[hi], return_inverse=True)
        out[hi] = np.array([_class(int(c)) for c in u], np.uint8)[inv]
    return out


def special_ids(pieces: list[bytes], types: np.ndarray) -> list[int]:
    """The special tokens of a GGUF vocabulary (its tokens' bytes and ``tokenizer.ggml.token_type``): the control ones and
    the user-defined ones spelled ``<...>`` (Qwen's ``<tool_call>``, ``<think>``)."""
    return [int(i) for i in np.flatnonzero((types == T_CONTROL) | (types == T_USER))
            if types[i] == T_CONTROL or (pieces[i].startswith(b"<") and pieces[i].endswith(b">"))]


def texts_of(names: list[str]) -> re.Pattern[str]:
    """A pattern that finds any of the texts (longest first; none: never matches)."""
    return re.compile("|".join(re.escape(t) for t in sorted(names, key=len, reverse=True)) or "(?!)")


class BpeTokenizer:
    """``encode`` / ``decode`` as llama.cpp tokenizes for this vocabulary (special tokens parsed)."""

    def __init__(self, gguf: GGUFFile):
        pre = gguf.meta.get("tokenizer.ggml.pre")
        if pre not in PATTERNS and pre not in SPLITS:
            raise NotImplementedError(f"{gguf.path}: pre-tokenizer {pre!r} (supported: {', '.join([*PATTERNS, *SPLITS])})")
        self.tokenizer = gigatoken.Tokenizer.from_json(bpe_tokenizer_json(gguf, PATTERNS.get(str(pre)),
                                                                          ignore_merges=pre in IGNORE_MERGES))
        types = np.asarray(gguf.meta["tokenizer.ggml.token_type"])
        self.types = types
        self._plain_json = lambda: bpe_tokenizer_json(gguf, PATTERNS.get(str(pre)), ignore_merges=pre in IGNORE_MERGES,
                                                      special=False)
        self._plain: gigatoken.Tokenizer | None = None
        self._added: re.Pattern[str] | None = None
        self._split = SPLITS.get(str(pre))
        if self._split is not None:  # (the merges natively; the special tokens, longest first: llama.cpp's partition)
            names = gguf.strings("tokenizer.ggml.tokens")
            self._bpe = _native.BpeEncoder(names, gguf.strings("tokenizer.ggml.merges"))
            self._special_ids = {names[i]: int(i) for i in np.flatnonzero((types == T_CONTROL) | (types == T_USER))}
            self._special = re.compile("(" + "|".join(re.escape(t) for t in sorted(self._special_ids, key=len,
                                                                                    reverse=True)) + ")")
        # the tokens decode_bytes writes: text and user-defined tokens; with special the control tokens too
        self._decoded = {False: np.isin(types, (T_NORMAL, T_USER)), True: np.isin(types, (T_NORMAL, T_USER, T_CONTROL))}
        self.eos = int(gguf.meta["tokenizer.ggml.eos_token_id"])

    def encode(self, text: str) -> list[int]:
        if self._split is None:
            return self.tokenizer.encode(text).tolist()
        out: list[int] = []
        for k, part in enumerate(self._special.split(text)):
            if k % 2:
                out.append(self._special_ids[part])
            elif part:
                cps = np.frombuffer(part.encode("utf-32-le"), np.uint32)
                out += self._bpe.encode(cps, self._split(cps, unicode_classes(cps))).tolist()
        return out

    def encode_plain(self, text: str) -> list[int]:
        """Token ids of ``text`` with no special tokens (their text encodes as text)."""
        if self._split is not None:
            cps = np.frombuffer(text.encode("utf-32-le"), np.uint32)
            return self._bpe.encode(cps, self._split(cps, unicode_classes(cps))).tolist()
        if self._added is None:  # (the control and user-defined tokens' texts: gigatoken's added tokens)
            vocab = self.tokenizer.vocab
            self._added = texts_of([vocab[int(i)].decode("utf-8", "replace")
                                    for i in np.flatnonzero((self.types == T_CONTROL) | (self.types == T_USER))])
        if not self._added.search(text):
            return self.tokenizer.encode(text).tolist()
        if self._plain is None:  # (gigatoken parses special tokens: a vocabulary without them)
            self._plain = gigatoken.Tokenizer.from_json(self._plain_json())
        return self._plain.encode(text).tolist()

    def vocabulary(self) -> tuple[list[bytes], list[int]]:
        """Every token's bytes as a generation's text has them (a control token's text too) and the special tokens: the
        control ones and the user-defined ones spelled ``<...>`` (llguidance's llama.cpp adapter's rule)."""
        vocab = self.tokenizer.vocab
        pieces = [vocab.get(i, b"") for i in range(len(self.types))]
        return pieces, special_ids(pieces, self.types)

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
