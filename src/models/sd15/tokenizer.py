"""The CLIP tokenizer of Stable Diffusion 1.x (openai/clip-vit-large-patch14: ``tokenizer/vocab.json`` and
``merges.txt``), as Hugging Face's ``CLIPTokenizer`` tokenizes.

``<|startoftext|>`` and ``<|endoftext|>`` written in the text are their tokens. The text between them is
NFC-normalized and lowercased, then split into words: the contractions ``'s 't 're 've 'm 'll 'd``, runs of letters
(Unicode category L), single numbers (category N) and runs of other non-space characters; whitespace separates
words and is dropped. Each word's UTF-8 bytes, mapped to
GPT-2's byte characters, go through BPE with ``</w>`` marking the word's end. A prompt's ids: ``<|startoftext|>``,
at most 75 word tokens, ``<|endoftext|>``, padded with ``<|endoftext|>`` to 77. Checked against ``CLIPTokenizer``
(tests/rtc/test_sd15_tokenizer.py).
"""

import functools
import itertools
import json
import os
import re
import unicodedata

SPECIAL = ("<|startoftext|>", "<|endoftext|>")
CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
LENGTH = 77  # the text encoder's positions
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")
# characters str.isspace() counts as whitespace that Unicode's White_Space property does not
_NOT_SPACE = frozenset("\x1c\x1d\x1e\x1f")


@functools.cache
def byte_chars() -> tuple[str, ...]:
    """GPT-2's printable character of each byte value."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    return tuple(chars[b] for b in range(256))


def _kind(c: str) -> int:
    """0: whitespace, 1: letter, 2: number, 3: other."""
    if c.isspace() and c not in _NOT_SPACE:
        return 0
    cat = unicodedata.category(c)[0]
    return 1 if cat == "L" else 2 if cat == "N" else 3


def words(text: str) -> list[str]:
    """The words of a normalized text without special tokens (CLIP's split pattern)."""
    out, i, n = [], 0, len(text)
    while i < n:
        kind = _kind(text[i])
        if kind == 0:
            i += 1
            continue
        word = next((s for s in CONTRACTIONS if text.startswith(s, i)), None)
        if word is None:
            j = i + 1
            if kind != 2:  # (numbers one by one)
                while j < n and _kind(text[j]) == kind:
                    j += 1
            word = text[i:j]
        out.append(word)
        i += len(word)
    return out


class ClipTokenizer:
    """``encode(prompt)``: the 77 token ids of a prompt."""

    def __init__(self, directory: str):
        with open(os.path.join(directory, "vocab.json"), encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        with open(os.path.join(directory, "merges.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")
        merges = [line.split() for line in lines[lines[0].startswith("#version"):] if line]
        self.ranks = {(a, b): i for i, (a, b) in enumerate(merges)}
        self.bos, self.eos = self.vocab[SPECIAL[0]], self.vocab[SPECIAL[1]]
        self._cache: dict[str, list[int]] = {}

    def bpe(self, word: str) -> list[int]:
        """The token ids of a word (its byte characters)."""
        if word in self._cache:
            return self._cache[word]
        parts = [*word[:-1], word[-1] + "</w>"]
        while len(parts) > 1:
            rank, i = min((self.ranks.get(pair, 1 << 30), i) for i, pair in enumerate(itertools.pairwise(parts)))
            if rank == 1 << 30:
                break
            pair, merged, j = (parts[i], parts[i + 1]), [], 0
            while j < len(parts):  # every occurrence of the pair
                if j + 1 < len(parts) and (parts[j], parts[j + 1]) == pair:
                    merged.append(parts[j] + parts[j + 1])
                    j += 2
                else:
                    merged.append(parts[j])
                    j += 1
            parts = merged
        ids = self._cache[word] = [self.vocab[p] for p in parts]
        return ids

    def tokenize(self, text: str) -> list[int]:
        """The token ids of a text, without BOS, EOS and padding."""
        chars = byte_chars()
        ids = []
        for part in _SPECIAL_RE.split(text):
            if part in SPECIAL:
                ids.append(self.vocab[part])
                continue
            # (lowercase without str.lower's final sigma: Hugging Face lowercases character by character)
            for word in words(unicodedata.normalize("NFC", part).replace("Σ", "σ").lower()):
                ids += self.bpe("".join(chars[b] for b in word.encode("utf-8")))
        return ids

    def encode(self, text: str, length: int = LENGTH) -> list[int]:
        """BOS, the text's tokens (truncated), EOS, then EOS up to ``length`` ids."""
        ids = [self.bos, *self.tokenize(text)[:length - 2], self.eos]
        return ids + [self.eos] * (length - len(ids))
