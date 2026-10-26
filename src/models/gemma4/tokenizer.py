"""The Gemma 4 tokenizer and chat template, from the GGUF file's own metadata.

``tokenizer.ggml.model == "gemma4"`` is a SentencePiece-style BPE (llama.cpp
``LLAMA_VOCAB_PRE_TYPE_GEMMA4``): special tokens are cut out of the text, spaces become
U+2581, the text splits into runs of newlines and runs of anything else, and the merges
(lowest rank first, leftmost on ties) run on the UTF-8 characters of each run. A piece
that is not in the vocabulary falls back to its ``<0xXX>`` byte tokens. Checked against
``llama-tokenize``.
"""

import heapq
import re

from artlite.gguf_model import GGUFFile

from ..common import chat_template

SPACE = "▁"
T_UNKNOWN, T_CONTROL, T_USER, T_BYTE = 2, 3, 4, 6  # tokenizer.ggml.token_type (1: normal, 5: unused)
_RUNS = re.compile(r"[^\n]+|\n+")


class GemmaTokenizer:
    """``encode`` / ``decode`` as llama.cpp tokenizes for this vocabulary (with its ``parse_special``)."""

    def __init__(self, gguf: GGUFFile):
        assert gguf.meta.get("tokenizer.ggml.model") == "gemma4", gguf.meta.get("tokenizer.ggml.model")
        self.tokens = gguf.strings("tokenizer.ggml.tokens")
        self.types = gguf.meta["tokenizer.ggml.token_type"]
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.bos = int(gguf.meta["tokenizer.ggml.bos_token_id"])
        self._gguf = gguf
        self._ranks: dict[str, int] | None = None  # "left right" -> merge rank (built on first use)
        special = [i for i, t in enumerate(self.types) if t in (T_CONTROL, T_USER, T_UNKNOWN)]
        self.special = {self.tokens[i]: i for i in special}
        # longest first, as llama.cpp's special-token cache
        names = sorted(self.special, key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(n) for n in names))
        self._bytes = {i: int(t[3:5], 16) for i, t in enumerate(self.tokens) if self.types[i] == T_BYTE}

    def _merge_ranks(self) -> dict[str, int]:
        if self._ranks is None:
            # a merge "left right" (no literal space inside a piece: spaces are U+2581)
            self._ranks = {m: i for i, m in enumerate(self._gguf.strings("tokenizer.ggml.merges"))}
        return self._ranks

    def _bpe(self, word: str, out: list[int]) -> None:
        """The tokens of one run (no special tokens inside)."""
        ids = self.ids
        if word[0] == "\n" and word in ids:  # a run of newlines that is a token
            out.append(ids[word])
            return
        ranks = self._merge_ranks()
        sym = list(word)  # pieces; "" once merged into the left neighbour
        n = len(sym)
        prev = list(range(-1, n - 1))
        nxt = [*range(1, n), -1]
        heap: list[tuple[int, int, str, str]] = []

        def push(left: int, right: int) -> None:
            if left < 0 or right < 0:
                return
            r = ranks.get(sym[left] + " " + sym[right])
            if r is not None:
                heap.append((r, left, sym[left], sym[right]))

        for i in range(n - 1):
            push(i, i + 1)
        heapq.heapify(heap)
        while heap:
            _, left, lt, rt = heapq.heappop(heap)
            right = nxt[left]
            if right < 0 or sym[left] != lt or sym[right] != rt:
                continue  # outdated
            sym[left] = lt + rt
            sym[right] = ""
            nxt[left] = nxt[right]
            if nxt[right] >= 0:
                prev[nxt[right]] = left
            for a, b in ((prev[left], left), (left, nxt[left])):
                if a >= 0 and b >= 0:
                    r = ranks.get(sym[a] + " " + sym[b])
                    if r is not None:
                        heapq.heappush(heap, (r, a, sym[a], sym[b]))
        for s in sym:
            if not s:
                continue
            t = ids.get(s)
            if t is not None:
                out.append(t)
            else:  # unknown piece: its bytes
                out.extend(ids[f"<0x{b:02X}>"] for b in s.encode("utf-8") if f"<0x{b:02X}>" in ids)

    def _text(self, text: str, out: list[int]) -> None:
        for run in _RUNS.findall(text.replace(" ", SPACE)):
            self._bpe(run, out)

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        """Token ids of ``text``; special tokens written in the text (``<|turn>``, ...) become
        their ids."""
        out: list[int] = [self.bos] if add_bos else []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                self._text(text[pos:m.start()], out)
            out.append(self.special[m.group()])
            pos = m.end()
        if pos < len(text):
            self._text(text[pos:], out)
        return out

    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes:
        """The UTF-8 bytes of the tokens (control tokens only when ``special``)."""
        out = bytearray()
        for i in ids:
            if i in self._bytes:
                out.append(self._bytes[i])
            elif self.types[i] == T_CONTROL and not special:
                continue
            else:
                out += self.tokens[i].replace(SPACE, " ").encode("utf-8")
        return bytes(out)

    def decode(self, ids: list[int], *, special: bool = False) -> str:
        return self.decode_bytes(ids, special=special).decode("utf-8", errors="replace")


def chat_prompt(gguf: GGUFFile, messages: list[dict[str, object]], *, enable_thinking: bool = False,
                tools: list[dict[str, object]] | None = None) -> str:
    """The prompt text of a conversation, rendered with the file's ``tokenizer.chat_template``
    (it begins with ``<bos>``: encode it without ``add_bos``), up to the model's turn (``enable_thinking``: the
    model thinks first; ``tools``: the functions it may call)."""
    bos = gguf.strings("tokenizer.ggml.tokens")[int(gguf.meta["tokenizer.ggml.bos_token_id"])]
    text = chat_template(str(gguf.meta["tokenizer.chat_template"]), messages, tools=tools,
                         enable_thinking=enable_thinking, bos_token=bos)
    # (an answer with content after its tool calls' results closes the model's turn, and the template opens no new
    # one: the model's turn goes before the generation prompt, as llama.cpp does)
    head, end, tail = text.rpartition("<turn|>\n")
    return head + end + "<|turn>model\n" + tail if end and not tail.startswith("<|turn>") else text
