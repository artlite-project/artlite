"""Tokens as text: the protocols of a model's tokenizer and the incremental detokenizer of a generation's stream."""

from typing import Protocol


class Detokenizer(Protocol):
    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes: ...


class Tokenizer(Detokenizer, Protocol):
    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` (special tokens written in it become their ids; no BOS)."""
        ...


class StreamDecoder:
    """Incremental detokenizer: ``feed`` returns the text completed by a token (a multi-byte character split over
    byte tokens appears once its last byte arrives; ``pending``: its bytes so far)."""

    def __init__(self, tokenizer: Detokenizer, *, special: bool = False):
        self.tokenizer = tokenizer
        self.special = special
        self.pending = b""

    def feed(self, token: int) -> str:
        data = self.pending + self.tokenizer.decode_bytes([token], special=self.special)
        try:
            text = data.decode("utf-8")
            self.pending = b""
        except UnicodeDecodeError as e:
            text = data[:e.start].decode("utf-8", errors="replace")
            self.pending = data[e.start:]
            if len(self.pending) > 4:  # not an incomplete character: invalid bytes
                text += self.pending.decode("utf-8", errors="replace")
                self.pending = b""
        return text
