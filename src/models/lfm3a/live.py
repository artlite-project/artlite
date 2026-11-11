"""Live ASR with LFM3-Audio (``LiveAsr``): the text of a stream of audio while it arrives, from the language model's
``asr`` chat template ("Perform live ASR.").

The trained layout (the model card's ``transcribe_live``, llama.cpp's ``mtmd_audio_stream_lfm3a``): the prompt, then per
12 rows (0.96 s) of the stream its rows and their text up to ``<|text_end|>``, the last chunk's rows (12 at most) closed
by ``<|mixed_end|>``::

    <prompt> [A x 12] text<|text_end|> [A x 12] text<|text_end|> .. [A x r] <|mixed_end|> text<|text_end|>

A chunk's text is greedy (the model card's), never the tokens that end the turn or start speech (``BANNED``: only
``<|text_end|>`` ends a chunk's text), at most 48 tokens (the last chunk's 96), then closed as the model would. Training
samples are at most 30 s: the text restarts (the prompt again, a new line in the text) at a chunk after 15 s of the
sequence when the chunk before had no text, always after 30 s; the encoder keeps its state.

The rows (``Stream``): the mel frames of the samples so far (a frame once its 400 samples arrived), the encoder's band
in progress run again over them (``Encoder.run_band(partial=True)``: its rows before the frames' end the band's, but
for the rounding of the 8-key blocks of V^T the frames' end cuts), as many rows as there are whole ones, a chunk's 12
taken at a time. ``commit`` pads the samples with silence to the next chunk (llama.cpp's flush: the text of the audio
so far comes out), ``end`` as a clip's end (the last row's frames padded) and runs the last chunk.

Language model and encoder take turns on the NPU (two hardware contexts): a chunk's dispatches are the encoder's band
(~18 ms) and the language model's decode steps for its rows and text."""

import time
from collections.abc import Callable

import jinja2
import numpy as np

from ..common import NpuModel, Penalties, Sampling, StreamDecoder
from ..sequence import Sequence
from . import host
from .encoder import FRAMES_BAND, ROWS_BAND, Encoder

CHUNK_ROWS = 12  # (rows of a chunk: 0.96 s)
CHUNK_TOKENS, FINAL_TOKENS = 48, 96  # (a chunk's text tokens at most; the last chunk's)
SOFT_ROWS, MAX_ROWS = 188, 375  # (a sequence's rows before a chunk without text restarts it, 15 s; always, 30 s)
TEXT_END, MIXED_END = "<|text_end|>", "<|mixed_end|>"
BANNED = ("<|mixed_end|>", "<|mixed_start|>", "<|audio_start|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")


class Stream:
    """A stream's rows on the NPU encoder as its samples arrive (module docstring): ``push`` samples, ``silence``,
    ``end``; ``available`` rows, ``take`` them."""

    def __init__(self, encoder: Encoder, *, ran: Callable[[], None] | None = None):
        """ran: called after each encoder dispatch (another hardware context ran)"""
        self.enc, self._ran = encoder, ran
        self.reset()

    def reset(self) -> None:
        self.enc.reset()
        self.enc.owner = self
        self.band, self._saved = 0, None  # (the band in progress, the state at its start: ``Encoder.saved``)
        self.x = np.zeros(0, np.float32)  # (the samples from sample x0 on not yet in a frame's start)
        self.x0 = 0
        self.mel = np.zeros((host.N_MEL, 0), np.float32)  # (the frames from frame m0 on: the band in progress's)
        self.m0 = 0
        self.frames = 0
        self.ended = False
        self.encoded = 0  # (rows encoded)
        self.taken = 0  # (rows taken)
        self._rows: list[np.ndarray] = []  # (rows encoded, not taken)
        self.t_encode, self.dispatches = 0.0, 0

    @property
    def samples(self) -> int:
        return self.x0 + len(self.x)

    def push(self, samples: np.ndarray) -> None:
        """More samples (16 kHz mono)."""
        if self.ended:
            raise ValueError("the stream ended")
        self.x = np.concatenate([self.x, np.asarray(samples, np.float32).reshape(-1)])
        self._frame()

    def silence(self, n: int) -> None:
        self.push(np.zeros(max(n, 0), np.float32))

    def to_chunk(self) -> None:
        """Silence up to the next chunk's end (llama.cpp's flush: the frames a multiple of a chunk's)."""
        if self.samples == 0:
            return
        per = CHUNK_ROWS * host.STRIDE
        n = 1 if self.samples <= host.N_FFT else -(-(self.samples - host.N_FFT) // host.HOP) + 1
        self.silence(host.N_FFT + (-(-n // per) * per - 1) * host.HOP - self.samples)

    def end(self) -> None:
        """The stream's end: its samples padded as a clip's (``host.padded_length``), every frame."""
        if not self.ended:
            self.silence(host.padded_length(self.samples) - self.samples)
            self.ended = True

    def _frame(self) -> None:
        nf = (self.samples - host.N_FFT) // host.HOP + 1 if self.samples >= host.N_FFT else 0
        if nf <= self.frames:
            return
        s = self.frames * host.HOP - self.x0
        mel = host.log_mel(self.x[s:s + (nf - self.frames - 1) * host.HOP + host.N_FFT])
        assert mel.shape[1] == nf - self.frames
        self.mel = np.concatenate([self.mel, mel], axis=1)
        self.frames = nf
        cut = nf * host.HOP - self.x0
        self.x, self.x0 = self.x[cut:], self.x0 + cut

    def rows(self) -> int:
        """The stream's rows so far (whole ones; at its end every one)."""
        return host.rows(self.frames) if self.ended else self.frames // host.STRIDE

    def available(self) -> int:
        """Rows not taken yet."""
        return self.rows() - self.taken

    def take(self, n: int) -> np.ndarray:
        """The next n rows (n <= ``available``): the whole rows so far are encoded (a band's dispatch at a time), up to
        a multiple of 8 where that covers them (a block of V^T's 8 keys the frames' end does not cut)."""
        assert 0 < n <= self.available(), (n, self.available())
        if self.encoded < self.taken + n:
            r = self.rows()
            self._encode(r if self.ended else max(r - r % 8, self.taken + n))
        out = np.concatenate(self._rows)
        self._rows = [out[n:]] if len(out) > n else []
        self.taken += n
        return out[:n]

    def _encode(self, r: int) -> None:
        """Encode the rows up to r: the bands they complete, then the band in progress."""
        enc = self.enc
        while self.encoded < r:
            b = self.band
            end = ROWS_BAND * (b + 1)
            t0 = time.perf_counter()
            if enc.owner is not self:  # (a clip or another stream ran meanwhile: this one's state back)
                enc.restore(b, self._saved)
                enc.owner = self
            out = enc.run_band(self.mel, self.frames, start=self.m0, partial=r < end and not self.ended)
            if r >= end:  # (the band complete: its state kept, its frames gone but the next band's frame -1)
                enc.carry()
                self.band, self._saved = b + 1, enc.saved()
                keep = FRAMES_BAND * (b + 1) - 1 - self.m0
                self.mel, self.m0 = self.mel[:, keep:], self.m0 + keep
            self.t_encode += time.perf_counter() - t0
            self.dispatches += 1
            if self._ran is not None:
                self._ran()
            hi = min(r, end)
            self._rows.append(out[self.encoded - ROWS_BAND * b:hi - ROWS_BAND * b])
            self.encoded = hi


class LiveAsr:
    """Live ASR on an LFM3-Audio model (module docstring): ``start`` a stream, ``push`` samples, ``commit``, ``end``;
    each returns the text that came out meanwhile (a new line where the text restarted).

    ``sync``: makes the device hold a sequence (the server's prompt cache: other requests run between chunks) and is
    told what the device holds after a chunk; without it the stream owns the language model's caches."""

    def __init__(self, model: object, npu: NpuModel, encoder: Encoder, *,
                 sync: tuple[Callable[[Sequence], None], Callable[[Sequence], None]] | None = None):
        self.model, self.npu = model, npu
        meta = model.gguf.meta
        if "tokenizer.chat_template.asr" not in meta:
            raise ValueError("the model has no live ASR template (tokenizer.chat_template.asr): LFM3-Audio's")
        pieces = model.gguf.strings("tokenizer.ggml.tokens")
        text = jinja2.Environment().from_string(str(meta["tokenizer.chat_template.asr"])).render(
            bos_token=pieces[model.bos], messages=[], add_generation_prompt=True)
        self.prompt = model.tokenizer.encode(text)
        self.text_end, self.mixed_end = pieces.index(TEXT_END), pieces.index(MIXED_END)
        self.sampling = Sampling(greedy=True, banned=tuple(pieces.index(p) for p in BANNED))
        self.stream = Stream(encoder, ran=model._vision_ran)
        self._sync = sync
        self.start()

    def start(self) -> None:
        """A new stream."""
        self.stream.reset()
        self.seq: Sequence = Sequence.text([], self.model.cfg.hidden_size)  # (the device's tokens of the text sequence)
        self.closing: list[int] = []  # (the tokens before the next chunk's rows: its prompt, the last chunk's end)
        self.started = self.done = False
        self.seq_rows = 0
        self.had_text = False  # (the last chunk's)
        self.sent = False  # (any text)
        self.decoder = StreamDecoder(self.model.tokenizer)
        self.chunks, self.t_text = 0, 0.0

    def push(self, samples: np.ndarray) -> str:
        """More samples (16 kHz mono float32): the text of the chunks they complete."""
        self.stream.push(samples)
        return self._run()

    def commit(self) -> str:
        """The audio so far padded with silence to a chunk's end: its text."""
        self.stream.to_chunk()
        return self._run()

    def end(self) -> str:
        """The stream's end: the rest of the text."""
        self.stream.end()
        return self._run()

    def _run(self) -> str:
        out = []
        while not self.done:
            n = self.stream.available()
            final = self.stream.ended and n <= CHUNK_ROWS
            if not final and n < CHUNK_ROWS:
                break
            if final and n == 0 and not self.started:
                self.done = True
                break
            out.append(self._chunk(n if final else CHUNK_ROWS, final))
        return "".join(out)

    def _chunk(self, n: int, final: bool) -> str:
        """A chunk: its rows (and the tokens before them) in, its text out."""
        npu, out = self.npu, ""
        rows = self.stream.take(n) if n else np.zeros((0, self.model.cfg.hidden_size), np.float32)
        t0 = time.perf_counter()
        if not self.started or self.seq_rows >= MAX_ROWS or (self.seq_rows >= SOFT_ROWS and not self.had_text):
            self.closing = list(self.prompt)
            self.seq = Sequence.text([], self.model.cfg.hidden_size)
            self.seq_rows = 0
            if self.sent:
                out += "\n"
        if self._sync is not None:
            self._sync[0](self.seq)
        elif not len(self.seq):
            npu.reset()
        feed = self.closing + host.clip_sequence(rows) + ([self.mixed_end] if final else [])
        logits = npu.forward(feed)
        self.seq = self.seq + feed
        self.closing = []
        self.started, self.seq_rows, self.chunks = True, self.seq_rows + n, self.chunks + 1
        penalties = Penalties(self.sampling)
        budget = FINAL_TOKENS if final else CHUNK_TOKENS
        token = int(penalties.top(logits, 1)[1][0])
        count, self.had_text = 0, False
        while token != self.text_end:
            self.had_text = True
            out += self.decoder.feed(token)
            if count + 1 >= budget:  # (a runaway chunk: closed as the model should have)
                self.closing.append(token)
                break
            count += 1
            self.seq = self.seq + [token]
            token = int(npu.top_k(token, 1, penalties)[1][0])
        self.closing.append(self.text_end)
        if self._sync is not None:
            self._sync[1](self.seq)
        if final:
            self.done = True
        self.sent = self.sent or bool(out)
        self.t_text += time.perf_counter() - t0
        return out
