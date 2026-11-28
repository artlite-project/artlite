"""MOSS-Transcribe-Diarize on the NPU (GGUF architecture ``moss``: OpenMOSS's checkpoint directory, a transcribe.cpp
GGUF file of it: ``artlite.transcribe_cpp``): a Qwen3 language
model (``models.llama.qwen3``: Llama's decode overlay and prefill) that reads a recording as the rows of its audio
tower (``models.moss.npu.NpuAudio``: a Whisper encoder and the VQ adaptor on the vision overlay) and answers with the
recording's transcript, ``[start][Sxx]text[end]`` segments: times in seconds, anonymous speaker tags.

A prompt is the chat template's text with a recording where a message has an audio part (``{"type": "audio", "audio":
a file, its bytes or 16 kHz samples}``, OpenAI's ``input_audio`` and ``audio_url`` ones): the template's
``<|audio_pad|>`` becomes the recording's span (``host.Span``: a row per 80 ms of every 30 s chunk, the seconds as digits
every 5 s), a ``sequence.Sequence``, so a recording of any length is one prompt and its speakers' tags and times run
through it (the model card: up to 90 minutes; a minute is 762 positions, the KV cache's size the limit). The model
transcribes greedily (the checkpoint's ``generation_config.json``) and has no thinking mode: the template's
``enable_thinking`` is never set, as the model's own code never sets it.

``transcription``: the model card's instruction after the recording, with a request's ``prompt`` as its hotword hint.

The KV cache holds K as hi + lo bfp16 blocks (Qwen3's default): with one block per 8 dims LibriSpeech test-clean's WER
goes from 1.99% to 2.12% and its empty transcripts from the official model's 3 to 5, for a step ~20% faster 10
minutes into a recording (``kv_k="single"``).
"""

import functools
from collections.abc import Sequence as Seq

from ..audio import AUDIO_PARTS, part_audio
from ..llama.llama import NpuLlama
from ..llama.qwen3 import Qwen3
from ..model import check_tower
from ..sampling import Sampling
from ..sequence import Sequence, splice
from ..templating import render_chat
from . import host
from .npu import PROJECTORS, NpuAudio

HOTWORDS = "热词提示："  # (the model card's hotword hint, after the instruction)


class Moss(Qwen3):
    """A MOSS-Transcribe-Diarize file (a checkpoint directory, a transcribe.cpp GGUF file): its language model
    (``Qwen3``: the weight image, the tokenizer, the chat template) and its audio tower on the NPU (``NpuAudio``,
    loaded here: a hardware context of its own)."""

    name = "MOSS-Transcribe-Diarize"
    thinks = False
    reasoning = None
    thought_first = False
    transcript_tokens = (128, 16)  # (its segments' times and tags: about as many tokens as its words)
    media = ("audio",)

    def __init__(self, path: str, *, kv_k: str = "hilo"):
        super().__init__(path, kv_k=kv_k)
        self.audio = NpuAudio(path)
        check_tower(path, self.audio.cfg, PROJECTORS, what="audio projector", hidden=self.cfg.hidden_size)
        self.span = host.Span.of(self.gguf, self.tokenizer)
        self.marks = host.Marks.of(self.gguf, self.tokenizer)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int] | Sequence:
        if think:
            raise ValueError(f"{self.name} has no thinking mode")
        clips: list[object] = []

        def part(p: object) -> object:  # (an audio part as the template writes its place)
            if not isinstance(p, dict) or p.get("type") not in AUDIO_PARTS:
                return p
            clips.append(part_audio(p))
            return {"type": "audio", "audio": ""}
        messages = [{**m, "content": [part(p) for p in m["content"]]} if isinstance(m.get("content"), list) else m
                    for m in messages]
        variables = {k: v for k, v in (variables or {}).items() if k != "enable_thinking"}
        text = render_chat(self.gguf, messages, tools=tools, template=self.chat_template, **variables)
        if not clips:
            return self.tokenizer.encode(text)
        try:
            return splice(text, host.PAD, [self.span.sequence(self.audio.rows(c)) for c in clips], self.tokenizer.encode)
        finally:
            self.vision_ran()

    def transcription(self, clips: Seq[dict[str, object]], *, language: str = "", prompt: str = ""
                      ) -> list[list[dict[str, object]]]:
        """(a conversation per recording: the model reads one; it finds the language itself)"""
        text = host.INSTRUCTION + (HOTWORDS + prompt if prompt else "")
        return [[{"role": "user", "content": [clip, {"type": "text", "text": text}]}] for clip in clips]

    def transcript(self, text: str) -> tuple[str, list[dict[str, object]]]:
        return host.plain(text), [s._asdict() for s in host.segments(text)]

    def sampling(self, *, think: bool = False) -> Sampling:
        return Sampling(greedy=True)

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> NpuLlama:
        self._npu = npu = super().load(capacity=capacity, prefill=prefill, sequences=sequences)
        for lane in npu.lanes:  # (the transcripts' drafts: ``host.Drafts``)
            lane.context = functools.partial(host.Drafts, self.marks)
        return npu
