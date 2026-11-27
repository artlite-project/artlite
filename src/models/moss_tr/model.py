"""MOSS-Transcribe-Diarize on the NPU: its audio tower (``models.moss_tr.tower``, the vision overlay's four programs),
its prompt (the chat template's text with the tower's rows where the audio span's pads are, ``models.moss_tr.host``) and
its decoder (``models.moss_tr.decoder``: the checkpoint's Qwen3-0.6B-shaped language model on the llama decode overlay).

The architecture is ``moss`` (``artlite.hf_model._Moss``: the language model's tensors in the Qwen3 map, the metadata's
keys ``moss.*``), so ``models.open_model`` opens a checkpoint directory with this class; the tower's tensors are the same
checkpoint's other names (``model.whisper_encoder.*``, ``model.vq_adaptor.*``), read by ``HfModel``'s checkpoint names.

``load`` puts the decoder on the NPU (``models.moss_tr.decoder.decoder``: the same path as a Qwen3's) and builds the
tower's four programs on one vision device, dispatching each once (``Tower.load``: the cold cost of the five programs'
contexts, paid at load, so a chunk's timing is the steady one). The tower's dispatch invalidates the decoder's
overlay (one hardware context on the array): ``chat`` calls ``vision_ran`` as the image encoders' path does.

The conversation: ``chat`` renders the template, then replaces the ``<|audio_pad|>`` place(s) with the tower's rows - a
``Sequence`` of the clip's given rows (``host.span_tokens``: ``(n - 1) // 1,280 + 1`` of its samples, at most a 30 s
chunk's 375) and its span's ids (the pads plus the time markers' digit tokens, which keep their text ids) - and the
decoder reads the rows at their positions through the prefill's chunks and the decode steps alike
(``decoder.rows_of``). A prompt's audio span is one chunk's encoder run read at the clip's own row count (the
processor's slice); a clip longer than 30 s is cut to the chunk.

Sampling: greedy, as transformers' ``generate`` on the checkpoint's ``generation_config.json`` (no do_sample, no
temperature).
"""

import numpy as np

from models.llama.qwen3 import Qwen3
from models.model import Capabilities
from models.sampling import Sampling
from models.sequence import Sequence
from models.templating import render_chat

from .decoder import Decoder, decoder, digits
from .host import INSTRUCTION, splice_clips
from .tower import Tower

GREEDY = Sampling(greedy=True)


class Moss(Qwen3):
    """A MOSS-Transcribe-Diarize checkpoint (architecture ``moss``): the decoder of a qwen3 (``Qwen3.__init__``: the
    weight image, the tokenizer, the chat template) and the audio tower (``Tower``: built at ``load``, on the first
    clip's call if none). A speech model: ``capabilities.audio`` and the prompts' audio parts."""

    name = "MOSS-Transcribe"
    thinks = False  # (the template has no thinking switch: a transcription is its own answer)
    parallel = False  # (a decoder of many sequences would need the tower's device between them: one at a time)
    turn, user_turn = "<|im_start|>", "<|im_start|>user"

    def __init__(self, path: str, *, kv_k: str = "single"):
        super().__init__(path, kv_k=kv_k)
        self.audio = Tower(self.gguf)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(audio=self.audio is not None)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int] | Sequence:
        """The conversation's prompt: the template's text (its ``<|audio_pad|>`` place(s) where a message has an audio
        part) with the tower's rows at the pads (``splice_clips``: a ``Sequence``). A message's audio part is a file, its
        bytes or 16 kHz samples; the tower runs on the NPU (its first call builds it), and the decoder's overlay is told
        that the array ran another context (``vision_ran``)."""
        if think:
            raise ValueError(f"{self.name} has no thinking mode")
        text = render_chat(self.gguf, messages, tools=tools, template=self.chat_template, **(variables or {}))
        parts = [p for m in messages if isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict)]
        clips = [p.get("audio", p.get("audio_url")) for p in parts if p.get("type") == "audio"]
        if any(c is None for c in clips):
            raise ValueError(f"{self.name}: an audio part without its clip (``audio``: a file, its bytes or samples)")
        if not clips:
            return self.tokenizer.encode(text)
        try:
            seqs = [self.audio.sequence(c, digits=digits(self.tokenizer)) for c in clips]
        finally:
            self.vision_ran()
        return splice_clips(text, seqs, self.tokenizer.encode)

    def silent_turn(self, logits: np.ndarray) -> bool:
        """Whether a turn's first token is an end-of-sequence (a `single` bf16 cache's near-tie on the instruction: the
        ``[`` of a transcript's first timestamp and EOS within a logit)."""
        return int(np.argmax(np.asarray(logits, np.float32))) in self.eos_token_ids

    def sampling(self, *, think: bool = False) -> Sampling:
        return GREEDY


    def transcribe(self, npu: Decoder, audio: str | bytes | np.ndarray, *, instruction: str | None = INSTRUCTION,
                   max_tokens: int = 400, retry: bool = True) -> str:
        """A clip's transcript, greedy: the instruction's timestamped, speaker-tagged format. A turn whose first
        token is an end-of-sequence (``silent_turn``: the instruction reaching the single bf16 block's near-tie) has
        its **first token forced to the best other one** - the format the model answers in starts with a timestamp's
        ``[`` - a guard for that near-tie; ``retry``: a turn that still ends at once is transcribed with the audio
        alone."""
        def turn(ids: list[int]) -> list[int]:
            npu.prepare(len(ids))
            logits = npu.forward(ids)
            if self.silent_turn(logits):
                lg = np.asarray(logits, np.float32).copy()
                lg[list(self.eos_token_ids)] = -np.inf
                first = int(np.argmax(lg))
                logits = npu.forward([first])
                return [first] + [t for t in npu.generate(logits, max_tokens - 1, self.sampling(), ids + [first])
                                  if t not in self.eos_token_ids]
            return [t for t in npu.generate(logits, max_tokens, self.sampling(), ids) if t not in self.eos_token_ids]

        parts: list[dict[str, object]] = [{"type": "audio", "audio": audio}]
        if instruction:
            parts.append({"type": "text", "text": instruction})
        npu.reset()
        tokens = turn(self.chat([{"role": "user", "content": parts}]))
        if not tokens and retry and instruction:  # (the audio alone: no instruction to reach the near-tie)
            npu.reset()
            tokens = turn(self.chat([{"role": "user", "content": [{"type": "audio", "audio": audio}]}]))
        return self.tokenizer.decode(tokens)

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> Decoder:
        """The decoder on the NPU and the tower's four programs (each dispatched once at load: the five contexts' cold
        cost). ``capacity``: the decoder's KV cache positions (67.7 KB a position: 414 + the answer); ``prefill``:
        prompts through the prefill."""
        if sequences != 1:
            raise ValueError(f"{self.name}: one sequence (its audio tower holds one vision device)")
        npu = decoder(self.gguf, capacity=capacity, kv_k=self.kv_k, prefill=prefill)
        self.audio.load()  # (the tower's four programs, each dispatched once: the cold cost, paid at load)
        npu.overlay.dev.open()  # (the decode overlay's context: the fifth program's share of the cold cost)
        self._npu = npu
        return npu
