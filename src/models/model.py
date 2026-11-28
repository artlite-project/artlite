"""The host side of a model file (``Model``: tokenizer, chat template, default sampling, what it takes and gives:
``Capabilities``); ``load`` puts it on the NPU (``npu_model.NpuModel``)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from artlite.gguf_model import GGUFFile

from .detokenizer import Tokenizer
from .npu_model import NpuModel
from .sampling import Sampling
from .sequence import Sequence as Tokens

MEDIA_MARKER = "<__media__>"  # where an image goes in a prompt's text (llama.cpp's mtmd marker)


def media_parts(*, images: Sequence[object] = (), videos: Sequence[object] = (), audio: Sequence[object] = ()
                ) -> list[dict[str, object]]:
    """A message's media parts (``Model.chat``): the ``images``, then the ``videos``, then the ``audio`` clips (files or
    their bytes)."""
    return [{"type": "image", "image": im} for im in images] + [{"type": "video", "video": v} for v in videos] + \
        [{"type": "audio", "audio": a} for a in audio]


def check_tower(mmproj: str, config: Any, supported: tuple[str, ...], *, what: str, hidden: int) -> None:
    """NotImplementedError unless the mmproj's tower (its ``model_config.ClipConfig`` or ``AudioConfig``) has a
    ``supported`` projector (``what`` names it in the message); ValueError unless it embeds into the language model's
    ``hidden`` size."""
    if config.projector not in supported:
        raise NotImplementedError(f"{mmproj}: {what} {config.projector!r} (supported: {', '.join(supported)})")
    if config.projection_dim != hidden:
        raise ValueError(f"{mmproj}: embeddings of {config.projection_dim}, the language model's {hidden}")


@dataclass(frozen=True)
class Blocks:
    """A turn the model writes as blocks (Inkling's; ``Model.blocks``): each ``header``, a kind, its text, ``end``.
    The kinds: ``thought`` (a thought), ``text`` (the answer's text), any other (a tool call's: ``header`` NAME
    ``call`` JSON ``end``); the template opens no block for the reply, the model writes its own. The markers are control
    tokens that the generation's text keeps (llama.cpp's preserved tokens: ``tokens``)."""

    header: str
    thought: str
    text: str
    end: str
    call: str = ""

    @property
    def answer(self) -> str:
        """The text that opens an answer's text block."""
        return self.header + self.text

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(t for t in (self.header, self.thought, self.text, self.call, self.end) if t)


@dataclass(frozen=True)
class Capabilities:
    """What a loaded model takes and gives beyond text (the server's listings and routes, the tools' options read it)."""

    vision: bool = False  # images in prompts (``Model.vision``)
    audio: bool = False  # speech clips in prompts (``Model.audio``)
    video: bool = False  # videos in prompts (``video_sequence``)
    speech: bool = False  # speech replies (``tts``, ``voice_reply``)
    live_asr: bool = False  # live transcription (the file's ``tokenizer.chat_template.asr``)


class Model:
    """A model file opened on the host: its tokenizer, chat template and default sampling; ``load`` puts it on the NPU.
    One subclass per architecture (``models.ARCHITECTURES``); constructor keywords are its options."""

    name: str  # for messages: "Llama 3", ...
    vision: Any = None  # the image encoder (an option of the model's: its mmproj file), or None: text only
    audio: Any = None  # the speech encoder (an mmproj file's), or None
    projectors: tuple[str, ...] = ()  # the projector types (``clip.projector_type``) of the mmproj files it takes
    thinks = False  # the chat template has a thinking mode
    reasoning: tuple[str, str] | None = None  # the text that opens and closes a thought (the model writes them)
    # a reasoning_effort level as the chat template takes it, where the template refuses levels clients send (Spark-X's
    # takes high and low); a level not in it goes as it is
    efforts: dict[str, str] = {}  # noqa: RUF012
    thought_first = False  # a thought only begins an answer (its markers later in the answer are text)
    blocks: Blocks | None = None  # a turn the model writes as blocks (``Blocks``: Inkling's), or None
    turn: str | None = None  # the text that begins every message in the chat template (a prompt cache's marks)
    user_turn: str | None = None  # the text that begins a user message
    chat_template: str | None = None  # a Jinja chat template in place of the file's (``open_model``'s); None: the file's
    generates = True  # False: a readout model's file (``readouts.readout_model``): no LM head on the NPU, no text
    parallel = False  # ``load`` takes ``sequences``: the NPU holds several sequences, a lane each (``NpuModel.lanes``)
    transcript_tokens = (64, 8)  # a transcription's tokens at most by default: this many, and these per second of audio
    media: tuple[str, ...] = ()  # what the file takes with no mmproj (a tower of its own: "audio"), for a server's listing
    gguf: GGUFFile  # the model file
    cfg: Any  # its configuration (``artlite.model_config``)
    tokenizer: Tokenizer
    eos_token_ids: tuple[int, ...] | set[int]
    _npu: NpuModel | None = None  # (the model on the NPU, once ``load``ed)

    @property
    def capabilities(self) -> Capabilities:
        """What the model takes and gives beyond text, as it is opened (its mmproj's encoders); a model that takes
        videos or speaks, or transcribes live, says so."""
        return Capabilities(vision=self.vision is not None, audio=self.audio is not None)

    def encode(self, text: str) -> list[int]:
        """Token ids of a raw text (no chat template), after the model's BOS if it has one."""
        raise NotImplementedError

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        """Token ids of a conversation through the chat template (``chat_template``, else the file's), up to the
        assistant's turn (``think``: let it think first; otherwise an empty thought where the template has one;
        ``tools``: the functions it may call, ``chat_template``; ``preserve``: the earlier answers keep their thoughts,
        the template's ``preserve_thinking``, where it has one; ``variables``: the template's others, over the model's
        own: ``reasoning_effort``, ``bos_token``, ...). A message's content may be a list of parts:
        ``{"type": "text", "text": ...}``, ``{"type": "image", "image": a file, its bytes or a PIL image}`` (a model
        with ``vision``: its prompt then a sequence of ids and embedding rows; the encoder runs on the NPU)."""
        raise NotImplementedError

    def transcription(self, clips: Sequence[dict[str, object]], *, language: str = "", prompt: str = ""
                      ) -> list[list[dict[str, object]]]:
        """The conversations (``chat``'s messages) that transcribe ``clips`` (a message's audio parts), each answered by
        a transcript: the model's own ASR prompt, as llama.cpp keeps one per model (``common_chat_get_asr_prompt``);
        here its default, the clips after "Transcribe audio to text". ``language``: a code or a name ("": the model
        finds it); ``prompt``: a text that guides the transcript (OpenAI's ``prompt``). ValueError: no speech
        encoder."""
        if self.audio is None:
            raise ValueError(f"{self.name}: no transcription (audio needs a speech encoder)")
        text = (prompt + "\n" if prompt else "") + "Transcribe audio to text"
        return [[{"role": "user", "content": [{"type": "text", "text": text}, *clips]}]]

    def transcript(self, text: str) -> tuple[str, list[dict[str, object]]]:
        """A transcription's answer as its words alone and its segments (``{"start", "end", "speaker", "text"}``:
        seconds, a speaker's tag), where the model writes them (else the answer, no segments)."""
        return text, []

    def image_tokens(self, image: object) -> Tokens:
        """An image (a file, its bytes or a PIL image) as the model reads it where a prompt's text has ``MEDIA_MARKER``
        (llama.cpp's mtmd: the encoder's markup ``vision.markup`` around the image's tokens); the encoder runs on the
        NPU."""
        markup = getattr(self.vision, "markup", None)
        if markup is None:
            raise ValueError(f"{self.name}: images need its vision encoder (an mmproj file)")
        try:
            return self.tokenizer.encode(markup[0]) + self.vision.sequence(image) + self.tokenizer.encode(markup[1])
        finally:
            self.vision_ran()

    def vision_ran(self) -> None:
        """The image encoder's hardware context ran: the language model's overlay comes back before its next dispatch
        (``OverlayDevice.set_prologue``)."""
        if self._npu is not None:
            self._npu.stale()

    def sampling(self, *, think: bool = False) -> Sampling:
        """The model's default sampling (in the thinking mode)."""
        raise NotImplementedError

    def load(self, *, capacity: int, prefill: bool = True) -> NpuModel:
        """The model on the NPU with a KV cache of ``capacity`` positions; ``prefill``: prompts through the prefill
        (else as decode steps). NotImplementedError: a file whose shapes or formats the NPU path does not handle."""
        raise NotImplementedError

    def prompt(self, text: str, *, system: str | None = None, raw: bool = False, think: bool = False,
               images: list[object] = (), audio: list[object] = (), videos: list[object] = (),
               variables: dict[str, object] | None = None) -> list[int]:
        """Token ids of a user message after an optional system message, through the chat template (``raw``: the text
        alone; ``variables``: the template's, as ``chat``'s); ``images``, then ``videos``, then ``audio`` clips (files
        or their bytes), where the text has ``MEDIA_MARKER``, else before it."""
        if (images or videos) and self.vision is None:
            raise ValueError(f"{self.name}: images and videos need a vision encoder (an mmproj file)")
        if videos and not self.capabilities.video:
            raise ValueError(f"{self.name}: no video input (LFM3-VL's and LFM3-Omni's mmproj)")
        if audio and self.audio is None:
            raise ValueError(f"{self.name}: audio needs a speech encoder (an mmproj file)")
        media = media_parts(images=images, videos=videos, audio=audio)
        if raw:
            if media:
                raise ValueError("images, videos and audio go through the chat template (not a raw prompt)")
            return self.encode(text)
        if think and not self.thinks:
            raise ValueError(f"{self.name} has no thinking mode")
        content: str | list[dict[str, object]] = text
        if media:
            parts = text.split(MEDIA_MARKER) if MEDIA_MARKER in text else ["", text]
            if len(parts) != len(media) + 1:
                raise ValueError(f"{len(media)} media file(s) for {len(parts) - 1} {MEDIA_MARKER} marker(s) in the "
                                 "prompt")
            content = [{"type": "text", "text": parts[0]}] if parts[0] else []
            for part, t in zip(media, parts[1:], strict=True):
                content += [part] + ([{"type": "text", "text": t}] if t else [])
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.chat(messages, think=think, variables=variables)
