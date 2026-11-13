"""llama-server's model JSON, as ``GET /v1/models`` of a model's server and of the router answer it: ``data`` (the
OpenAI-style entries) and ``models`` (the Ollama-style details) built from the same fields, and what the model takes and
gives (``Modalities``)."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .engine import Engine
from .speech import speaks

MEDIA = ("image", "audio", "video", "speech")  # (what a model takes and gives, in the order the API lists them)


def takes_video(model: object) -> bool:
    """Whether a model takes videos (LFM3-VL's towers: ``models.lfm.lfm.Lfm.video_sequence``)."""
    return getattr(model, "vision", None) is not None and hasattr(model, "video_sequence")


@dataclass(frozen=True)
class Modalities:
    """What a loaded model takes (images, audio clips, videos) and gives (speech)."""

    vision: bool = False
    audio: bool = False
    video: bool = False
    speech: bool = False

    @classmethod
    def of(cls, engine: Engine) -> "Modalities":
        """The engine's model's (none before it is loaded)."""
        m = engine.model
        if m is None:
            return cls()
        return cls(vision=m.vision is not None, audio=m.audio is not None, video=takes_video(m), speech=speaks(engine))

    @property
    def media(self) -> list[str]:
        """The media (``MEDIA``) it takes and gives."""
        flags = (self.vision, self.audio, self.video, self.speech)
        return [m for m, on in zip(MEDIA, flags, strict=True) if on]


def architecture(media: Iterable[str], *, decision: bool) -> dict[str, list[str]]:
    """An entry's modalities: ``media`` (``MEDIA``: speech is an output) and whether the model answers decisions
    (``/v1/systemone``) instead of text."""
    media = list(media)
    return {"input_modalities": ["text"] + [m for m in media if m != "speech"],
            "output_modalities": ["decisions" if decision else "text"] + ["audio"] * ("speech" in media)}


def model_meta(*, n_vocab: int, n_ctx: int, n_ctx_train: int, n_embd: int, n_params: int, size: int, ftype: str,
               arch: str | None = None) -> dict[str, Any]:
    """An entry's ``meta`` (``arch``: the router lists it too)."""
    return {**({"architecture": arch} if arch is not None else {}), "n_vocab": n_vocab, "n_ctx": n_ctx,
            "n_ctx_train": n_ctx_train, "n_embd": n_embd, "n_params": n_params, "size": size, "ftype": ftype}


def model_entries(name: str, *, aliases: list[str], arch: dict[str, list[str]], meta: dict[str, Any], created: int,
                  multimodal: bool, family: str = "", parameter_size: str = "", quantization: str = "",
                  extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """A model's entries of ``data`` and of ``models``: ``arch``: ``architecture``'s, ``meta``: ``model_meta``'s,
    ``extra``: more fields of the ``data`` entry (the router's)."""
    data = {"id": name, "aliases": aliases, "tags": [], "object": "model", "architecture": arch, "created": created,
            "owned_by": "artlite", **(extra or {}), "meta": meta}
    details = {"name": name, "model": name, "modified_at": "", "size": "", "digest": "", "type": "model",
               "description": "", "tags": [""], "capabilities": ["completion"] + ["multimodal"] * multimodal,
               "parameters": "",
               "details": {"parent_model": "", "format": "gguf", "family": family, "families": [family],
                           "parameter_size": parameter_size, "quantization_level": quantization}}
    return data, details
