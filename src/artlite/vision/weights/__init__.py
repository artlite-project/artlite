"""The vision networks' weights: index maps on sources (``algebra``: ``Wt``), the model files' tensors (``files``), and
their packing into the overlay's B slots (``jobs``: ``Slots``, the jobs ``Builder.gemm`` adds and ``_native.pack_slots``
packs: the end of a build, and every restore of a cached build: the cache keeps the jobs, never the weights)."""

from .algebra import KINDS, Part, Source, Wt, as_wt, cat, computed, eye, f32, flat, memory, pad_index, take, values
from .files import Sources, gguf_getter, gguf_matrix, gguf_sources, safetensors, sources
from .jobs import Slots, job_arrays, pack_arrays

__all__ = ["KINDS", "Part", "Slots", "Source", "Sources", "Wt", "as_wt", "cat", "computed", "eye", "f32", "flat",
           "gguf_getter", "gguf_matrix", "gguf_sources", "job_arrays", "memory", "pack_arrays", "pad_index", "safetensors", "sources", "take", "values"]
