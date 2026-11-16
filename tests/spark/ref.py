"""References of the Spark-X tests: the files (``gguf``), token sequences (an essay's tokens, a chat turn with a fixed
reply), the fp32 CPU model's logits (``cpu.SparkCpu``, cached in ``tests/refs/<model>/``: ``common.cached_ref``) and
llama.cpp's CPU logits of a sequence (the golden dump tool: ``tests.goldens.text.golden`` runs any architecture)."""

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import spark_gguf

from .. import common
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from . import cpu

SOURCES = [cpu.__file__]
BOS = 0
PROMPT = "Explain how photosynthesis works in plants."
# a fixed reply (teacher forcing: the CPU and NPU models see the same tokens)
REPLY = ("Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical "
         "energy. It takes place in the chloroplasts, mainly in the leaves.")


def path(size: str = "1.7B", quant: str = "Q4_K_M", org: str = "XHToken") -> str:
    p = spark_gguf(size, quant, org)
    require(p)
    return p


def gguf(size: str = "1.7B", quant: str = "Q4_K_M", org: str = "XHToken") -> GGUFFile:
    return GGUFFile(path(size, quant, org))


def essay_ids(n: int, size: str = "1.7B", name: str = "addiction.txt") -> list[int]:
    """The BOS and the first n - 1 tokens of an essay of the corpus."""
    return [BOS] + llama_tokenize(path(size), common.essay(name))[:n - 1]


def chat_ids(size: str = "1.7B") -> list[int]:
    """A chat turn (thinking off, the template's default system prompt) and the fixed reply, as llama.cpp tokenizes
    them."""
    text = ("<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜>"
            f"<｜start▁of▁sentence｜><|User|>{PROMPT}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>{REPLY}")
    return llama_tokenize(path(size), text)


def cpu_logits(spec: tuple[str, str, str], ids: list[int], *, layers: list[int] | None = None,
               model: cpu.SparkCpu | None = None) -> torch.Tensor:
    """The fp32 CPU model's logits (len(ids), vocab) of a sequence from position 0 (cached); ``spec``: (size, quant,
    org)."""
    p = path(*spec)
    key = (common.model_key(p), tuple(ids), tuple(layers or ()))

    def compute() -> torch.Tensor:
        m = model or cpu.SparkCpu(GGUFFile(p), layers=layers)
        return m.logits(ids, slice(None)).half()

    return common.cached_ref(p, "cpu_logits", key, compute, sources=SOURCES).float()


def llama_logits(spec: tuple[str, str, str], ids: list[int]) -> torch.Tensor:
    """llama.cpp's CPU logits (len(ids), vocab) of a sequence (its prompt graph)."""
    g = golden(path(*spec), ids, logits="f32")
    return torch.from_numpy(np.asarray(g.logits, np.float32).copy())
