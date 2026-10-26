"""The apps' full designs, built as the apps build them (no NPU): the decode overlays and prefill libraries of the
models present, the Stable Diffusion UNet program. They compile (program memory, stack and BD limits of the whole
models, which the layer-slice tests do not reach), and ``--dump-designs`` records them: the refactoring check covers
the apps without running the slow tests."""

import os
from collections.abc import Callable

import pytest

from artlite import parallel_compile
from artlite.gguf_model import GGUFFile
from artlite.model_config import gemma_gguf, llama_gguf, occamy_gguf, qwen_gguf, sd15_dir


def _llama(size: str, quant: str, kv_k: str) -> tuple[str, Callable[[str], None]]:
    def build(path: str) -> None:
        from models.llama.overlay import LlamaOverlay
        from models.llama.prefill import LlamaPrefill

        ov = LlamaOverlay(GGUFFile(path), kv_k=kv_k)
        parallel_compile.compile_all([ov.op, LlamaPrefill(ov).lib])

    return llama_gguf(quant, size), build


def _gemma(path: str) -> None:
    from models.gemma4.gemma import BATCH
    from models.gemma4.overlay import GemmaOverlay
    from models.gemma4.prefill import GemmaPrefill

    ov = GemmaOverlay(GGUFFile(path), max_positions=4096 + BATCH)
    parallel_compile.compile_all([ov.op, GemmaPrefill(ov).lib])


def _qwen(path: str) -> None:
    from models.qwen35.overlay import QwenOverlay
    from models.qwen35.prefill import QwenPrefill

    ov = QwenOverlay(GGUFFile(path), max_positions=4096)
    parallel_compile.compile_all([ov.op, QwenPrefill(ov).lib])


def _sd15(path: str) -> None:
    from artlite.vision.design import VisionDesign

    parallel_compile.compile_all([VisionDesign()])


APPS: dict[str, tuple[str, Callable[[str], None]]] = {
    "llama-1B-Q4_K_M": _llama("1B", "Q4_K_M", "single"),
    "llama-1B-Q4_K_M-hilo": _llama("1B", "Q4_K_M", "hilo"),
    "llama-1B-Q5_K_M": _llama("1B", "Q5_K_M", "single"),
    "llama-3B-Q4_K_M": _llama("3B", "Q4_K_M", "single"),
    "llama-8B-Q4_K_S": _llama("8B", "Q4_K_S", "single"),
    "gemma-E4B-Q4_0": (gemma_gguf("Q4_0"), _gemma),
    "qwen-9B-Q4_K_M": (qwen_gguf("Q4_K_M", "9B"), _qwen),
    "qwen-27B-Q4_K_M": (qwen_gguf("Q4_K_M", "27B"), _qwen),
    "qwen-1B-Q4_K_M": (qwen_gguf("Q4_K_M", "1B"), _qwen),
    "qwen-35B-A3B-Q4_K_M": (qwen_gguf("Q4_K_M", "35B-A3B"), _qwen),
    "occamy-1.0-APEX": (occamy_gguf(), _qwen),
    "sd15": (sd15_dir(), _sd15),
}


@pytest.mark.parametrize("app", list(APPS))
def test_app_designs(app: str) -> None:
    path, build = APPS[app]
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    build(path)
