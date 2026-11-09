"""The apps' full designs, built as the apps build them (no NPU): the decode overlays and prefill libraries of the
models present (Qwen3 on Llama's path with its q / k norm weights; Qwen3.6-35B-A3B also with its MTP drafter file, the
app's default: the design runs the drafter's formats too), the Stable Diffusion UNet program. They compile (program
memory, stack and BD limits of the whole models, which the layer-slice tests do not reach), a paged design's code pages
link (each within the slot its core's resident code leaves: ``paging.link``, else only at load), and
``--dump-designs`` records them: the refactoring check covers the apps without running the slow tests."""

import os
from collections.abc import Callable

import pytest

from artlite import parallel_compile
from artlite.gguf_model import GGUFFile
from artlite.model_config import (
    gemma_gguf,
    llama_gguf,
    occamy_gguf,
    qwen3_gguf,
    qwen_gguf,
    qwen_mtp_of,
    sd15_dir,
    swift_gguf,
)


def _llama(path: str, kv_k: str) -> tuple[str, Callable[[str], None]]:
    def build(path: str) -> None:
        from models.llama.overlay import LlamaOverlay
        from models.llama.prefill import LlamaPrefill

        ov = LlamaOverlay(GGUFFile(path), kv_k=kv_k)
        parallel_compile.compile_all([ov.op, LlamaPrefill(ov).lib])
        ov.op.page_store()

    return path, build


def _gemma(path: str) -> None:
    from models.gemma4.gemma import BATCH
    from models.gemma4.overlay import GemmaOverlay
    from models.gemma4.prefill import GemmaPrefill

    ov = GemmaOverlay(GGUFFile(path), max_positions=4096 + BATCH)
    parallel_compile.compile_all([ov.op, GemmaPrefill(ov).lib])


def _qwen(path: str, mtp: bool = False) -> None:
    """mtp: with the model's MTP drafter file (``Qwen``'s default for an MoE model: ``model_config.qwen_mtp_of``)."""
    from models.qwen35.draft import QwenDraft, draft_formats, draft_gate_up
    from models.qwen35.overlay import QwenOverlay
    from models.qwen35.prefill import QwenPrefill

    drafter = None
    if mtp:
        if (file := qwen_mtp_of(path)) is None:
            pytest.skip(f"no MTP drafter of {path}")
        drafter = GGUFFile(file)
    ov = QwenOverlay(GGUFFile(path), max_positions=4096, formats=draft_formats(drafter) if drafter else (),
                     gu=draft_gate_up(drafter) if drafter else frozenset())
    parallel_compile.compile_all([ov.op, QwenPrefill(ov, draft=QwenDraft(ov, drafter) if drafter else None).lib])
    ov.op.page_store()


def _sd15(path: str) -> None:
    from artlite.vision.design import VisionDesign

    parallel_compile.compile_all([VisionDesign()])


APPS: dict[str, tuple[str, Callable[[str], None]]] = {
    "llama-1B-Q4_K_M": _llama(llama_gguf("Q4_K_M", "1B"), "single"),
    "llama-1B-Q4_K_M-hilo": _llama(llama_gguf("Q4_K_M", "1B"), "hilo"),
    "llama-1B-Q5_K_M": _llama(llama_gguf("Q5_K_M", "1B"), "single"),
    "llama-3B-Q4_K_M": _llama(llama_gguf("Q4_K_M", "3B"), "single"),
    "llama-8B-Q4_K_S": _llama(llama_gguf("Q4_K_S", "8B"), "single"),
    "qwen3-0.6B-Q4_1": _llama(qwen3_gguf("0.6B"), "hilo"),
    "qwen3-embedding-4B-Q4_K_M": _llama(qwen3_gguf("embedding-4B"), "hilo"),
    "gemma-E4B-Q4_0": (gemma_gguf("Q4_0"), _gemma),
    "qwen-9B-Q4_K_M": (qwen_gguf("Q4_K_M", "9B"), _qwen),
    "qwen-27B-Q4_K_M": (qwen_gguf("Q4_K_M", "27B"), _qwen),
    "qwen-1B-Q4_K_M": (qwen_gguf("Q4_K_M", "1B"), _qwen),
    "qwen-35B-A3B-Q4_K_M": (qwen_gguf("Q4_K_M", "35B-A3B"), _qwen),
    "qwen-35B-A3B-Q4_K_M-mtp": (qwen_gguf("Q4_K_M", "35B-A3B"), lambda path: _qwen(path, mtp=True)),
    "occamy-1.0-APEX": (occamy_gguf(), _qwen),
    "swift-1.5-APEX": (swift_gguf(), _qwen),
    "sd15": (sd15_dir(), _sd15),
}


@pytest.mark.parametrize("app", list(APPS))
def test_app_designs(app: str) -> None:
    path, build = APPS[app]
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    build(path)
