"""The apps' designs as the apps build them (no NPU): per app its model file and its builder, which compiles the app's
designs and returns what it built (``AppBuild``): the decode overlays and prefill libraries of the models present
(Qwen3 on Llama's path with its q / k norm weights, and Qwen3-0.6B-PARO's, Qwen3.5-0.8B-PARO's, Qwen3.6-35B-A3B-PARO's and
Qwen3.8-27B-PARO's (int4: gate and up as two GEMMs; int5) and Gemma 4 E4B PARO's checkpoint directories: their rotated
inputs, the 27B's also with its base model's DFlash2 drafter; Qwen3.6-35B-A3B (and its PARO checkpoint) also with its
MTP drafter file, the app's default: the design runs the drafter's formats too; Qwen3.8 27B on its design of 8 slots
with its MTP block and with its DFlash2 drafter; LFM3's QAD-Q4_0 and Q4_K_M images; Spark-X2.6-4B with its DSpark
drafter), the vision overlay (the Stable Diffusion UNet's and the media encoders' design). ``test_app_designs.py``
builds them, ``schedule_ledger.py`` reads their builds."""

import os
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

from artlite import parallel_compile
from artlite.gguf_model import GGUFFile
from artlite.hf_model import open_weights
from artlite.model_config import (
    checkpoint,
    flint_gguf,
    gemma_gguf,
    lfm3_gguf,
    llama_gguf,
    occamy_gguf,
    qwen3_gguf,
    qwen_dflash_of,
    qwen_gguf,
    qwen_mtp_of,
    sd15_dir,
    spark_dspark_of,
    spark_gguf,
    swift_gguf,
)
from artlite.operators import paging
from artlite.operators.base import ArtcOperator
from artlite.operators.fused.elf import FusedElf


class AppBuild(NamedTuple):
    """An app's compiled designs, its prefill (the chunks' runlists; None without one), its decode overlay (None: the
    vision overlay) and the pagers of its paged designs' cores."""

    designs: list[ArtcOperator | FusedElf]
    prefill: Any
    overlay: Any
    pagers: list[paging.Pager]


def _built(designs: list[ArtcOperator | FusedElf], prefill: Any, overlay: Any) -> AppBuild:
    """Compile ``designs`` and link the overlay's code pages."""
    parallel_compile.compile_all(designs)
    stores = overlay.op.page_store() or ()
    return AppBuild(designs, prefill, overlay, [pg for store in stores for pg in store.pagers])


def _llama(path: str, kv_k: str) -> tuple[str, Callable[[str], AppBuild]]:
    def build(path: str) -> AppBuild:
        from models.llama.overlay import LlamaOverlay
        from models.llama.prefill import LlamaPrefill

        ov = LlamaOverlay(open_weights(path), kv_k=kv_k)
        pf = LlamaPrefill(ov)
        return _built([ov.op, pf.lib], pf, ov)

    return path, build


def _gemma(path: str) -> AppBuild:
    from models.gemma4.gemma import BATCH
    from models.gemma4.overlay import GemmaOverlay
    from models.gemma4.prefill import GemmaPrefill

    ov = GemmaOverlay(open_weights(path), max_positions=4096 + BATCH)
    pf = GemmaPrefill(ov)
    return _built([ov.op, pf.lib], pf, ov)


def _qwen(path: str, mtp: bool = False) -> AppBuild:
    """mtp: with the model's MTP drafter file (``Qwen``'s default for an MoE model: ``model_config.qwen_mtp_of``)."""
    from models.qwen35.draft import QwenDraft, draft_formats, draft_gate_up
    from models.qwen35.overlay import QwenOverlay
    from models.qwen35.prefill import QwenPrefill

    drafter = None
    if mtp:
        if (file := qwen_mtp_of(path)) is None:
            pytest.skip(f"no MTP drafter of {path}")
        drafter = GGUFFile(file)
    ov = QwenOverlay(open_weights(path), max_positions=4096, formats=draft_formats(drafter) if drafter else (),
                     gu=draft_gate_up(drafter) if drafter else frozenset())
    pf = QwenPrefill(ov, draft=QwenDraft(ov, drafter) if drafter else None)
    return _built([ov.op, pf.lib], pf, ov)


def _qwen_drafts(path: str, dflash: bool) -> AppBuild:
    """A dense qwen35 file with an MTP block (Qwen3.8 27B: a design of 8 slots, the one-pass attention) as its app
    builds it: its own block as the drafter, or (``dflash``) its DFlash2 drafter's file (``model_config.qwen_dflash_of``:
    the app's default where there is one) on a design of its own beside the model's."""
    from artlite.decode.dflash.overlay import DFlashOverlay
    from artlite.model_config import QwenConfig
    from models.qwen35.draft import QwenDraft
    from models.qwen35.image import QwenImage
    from models.qwen35.overlay import QwenOverlay
    from models.qwen35.prefill import QwenPrefill

    gguf = GGUFFile(path)
    if not dflash:
        ov = QwenOverlay(gguf, max_positions=4096, drafts=True)
        pf = QwenPrefill(ov, draft=QwenDraft(ov))
        return _built([ov.op, pf.lib], pf, ov)
    if (file := qwen_dflash_of(path)) is None:
        pytest.skip(f"no DFlash2 drafter of {path}")
    ov = QwenOverlay(gguf, max_positions=4096, image=QwenImage(gguf, QwenConfig.from_gguf(gguf), mtp=False), drafts=True)
    drafter = DFlashOverlay(ov, GGUFFile(file))
    pf = QwenPrefill(ov, draft=None, dflash=drafter)
    return _built([ov.op, drafter.op, pf.lib], pf, ov)


def _qwen_paro_dflash(path: str) -> AppBuild:
    """A PARO checkpoint of a model with a DFlash2 drafter (Qwen3.8-27B-PARO, int4 or int5) as its app builds it: the
    drafter's head on the screened head's plane, its taps as plain copies (``qwen35.image.QwenImage``'s ``dflash``)."""
    from artlite.decode.dflash.overlay import DFlashOverlay
    from models.qwen35.image import QwenImage
    from models.qwen35.overlay import QwenOverlay
    from models.qwen35.prefill import QwenPrefill
    from models.qwen35.qwen import dflash_taps

    if (file := qwen_dflash_of(path)) is None:
        pytest.skip(f"no DFlash2 drafter of {path}")
    g, taps = open_weights(path), dflash_taps(file)
    ov = QwenOverlay(g, max_positions=4096, image=QwenImage(g, mtp=False, dflash=taps), drafts=True, taps=taps)
    drafter = DFlashOverlay(ov, GGUFFile(file))
    pf = QwenPrefill(ov, draft=None, dflash=drafter)
    return _built([ov.op, drafter.op, pf.lib], pf, ov)


def _lfm(path: str) -> AppBuild:
    from models.lfm.overlay import LfmOverlay
    from models.lfm.prefill import LfmPrefill

    ov = LfmOverlay(GGUFFile(path))
    pf = LfmPrefill(ov)
    return _built([ov.op, pf.lib], pf, ov)


def _flint(path: str) -> AppBuild:
    from models.flint.overlay import FlintOverlay
    from models.flint.prefill import FlintPrefill

    ov = FlintOverlay(GGUFFile(path), max_positions=32768)
    pf = FlintPrefill(ov)
    return _built([ov.op, pf.lib], pf, ov)


def _spark(path: str) -> AppBuild:
    """With the model's DSpark drafter (``spark_dspark_of``: the app's default): the overlay runs the drafter's formats
    too and the prefill's chunks write its records."""
    from models.spark.dspark import DSparkOverlay, needs
    from models.spark.overlay import SparkOverlay
    from models.spark.prefill import SparkPrefill

    if (file := spark_dspark_of(path)) is None:
        pytest.skip(f"no DSpark drafter of {path}")
    dg = GGUFFile(file)
    ov = SparkOverlay(GGUFFile(path), max_positions=4096, **needs(dg))
    pf = SparkPrefill(ov, dspark=DSparkOverlay(ov, dg))
    return _built([ov.op, pf.lib], pf, ov)


def _sd15(path: str) -> AppBuild:
    from artlite.vision.design import VisionDesign
    from artlite.vision.device import role_pages

    design = VisionDesign()
    parallel_compile.compile_all([design])
    return AppBuild([design], None, None, [role_pages(os.path.dirname(design._program.insts_path))])


APPS: dict[str, tuple[str, Callable[[str], AppBuild]]] = {
    "llama-1B-Q4_K_M": _llama(llama_gguf("Q4_K_M", "1B"), "single"),
    "llama-1B-Q4_K_M-hilo": _llama(llama_gguf("Q4_K_M", "1B"), "hilo"),
    "llama-1B-Q5_K_M": _llama(llama_gguf("Q5_K_M", "1B"), "single"),
    "llama-3B-Q4_K_M": _llama(llama_gguf("Q4_K_M", "3B"), "single"),
    "llama-8B-Q4_K_S": _llama(llama_gguf("Q4_K_S", "8B"), "single"),
    "qwen3-0.6B-Q4_1": _llama(qwen3_gguf("0.6B"), "hilo"),
    "qwen3-embedding-4B-Q4_K_M": _llama(qwen3_gguf("embedding-4B"), "hilo"),
    "qwen3-0.6B-PARO": _llama(checkpoint("qwen3-0.6B-paro"), "hilo"),
    "qwen3.5-0.8B-PARO": (checkpoint("qwen3.5-0.8B-paro"), _qwen),
    "qwen3.6-35B-A3B-PARO": (checkpoint("qwen3.6-35B-A3B-paro"), _qwen),
    "qwen3.6-35B-A3B-PARO-mtp": (checkpoint("qwen3.6-35B-A3B-paro"), lambda path: _qwen(path, mtp=True)),
    "qwen3.8-27B-PARO": (checkpoint("qwen3.8-27B-paro"), _qwen),
    "qwen3.8-27B-PARO-dflash": (checkpoint("qwen3.8-27B-paro"), _qwen_paro_dflash),
    "qwen3.8-27B-PARO-int5": (checkpoint("qwen3.8-27B-paro-int5"), _qwen),
    "qwen3.8-27B-PARO-int5-dflash": (checkpoint("qwen3.8-27B-paro-int5"), _qwen_paro_dflash),
    "gemma-E4B-Q4_0": (gemma_gguf("Q4_0"), _gemma),
    "gemma-E4B-PARO": (checkpoint("gemma-4-E4B-paro"), _gemma),
    "qwen-9B-Q4_K_M": (qwen_gguf("Q4_K_M", "9B"), _qwen),
    "qwen-27B-Q4_K_M": (qwen_gguf("Q4_K_M", "27B"), _qwen),
    "qwen-3.8-27B-Q4_K_M-mtp": (qwen_gguf("Q4_K_M", "3.8-27B"), lambda path: _qwen_drafts(path, dflash=False)),
    "qwen-3.8-27B-Q4_K_M-dflash": (qwen_gguf("Q4_K_M", "3.8-27B"), lambda path: _qwen_drafts(path, dflash=True)),
    "qwen-1B-Q4_K_M": (qwen_gguf("Q4_K_M", "1B"), _qwen),
    "qwen-35B-A3B-Q4_K_M": (qwen_gguf("Q4_K_M", "35B-A3B"), _qwen),
    "qwen-35B-A3B-Q4_K_M-mtp": (qwen_gguf("Q4_K_M", "35B-A3B"), lambda path: _qwen(path, mtp=True)),
    "occamy-1.0-APEX": (occamy_gguf(), _qwen),
    "swift-1.5-APEX": (swift_gguf(), _qwen),
    "lfm3-QAD-Q4_0": (lfm3_gguf("QAD-Q4_0"), _lfm),
    "lfm3-Q4_K_M": (lfm3_gguf("Q4_K_M"), _lfm),
    "spark-4B-Q4_K_M": (spark_gguf("4B", "Q4_K_M"), _spark),
    "flint-int4": (flint_gguf(), _flint),
    "sd15": (sd15_dir(), _sd15),
}
