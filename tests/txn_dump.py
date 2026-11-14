"""The apps' host-built transactions, dumped and compared: the refactoring check of the overlays' host side (``pytest
--dump-designs`` + ``design_diff.py`` check the designs): the language models' decode overlays and the media encoders on
the vision overlay.

    python tests/txn_dump.py DIR [--only REGEX]    # per app present: its overlay on a fake device, a scripted run
                                                    # (decode steps; images, a video, clips); per dispatch the
                                                    # transaction's words and digests of the host buffers the NPU reads
    python tests/txn_dump.py --restore [--only REGEX]  # the media apps' state caches: a run that builds and saves, then
                                                    # the same run restored: identical dispatches
    python tests/txn_dump.py --diff BEFORE AFTER   # per app the first dispatch that differs, with its first differing
                                                    # operations decoded

The fake device runs the real ``OverlayDevice``, ``Transaction`` and ``Sequence`` on numpy buffers with fixed device
addresses (in allocation order); a dispatch is recorded instead of run; the design is not compiled (its program a
stand-in, its code pages left out) and the weight image is not packed (no transaction depends on its bytes). No NPU
device is opened. A buffer of more than ``DIGEST_MAX`` bytes (the weight image, large KV caches) is recorded by size
only: the host writes into it only at load (the image) or not at all in these runs (the caches). The vision overlay's
design compiles for real (cached builds; its role pages link from it), only its program is a stand-in, and the media
models' weights are packed as usual (their digests cover them); their state caches are bypassed.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from unittest import mock

import numpy as np

from artlite import _native, txn
from artlite.decode.image import TileImage
from artlite.gguf_model import GGUFFile
from artlite.model_config import (
    QwenConfig,
    decision_mmproj,
    gemma_gguf,
    lfm3_gguf,
    lfm3a_mmproj,
    lfm3vl_mmproj,
    llama_gguf,
    minicpmv_mmproj,
    occamy_gguf,
    qwen3_gguf,
    qwen_dflash_of,
    qwen_gguf,
    qwen_mtp_of,
    sd15_dir,
    swift_gguf,
)
from artlite.moe import route as moe_route
from artlite.operators import cdo

BASE = 1 << 40  # the first buffer's device address (then STRIDE apart: every buffer fits, addresses stay < 2^48)
STRIDE = 1 << 38
DIGEST_MAX = 256 << 20


class _Buffer:
    def __init__(self, size: int, paddr: int):
        self.size, self.paddr = size, paddr
        self._a = np.zeros(size, np.uint8)

    def map(self) -> np.ndarray:
        return self._a

    def paddr_in(self, context: object) -> int:
        return self.paddr

    def sync_to_device(self, *args: object) -> None:
        pass

    def sync_from_device(self, *args: object) -> None:
        pass

    def sync_lines(self, *args: object) -> None:
        pass


class _Log:
    """The dispatches of a run: (label, transaction words, [(buffer size, digest or None)])."""

    def __init__(self):
        self.label = ""
        self.entries: list[tuple[str, np.ndarray, list[tuple[int, str | None]]]] = []

    def record(self, insts: _Buffer, buffers: list[_Buffer]) -> None:
        w = insts.map().view(np.uint32)
        words = w[:int(w[3]) // 4].copy()  # (header word 3: the transaction's bytes)
        bufs = [(b.size, hashlib.sha256(b.map()).hexdigest()[:16] if b.size <= DIGEST_MAX else None) for b in buffers]
        self.entries.append((self.label, words, bufs))


class _Device:
    def __init__(self):
        self.n = 0

    def _alloc(self, size: int) -> _Buffer:
        self.n += 1
        return _Buffer(size, BASE + self.n * STRIDE)

    alloc_host_buffer = alloc_user_buffer = alloc_insts_buffer = _alloc


class _Kernel:
    def __init__(self, log: _Log):
        self.log = log

    def dispatch(self, insts: _Buffer, buffers: list[_Buffer], **kw: object) -> None:
        self.log.record(insts, buffers)

    def dispatch_async(self, insts: _Buffer, buffers: list[_Buffer]) -> None:
        self.log.record(insts, buffers)

    def dispatch_wait(self, **kw: object) -> None:
        pass


class _Program:
    def __init__(self, device: _Device, log: _Log, insts_path: str):
        self._device, self._ctx, self.insts_path = device, None, insts_path
        self._k = _Kernel(log)

    def _ensure_loaded(self) -> None:
        pass

    def _kernel(self) -> _Kernel:
        return self._k


@contextlib.contextmanager
def fake_npu() -> Iterator[Callable[[object], _Log]]:
    """Within: ``attach(op)`` gives a decode design's operator a fake program (on one fake device) and returns the log
    of its dispatches. The NPU's turn is a no-op, the weight image is not packed, the checks that read the compiled
    design (its endpoints, the MoE routes) do not run."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as empty, \
            mock.patch.object(_native, "turn_enter", lambda: None), \
            mock.patch.object(_native, "turn_leave", lambda: None), \
            mock.patch.object(_native, "turn_foreign", lambda: 0), \
            mock.patch.object(TileImage, "pack", lambda self, out, **kw: False), \
            mock.patch.object(cdo, "values", lambda work_dir: {}), \
            mock.patch.object(moe_route, "check_route", lambda values, cols: None):  # (the compiled design's)
        dev, log = _Device(), _Log()

        def attach(op: object) -> _Log:
            op._program = _Program(dev, log, empty.name)
            op.check_endpoints = lambda insts: None  # (the compiled design's: none here)
            op.page_store = lambda: None  # (code pages: linked from the compiled design)
            return log

        yield attach


@contextlib.contextmanager
def fake_vision(cache: str | None = None) -> Iterator[_Log]:
    """Within: the vision overlay's devices on one fake device, the log of their dispatches. The vision design compiles
    for real (a cached build: its role pages link from it), only its program is a stand-in; the media models' state
    caches neither restore nor save (every run builds its programs; ``cache``: a directory they use, the run restores
    what an earlier one saved there); their weights pack as usual."""
    from artlite.vision import device as vision_device
    from artlite.vision import shell

    real = vision_device.OverlayDevice
    dev, log = _Device(), _Log()

    def overlay_device(op: object, **kw: object) -> object:
        if not isinstance(op._program, _Program):
            op.compile()
            op._program = _Program(dev, log, op._program.insts_path)
        return real(op, **kw)

    with contextlib.ExitStack() as stack:
        for name, value in (("turn_enter", lambda: None), ("turn_leave", lambda: None), ("turn_foreign", lambda: 0)):
            stack.enter_context(mock.patch.object(_native, name, value))
        stack.enter_context(mock.patch.object(vision_device, "OverlayDevice", overlay_device))
        if cache is None:
            for name in ("load_state", "save_state"):
                stack.enter_context(mock.patch.object(shell, name, lambda *a, **kw: None))
        else:
            stack.enter_context(mock.patch.object(vision_device, "CACHE_DIR", cache))
        yield log


# ----------------------------------------------------------------------------------------------------------------
# The runs: per app family a scripted sequence of steps (slot counts 1 .. 4, record and group boundaries, steps without
# logits, a reset); x: fixed random embeddings
# ----------------------------------------------------------------------------------------------------------------

def _steps(long: int) -> list[tuple[int, int, bool]]:
    """(S, position, logits) of a contiguous run: slot counts 1 .. 4 across record (8) and group (32) boundaries, steps
    without logits, then steps of 4 up to ``long`` (logits every 16th)."""
    out, p = [], 0
    for S, lg in ((1, True), (4, True), (3, False), (2, True), (4, True), (4, True), (1, True), (4, True), (4, True),
                  (4, True), (2, True), (3, True), (1, False), (4, True), (3, True), (4, True), (2, True), (4, True)):
        out.append((S, p, lg))
        p += S
    while p + 4 <= long:
        out.append((4, p, len(out) % 16 == 0))
        p += 4
    return out


def _x(rng: np.random.Generator, S: int, H: int) -> np.ndarray:
    return rng.standard_normal((S, H)).astype(np.float32)


def _run_plain(ov: object, log: _Log, step: Callable[[np.ndarray, int, bool], object], *, lm: bool = True) -> None:
    rng = np.random.default_rng(0)
    ov.load()
    for rnd, long in enumerate((600, 40)):
        for S, p, logits in _steps(long):
            log.label = f"round {rnd} S={S} p={p} logits={logits}"
            step(_x(rng, S, ov.H), p, logits)
        ov.reset()
    if lm:
        log.label = "logits_of"
        ov.logits_of(_x(rng, 1, ov.H)[0])


def _llama(path: str, kv_k: str) -> Callable[[Callable[[object], _Log]], None]:
    def run(attach: Callable[[object], _Log]) -> None:
        from models.llama.overlay import LlamaOverlay

        ov = LlamaOverlay(GGUFFile(path), kv_k=kv_k)
        log = attach(ov.op)
        _run_plain(ov, log, lambda x, p, lg: ov.step(x, p, logits=lg), lm=False)

    return run


def _lfm(path: str) -> Callable[[Callable[[object], _Log]], None]:
    def run(attach: Callable[[object], _Log]) -> None:
        from models.lfm.overlay import LfmOverlay

        ov = LfmOverlay(GGUFFile(path))
        log = attach(ov.op)
        _run_plain(ov, log, lambda x, p, lg: ov.step(x, p, logits=lg))

    return run


def _gemma(path: str) -> Callable[[Callable[[object], _Log]], None]:
    def run(attach: Callable[[object], _Log]) -> None:
        from models.gemma4.gemma import BATCH
        from models.gemma4.overlay import GemmaOverlay

        ov = GemmaOverlay(GGUFFile(path), max_positions=4096 + BATCH)
        log = attach(ov.op)
        rng = np.random.default_rng(1)
        e_shape = (len(ov.layers), ov.cfg.per_layer_dim) if getattr(ov.cfg, "per_layer_dim", 0) else None

        def step(x: np.ndarray, p: int, lg: bool) -> None:
            e = rng.standard_normal((x.shape[0], *e_shape)).astype(np.float32) if e_shape else None
            ov.step(x, e, p, logits=lg)

        _run_plain(ov, log, step, lm=False)

    return run


def _qwen_steps(ov: object, log: _Log, rng: np.random.Generator) -> None:
    """A qwen35 overlay's main steps: committing steps, lazy steps accepting some of their tokens, a reset between rounds;
    then a prompt's hand-off (the conv states in GHDR, the rings from the prefill's caches, a snapshot restored) and,
    where steps of 8 slots may branch, branching steps."""
    for rnd in range(2):
        p = 0
        # (S, commit, accepted): committing steps, lazy steps accepting some of their tokens
        for S, commit, n in ((1, True, 1), (4, True, 4), (4, False, 2), (3, False, 3), (2, False, 1), (4, True, 4),
                             (1, False, 1), (4, False, 4), (4, False, 0), (4, True, 4), (3, False, 2)):
            S = min(S, ov.SLOTS)
            n = min(n, S)
            log.label = f"round {rnd} S={S} p={p} commit={commit} accept={n}"
            ov.step(_x(rng, S, ov.H), p, logits=True, commit=commit)
            ov.accept(S if commit else n)
            p += S if commit else n
        while rnd == 0 and p + 4 <= 600:  # (then committing steps of 4, logits every 16th)
            S = min(4, ov.SLOTS)
            log.label = f"round {rnd} S={S} p={p} commit=True accept={S}"
            ov.step(_x(rng, S, ov.H), p, logits=p % 64 == 0, commit=True)
            ov.accept(S)
            p += S
        ov.reset()
    # a prompt's hand-off: the prefill's conv states and open V records, a decode step after it at p0 (a block's
    # start or inside one), the states carried to a second prompt chunk and back, a snapshot restored later
    dev = ov.dev
    for p0 in (40, 45):
        ov.conv.fresh()
        for name in ("GHDR", "GSTATE"):
            if name in dev.buffers:
                dev.view(name)[:] = rng.integers(0, 256, dev.view(name).size, np.uint8)
                dev.flush(name)
        ov.vring.from_prefill.update(ov.fatt)
        p = p0
        for i, (S, commit, n) in enumerate(((3, False, 2), (4, True, 4), (2, False, 2), (4, False, 1), (1, True, 1))):
            S = min(S, ov.SLOTS)
            n = min(n, S)
            log.label = f"prompt {p0} S={S} p={p} commit={commit} accept={n}"
            ov.step(_x(rng, S, ov.H), p, logits=True, commit=commit)
            ov.accept(S if commit else n)
            p += S if commit else n
            if i == 1:
                snap, snap_at = ov.snapshot(p), p
        digest, arrays = hashlib.sha256(), {**snap.extra, "rings": snap.rings}  # (the snapshot's arrays by name)
        for k in sorted(arrays):
            if isinstance(arrays[k], np.ndarray):
                digest.update(arrays[k].tobytes())
        ov.restore(snap)
        p = snap_at
        for S, commit, n in ((4, True, 4), (4, False, 3)):
            S = min(S, ov.SLOTS)
            n = min(n, S)
            log.label = f"restored {digest.hexdigest()[:8]} S={S} p={p} commit={commit} accept={n}"
            ov.step(_x(rng, S, ov.H), p, logits=True, commit=commit)
            ov.accept(S if commit else n)
            p += S if commit else n
        while p % 8 or ov.cur.nrep:  # (a prefill continues from a block's start: the states fully accepted)
            ov.step(_x(rng, 1, ov.H), p, logits=False, commit=True)
            ov.accept(1)
            p += 1
        ov.conv.for_prefill()
        log.label = f"after a prompt chunk p={p}"
        ov.step(_x(rng, min(4, ov.SLOTS), ov.H), p, logits=True, commit=True)
        ov.accept(min(4, ov.SLOTS))
        ov.reset()
    if getattr(ov, "can_branch", False):  # (steps of 8: chains and trees of two branches inside the slots' 8 positions)
        p, cands = 0, ((0, 1), (1, 2), (3, 2), (5, 2), (0, 2), (2, 1), (4, 1), (6, 1))
        for i in range(24):
            branch = next(((c, L) for c, L in cands[i % len(cands):] + cands[:i % len(cands)]
                           if p % 8 + c + L <= 7), None) if i % 5 != 4 else None
            n = (5, 3, 6, 2, 7, 1)[i % 6]
            log.label = f"branch S=8 p={p} branch={branch} accept={n}"
            ov.step(_x(rng, 8, ov.H), p, logits=True, commit=False, branch=branch)
            ov.accept(n)
            p += n
        ov.reset()


def _qwen(path: str, *, draft: str | None = None, dflash: str | None = None
          ) -> Callable[[Callable[[object], _Log]], None]:
    """A qwen35 model's run; ``draft``: its MTP drafter ("": the file's own block, else a drafter file), ``dflash``: a
    DFlash2 drafter's file (the drafter's steps follow the model's)."""
    def run(attach: Callable[[object], _Log]) -> None:
        from models.qwen35.dflash import DFlashOverlay
        from models.qwen35.draft import QwenDraft, draft_formats, draft_gate_up
        from models.qwen35.image import QwenImage
        from models.qwen35.overlay import QwenOverlay

        gguf = GGUFFile(path)
        mtp = GGUFFile(draft) if draft else None
        if dflash:
            ov = QwenOverlay(gguf, max_positions=4096, image=QwenImage(gguf, QwenConfig.from_gguf(gguf), mtp=False),
                             drafts=True)
        elif draft is not None:
            ov = QwenOverlay(gguf, max_positions=4096, drafts=True, formats=draft_formats(mtp) if mtp else (),
                             gu=draft_gate_up(mtp) if mtp else frozenset())
        else:
            ov = QwenOverlay(gguf, max_positions=4096)
        log = attach(ov.op)
        rng = np.random.default_rng(2)
        ov.load()
        if draft is not None:
            drafter = QwenDraft(ov, mtp)
            drafter.load()
            _qwen_mtp(ov, drafter, log, rng)
        elif dflash:
            drafter = DFlashOverlay(ov, GGUFFile(dflash))
            attach(drafter.op)
            drafter.load()
            _qwen_dflash(ov, drafter, log, rng)
        else:
            _qwen_steps(ov, log, rng)

    return run


def _qwen_mtp(ov: object, drafter: object, log: _Log, rng: np.random.Generator) -> None:
    """The MTP drafter's steps (its rows: [token embedding | the model's hidden state before]), between the model's: a
    step of every slot count, with and without logits, then the main steps."""
    for rnd in range(2):
        p = 0
        for S, lg in ((1, True), (2, False), (4, True), (3, True), (ov.SLOTS, True), (1, False), (ov.SLOTS, False),
                      (2, True), (1, True), (4, False)):
            S = min(S, ov.SLOTS)
            log.label = f"draft round {rnd} S={S} p={p} logits={lg}"
            drafter.step(_x(rng, S, ov.H), _x(rng, S, ov.H), p, logits=lg)
            p += S
        p = 0
        for S, n in ((1, 1), (4, 3), (ov.SLOTS, 5), (2, 2), (4, 1)):
            S, n = min(S, ov.SLOTS), min(n, S)
            log.label = f"model round {rnd} S={S} p={p}"
            ov.step(_x(rng, S, ov.H), p, logits=True, commit=S == 1)
            ov.accept(S if S == 1 else n)
            p += n
            log.label = f"draft after round {rnd} S={n} p={p - n}"
            drafter.step(_x(rng, n, ov.H), _x(rng, n, ov.H), p - n, logits=True)
        drafter.reset()
        ov.reset()
    _qwen_steps(ov, log, rng)


def _qwen_dflash(ov: object, drafter: object, log: _Log, rng: np.random.Generator) -> None:
    """The DFlash2 drafter's dispatches: the injection of the kept rows of a model's step (S slots, ``keep`` of them,
    from the slots ``src`` where they are not the first: ``DFlashDrafting.move_taps``) with the block, an injection alone,
    a block alone, rings wrapping past their window."""
    from types import SimpleNamespace

    from models.qwen35.drafting import DFlashDrafting, Injection

    H = ov.H
    for rnd in range(2):
        p = 0
        if rnd:  # (a prompt's end at 2300: the ring's slots wrap within the rounds)
            ov.conv.fresh()
            ov.vring.from_prefill.update(ov.fatt)
            p = drafter.next = 2300
        # (S, keep, whether a block follows, the slots of the kept rows when not the first ones)
        rounds = [(1, 1, True, None), (8, 5, True, None), (8, 8, True, None), (8, 3, False, None), (4, 4, True, None),
                  (8, 6, True, [0, 1, 2, 5, 6, 7]), (8, 7, True, [1, 2, 3, 4, 5, 6, 7]), (2, 2, True, None)]
        rounds += [(8, 8, True, None)] * (14 if rnd else 20)
        for S, keep, block, src in rounds:
            log.label = f"model step S={S} p={p} accept={keep}"
            ov.step(_x(rng, S, H), p, logits=False, commit=S == 1)
            ov.accept(keep)
            if src is not None:
                fd = DFlashDrafting(SimpleNamespace(overlay=ov, dflash=drafter, cfg=ov.cfg))
                fd.inj = Injection(S, keep, p, src)
                fd.move_taps()
            log.label = f"dflash S={S} keep={keep} q={p} block={block} p={p + keep}"
            drafter.step(S, keep, p, _x(rng, 8, H) if block else None, p + keep)
            p += keep
        log.label = f"dflash block alone p={p}"
        drafter.step(0, 0, 0, _x(rng, 8, H), p)
        drafter.reset()
        ov.reset()
    _qwen_steps(ov, log, rng)


def _image(rng: np.random.Generator, w: int, h: int) -> object:
    from PIL import Image

    return Image.fromarray(rng.integers(0, 256, (h, w, 3), np.uint8))


def _vision(pkg: str, sizes: tuple[tuple[int, int], ...], *, video: tuple[int, int] | None = None
            ) -> Callable[[str], Callable[[_Log], None]]:
    """An image encoder's run: images of ``sizes`` (w, h); ``video``: then 3 frames of that size (two pairs)."""
    def make(path: str) -> Callable[[_Log], None]:
        def run(log: _Log) -> None:
            npu = importlib.import_module(f"models.{pkg}.npu")
            rng = np.random.default_rng(3)
            log.label = "load"
            enc = npu.NpuVision(path)
            for w, h in sizes:
                log.label = f"image {w}x{h}"
                enc.embeddings(_image(rng, w, h))
            if video is not None:
                log.label = f"video 3 frames {video[0]}x{video[1]}"
                enc.pairs(rng.integers(0, 256, (3, video[1], video[0], 3), np.uint8))

        return run

    return make


def _audio(path: str) -> Callable[[_Log], None]:
    def run(log: _Log) -> None:
        from models.lfm3a.npu import NpuAudio

        rng = np.random.default_rng(4)
        log.label = "load"
        enc = NpuAudio(path)
        for seconds in (1.5, 7.0):
            log.label = f"clip {seconds} s"
            enc.embeddings((0.1 * rng.standard_normal(int(16000 * seconds))).astype(np.float32))

    return run


def _diffusion(path: str) -> Callable[[_Log], None]:
    """Stable Diffusion 1.5's UNet step (two timesteps, a new context between them) and VAE decode."""
    def run(log: _Log) -> None:
        from models.sd15 import npu

        rng = np.random.default_rng(5)
        log.label = "load"
        unet, vae = npu.load(path)
        context = rng.standard_normal((2, 77, 768)).astype(np.float32)
        latents = rng.standard_normal((2, 4, 64, 64)).astype(np.float32)
        for step, t in enumerate((981.0, 481.0)):
            log.label = f"unet step {step}"
            unet(latents, t, context)
        log.label = "unet new context"
        unet(latents, 1.0, context + 1.0)
        log.label = "vae"
        vae(latents[:1])

    return run


MEDIA_APPS: dict[str, tuple[str, Callable[[str], Callable[[_Log], None]]]] = {
    "qwen3vl-d3-lite": (decision_mmproj("d3-lite"), _vision("qwen3vl", ((224, 224), (640, 480)))),
    "minicpmv-4.7": (minicpmv_mmproj(), _vision("minicpmv", ((448, 448), (1024, 640)))),
    "lfm3vl-2B": (lfm3vl_mmproj(), _vision("lfm3vl", ((256, 256), (640, 480)), video=(320, 256))),
    "lfm3a-audio-2B": (lfm3a_mmproj(), _audio),
    "sd15": (sd15_dir(), lambda path: _diffusion(path)),
}

APPS: dict[str, tuple[str, Callable[[str], Callable[[Callable[[object], _Log]], None]]]] = {
    "llama-1B-Q4_K_M": (llama_gguf("Q4_K_M", "1B"), lambda p: _llama(p, "single")),
    "llama-1B-Q4_K_M-hilo": (llama_gguf("Q4_K_M", "1B"), lambda p: _llama(p, "hilo")),
    "llama-3B-Q4_K_M": (llama_gguf("Q4_K_M", "3B"), lambda p: _llama(p, "single")),
    "llama-8B-Q4_K_S": (llama_gguf("Q4_K_S", "8B"), lambda p: _llama(p, "single")),
    "qwen3-0.6B-Q4_1": (qwen3_gguf("0.6B"), lambda p: _llama(p, "hilo")),
    "gemma-E4B-Q4_0": (gemma_gguf("Q4_0"), _gemma),
    "qwen-1B-Q4_K_M": (qwen_gguf("Q4_K_M", "1B"), _qwen),
    "qwen-9B-Q4_K_M": (qwen_gguf("Q4_K_M", "9B"), _qwen),
    "qwen-27B-Q4_K_M": (qwen_gguf("Q4_K_M", "27B"), _qwen),
    "qwen-35B-A3B-Q4_K_M": (qwen_gguf("Q4_K_M", "35B-A3B"), _qwen),
    "occamy-1.0-APEX": (occamy_gguf(), _qwen),
    "swift-1.5-APEX": (swift_gguf(), _qwen),
    "qwen-35B-A3B-mtp": (qwen_gguf("Q4_K_M", "35B-A3B"), lambda p: _qwen(p, draft=qwen_mtp_of(p))),
    "swift-1.5-APEX-mtp": (swift_gguf(), lambda p: _qwen(p, draft="")),
    "qwen-3.8-27B-dflash": (qwen_gguf("Q4_K_M", "3.8-27B"), lambda p: _qwen(p, dflash=qwen_dflash_of(p))),
    "lfm3-QAD-Q4_0": (lfm3_gguf("QAD-Q4_0"), _lfm),
    "lfm3-Q4_K_M": (lfm3_gguf("Q4_K_M"), _lfm),
}


def _record(path: str, make: Callable[[str], Callable[[Callable[[object], _Log]], None]]) -> _Log:
    """The dispatches of an app's run on the fake device."""
    with fake_npu() as attach:
        logs: list[_Log] = []

        def attach_logged(op: object) -> _Log:
            logs.append(attach(op))
            return logs[-1]

        make(path)(attach_logged)
    return logs[0]


def _record_media(path: str, make: Callable[[str], Callable[[_Log], None]], cache: str | None = None) -> _Log:
    with fake_vision(cache) as log:
        make(path)(log)
    return log


def restore_check(only: str | None) -> int:
    """Per media app: a run that builds and saves its state, then the same run restored from it, on fresh fake devices;
    their dispatches' words and buffers must be identical."""
    bad = 0
    for app, (path, make) in MEDIA_APPS.items():
        if (only and not re.search(only, app)) or not os.path.exists(path):
            continue
        with tempfile.TemporaryDirectory() as cache:
            built = _record_media(path, make, cache).entries
            assert any(os.scandir(cache)), f"{app}: no state saved"
            from artlite.vision import shell

            def no_save(*a: object, **kw: object) -> None:
                raise AssertionError("the state was rebuilt, not restored")

            with mock.patch.object(shell, "save_state", no_save):
                restored = _record_media(path, make, cache).entries
        same = len(built) == len(restored) and all(
            a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2] for a, b in zip(built, restored, strict=True))
        print(f"{app}: restored run {'identical' if same else 'DIFFERS'} ({len(built)} / {len(restored)} dispatches)")
        bad += not same
    return 1 if bad else 0


def dump(out: str, only: str | None) -> None:
    os.makedirs(out, exist_ok=True)
    runs = [(app, path, lambda p, m=make: _record(p, m)) for app, (path, make) in APPS.items()]
    runs += [(app, path, lambda p, m=make: _record_media(p, m)) for app, (path, make) in MEDIA_APPS.items()]
    for app, path, record in runs:
        if only and not re.search(only, app):
            continue
        if not os.path.exists(path):
            print(f"{app}: {path} missing, skipped")
            continue
        entries = record(path).entries
        np.savez_compressed(os.path.join(out, f"{app}.npz"), *[w for _, w, _ in entries])
        with open(os.path.join(out, f"{app}.json"), "w") as f:
            json.dump([{"label": lb, "words": hashlib.sha256(w).hexdigest()[:16], "n": int(w.size), "buffers": bufs}
                       for lb, w, bufs in entries], f, indent=0)
        print(f"{app}: {len(entries)} dispatches, {sum(w.size for _, w, _ in entries) * 4 >> 10} KB of words")


def _entries(d: str, name: str) -> list[dict]:
    with open(os.path.join(d, name)) as f:
        return json.load(f)


def _first_ops(a: np.ndarray, b: np.ndarray, n: int = 3) -> str:
    oa, ob = txn.op_offsets(a[4:]), txn.op_offsets(b[4:])
    for i, ((pa, ca), (pb, cb)) in enumerate(zip(oa, ob, strict=False)):
        na = (oa[i + 1][0] if i + 1 < len(oa) else a.size - 4) - pa
        nb = (ob[i + 1][0] if i + 1 < len(ob) else b.size - 4) - pb
        if ca != cb or not np.array_equal(a[4 + pa:4 + pa + na], b[4 + pb:4 + pb + nb]):
            da, db = txn.decode(a), txn.decode(b)
            return f"operation {i} of {len(oa)} / {len(ob)}: {da[i:i + n]} vs {db[i:i + n]}"
    return f"operation counts {len(oa)} vs {len(ob)}"


def diff(before: str, after: str) -> int:
    bad = 0
    for name in sorted(f for f in os.listdir(before) if f.endswith(".json")):
        app = name[:-5]
        if not os.path.exists(os.path.join(after, name)):
            print(f"{app}: only before")
            bad += 1
            continue
        ea, eb = _entries(before, name), _entries(after, name)
        if len(ea) != len(eb):
            print(f"{app}: {len(ea)} vs {len(eb)} dispatches")
            bad += 1
        for i, (x, y) in enumerate(zip(ea, eb, strict=False)):
            if x == y:
                continue
            bad += 1
            msg = f"{app}: dispatch {i} ({x['label']} / {y['label']}) differs"
            if x["words"] != y["words"]:
                wa = np.load(os.path.join(before, f"{app}.npz"))[f"arr_{i}"]
                wb = np.load(os.path.join(after, f"{app}.npz"))[f"arr_{i}"]
                msg += f": words, {_first_ops(wa, wb)}"
            if x["buffers"] != y["buffers"]:
                msg += f": buffers {[j for j, (p, q) in enumerate(zip(x['buffers'], y['buffers'], strict=False)) if p != q]}"
            print(msg)
            break
        else:
            if len(ea) == len(eb):
                print(f"{app}: {len(ea)} dispatches identical")
    for name in sorted(f for f in os.listdir(after) if f.endswith(".json") and not os.path.exists(os.path.join(before, f))):
        print(f"{name[:-5]}: only after")
        bad += 1
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--diff", action="store_true", help="compare two dumps")
    ap.add_argument("--restore", action="store_true", help="the media apps built, saved, restored: runs compared")
    ap.add_argument("--only", help="the apps whose names match (a regular expression)")
    a = ap.parse_args()
    if a.diff:
        sys.exit(diff(*a.dirs))
    if a.restore:
        sys.exit(restore_check(a.only))
    dump(a.dirs[0], a.only)


if __name__ == "__main__":
    main()
