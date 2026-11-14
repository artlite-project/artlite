"""A built dispatch as data (``Program``): its transaction's words, the shim addresses in them by DDR image
(``VisionDevice.refs``) and the addresses the host sets per run (tags), so that one built program runs on any device
whose images were restored (``VisionDevice.restore``): resolved there, it is the transaction a build gives on the
device it was built on. The one code path of every network's programs, built (``Program.build``) or restored from a
cached state (the pickled Program)."""

from __future__ import annotations

import glob
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from ..overlay import Transaction
from .builder import Builder
from .device import VisionDevice


@dataclass
class Program:
    """A dispatch of a network: the transaction's ``words``, their shim addresses by image (``refs``: ``VisionDevice.
    refs``) and its tagged shim addresses (``tags``: [word, image, offset, tag], from the build's ``MemTask`` tags: the
    host sets them before a run, ``retarget``)."""

    words: np.ndarray
    refs: tuple
    tags: list = field(default_factory=list)

    @classmethod
    def of(cls, npu: VisionDevice, b: Builder, **extra: object) -> Self:
        """The program ``b`` has written (finished here); ``extra``: a subclass's fields."""
        words = b.finish()
        tags = []
        if b.s.tags:  # (each tagged address's image and offset at build time)
            refs = npu.refs([(i, int(words[i]) | int(words[i + 1]) << 32) for i, _ in b.s.tags])
            tags = [(int(i), refs[1][k], int(off), tag) for (i, tag), (k, off) in zip(b.s.tags, refs[2], strict=True)]
        return cls(words, npu.refs(b.s.t.addrs), tags, **extra)

    @classmethod
    def build(cls, npu: VisionDevice, bd: Builder, build: Callable[[Builder], None], **extra: object) -> Self:
        """The program ``build`` writes on a fork of ``bd`` (its DDR images shared with ``bd``'s other programs)."""
        b = bd.fork()
        build(b)
        return cls.of(npu, b, **extra)

    @property
    def images(self) -> set[str]:
        """The images whose addresses the program holds."""
        return set(self.refs[1]) | {t[1] for t in self.tags}

    def resolved(self, npu: VisionDevice, rename: Callable[[str], str] = lambda n: n) -> np.ndarray:
        """The words with their shim addresses on ``npu``'s images (``rename`` maps an image to the one taken in its
        place: each of at least its bytes); a copy."""
        words = self.words.copy()
        npu.resolve(words, self.refs, rename)
        return words

    def transaction(self, npu: VisionDevice, rename: Callable[[str], str] = lambda n: n) -> Transaction:
        """The program as a transaction on ``npu`` (``resolved``)."""
        return Transaction(npu.dev, self.resolved(npu, rename))

    def retarget(self, npu: VisionDevice, tr: Transaction, address: Callable[[object, int], int]) -> None:
        """Set the tagged addresses in ``tr`` (the program's transaction): each tag's address is ``address(tag, base)``,
        base its image and offset on ``npu``; ``tr.flush()`` after."""
        for idx, name, off, tag in self.tags:
            a = address(tag, npu.address(name, off))
            tr.words[idx], tr.words[idx + 1] = a & 0xFFFFFFFF, a >> 32


class ProgramStore:
    """The programs a network builds after its state was cached (``state.save_state``: the first ones), as pickles in
    the state's directory (``program_<key>.pkl``): the next load takes them with the state (``restore``)."""

    def __init__(self, cache: str):
        self.cache = cache

    def save(self, key: int | tuple[str, int], pieces: list[Program], same_layout: bool) -> None:
        """A program (key: its level, ``("pair", level)`` a video pair's) into the directory, if it takes only the
        cached state's images (``same_layout``: ``WindowEncoder.state``)."""
        if not same_layout or not os.path.isdir(self.cache):
            return
        path = os.path.join(self.cache, f"program_{key if isinstance(key, int) else '_'.join(map(str, key))}.pkl")
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump((key, pieces), f)
        os.replace(tmp, path)

    def restore(self, programs: dict) -> None:
        """The saved programs into ``programs`` (by key)."""
        for f in glob.glob(os.path.join(glob.escape(self.cache), "program_*.pkl")):
            with open(f, "rb") as fh:
                key, pieces = pickle.load(fh)
            programs[key] = pieces
