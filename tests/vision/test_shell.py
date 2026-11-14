"""The media encoders' shared shell (artlite.vision.shell, ``program``, ``sources``; no NPU): the embedding cache's
LRU, the program pickles, the configuration checks, the tagged addresses of a program, and the state key's source files
derived from the loaders' imports (every module a loader imports at run time is in them)."""

import json
import os
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

from artlite.vision import sources
from artlite.vision.program import Program, ProgramStore
from artlite.vision.shell import EmbeddingCache, digest, require_dims, require_token_bounds

LOADERS = ("models.qwen3vl.npu", "models.minicpmv.npu", "models.lfm3vl.npu", "models.lfm3a.npu", "models.sd15.npu")


def test_embedding_cache_lru() -> None:
    cache, runs = EmbeddingCache(keep=2), []

    def get(key: bytes, count: int = 1) -> np.ndarray:
        return cache.get(key, lambda: runs.append(key) or np.full(1, len(runs), np.float32), count=count)
    a, b = get(b"a"), get(b"b", count=3)
    assert get(b"a") is a and runs == [b"a", b"b"]  # (a hit: not run, and now the newest)
    get(b"c")  # (evicts b, the oldest)
    assert get(b"a") is a and runs == [b"a", b"b", b"c"]
    assert get(b"b") is not b and runs[-1] == b"b"
    assert cache.entries == 1 + 3 + 1 + 1 and cache.t_encode >= 0
    cache.add(2.0, 5)
    assert cache.entries == 11 and cache.t_encode >= 2.0


def test_digest_covers_shape_and_bytes() -> None:
    x = np.arange(12, dtype=np.uint8)
    assert digest(x) == digest(x.copy()) and digest(x) != digest(x.reshape(3, 4))
    assert digest(x) != digest(x + 1) and digest(x, x) != digest(x)


def test_program_store(tmp_path: str) -> None:
    store = ProgramStore(str(tmp_path))
    prog = Program(np.arange(4, dtype=np.uint32), (np.zeros(0, np.int64), [], np.zeros((0, 2), np.int64)))
    store.save(3, [prog], same_layout=False)  # (images the cached state lacks: not kept)
    store.save(("pair", 3), [prog, prog], same_layout=True)
    store.save(5, [prog], same_layout=True)
    got: dict = {}
    store.restore(got)
    assert sorted(map(str, got)) == sorted(["5", "('pair', 3)"])
    assert len(got[("pair", 3)]) == 2 and (got[5][0].words == prog.words).all()
    ProgramStore(str(tmp_path / "missing")).save(1, [prog], same_layout=True)  # (no directory: nothing)


class _Dev:
    """An ``npu`` with images at fixed addresses, a transaction as its words."""

    def __init__(self) -> None:
        self.dev = None
        self.base = {"x": 0x1_0000_0000, "y": 0x2_0000_0000}

    def address(self, name: str, off: int = 0) -> int:
        return self.base[name] + off

    def resolve(self, words: np.ndarray, refs: tuple, rename: Callable[[str], str]) -> None:
        idx, names, at = refs
        a = np.array([self.base[rename(n)] for n in names], np.int64)[at[:, 0]] + at[:, 1]
        words[idx], words[idx + 1] = a & 0xFFFFFFFF, a >> 32


class _Tr:
    def __init__(self, words: np.ndarray) -> None:
        self.words = words


def test_program_resolve_and_retarget() -> None:
    npu = _Dev()
    prog = Program(np.zeros(8, np.uint32), (np.array([0, 4]), ["x", "y"], np.array([[0, 64], [1, 8]])),
                   [(2, "y", 16, "zrow")])
    assert prog.images == {"x", "y"}
    words = prog.resolved(npu, lambda n: {"x": "y"}.get(n, n))  # (an image taken in another's place)
    assert (words[0], words[1], words[4], words[5]) == (64, 2, 8, 2)
    tr = _Tr(prog.resolved(npu))
    seen = []
    prog.retarget(npu, tr, lambda tag, base: seen.append((tag, base)) or base + 100)
    assert seen == [("zrow", 0x2_0000_0010)] and (tr.words[2], tr.words[3]) == (0x74, 2)


def test_require_dims_names_the_constants() -> None:
    require_dims("f", "an encoder", {"hidden": (4, 4), "heads": (2, 2)})
    with pytest.raises(NotImplementedError, match=r"f: an encoder with hidden 4, heads 3 \(the encoder runs hidden 4, heads 2\)"):
        require_dims("f", "an encoder", {"hidden": (4, 4), "heads": (3, 2)})
    require_token_bounds(1, 16384)
    for lo, hi in ((0, 4), (5, 4), (1, 16385)):
        with pytest.raises(ValueError, match="image tokens"):
            require_token_bounds(lo, hi)


def _loaded(root: str) -> set[str]:
    """The modules of artlite and models a fresh interpreter has after importing ``root``."""
    code = f"import {root}, sys, json; print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {sources.ROOTS})))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=os.environ)
    return {m for m in json.loads(out.stdout.splitlines()[-1]) if sources.module_path(m) is not None}


@pytest.mark.parametrize("root", LOADERS)
def test_state_key_covers_the_loaders_imports(root: str) -> None:
    """Everything the loader imports is in its state key's sources, but what only its packages' ``__init__`` import (and
    what no network code uses: ``models.common``): a module loaded at run time that the static graph does not know
    (a dynamic import) fails here."""
    closure = set(sources.import_closure(root))
    inits = [m for m in closure if sources.module_path(m).endswith("__init__.py")]
    known = closure | set(sources.import_closure(*inits))
    assert _loaded(root) <= known
    assert {"artlite.vision.builder", "artlite.vision.device", "artlite.vision.program", "artlite.txn", "artlite.overlay",
            "artlite.operators.kmath", root} <= closure
    files = sources.source_files(root)
    assert any(f.endswith("artlite/vision/core.cc.jinja") for f in files)  # (the kernel templates ride with their module)
    assert all(os.path.isfile(f) for f in files)


def test_state_key_changes_with_a_source(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from artlite.vision import device

    files = [str(tmp_path / "model.gguf")]
    with open(files[0], "wb") as f:
        f.write(b"x")
    monkeypatch.setattr(device, "CACHE_DIR", str(tmp_path))
    key = device.state_dir("k", files[0], ("models.sd15.npu",), files)
    assert key == device.state_dir("k", files[0], ("models.sd15.npu",), files)
    assert key != device.state_dir("k", files[0], ("models.qwen3vl.npu",), files)  # (other modules: other sources)
    with open(files[0], "wb") as f:
        f.write(b"xy")
    assert key != device.state_dir("k", files[0], ("models.sd15.npu",), files)  # (the model's file counts)
