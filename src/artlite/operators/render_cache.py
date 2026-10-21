"""The render manifest: compile an operator whose build is cached without rendering it.

``compile()`` / ``compile_all`` must render an operator (``build_spec``: Python and aie.iron) to find its build's
cache key, and for a FusedOperator of many designs that costs seconds even when everything is built (the 27B qwen35
prefill ELFs: 11.7 s). The manifest (``CACHE_DIR/render/<key>.json``) maps a key of what a render depends on to the
build directory it produced:

- an ArtcOperator: its class (module, qualified name) and constructor arguments (``ArtcOperator.__new__`` records
  them), its clang flags;
- a fused ELF (``FusedOperator``, ``Library``): its operators' keys, the runlist (which operator object, buffer
  names) and what else its render depends on (``render_args``);
- a hash of the artlite package's sources (and of the files defining the classes involved, for classes defined
  outside it, with the user-code files their modules import from), the toolchain's versions and modification times
  (as the builds' keys).

A hit attaches the build directory (when its artifacts are there: a fused ELF's ``artifacts``).
Rendered every time (no key): an argument the key cannot describe (a function, an object without plain data, an
array above 16 MB), a method replaced on the instance or on a class by a function from another module (variant
builds, experiments), profiling, ``ARTLITE_IGNORE_CACHE``, ``ARTLITE_MEASURE_II`` and ``ARTLITE_GENERATE_ASM`` (their
reports come from a build). Not seen: a module global changed at runtime (``fused.layout.SEGMENT_BYTES`` is part of the
key): a test that monkeypatches one gives its operators or ELF their own names, so that their keys differ.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import os
import sys
import threading
import types
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .. import profiling
from .base import AIECC, CACHE_DIR, CLANGPP, ArtcOperator

if TYPE_CHECKING:
    from .fused.elf import FusedElf

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # (the artlite package)
_MAX_ARRAY = 16 << 20
_ENV = ("ARTLITE_IGNORE_CACHE", "ARTLITE_MEASURE_II", "ARTLITE_GENERATE_ASM")


class _NoKey(Exception):
    """An operator whose render the key cannot describe."""


@functools.cache
def _sources() -> str:
    """Hash of the artlite package's sources (Python, kernel headers)."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if not name.endswith((".so", ".pyc")):
                path = os.path.join(root, name)
                h.update(os.path.relpath(path, _PKG).encode())
                with open(path, "rb") as f:
                    h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


@functools.cache
def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _user_file(x: Any) -> str | None:
    """The source file of a module, class or function outside the artlite package and the installed packages (user
    code: experiments, scripts), else None."""
    try:
        path = os.path.abspath(inspect.getfile(x))
    except (TypeError, OSError):
        return None
    if path.startswith(_PKG + os.sep) or not path.endswith(".py") or any(
            path.startswith(os.path.abspath(p) + os.sep) for p in (sys.prefix, sys.base_prefix)):
        return None
    return path


def _class(cls: type, files: set[str]) -> str:
    """A class's name; the file defining it joins the key when it is outside the artlite package, with the user-code
    files its module imports from (a design rendered by helpers of a sibling module)."""
    try:
        path = os.path.abspath(inspect.getfile(cls))
    except (TypeError, OSError) as e:
        raise _NoKey(f"{cls.__qualname__}: no source file") from e
    if not path.startswith(_PKG + os.sep):
        files.add(path)
        module = sys.modules.get(cls.__module__)
        for v in vars(module).values() if module is not None else ():
            if isinstance(v, (types.ModuleType, type, types.FunctionType)) and (f := _user_file(v)) is not None:
                files.add(f)
    return f"{cls.__module__}.{cls.__qualname__}"


def _patched(cls: type) -> bool:
    """A method of the class or its bases replaced by a function of another module."""
    for c in cls.__mro__[:-1]:
        for v in vars(c).values():
            f = v.__func__ if isinstance(v, (staticmethod, classmethod)) else v
            if isinstance(f, types.FunctionType) and f.__module__ != c.__module__:
                return True
    return False


def _describe(x: Any, files: set[str], active: set[int]) -> str:
    """A deterministic description of a constructor argument (raises _NoKey)."""
    if isinstance(x, ArtcOperator):
        k = _op_key(x)
        if k is None:
            raise _NoKey(type(x).__qualname__)
        return f"op:{k}"
    if isinstance(x, enum.Enum):
        return f"{_class(type(x), files)}.{x.name}"
    if x is None or type(x) in (bool, int, float, str, bytes):
        return repr(x)
    if isinstance(x, np.generic):
        return f"np:{x.dtype}:{x.dtype.str}:{x.item()!r}"
    if isinstance(x, np.dtype):
        return f"dtype:{x}:{x.str}"
    if isinstance(x, torch.dtype):
        return str(x)
    if isinstance(x, type):
        return f"type:{x.__module__}.{x.__qualname__}"
    if isinstance(x, types.GenericAlias):
        return f"alias:{x!r}"
    if isinstance(x, np.ndarray):
        if isinstance(x, np.memmap) or x.nbytes > _MAX_ARRAY or x.dtype.hasobject:
            raise _NoKey("array")
        return f"nd:{x.dtype}:{x.dtype.str}:{x.shape}:{hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}"
    if isinstance(x, torch.Tensor):
        if x.device.type != "cpu" or x.numel() * x.element_size() > _MAX_ARRAY:
            raise _NoKey("tensor")
        data = x.detach().contiguous().view(torch.uint8).numpy().tobytes() if x.numel() else b""
        return f"t:{x.dtype}:{tuple(x.shape)}:{hashlib.sha256(data).hexdigest()}"
    if id(x) in active:
        raise _NoKey("a cycle")
    active.add(id(x))
    try:
        if isinstance(x, (tuple, list)):
            name = type(x).__name__ if type(x) in (tuple, list) else _class(type(x), files)
            return f"{name}({','.join(_describe(v, files, active) for v in x)})"
        if isinstance(x, dict) and type(x) is dict:
            return "{" + ",".join(f"{_describe(k, files, active)}:{_describe(v, files, active)}" for k, v in x.items()) + "}"
        if isinstance(x, (set, frozenset)):
            return f"set({','.join(sorted(_describe(v, files, active) for v in x))})"
        if dataclasses.is_dataclass(x):
            fields = ",".join(f"{f.name}={_describe(getattr(x, f.name), files, active)}" for f in dataclasses.fields(x))
            return f"{_class(type(x), files)}({fields})"
        if isinstance(x, (types.FunctionType, types.MethodType, types.ModuleType, functools.partial)) or \
                not hasattr(x, "__dict__") or hasattr(type(x), "__slots__"):
            raise _NoKey(type(x).__qualname__)
        return f"{_class(type(x), files)}{_describe(vars(x), files, active)}"
    finally:
        active.discard(id(x))


def _op_key(op: ArtcOperator) -> str | None:
    """The key of an operator's render (memoized: operators do not change after construction), None: no key."""
    if "_render_key" in op.__dict__:
        return op._render_key
    try:
        if _patched(type(op)) or any(isinstance(v, (types.FunctionType, types.MethodType, functools.partial))
                                     for v in vars(op).values()):
            raise _NoKey("a replaced method")
        args, kwargs = op.__dict__.get("_ctor", ((), {}))
        files: set[str] = set()
        desc = (_class(type(op), files), _describe(args, files, set()), _describe(kwargs, files, set()),
                tuple(op.clang_flags))
        key = hashlib.sha256(repr((desc, sorted((p, _file_hash(p)) for p in files))).encode()).hexdigest()[:32]
    except _NoKey:
        key = None
    op._render_key = key
    return key


def _fused_key(f: FusedElf) -> str | None:
    ops = f.operators
    keys = [_op_key(op) for op in ops]
    if None in keys:
        return None
    index = {id(op): i for i, op in enumerate(ops)}
    try:
        desc = (type(f).__name__, f.name, keys, [(index[id(op)], *bufs) for op, *bufs in f.runlist],
                _describe(f.render_args(), set(), set()))
    except _NoKey:
        return None
    return "fused_" + hashlib.sha256(repr(desc).encode()).hexdigest()[:32]


def _key(op: ArtcOperator | FusedElf) -> str | None:
    """The manifest key of a render, None: render it."""
    if profiling.get_config() is not None or any(os.environ.get(v, "0") == "1" for v in _ENV):
        return None
    k = _op_key(op) if isinstance(op, ArtcOperator) else _fused_key(op)
    if k is None:
        return None
    # (the builds' keys: also the tools' modification times, a reinstalled or locally built toolchain)
    versions = (_sources(), version("mlir_aie"), version("llvm_aie"), os.path.getmtime(AIECC), os.path.getmtime(CLANGPP),
                sys.version_info[:2])
    return hashlib.sha256(repr((k, versions)).encode()).hexdigest()[:32]


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, "render", f"{key}.json")


def lookup(op: ArtcOperator | FusedElf) -> str | None:
    """The build directory of an operator whose render is in the manifest (its artifacts present), else None."""
    key = _key(op)
    if key is None:
        return None
    try:
        with open(_path(key)) as f:
            entry = json.load(f)
    except (OSError, ValueError):
        return None
    work_dir = os.path.join(CACHE_DIR, entry["work_dir"])
    artifacts = ("main.pdi", "insts.bin") if isinstance(op, ArtcOperator) else op.artifacts
    if not all(os.path.exists(os.path.join(work_dir, n)) for n in artifacts):
        return None
    return work_dir


def store(op: ArtcOperator | FusedElf, work_dir: str) -> None:
    """Record the build directory of an operator just rendered and built."""
    key = _key(op)
    if key is None:
        return
    entry: dict[str, Any] = {"work_dir": os.path.relpath(work_dir, CACHE_DIR)}
    path = _path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f)
    os.replace(tmp, path)
