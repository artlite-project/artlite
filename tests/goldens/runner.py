"""What every llama.cpp golden shares: a dump tool compiled on first use against the llama.cpp tree of
``model_config.LLAMA_CPP_DIR`` (into the artlite cache, named by a hash of its source and of the libraries it links), and
a run's output folder cached under ``tests/refs/<model>/<prefix>_<hash>/`` (``--refresh-refs`` recomputes): the hash covers
a key (the model files' identity, the inputs, the libraries) and the tool's source; a run writes into a temporary folder
that replaces the cached one only when it succeeded."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

from artlite.model_config import LLAMA_CPP_DIR
from artlite.operators.base import CACHE_DIR

from .. import common
from ..llama_cpp import lib_identity, llama_cpp_bin


def build_tool(name: str, src: str, *, mtmd: bool = False) -> str:
    """The path of the dump tool ``name`` (``src``: its C++ file), compiled on first use; ``mtmd``: it links libmtmd
    (llama.cpp's media library) and includes its headers."""
    ident = lib_identity() + (lib_identity("libmtmd.so") if mtmd else "")
    with open(src, "rb") as f:
        key = hashlib.sha256(f.read() + ident.encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{name}_{key}")
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        bindir = os.path.dirname(llama_cpp_bin("libllama.so"))
        tmp = f"{path}.{os.getpid()}.tmp"
        cmd = ["g++", "-O2", "-std=c++17"]
        if mtmd:
            cmd += ["-Wno-deprecated-declarations", f"-I{LLAMA_CPP_DIR}/include", f"-I{LLAMA_CPP_DIR}/ggml/include",
                    f"-I{LLAMA_CPP_DIR}/tools/mtmd"]
        else:
            cmd += [f"-I{LLAMA_CPP_DIR}/include", f"-I{LLAMA_CPP_DIR}/src", f"-I{LLAMA_CPP_DIR}/ggml/include"]
        cmd += [src, "-o", tmp, f"-L{bindir}", *(["-lmtmd"] if mtmd else []), "-lllama", "-lggml", "-lggml-base",
                f"-Wl,-rpath,{bindir}"]
        subprocess.run(cmd, check=True)
        os.replace(tmp, path)
    return path


def run_tool(cmd: list[str]) -> None:
    """Run a dump tool; a failure raises with its command and the end of its stderr."""
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: exit {r.returncode}\n{r.stderr[-4000:]}")


def cached_golden(model: str, prefix: str, key: object, src: str, run: Callable[[str], None], *,
                  marker: str = "meta.json") -> str:
    """The folder of a golden: ``tests/refs/<model>/<prefix>_<hash of key and the tool's source>``; where it lacks its
    ``marker`` file (or ``--refresh-refs``) ``run(folder)`` fills a temporary folder, which then replaces it."""
    h = hashlib.sha256(repr(key).encode())
    with open(src, "rb") as f:
        h.update(f.read())
    folder = common.model_refs(model)
    path = os.path.join(folder, f"{prefix}_{h.hexdigest()[:16]}")
    if os.path.exists(os.path.join(path, marker)) and not common.REFRESH_REFS:
        return path
    os.makedirs(folder, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=folder, prefix=".tmp_")
    try:
        run(tmp)
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path
