"""The NPU runtime's backend: which build of the native module is ``artlite._native``.

- ``xdna`` (Linux): Artlite's shim on the amdxdna driver's ioctls (``csrc/shim_xdna``). A buffer's device address is
  fixed at its allocation; a fused ELF's control program runs as an ELF (``ERT_START_NPU_PREEMPT_ELF``).
- ``xrt`` (Windows; on Linux ``artlite._native_xrt``, built where XRT is installed): the same runtime over XRT's API
  (``csrc/shim_xrt``). A context opens from an xclbin of the program's PDI, built in memory. A command reaches only the
  XRT buffers it lists, at most 5 (aiecc's kernel): data buffers lie in shared regions and a command lists theirs. A
  buffer's device address exists only once a context does, and on Windows differs between contexts, so programs open
  their context before they resolve addresses (``ADDRESSES_NEED_CONTEXT``). A fused ELF's control program runs as a
  transaction without its DDR patches (``txn.without_ddr_patches``). No profiler register reads, clocks, busy time or
  firmware error queries.

``ARTLITE_BACKEND`` picks one (default: the platform's own, ``xdna`` on Linux); ``artlite`` imports this module first.
"""

import hashlib
import importlib
import os
import shutil
import sys

_DEFAULT = "xrt" if sys.platform == "win32" else "xdna"


def _select() -> object:
    """``artlite._native`` of the backend ``ARTLITE_BACKEND`` names."""
    name = os.environ.get("ARTLITE_BACKEND") or _DEFAULT
    if name == _DEFAULT:
        return importlib.import_module("artlite._native")
    try:
        module = importlib.import_module(f"artlite._native_{name}")
    except ModuleNotFoundError:
        raise ImportError(f"ARTLITE_BACKEND={name}: this build has no such backend (Linux builds artlite._native_xrt "
                          f"where XRT is installed; the platform's own: {_DEFAULT})") from None
    sys.modules["artlite._native"] = module
    return module


_native = _select()
NAME: str = _native.BACKEND
XRT = NAME == "xrt"
# A buffer's device address needs an open context (txn.device_address of a context's buffers)
ADDRESSES_NEED_CONTEXT = XRT


def _source_hash(root: str) -> str:
    """CMakeLists.txt's ARTLITE_SOURCE_HASH of the native module's sources in the checkout at root."""
    files = ["CMakeLists.txt"]
    for d, _, names in os.walk(os.path.join(root, "csrc")):
        rel = os.path.relpath(d, root).replace(os.sep, "/")
        files += [f"{rel}/{n}" for n in names if n.endswith((".cpp", ".h", ".hpp")) or n == "CMakeLists.txt"]
    listing = ""
    for f in sorted(files):
        with open(os.path.join(root, f), "rb") as fh:
            listing += f"{f} {hashlib.sha256(fh.read()).hexdigest()}\n"
    return hashlib.sha256(listing.encode()).hexdigest()


def stale_native() -> str | None:
    """Why the native module is not built from the checkout's csrc/ (a ``git pull`` or an edit without a rebuild: the
    Python side runs on another native side, e.g. another weight packer), else None. Without a checkout around the
    package (an installed wheel) there is nothing to compare."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    if not os.path.isdir(os.path.join(root, "csrc")):
        return None
    built = _native.source_hash() if hasattr(_native, "source_hash") else None
    if built == _source_hash(root):
        return None
    return (f"the native module ({_native.__file__}) is not built from this checkout's csrc/; rebuild it: "
            "uv pip install --no-build-isolation --no-deps -ve . (docs/install.md)")


def xrt_tool(name: str) -> str | None:
    """An XRT command-line tool: on PATH, else in the directory of the XRT the module was built against."""
    found = shutil.which(name)
    if not found and XRT:
        root = _native.XRT_DIR
        found = shutil.which(name, path=os.pathsep.join((root, os.path.join(root, "bin"))))
    return found
