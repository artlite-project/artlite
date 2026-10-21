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


def xrt_tool(name: str) -> str | None:
    """An XRT command-line tool: on PATH, else in the directory of the XRT the module was built against."""
    found = shutil.which(name)
    if not found and XRT:
        root = _native.XRT_DIR
        found = shutil.which(name, path=os.pathsep.join((root, os.path.join(root, "bin"))))
    return found
