"""The host environment the tools check at start: whether the install is the checkout's (``stale_install``) and, on
Windows, the NPU's power mode (``xrt-smi``)."""

import importlib.metadata
import os
import re
import subprocess
import threading
import tomllib
from collections.abc import Callable

from artlite import backend, portable


def stale_toolchain() -> str | None:
    """What differs when the installed AIE toolchain (mlir_aie, llvm_aie) is not the one the checkout's pyproject.toml
    pins (a ``git pull`` without updating the venv: designs compile with another toolchain, and a measurement is not
    this commit's), else None. Without a checkout (an installed wheel): the pins of the installed artlite."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "pyproject.toml")
    try:
        with open(path, "rb") as f:
            deps = tomllib.load(f)["project"]["dependencies"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        deps = importlib.metadata.requires("artlite") or []
    stale = []
    for dep in deps:
        if pin := re.match(r"(mlir_aie|llvm_aie)\s*==\s*([^\s;]+)", dep.strip()):
            try:
                have = importlib.metadata.version(pin[1])
            except importlib.metadata.PackageNotFoundError:
                have = "not installed"
            if have != pin[2]:
                stale.append(f"{pin[1]} {have} (pyproject.toml: {pin[2]})")
    if not stale:
        return None
    return (f"the installed AIE toolchain is not this checkout's: {', '.join(stale)}; update the venv: "
            "uv sync --no-dev (a development venv: uv sync; docs/install.md)")


def stale_install() -> str | None:
    """What of the install is not this checkout's (``stale_toolchain``, ``backend.stale_native``), else None."""
    return "; ".join(s for s in (stale_toolchain(), backend.stale_native()) if s) or None


def xrt_smi(*args: str) -> str | None:
    """Windows: the output of ``xrt-smi`` with ``args`` (the driver's copy, else XRT's); None without one, or when it
    fails."""
    smi = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "AMD", "xrt-smi.exe")
    smi = smi if os.path.exists(smi) else backend.xrt_tool("xrt-smi")
    try:
        return subprocess.run([smi, *args], capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, TypeError, subprocess.SubprocessError):
        return None


def windows_power_mode() -> str:
    """Windows: the NPU's power mode as ``xrt-smi`` tells it (``default``, ``performance``, ...; '' unknown)."""
    mode = re.search(r"Power Mode\s*:\s*(\w+)", xrt_smi("examine", "-r", "platform") or "")
    return mode[1].lower() if mode else ""


def performance_mode(report: Callable[[str], None]) -> None:
    """Windows: raise the NPU's power mode from ``default`` to ``performance`` (``xrt-smi configure --pmode``, as
    FastFlowLM does at its start), in a thread while the model loads; ``report``: told the change. A mode set by hand
    stays (``turbo`` among them). Not on Linux: its driver takes a power mode from root only, and in ``default`` it runs
    a context at the top DPM level."""
    if not portable.WINDOWS:
        return

    def run() -> None:  # (no xrt-smi, or it failed: the mode stays)
        if windows_power_mode() == "default" and xrt_smi("configure", "--pmode", "performance") is not None:
            report("NPU power mode: default -> performance (xrt-smi)")

    threading.Thread(target=run, name="power-mode", daemon=True).start()
