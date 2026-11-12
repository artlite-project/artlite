"""Production code runs without torch (a dev dependency: the tests' CPU references): every module of ``artlite`` and
``models`` imports in a fresh interpreter whose import system refuses torch."""

import subprocess
import sys

_SCRIPT = r"""
import importlib
import importlib.abc
import pkgutil
import sys


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            raise ModuleNotFoundError(f"torch refused: {name}", name=name)
        return None


sys.meta_path.insert(0, NoTorch())
import artlite
import models

failed = []
names = []
for pkg in (artlite, models):
    names += [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".",
                                                     onerror=lambda n: failed.append(f"{n}: walk"))]
for name in names:
    try:
        importlib.import_module(name)
    except Exception as e:
        failed.append(f"{name}: {type(e).__name__}: {e}")
print(len(names), "modules")
print("\n".join(failed))
sys.exit(1 if failed else 0)
"""


def test_no_torch() -> None:
    r = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
