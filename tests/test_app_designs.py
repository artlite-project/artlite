"""The apps' full designs (``apps.py``), built as the apps build them (no NPU). They compile (program memory, stack and
BD limits of the whole models, which the layer-slice tests do not reach; Flint's overlay of 49 layers and its prefill's
chunks to 8192), a paged design's code pages link (each within the slot its core's resident code leaves:
``paging.link``, else only at load), their schedule ledger equals the committed baseline's rows (``schedule_ledger.py``:
the kernels' loop schedules, call cycles, code bytes and stack, the cores' program memory; a change of kernel code or of
the toolchain refreshes the baseline in its commit), and ``--dump-designs`` records them: the refactoring check covers
the apps without running the slow tests."""

import pytest

from . import schedule_ledger as sl
from .apps import APPS
from .model_files import require


@pytest.fixture(scope="module")
def baseline() -> sl.Ledger:
    return sl.Ledger.load(sl.BASELINE)


@pytest.mark.parametrize("app", list(APPS))
def test_app_designs(app: str, baseline: sl.Ledger) -> None:
    path, build = APPS[app]
    require(path)
    new = sl.ledger_of(app, build(path))
    if app not in baseline.apps:
        pytest.fail(f"{app} is not in {sl.BASELINE}: python -m tests.schedule_ledger record {app} --update")
    changes = sl.diff(baseline, new, [app])
    if changes:
        pytest.fail(f"{app}'s schedule ledger differs from the baseline (python -m tests.schedule_ledger record {app} "
                    f"--update, after checking the changes):\n{sl.report(changes, limit=15)}")
