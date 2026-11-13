from collections.abc import Iterator

import pytest

from .fakes import ENGINES


@pytest.fixture(autouse=True)
def _stop_engines() -> Iterator[None]:
    """The engines the test's ``fakes.make`` started are stopped after it: a worker left running keeps pytest from
    exiting."""
    yield
    while ENGINES:
        e = ENGINES.pop()
        e.stop()
        e.join(5)
