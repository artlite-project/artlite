"""``OverlayModel._open``: the order of the load stages (the prefill compiles in a thread while the overlay loads, then the
drafters, the steps' transactions, the prefill's buffers, the wait for the weights). No NPU."""

from types import SimpleNamespace

from models.overlay_model import OverlayModel


class _Overlay:
    SLOTS = 2

    def __init__(self, log: list[str]):
        self.log = log

    def load(self, *, threads: int) -> None:
        self.log.append(f"load {threads}")

    def prepare(self, S: int, logits: bool = True) -> None:
        self.log.append(f"prepare {S} {logits}")

    def ready(self) -> None:
        self.log.append("ready")


class _Model(OverlayModel):
    causal = True

    def __init__(self, log: list[str], *, prefill: bool):
        self.overlay = _Overlay(log)
        self.prefill = SimpleNamespace(compile=lambda: self._compile(log), bind=lambda: log.append("bind")) \
            if prefill else None

    def _compile(self, log: list[str]) -> None:
        log.append("compile")

    def _load_drafters(self, threads: int) -> None:
        self.overlay.log.append("drafters")


def test_open_order_with_prefill() -> None:
    log: list[str] = []
    _Model(log, prefill=True)._open(threads=3)
    assert set(log[:2]) == {"compile", "load 3"}  # (concurrent: either first)
    assert log[2:] == ["drafters", "prepare 1 False", "prepare 1 True", "prepare 2 False", "prepare 2 True", "bind", "ready"]


def test_open_without_prefill() -> None:
    log: list[str] = []
    _Model(log, prefill=False)._open(threads=0)
    assert log == ["load 0", "drafters", "prepare 1 False", "prepare 1 True", "prepare 2 False", "prepare 2 True", "ready"]


def test_non_causal_model_prepares_no_steps() -> None:
    log: list[str] = []
    model = _Model(log, prefill=False)
    model.causal = False
    model._open(threads=0)
    assert log == ["load 0", "drafters", "ready"]
