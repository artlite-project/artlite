"""The vision overlay's DDR image store on the host (``images.ImageStore`` over a stand-in device): sub-allocation from
chunks, a layout restored on another device, the shim addresses of a transaction found by image and rewritten for
other images."""

import numpy as np

from artlite.vision.images import ImageStore


class FakeDevice:
    """The overlay device's buffers by name: host arrays at fixed addresses."""

    def __init__(self) -> None:
        self.buffers: dict[str, np.ndarray] = {}

    def alloc(self, name: str, nbytes: int) -> None:
        self.buffers[name] = np.zeros(nbytes, np.uint8)

    def view(self, name: str) -> np.ndarray:
        return self.buffers[name]

    def address(self, name: str, offset: int = 0) -> int:
        return (1 << 40) * (1 + list(self.buffers).index(name)) + offset

    def flush(self, name: str, offset: int, nbytes: int) -> None:
        pass

    def fetch(self, name: str, offset: int, nbytes: int) -> None:
        pass


class Store(ImageStore):
    CHUNK = 1 << 16


def test_alloc_and_layout_restore() -> None:
    """Images land in chunks (a larger one in a buffer of its own); the layout rebuilds the same buffers and images on
    another device."""
    a = Store(FakeDevice())
    for name, n in (("a", 5000), ("b", 40000), ("c", 30000), ("big", 200000), ("a", 100)):
        a.alloc(name, n)
    a.write("b", np.arange(100, dtype=np.uint8), 8)
    assert a.names() == ["a", "b", "c", "big"] and a.span("a")[2] == 8192 and a.span("big")[0].startswith("_big")
    b = Store(FakeDevice())
    b.restore(a.layout())
    assert b.names() == a.names() and [b.address(n) for n in b.names()] == [a.address(n) for n in a.names()]
    assert not b.read("b", 100, 8).any()  # (restored images are zero: the owner writes them)
    b.alloc("d", 1000)  # (the restored device goes on in the newest chunk, where the first left off)
    a.alloc("d", 1000)
    assert b.span("d") == a.span("d")


def test_refs_resolve() -> None:
    """The shim addresses of a transaction by image, then the same words for other images of the layout."""
    s = Store(FakeDevice())
    for name in ("x.0", "x.1", "y"):
        s.alloc(name, 8192)
    words = np.zeros(20, np.uint32)
    addrs = [(2, s.address("x.0", 100)), (7, s.address("y", 4096)), (11, s.address("x.0", 8000))]
    for i, a in addrs:
        words[i], words[i + 1] = a & 0xFFFFFFFF, a >> 32
    refs = s.refs(addrs)
    assert refs[1] == ["x.0", "y"] and refs[2].tolist() == [[0, 100], [1, 4096], [0, 8000]]
    s.resolve(words, refs, lambda n: "x.1" if n == "x.0" else n)
    got = [int(words[i]) | int(words[i + 1]) << 32 for i, _ in addrs]
    assert got == [s.address("x.1", 100), s.address("y", 4096), s.address("x.1", 8000)]
