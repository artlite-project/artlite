"""What the models' prefills share (``prefill.chunks.ChunkPrefill``) on the host: the chunk's input and output rows
of the residual stream, the padding check, the norm blocks, the layouts' keys."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.prefill import gemm as PG
from artlite.prefill.chunks import ChunkPrefill, norm_blocks

H = 24


class _Dev:
    """A device with a host-memory PX that records flushes and fetches."""

    def __init__(self, nbytes: int):
        self.px = np.full(nbytes, 0xAB, np.uint8)
        self.log: list[tuple] = []

    def view(self, name: str) -> np.ndarray:
        assert name == "PX"
        return self.px

    def flush(self, name: str, off: int, n: int) -> None:
        self.log.append(("flush", off, n))

    def fetch(self, name: str, off: int, n: int) -> None:
        self.log.append(("fetch", off, n))


class _Prefill(ChunkPrefill):
    SUB, SMALL, MAX_T, NAME, KV_PAD = 64, (16,), 512, "test", 448


def _prefill(sizes: tuple[int, ...] = (16, 64, 512), kv_pad: int = 448, positions: int = 4096) -> _Prefill:
    ov = SimpleNamespace(max_positions=positions, cfg=SimpleNamespace(hidden_size=H), image=None,
                         dev=_Dev(4 * 512 * H))
    cls = type("P", (_Prefill,), {"KV_PAD": kv_pad})
    return cls(ov, sizes)


@pytest.mark.parametrize("dtype", [bfloat16, np.float32])
def test_stage_and_read_rows(dtype: type) -> None:
    pf = _prefill()
    x = np.random.default_rng(0).standard_normal((10, H)).astype(np.float32) * 3
    pf._stage_px(x, 16, dtype)
    size = np.dtype(dtype).itemsize
    assert pf.ov.dev.log == [("flush", 0, size * 16 * H)]
    staged = pf.ov.dev.px[:size * 16 * H].view(dtype).reshape(16, H)
    assert (staged[:10].astype(np.float32) == x.astype(dtype).astype(np.float32)).all() and not staged[10:].any()
    assert (pf._px_rows(2, 5, dtype) == staged[2:7].astype(np.float32)).all()
    assert pf._px_rows(0, 1, dtype).dtype == np.float32 and pf.ov.dev.log[-1] == ("fetch", 0, size * H)
    pf._stage_px(x, 16, dtype, zero=np.array([1, 3]))
    assert not pf.ov.dev.px[:size * 16 * H].view(dtype).reshape(16, H)[[1, 3]].any()


def test_stage_matches_the_bf16_rounding_of_a_cast() -> None:
    pf = _prefill()
    x = np.random.default_rng(1).standard_normal((16, H)).astype(np.float32)
    pf._stage_px(x, 16, bfloat16)
    assert (pf.ov.dev.px[:2 * 16 * H].view(np.uint16).reshape(16, H) == x.astype(bfloat16).view(np.uint16)).all()


def test_kv_pad_covers_the_largest_chunk() -> None:
    _prefill(kv_pad=512 - 64)
    with pytest.raises(AssertionError):  # (a chunk of 512 slices a cache 448 positions past it)
        _prefill(kv_pad=512 - 65)
    _prefill(sizes=(16, 64), kv_pad=0)  # (the padding is for the largest chunk of this library)


def test_kv_pad_of_shared_offset_ranges() -> None:
    """Sizes sharing one offset parameter: a smaller chunk at the caches' end ranges over the largest chunk's."""
    shared = type("S", (_Prefill,), {"SHARED_KV_RANGE": True, "KV_PAD": 512 - 16})
    ov = SimpleNamespace(max_positions=4096, cfg=SimpleNamespace(hidden_size=H), image=None, dev=_Dev(0))
    shared(ov, (16, 64, 512))
    with pytest.raises(AssertionError):  # (the sizes' own offsets: 512 - 64 would do)
        type("S", (shared,), {"KV_PAD": 512 - 17})(ov, (16, 64, 512))
    type("S", (shared,), {"KV_PAD": 512 - 64})(ov, (64, 512))


def test_norm_blocks() -> None:
    assert norm_blocks(2560, 256, [2048, 1024, 512]) == [(0, 2048), (2048, 512)]
    assert norm_blocks(1280, 256, [1024, 512]) == [(0, 1024), (1024, 256)]
    assert norm_blocks(192, 64, [1024]) == [(0, 64), (64, 64), (128, 64)]


def test_layout_keys_share_operators() -> None:
    a, b = PG.Layout(512, {"q": 0, "k": 128}), PG.Layout(512, {"k": 128, "q": 0})
    assert a.key == b.key != PG.Layout(512, {"q": 0, "k": 256}).key
