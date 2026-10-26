"""PNG writing (models.sd15.png): a written file parsed back chunk by chunk (no NPU)."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from models.sd15.png import SIGNATURE, encode_png, write_png


def read_png(data: bytes) -> np.ndarray:
    """An 8-bit truecolor, non-interlaced PNG's pixels (H, W, 3), its chunks' CRCs checked."""
    assert data[:8] == SIGNATURE
    pos, chunks = 8, []
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + body)
        chunks.append((tag, body))
        pos += 12 + n
    assert chunks[0][0] == b"IHDR" and chunks[-1] == (b"IEND", b"")
    w, h, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunks[0][1])
    assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
    raw = np.frombuffer(zlib.decompress(b"".join(b for t, b in chunks if t == b"IDAT")), np.uint8)
    rows = raw.reshape(h, 1 + w * 3)
    assert (rows[:, 0] == 0).all()  # (rows unfiltered)
    return rows[:, 1:].reshape(h, w, 3)


@pytest.mark.parametrize("shape", [(512, 512, 3), (3, 5, 3), (1, 1, 3)])
def test_round_trip(shape: tuple[int, int, int], tmp_path: Path) -> None:
    rgb = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    path = str(tmp_path / "image.png")
    write_png(path, rgb)
    with open(path, "rb") as f:
        data = f.read()
    assert data == encode_png(rgb)
    np.testing.assert_array_equal(read_png(data), rgb)
    # a strided view (a channel order flipped) is written as its values
    np.testing.assert_array_equal(read_png(encode_png(rgb[:, :, ::-1])), rgb[:, :, ::-1])


def test_rejects() -> None:
    for bad in (np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 3), np.float32)):
        with pytest.raises(ValueError):
            encode_png(bad)
